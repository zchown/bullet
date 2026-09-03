mod filter;
mod inputs;

use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, SparseInputType, get_num_buckets},
        outputs::MaterialCount,
    },
    trainer::schedule::{
        lr::{self, LrScheduler},
        wdl,
    },
    value::{
        loader::{ViriBinpackLoader, viribinpack::ViriFilter},
        save::save_to_checkpoint,
    },
};
use bullet_trainer::{
    model::{InitSettings, ModelDefinition, ModelEvaluator, ModelInputs, ModelWeights, SavedFormat},
    optimiser::{
        Optimiser,
        adam::{AdamW, AdamWParams},
    },
    reader::ReadMapLoader,
    run::{DefaultDevice, TrainingSchedule, TrainingSteps, train},
};

const NET_NAME: &str = "zig-nnue";
const CHECKPOINT_DIR: &str = "/kaggle/working";

const READ_BUF_MB: usize = 2048;
const READ_THREADS: usize = 4;
const MAP_THREADS: u8 = 4;
const SAVE_RATE: usize = 10;

const HIDDEN_SIZE: usize = 1536;
pub const NUM_OUTPUT_BUCKETS: usize = 8;
const QA: i16 = 255;
const QB: i16 = 64;

#[rustfmt::skip]
const KING_BUCKETS: [usize; 32] = [
     0,  1,  2,  3,
     4,  5,  6,  7,
     8,  8,  9,  9,
    10, 10, 11, 11,
    12, 12, 13, 13,
    12, 12, 13, 13,
    14, 14, 15, 15,
    14, 14, 15, 15,
];

const NUM_KING_BUCKETS: usize = get_num_buckets(&KING_BUCKETS);
const _: () = assert!(NUM_KING_BUCKETS == 16);

const SUPERBATCHES_STAGE0: usize = 25;
const SUPERBATCHES_STAGE1: usize = 250;
const SUPERBATCHES_STAGE2: usize = 50;

const WARMUP_SBS: usize = SUPERBATCHES_STAGE0 / 2;
const COOLDOWN_SBS: usize = SUPERBATCHES_STAGE0 - WARMUP_SBS;

const BASE_LR: f32 = 1e-3;
const WARMUP_PEAK_LR: f32 = 2e-3;
const WARMUP_FLOOR_LR: f32 = 5e-5;

/// Weight clip for the feature transformer. Halved from the usual 1.98 because
/// the effective FT weight is now the sum of two clipped tensors (l0w + l0f),
/// so each half has to fit in half the range.
const FT_CLIP: f32 = 0.99;

struct Args {
    data: Vec<String>,
    tune: Vec<String>,
}

fn parse_args() -> Args {
    let mut argv = std::env::args();
    let program = argv.next().unwrap_or_else(|| NET_NAME.to_string());

    let mut data = Vec::new();
    let mut tune = Vec::new();
    let mut current_list = None;

    for arg in argv {
        match arg.as_str() {
            "--data" => current_list = Some(1),
            "--tune" => current_list = Some(2),
            _ => {
                match current_list {
                    Some(1) => data.push(arg),
                    Some(2) => tune.push(arg),
                    _ => {
                        eprintln!("usage: {program} --data <file1> ... --tune <file1> ...");
                        eprintln!("Error: specify --data or --tune before providing file paths.");
                        std::process::exit(1);
                    }
                }
            }
        }
    }

    if data.is_empty() || tune.is_empty() {
        eprintln!("usage: {program} --data <file1> ... --tune <file1> ...");
        eprintln!("Error: You must provide at least one file for both --data and --tune.");
        std::process::exit(1);
    }

    if data.len() > 8 || tune.len() > 8 {
        eprintln!("Error: A maximum of 8 files are allowed per category.");
        eprintln!("Provided: {} data files, {} tune files", data.len(), tune.len());
        std::process::exit(1);
    }

    for path in data.iter().chain(tune.iter()) {
        if !std::path::Path::new(path).is_file() {
            eprintln!("not a readable file: {path}");
            std::process::exit(1);
        }
    }

    Args { data, tune }
}

fn main() {
    let args = parse_args();
    let data_paths: Vec<&str> = args.data.iter().map(String::as_str).collect();
    let tune_paths: Vec<&str> = args.tune.iter().map(String::as_str).collect();

    println!("train data: {data_paths:?}");
    println!("tune data:  {tune_paths:?}");

    let king_buckets = ChessBucketsMirrored::new(KING_BUCKETS);
    let output_buckets = MaterialCount::<NUM_OUTPUT_BUCKETS>;

    let l0_inputs = king_buckets.num_inputs();
    let l0_max_active = king_buckets.max_active();

    let inputs = ModelInputs::default()
        .add_sparse("stm", (l0_inputs, 1), l0_max_active)
        .add_sparse("ntm", (l0_inputs, 1), l0_max_active)
        .add_sparse("buckets", (NUM_OUTPUT_BUCKETS, 1), 1)
        .add_dense("targets", (1, 1));

    let defn = ModelDefinition::build(&inputs, |builder, (((stm, ntm), buckets), target)| {
        // Factorised feature transformer.
        //
        // l0w is the usual (HIDDEN, 768 * 10) bucketed matrix. l0f is a single
        // unbucketed (HIDDEN, 768) matrix tiled across all ten buckets, so every
        // position trains it regardless of where its king sits — that's what
        // stops the rare buckets learning from noise. l0f is zero-initialised,
        // so step 0 is numerically identical to the unfactorised net.
        //
        // l0f is a training-time device only: it gets folded into l0w in the
        // save format below, so the exported net has exactly the same shape and
        // inference cost as before.
        let ft_init = InitSettings::Normal { mean: 0.0, stdev: (2f32 / 32.0).sqrt() };
        let l0f = builder.new_weights("l0f", (HIDDEN_SIZE, 768), InitSettings::Zeroed);
        let l0w = builder.new_weights("l0w", (HIDDEN_SIZE, l0_inputs), ft_init)
            + l0f.repeat(NUM_KING_BUCKETS);
        let l0b = builder.new_weights("l0b", (HIDDEN_SIZE, 1), InitSettings::Zeroed);

        let ft = |x| (l0w.matmul(x) + l0b).screlu();

        // If the `+ l0b` above doesn't type-check, swap the four lines starting
        // at `let l0f` for the block below. Same maths, only ops the `advanced`
        // example already uses, at the cost of a second sparse matmul per
        // perspective (noticeably slower — the FT dominates training time):
        //
        //     let l0 = builder.new_affine("l0", l0_inputs, HIDDEN_SIZE);
        //     let l0f = builder.new_weights("l0f", (HIDDEN_SIZE, 768), InitSettings::Zeroed);
        //     let l0fr = l0f.repeat(NUM_KING_BUCKETS);
        //     let ft = |x| (l0.forward(x) + l0fr.matmul(x)).screlu();
        //
        // (W + F)x + b == Wx + b + Fx, and `new_affine` names its weights "l0w"
        // and "l0b", so everything downstream is unchanged either way.

        let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, NUM_OUTPUT_BUCKETS);

        let stm_hidden = ft(stm);
        let ntm_hidden = ft(ntm);
        let hidden_layer = stm_hidden.concat(ntm_hidden);

        let out = l1.forward(hidden_layer).select(buckets);

        let loss = out.sigmoid().squared_error(target);

        (Some(loss.reduce_sum_batch()), vec![("output".to_string(), out)])
    });

    let weights = ModelWeights::new(&defn, 12412421);
    let device = DefaultDevice::new(0).unwrap();

    let mut evaluator = ModelEvaluator::new(&defn, device.clone()).unwrap();
    let mut optimiser =
        Optimiser::<_, AdamW<_>>::new(defn, weights, device.clone(), AdamWParams::default()).unwrap();

    let ft_clip = AdamWParams { max_weight: FT_CLIP, min_weight: -FT_CLIP, ..Default::default() };
    optimiser.set_params_for_weight("l0w", ft_clip);
    optimiser.set_params_for_weight("l0f", ft_clip);

    let l1_clip = AdamWParams { max_weight: 1.98, min_weight: -1.98, ..Default::default() };
    optimiser.set_params_for_weight("l1w", l1_clip);

    let saved_format = vec![
        // Fold the factoriser into each bucket on the way out. Without this
        // transform you'd export only the bucket-specific half and the net
        // would be silently wrong rather than fail loudly.
        SavedFormat::id("l0w")
            .transform(|weights, values| {
                let fac = weights.get("l0f").values.f32().repeat(NUM_KING_BUCKETS);
                assert_eq!(values.len(), fac.len());
                values.iter().zip(fac).map(|(&a, b)| a + b).collect()
            })
            .round()
            .quantise::<i16>(QA),
        SavedFormat::id("l0b").round().quantise::<i16>(QA),
        SavedFormat::id("l1w").round().quantise::<i16>(QB).transpose(),
        SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
    ];

    let all_data = ViriBinpackLoader::new_concat_multiple(
        &data_paths,
        READ_BUF_MB,
        READ_THREADS,
        ViriFilter::Custom(filter::should_keep),
    );

    let tune_data = ViriBinpackLoader::new_concat_multiple(
        &tune_paths,
        READ_BUF_MB,
        READ_THREADS,
        ViriFilter::Custom(filter::should_keep),
    );

    let params = (&inputs, king_buckets, output_buckets);

    let mut run = |stage, end_superbatch, lr_schedule, mapper, reader| {
        train(
            &mut optimiser,
            TrainingSchedule {
                steps: TrainingSteps {
                    batch_size: 16_384,
                    batches_per_superbatch: 8192,
                    start_superbatch: 1,
                    end_superbatch,
                },
                lr_schedule,
                log_rate: 128,
            },
            ReadMapLoader::new(reader, mapper, MAP_THREADS),
            |_, _, _| {},
            |optimiser, step| {
                let superbatch = step.superbatch();
                if superbatch.is_multiple_of(SAVE_RATE) || superbatch == step.final_superbatch() {
                    let name = format!("{NET_NAME}-stage{stage}-{superbatch}");
                    save_to_checkpoint(optimiser, &saved_format, &format!("{CHECKPOINT_DIR}/{name}"));
                    println!("Saved [{name}]");
                }
            },
        )
        .unwrap();
    };

    // stage 0: warmup + cooldown
    run(
        0,
        SUPERBATCHES_STAGE0,
        lr::Sequence {
            first: lr::LinearDecayLR {
                initial_lr: WARMUP_FLOOR_LR,
                final_lr: WARMUP_PEAK_LR,
                final_superbatch: WARMUP_SBS,
            },
            second: lr::LinearDecayLR {
                initial_lr: WARMUP_PEAK_LR,
                final_lr: WARMUP_FLOOR_LR,
                final_superbatch: COOLDOWN_SBS,
            },
            first_scheduler_final_superbatch: WARMUP_SBS,
        }
        .boxed(),
        inputs::make_inputs_mapper(params, wdl::ConstantWDL { value: 0.3 }),
        all_data.clone(),
    );

    run(
        1,
        SUPERBATCHES_STAGE1,
        lr::LinearDecayLR { initial_lr: BASE_LR, final_lr: 1e-6, final_superbatch: SUPERBATCHES_STAGE1 }.boxed(),
        inputs::make_inputs_mapper(params, wdl::LinearWDL { start: 0.0, end: 0.5 }),
        all_data.clone(),
    );

    run(
        2,
        SUPERBATCHES_STAGE2,
        lr::LinearDecayLR { initial_lr: 1e-6, final_lr: 1e-8, final_superbatch: SUPERBATCHES_STAGE2 }.boxed(),
        inputs::make_inputs_mapper(params, wdl::ConstantWDL { value: 0.7 }),
        tune_data.clone(),
    );

    // sanity check
    evaluator.load_device_weights(optimiser.weights()).unwrap();
    let evaluator_mapper = inputs::make_inputs_mapper(params, wdl::ConstantWDL { value: 0.0 });

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
    ] {
        let pos = format!("{fen} | 0 | 0.0").parse().unwrap();
        let model_inputs = evaluator_mapper.map(&[pos], Default::default(), 1).to_device(&device).unwrap();
        let output = evaluator.evaluate(&model_inputs).unwrap().get("output").unwrap();
        let [value] = output.to_host().unwrap().f32()[..] else { panic!() };
        println!("FEN:  {fen}");
        println!("EVAL: {}", inputs::EVAL_SCALE * value);
    }
}
