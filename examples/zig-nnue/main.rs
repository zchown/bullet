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
    model::{ModelDefinition, ModelEvaluator, ModelInputs, ModelWeights, SavedFormat},
    optimiser::{
        Optimiser,
        adam::{AdamW, AdamWParams},
    },
    reader::ReadMapLoader,
    run::{DefaultDevice, TrainingSchedule, TrainingSteps, train},
};

const NET_NAME: &str = "zig-nnue";
const CHECKPOINT_DIR: &str = "/checkpoints";

const READ_BUF_MB: usize = 2048;
const READ_THREADS: usize = 4;
const MAP_THREADS: u8 = 4;
const SAVE_RATE: usize = 10;

const HIDDEN_SIZE: usize = 1536;
pub const NUM_OUTPUT_BUCKETS: usize = 8;
const QA: i16 = 255;
const QB: i16 = 64;

#[rustfmt::skip]
pub const KING_BUCKETS: [usize; 32] = [
    0, 1, 2, 3,
    4, 4, 5, 5,
    6, 6, 6, 6,
    7, 7, 7, 7,
    8, 8, 8, 8,
    8, 8, 8, 8,
    9, 9, 9, 9,
    9, 9, 9, 9,
];

const NUM_KING_BUCKETS: usize = get_num_buckets(&KING_BUCKETS);
const _: () = assert!(NUM_KING_BUCKETS == 10);

// stage 0: LR warmup + cooldown, low WDL. Gets the feature transformer out of
//          its random init without an early loss spike, and lets the optimiser
//          moments settle before the long run.
// stage 1: the bulk of training. LR decays across the whole stage, WDL ramps
//          from eval-heavy to result-heavy (your old 0.4 -> 0.8).
// stage 2: low-LR anneal at fixed high WDL. Cheap and usually worth 5-15 Elo.

const SUPERBATCHES_STAGE0: usize = 50;
const SUPERBATCHES_STAGE1: usize = 300;
const SUPERBATCHES_STAGE2: usize = 100;

const WARMUP_SBS: usize = SUPERBATCHES_STAGE0 / 2;
const COOLDOWN_SBS: usize = SUPERBATCHES_STAGE0 - WARMUP_SBS;

const BASE_LR: f32 = 1e-3;
const WARMUP_PEAK_LR: f32 = 2e-3;
const WARMUP_FLOOR_LR: f32 = 5e-5;

struct Args {
    data: [String; 3],
    tune: String,
}

fn parse_args() -> Args {
    let mut argv = std::env::args();
    let program = argv.next().unwrap_or_else(|| NET_NAME.to_string());
    let rest: Vec<String> = argv.collect();

    let [d0, d1, d2, tune] = <[String; 4]>::try_from(rest).unwrap_or_else(|rest| {
        eprintln!("usage: {program} <data0> <data1> <data2> <tune>");
        eprintln!("  data0..2  main training binpacks (stages 0 and 1)");
        eprintln!("  tune      fine-tune binpack (stage 2)");
        eprintln!("expected 4 paths, got {}", rest.len());
        std::process::exit(1);
    });

    for path in [&d0, &d1, &d2, &tune] {
        if !std::path::Path::new(path).is_file() {
            eprintln!("not a readable file: {path}");
            std::process::exit(1);
        }
    }

    Args { data: [d0, d1, d2], tune }
}

fn main() {
    let args = parse_args();
    let data_paths: [&str; 3] = args.data.each_ref().map(String::as_str);
    let tune_paths: [&str; 1] = [args.tune.as_str()];

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
        let l0 = builder.new_affine("l0", l0_inputs, HIDDEN_SIZE);
        let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, NUM_OUTPUT_BUCKETS);

        let stm_hidden = l0.forward(stm).screlu();
        let ntm_hidden = l0.forward(ntm).screlu();
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

    let clip = AdamWParams { max_weight: 1.98, min_weight: -1.98, ..Default::default() };
    optimiser.set_params_for_weight("l0w", clip);
    optimiser.set_params_for_weight("l1w", clip);

    let saved_format = vec![
        SavedFormat::id("l0w").round().quantise::<i16>(QA),
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
        inputs::make_inputs_mapper(params, wdl::ConstantWDL { value: 0.4 }),
        all_data.clone(),
    );

    run(
        1,
        SUPERBATCHES_STAGE1,
        lr::LinearDecayLR { initial_lr: BASE_LR, final_lr: 1e-6, final_superbatch: SUPERBATCHES_STAGE1 }.boxed(),
        inputs::make_inputs_mapper(params, wdl::LinearWDL { start: 0.4, end: 0.8 }),
        all_data.clone(),
    );

    run(
        2,
        SUPERBATCHES_STAGE2,
        lr::LinearDecayLR { initial_lr: 1e-5, final_lr: 1e-7, final_superbatch: SUPERBATCHES_STAGE2 }.boxed(),
        inputs::make_inputs_mapper(params, wdl::ConstantWDL { value: 0.9 }),
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
