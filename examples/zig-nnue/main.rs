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
    run::{DefaultDevice, HostPool, TrainingSchedule, TrainingSteps, train},
};

const NET_NAME: &str = "Alkaid";
const READ_BUF_MB: usize = 8192;
const READ_THREADS: usize = 8;
const MAP_THREADS: u8 = 8;
const SAVE_RATE: usize = 50;
const DATA_PATHS: &[&str] = &["data/moredata.vf", "data/5k1.vf", "data/5k2.vf", "data/20kdata.vf", "data/somedata.vf"];

const SUPERBATCHES_STAGE0: usize = 50;
const SUPERBATCHES_STAGE1: usize = 600;
const SUPERBATCHES_STAGE2: usize = 100;
const L1: usize = 1280;
const L2: usize = 16;
const L3: usize = 32;
const Q0: i16 = 255;
const Q1: i16 = 128;
const Q: i16 = 64;
const INPUT_BUCKETS: usize = get_num_buckets(&BUCKET_LAYOUT);
const OUTPUT_BUCKETS: usize = 8;

const FT_SHIFT: usize = 8;
const FT_SHIFT_SCALE: f32 = Q0 as f32 / ((1 << FT_SHIFT) as f32);
const I8_RANGE: f32 = i8::MAX as f32 / (Q1 as f32);
const L1_RANGE: f32 = I8_RANGE * FT_SHIFT_SCALE * FT_SHIFT_SCALE;

#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
     0,  1,  2,  3,
     4,  5,  6,  7,
     8,  8,  9,  9,
    10, 10, 11, 11,
    12, 12, 13, 13,
    12, 12, 13, 13,
    14, 14, 15, 15,
    14, 14, 15, 15,
];

fn main() {
    let psqt = ChessBucketsMirrored::new(BUCKET_LAYOUT);
    let output_buckets = MaterialCount::<OUTPUT_BUCKETS>;

    let inputs = ModelInputs::default()
        .add_sparse("stm/psqt", (psqt.num_inputs(), 1), psqt.max_active())
        .add_sparse("ntm/psqt", (psqt.num_inputs(), 1), psqt.max_active())
        .add_sparse("buckets", (OUTPUT_BUCKETS, 1), 1)
        .add_dense("targets", (1, 1));

    let defn = ModelDefinition::build(
        &inputs,
        |builder, (((stm_psqt, ntm_psqt), output_buckets), target)| {
            let l0f = builder.new_weights("l0/fac", (L1, 768), InitSettings::Zeroed);
            let psqt_init = InitSettings::Normal { mean: 0.0, stdev: (2f32 / 32.0).sqrt() };
            let mut l0_psqt = builder.new_weights("l0/psqt/w", (L1, psqt.num_inputs()), psqt_init);
            l0_psqt = l0_psqt + l0f.repeat(psqt.num_inputs() / 768);
            let l0_b = builder.new_weights("l0/psqt/b", (L1, 1), InitSettings::Zeroed);

            let l1 = builder.new_affine("l1/", L1, OUTPUT_BUCKETS * L2);
            let l2 = builder.new_affine("l2/", L2 * 2, OUTPUT_BUCKETS * L3);
            let l3 = builder.new_affine("l3/", L3, OUTPUT_BUCKETS);

            let ft = |psqt, start, end| {
                (l0_psqt.slice_rows(start, end).matmul(psqt) + l0_b.slice_rows(start, end)).crelu()
            };
            let stm_hidden = ft(stm_psqt, 0, L1 / 2) * ft(stm_psqt, L1 / 2, L1);
            let ntm_hidden = ft(ntm_psqt, 0, L1 / 2) * ft(ntm_psqt, L1 / 2, L1);
            let l0_out = stm_hidden.concat(ntm_hidden);

            let l0_out_norm = l0_out.reduce_sum_rows() / (L1 as f32);

            let l1_out = l1.forward(l0_out).select(output_buckets);
            let hl2 = l1_out.concat(l1_out.abs_pow(2.0)).crelu();

            let l2_out = l2.forward(hl2).select(output_buckets);
            let hl3 = l2_out.crelu();

            let l3_out = l3.forward(hl3).select(output_buckets);

            let loss = l3_out.sigmoid().squared_error(target);

            let loss = loss + 0.005 * l0_out_norm;

            (Some(loss.reduce_sum_batch()), vec![("output".to_string(), l3_out)])
        },
    );

    let weights = ModelWeights::new(&defn, 12412421);
    let device = DefaultDevice::new(0).unwrap();
    let params = AdamWParams::default();

    let mut evaluator = ModelEvaluator::new(&defn, device.clone()).unwrap();
    let mut optimiser = Optimiser::<_, AdamW<_>>::new(defn, weights, device.clone(), params).unwrap();

    let l0_clip = AdamWParams { max_weight: 0.99, min_weight: -0.99, ..Default::default() };
    optimiser.set_params_for_weight("l0/fac", l0_clip);
    optimiser.set_params_for_weight("l0/psqt/w", l0_clip);

    let l1_clip = AdamWParams { max_weight: L1_RANGE, min_weight: -L1_RANGE, ..Default::default() };
    optimiser.set_params_for_weight("l1/w", l1_clip);

    let saved_format = vec![
        SavedFormat::id("l0/psqt/w")
            .transform(|weights, values| {
                let fac = weights.get("l0/fac").values.f32().repeat(INPUT_BUCKETS);
                assert_eq!(values.len(), fac.len());
                values.iter().zip(fac).map(|(&a, b)| a + b).collect()
            })
            .round()
            .quantise::<i16>(Q0),
        SavedFormat::id("l0/psqt/b").round().quantise::<i16>(Q0),
        SavedFormat::id("l1/w")
            .transform(|_, values| values.iter().map(|f| f / (FT_SHIFT_SCALE * FT_SHIFT_SCALE)).collect())
            .round()
            .quantise::<i8>(Q1),
        SavedFormat::id("l1/b").round().quantise::<i32>(i32::from(Q) * 256),
        SavedFormat::id("l2/w").round().quantise::<i32>(i32::from(Q)),
        SavedFormat::id("l2/b").round().quantise::<i32>(i32::from(Q).pow(3)),
        SavedFormat::id("l3/w").round().quantise::<i32>(i32::from(Q)),
        SavedFormat::id("l3/b").round().quantise::<i32>(i32::from(Q).pow(4)),
    ];

    let reader =
        ViriBinpackLoader::new_concat_multiple(DATA_PATHS, READ_BUF_MB, READ_THREADS, ViriFilter::Custom(filter::should_keep));

    let params = (&inputs, psqt, output_buckets);

    let mut run = |stage, end_superbatch, lr_schedule, mapper| {
        train(
            &mut optimiser,
            TrainingSchedule {
                steps: TrainingSteps {
                    batch_size: 16_384 * 8,
                    batches_per_superbatch: 6104 / 8,
                    start_superbatch: 1,
                    end_superbatch,
                },
                lr_schedule,
                log_rate: 128,
                track_grad_norms: false,
            },
            ReadMapLoader::new(reader.clone(), mapper, MAP_THREADS),
            |_, _, _| {},
            |optimiser, step| {
                let superbatch = step.superbatch();
                if superbatch.is_multiple_of(SAVE_RATE) || superbatch == step.final_superbatch() {
                    let name = format!("{NET_NAME}-stage{stage}-{superbatch}");
                    save_to_checkpoint(optimiser, &saved_format, &format!("checkpoints/{name}"));
                    println!("Saved [{name}]");
                }
            },
        )
        .unwrap();
    };

    const WARMUP_SBS: usize = SUPERBATCHES_STAGE0 / 2;
    const COOLDOWN_SBS: usize = SUPERBATCHES_STAGE0 - WARMUP_SBS;
    run(
        0,
        SUPERBATCHES_STAGE0,
        lr::Sequence {
            first: lr::LinearDecayLR { initial_lr: 1e-4, final_lr: 5e-3, final_superbatch: WARMUP_SBS },
            second: lr::LinearDecayLR { initial_lr: 5e-3, final_lr: 1e-4, final_superbatch: COOLDOWN_SBS },
            first_scheduler_final_superbatch: WARMUP_SBS,
        }
        .boxed(),
        inputs::make_inputs_mapper(params, wdl::ConstantWDL { value: 0.2 }),
    );

    run(
        1,
        SUPERBATCHES_STAGE1,
        lr::LinearDecayLR { initial_lr: 1e-3, final_lr: 1e-6, final_superbatch: SUPERBATCHES_STAGE1 }.boxed(),
        inputs::make_inputs_mapper(params, wdl::LinearWDL { start: 0.2, end: 0.5 }),
    );

    run(
        2,
        SUPERBATCHES_STAGE2,
        lr::LinearDecayLR { initial_lr: 1e-5, final_lr: 1e-7, final_superbatch: SUPERBATCHES_STAGE2 }.boxed(),
        inputs::make_inputs_mapper(params, wdl::ConstantWDL { value: 0.5 }),
    );

    evaluator.load_device_weights(optimiser.weights()).unwrap();
    let evaluator_mapper = inputs::make_inputs_mapper(params, wdl::ConstantWDL { value: 0.0 });

    let pool = HostPool::new(device.clone());

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/P2P2PP/q2Q1R1K w kq - 0 2",
        "rn1qkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQka - 0 1",
    ] {
        let pos = format!("{fen} | 0 | 0.0").parse().unwrap();
        let inputs = evaluator_mapper.map(&pool, &[pos], Default::default(), 1).unwrap().to_device(&device).unwrap();
        let output = evaluator.evaluate(&inputs).unwrap().get("output").unwrap();
        let [value] = output.to_host().unwrap().f32()[..] else { panic!() };
        println!("FEN: {fen}");
        println!("EVAL: {}", 400.0 * value);
    }
}
