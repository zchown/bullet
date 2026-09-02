use bullet_lib::{
    game::{
        formats::bulletformat::ChessBoard,
        inputs::{ChessBucketsMirrored, SparseInputType},
        outputs::OutputBuckets,
    },
    wdl::WdlScheduler,
};
use bullet_trainer::model::{DenseInput, ModelInputs, ModelInputsMapper, SparseInput};

// stm, ntm, output bucket, target.
pub type InputTy = (((SparseInput, SparseInput), SparseInput), DenseInput<f32>);

pub const EVAL_SCALE: f32 = 128.0;

pub fn make_inputs_mapper(
    params: (&ModelInputs<InputTy>, ChessBucketsMirrored, impl OutputBuckets<ChessBoard>),
    wdl: impl WdlScheduler,
) -> ModelInputsMapper<ChessBoard> {
    ModelInputsMapper::build(params.0, move |pos, step, (((stm, ntm), bucket), target)| {
        let mut cnt = 0;
        params.1.map_features(pos, |stm_feat, ntm_feat| {
            stm[cnt] = stm_feat.try_into().unwrap();
            ntm[cnt] = ntm_feat.try_into().unwrap();
            cnt += 1;
        });

        if cnt < params.1.max_active() {
            stm[cnt] = -1;
            ntm[cnt] = -1;
        }

        bucket[0] = i32::from(params.2.bucket(pos));

        let result = f32::from(pos.result) / 2.0;
        let score = 1.0 / (1.0 + (f32::from(-pos.score) / EVAL_SCALE).exp());
        let lambda = wdl.blend(step.batch(), step.superbatch(), step.final_superbatch());
        assert!((0.0..=1.0).contains(&lambda), "WDL lambda must be in [0, 1]");
        target[0] = lambda * result + (1. - lambda) * score;
    })
}
