use std::{
    cell::{Cell, RefCell},
    sync::LazyLock,
};

use bullet_lib::game::formats::viriformat::{
    chess::{board::Board, chessmove::Move},
    dataformat::{Filter, WDL},
};
use rand::{Rng, rng};

const REBALANCE_PIECE_COUNTS: bool = false;

static BASE_FILTER: LazyLock<Filter> = LazyLock::new(|| Filter {
    min_ply: 16,
    min_pieces: 4,
    filter_check: true,
    filter_tactical: true,
    filter_castling: true,
    max_eval: 15000,
    max_eval_incorrectness: 2500,
    random_fen_skipping: true,
    random_fen_skip_probability: 0.15, 
    ..Filter::default()
});

fn piece_count_acceptance(board: &Board) -> f64 {
    #[rustfmt::skip]
    const DESIRED_DISTRIBUTION: [f64; 33] = [
        0.018411966423, 0.020641545085, 0.022727271053,
        0.024669162740, 0.026467201733, 0.028121406444,
        0.029631758462, 0.030998276198, 0.032220941240,
        0.033299772000, 0.034234750067, 0.035025893853,
        0.035673184944, 0.036176641754, 0.036536245870,
        0.036752015705, 0.036823932846, 0.036752015705,
        0.036536245870, 0.036176641754, 0.035673184944,
        0.035025893853, 0.034234750067, 0.033299772000,
        0.032220941240, 0.030998276198, 0.029631758462,
        0.028121406444, 0.026467201733, 0.024669162740,
        0.022727271053, 0.020641545085, 0.018411966423,
    ];

    thread_local! {
        static PIECE_COUNT_STATS: RefCell<[u64; 33]> = const { RefCell::new([0; 33]) };
        static PIECE_COUNT_TOTAL: Cell<u64> = const { Cell::new(0) };
    }

    let pc = board.pieces.occupied().count() as usize;
    let count = PIECE_COUNT_STATS.with_borrow_mut(|stats| {
        stats[pc] += 1;
        stats[pc]
    });
    let total = PIECE_COUNT_TOTAL.with(|t| {
        let total = t.get() + 1;
        t.set(total);
        total
    });
    let frequency = count as f64 / total as f64;

    let acceptance = 0.5 * DESIRED_DISTRIBUTION[pc] / frequency;
    acceptance.clamp(0., 1.)
}

pub fn should_keep(board: &Board, mv: Move, eval: i16, wdl: f32) -> bool {
    let mut rng = rng();

    let wdl = match wdl {
        1.0 => WDL::Win,
        0.5 => WDL::Draw,
        0.0 => WDL::Loss,
        _ => unreachable!(),
    };

    if BASE_FILTER.should_filter(mv, eval as i32, board, wdl, &mut rng) {
        return false;
    }

    if REBALANCE_PIECE_COUNTS && !rng.random_bool(piece_count_acceptance(board)) {
        return false;
    }

    true
}
