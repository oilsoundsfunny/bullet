use std::{
    cell::{Cell, RefCell},
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::PathBuf,
};

use anyhow::Context;
use bulletformat::ChessBoard;
use rand::{rng, Rng};
use structopt::StructOpt;
use viriformat::{
    chess::{board::{Board, movegen}, chessmove::Move, piece::PieceType, squareset::SquareSet},
    dataformat::{Filter, Game, WDL as Wdl},
};

fn see(board: &Board, m: Move, threshold: i32) -> bool {
    const SEE_PIECE_VALUES: [i32; 6] = [100, 300, 300, 500, 900, 0];

    fn estimate(board: &Board, m: Move) -> i32 {
        let mut v = board.piece_array[m.to()].map_or(0, |p| SEE_PIECE_VALUES[p.piece_type()]);
        if let Some(promo) = m.promotion_type() {
            v += SEE_PIECE_VALUES[promo] - SEE_PIECE_VALUES[0];
        } else if m.is_ep() {
            v = SEE_PIECE_VALUES[0];
        }
        v
    }

    let from = m.from();
    let to = m.to();
    let bbs = &board.pieces;

    let mut next_victim = m.promotion_type().unwrap_or_else(|| board.piece_array[from].unwrap().piece_type());
    let mut balance = estimate(board, m) - threshold;
    if balance < 0 {
        return false;
    }

    balance -= SEE_PIECE_VALUES[next_victim];
    if balance >= 0 {
        return true;
    }

    let diag_sliders = bbs.pieces[PieceType::Bishop] | bbs.pieces[PieceType::Queen];
    let orth_sliders = bbs.pieces[PieceType::Rook] | bbs.pieces[PieceType::Queen];

    let mut occupied = (bbs.occupied() ^ from.as_set()) | to.as_set();
    if m.is_ep() {
        occupied ^= board.ep_sq().unwrap().as_set();
    }

    let mut colour = board.turn().flip();
    let mut attackers = bbs.all_attackers_to_sq(to, occupied);

    loop {
        let my_attackers = attackers & bbs.colours[colour];
        if my_attackers == SquareSet::EMPTY {
            break;
        }

        for victim in PieceType::all() {
            next_victim = victim;
            if (my_attackers & bbs.pieces[victim]) != SquareSet::EMPTY {
                break;
            }
        }

        fn isolate_lsb(s: SquareSet) -> SquareSet {
            s & SquareSet::from_inner(s.inner().wrapping_neg())
        }

        occupied ^= isolate_lsb(my_attackers & bbs.pieces[next_victim]);

        if next_victim == PieceType::Pawn || next_victim == PieceType::Bishop || next_victim == PieceType::Queen {
            attackers |= movegen::bishop_attacks(to, occupied) & diag_sliders;
        }

        if next_victim == PieceType::Rook || next_victim == PieceType::Queen {
            attackers |= movegen::rook_attacks(to, occupied) & orth_sliders;
        }

        attackers &= occupied;
        colour = colour.flip();
        balance = -balance - 1 - SEE_PIECE_VALUES[next_victim];

        if balance >= 0 {
            if next_victim == PieceType::King && (attackers & bbs.colours[colour]) != SquareSet::EMPTY {
                colour = colour.flip();
            }
            break;
        }
    }

    board.turn() != colour
}

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
    acceptance.clamp(0.0, 1.0)
}

fn should_keep(board: &Board, mv: Move, eval: i32, wdl: Wdl) -> bool {
    const DEFAULT_VIRI_FILTER: Filter = Filter {
        min_ply: 16,
        min_pieces: 4,

        filter_tactical: false,
        filter_check: true,
        filter_castling: false,

        max_eval: 10000,
        max_eval_incorrectness: 2500,

        random_fen_skipping: false,
        random_fen_skip_probability: 0.0,

        wdl_filtered: false,
        wdl_model_params_a: [0.0; 4],
        wdl_model_params_b: [0.0; 4],
        material_min: 17,
        material_max: 78,
        mom_target: 58,
        wdl_heuristic_scale: 1.5,

        material_count_filtered: false,
        material_count_probabilities: [0.0; 33],
    };
    let mut rng = rng();

    !DEFAULT_VIRI_FILTER.should_filter(mv, eval as i32, board, wdl, &mut rng)
        && !(board.is_tactical(mv) && see(board, mv, 0))
        && rng.random_bool(piece_count_acceptance(board))
}

#[derive(StructOpt)]
pub struct SplatOptions {
    /// Path to input viriformat file.
    #[structopt(required = true)]
    pub input: PathBuf,
    /// Path to output bulletformat file.
    #[structopt(required = true)]
    pub output: PathBuf,
    /// Optional path to a viriformat filter config toml.
    pub cfg: PathBuf,
}

impl SplatOptions {
    pub fn run(&self) -> anyhow::Result<()> {
        println!("Reading from [{:#?}]", self.input);

        let input = File::open(&self.input).with_context(|| format!("Failed to open {}", self.input.display()))?;
        let output =
            File::create(&self.output).with_context(|| format!("Failed to create {}", self.output.display()))?;
        let bytes = input.metadata()?.len();

        let mut reader = BufReader::new(input);
        let mut writer = BufWriter::new(output);
        let mut games = 0usize;
        let mut positions = 0usize;

        let mut buffer = Vec::new();

        while let Ok(game) = Game::deserialise_from(&mut reader, buffer) {
            games += 1;
            positions += game.moves.len();

            game.splat_to_bulletformat_with_filter_callback(
                |bf_board| {
                    let bytes = unsafe { std::mem::transmute::<ChessBoard, [u8; 32]>(bf_board) };
                    writer.write_all(&bytes)?;
                    Ok(())
                },
                |mv, eval, board, wdl, _| { !should_keep(board, mv, eval, wdl) },
            )?;

            if games.is_multiple_of(16384) {
                print!("Splatted {games} games\r");
            }

            buffer = game.moves;
            buffer.clear();
        }

        println!();
        println!("Summary:");
        println!("Games = {games}");
        println!("Positions = {positions}");
        println!("Bytes per position = {}", bytes as f64 / positions as f64);

        Ok(())
    }
}
