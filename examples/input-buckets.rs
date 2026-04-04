use acyclib::trainer::logger::ansi;
use bullet_lib::{
    game::{
        inputs::{get_num_buckets, ChessBucketsMirrored},
        outputs::OutputBuckets,
    },
    nn::{
        optimiser::{Ranger, RangerParams},
        InitSettings, Shape,
    },
    trainer::{
        save::SavedFormat,
        schedule::{
            lr::{CosineDecayLR as CosLr, LinearDecayLR as LinearLr, LrScheduler},
            wdl::{ConstantWDL as ConstWdl, LinearWDL as LinearWdl, Sequence as WdlSequence},
            TrainingSchedule, TrainingSteps,
        },
        settings::LocalSettings,
    },
    value::{
        loader::{viribinpack::ViriFilter, ViriBinpackLoader},
        ValueTrainerBuilder,
    },
};
use bulletformat::ChessBoard;
use rand::{rng, Rng};
use std::{
    mem::MaybeUninit,
    ops::Div,
    sync::atomic::{AtomicU64, Ordering},
};
use viriformat::{
    chess::{
        board::{movegen, Board},
        chessmove::Move,
        piece::PieceType,
        squareset::SquareSet,
    },
    dataformat::{Filter, WDL as Result},
};

#[derive(Clone, Copy, Default)]
pub struct CjBuckets;
impl OutputBuckets<ChessBoard> for CjBuckets {
    const BUCKETS: usize = 8;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        let p = pos.occ().count_ones();
        let n = (63 - p) * (32 - p);
        n.div(225).min(7) as u8
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OneCycleLr {
    pub max_lr: f32,
    pub div: f32,
    pub final_div: f32,
    pub warmup_pct: f32,
    pub anneal_cos: bool,
    pub three_phase: bool,
    pub final_superbatch: usize,
}

impl OneCycleLr {
    fn initial_lr(&self) -> f32 {
        self.max_lr / self.div
    }

    fn final_lr(&self) -> f32 {
        self.initial_lr() / self.final_div
    }

    fn warmup_sb(&self) -> usize {
        (self.final_superbatch as f32 * self.warmup_pct) as usize
    }

    fn annealing_sb(&self) -> usize {
        self.final_superbatch - self.warmup_sb() - self.annihilation_sb()
    }

    fn annihilation_sb(&self) -> usize {
        if self.three_phase {
            self.final_superbatch - self.warmup_sb() * 2
        } else {
            0
        }
    }
}

impl LrScheduler for OneCycleLr {
    fn lr(&self, batch: usize, sb: usize) -> f32 {
        let warmup = self.warmup_sb();
        let anneal = self.annealing_sb();

        if sb <= warmup {
            let inner =
                CosLr { initial_lr: self.initial_lr(), final_lr: self.max_lr, final_superbatch: warmup * BATCHES };
            inner.lr(1, (sb - 1) * BATCHES + batch)
        } else if sb <= warmup + anneal && self.anneal_cos {
            let inner = CosLr {
                initial_lr: self.max_lr,
                final_lr: if self.three_phase { self.initial_lr() } else { self.final_lr() },
                final_superbatch: anneal * BATCHES,
            };
            inner.lr(1, (sb - warmup - 1) * BATCHES + batch)
        } else if sb <= warmup + anneal {
            let inner = LinearLr {
                initial_lr: self.max_lr,
                final_lr: if self.three_phase { self.initial_lr() } else { self.final_lr() },
                final_superbatch: anneal * BATCHES,
            };
            inner.lr(1, (sb - warmup - 1) * BATCHES + batch)
        } else if self.three_phase {
            let inner = CosLr {
                initial_lr: self.initial_lr(),
                final_lr: self.final_lr(),
                final_superbatch: self.annihilation_sb() * BATCHES,
            };
            inner.lr(1, (sb - warmup - anneal - 1) * BATCHES + batch)
        } else {
            self.final_lr()
        }
    }

    fn colourful(&self) -> String {
        format!(
            "1cycle(\
            initial = {}, max = {}, final = {}, \
            warmup = {} sb, annealing = {} sb, annihilation = {} sb)",
            ansi(self.initial_lr(), 31),
            ansi(self.max_lr, 31),
            ansi(self.final_lr(), 31),
            ansi(self.warmup_sb(), 31),
            ansi(self.annealing_sb(), 31),
            ansi(self.annihilation_sb(), 31),
        )
    }
}

const SEE_PIECE_VALUES: [i32; 6] = [100, 300, 300, 500, 900, 0];

const HL_SIZE: usize = 1024;
#[rustfmt::skip]
const INPUT_BUCKETS: [usize; 32] = [
     0,  1,  2,  3,
     4,  5,  6,  7,
     8,  9, 10, 11,
     8,  9, 10, 11,
    12, 12, 13, 13,
    12, 12, 13, 13,
    14, 14, 15, 15,
    14, 14, 15, 15,
];
const INPUT_BUCKETS_NUM: usize = get_num_buckets(&INPUT_BUCKETS);
const OUTPUT_BUCKETS_NUM: usize = CjBuckets::BUCKETS;

const BATCH_SIZE: usize = 16384 * 8;
const BATCHES: usize = 16384 * 6144 / BATCH_SIZE;
const SUPERBATCHES: usize = 200;
const CHECKPOINT: usize = 0;

const QA: i16 = 255;
const QB: i16 = 64;
const SCALE: f32 = 400.0;

fn estimated_see(board: &Board, m: Move) -> i32 {
    let mut value = board.piece_array[m.to()].map_or(0, |p| SEE_PIECE_VALUES[p.piece_type()]);

    if let Some(promo) = m.promotion_type() {
        value += SEE_PIECE_VALUES[promo] - SEE_PIECE_VALUES[PieceType::Pawn];
    } else if m.is_ep() {
        value = SEE_PIECE_VALUES[PieceType::Pawn];
    }

    value
}

fn static_exchange_eval(board: &Board, m: Move, threshold: i32) -> bool {
    let from = m.from();
    let to = m.to();
    let bbs = &board.pieces;

    let mut next_victim = m.promotion_type().unwrap_or_else(|| board.piece_array[from].unwrap().piece_type());

    let mut balance = estimated_see(board, m) - threshold;

    if balance < 0 {
        return false;
    }

    balance -= SEE_PIECE_VALUES[next_victim];

    if balance >= 0 {
        return true;
    }

    let diag_sliders = bbs.pieces[PieceType::Queen] | bbs.pieces[PieceType::Bishop];
    let orth_sliders = bbs.pieces[PieceType::Queen] | bbs.pieces[PieceType::Rook];

    let mut occupied = bbs.occupied();
    occupied ^= from.as_set();
    occupied |= to.as_set();
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
    const TARGET_DISTR: [f64; 33] = [
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

    static STATS: [AtomicU64; 33] = {
        let mut a: [MaybeUninit<AtomicU64>; 33] = [const { MaybeUninit::uninit() }; 33];
        let mut i: usize = 0;
        while i < 33 {
            a[i].write(AtomicU64::new(0));
            i += 1;
        }
        unsafe { std::mem::transmute::<_, [AtomicU64; 33]>(a) }
    };
    static TOTAL: AtomicU64 = AtomicU64::new(0);

    let pc = board.pieces.occupied().count() as usize;
    let n = STATS[pc].fetch_add(1, Ordering::Relaxed) + 1;
    let d = TOTAL.fetch_add(1, Ordering::Relaxed) + 1;

    let f = n as f64 / d as f64;
    (0.5 * TARGET_DISTR[pc] / f).clamp(0., 1.)
}

fn filter(board: &Board, mv: Move, eval: i16, wdl: f32) -> bool {
    let default_filter = Filter {
        max_eval: 32000,
        filter_tactical: false,
        random_fen_skipping: true,
        random_fen_skip_probability: 0.5,
        ..Default::default()
    };
    let wdl = match wdl {
        1.0 => Result::Win,
        0.5 => Result::Draw,
        0.0 => Result::Loss,
        _ => unreachable!(),
    };
    let mut rng = rng();

    fn see_filter(board: &Board, mv: Move) -> bool {
        board.is_tactical(mv) && static_exchange_eval(board, mv, 0)
    }

    !default_filter.should_filter(mv, eval as i32, board, wdl, &mut rng)
        && !see_filter(board, mv)
        && rng.random_bool(piece_count_acceptance(board))
}

fn main() {
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(Ranger)
        .inputs(ChessBucketsMirrored::new(INPUT_BUCKETS))
        .output_buckets(CjBuckets)
        .save_format(&[
            SavedFormat::id("l0w")
                .transform(|store, weights| {
                    let factoriser = store.get("l0f").values.repeat(INPUT_BUCKETS_NUM);
                    weights.into_iter().zip(factoriser).map(|(a, b)| a + b).collect()
                })
                .round()
                .quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB).transpose(),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0f = builder.new_weights("l0f", Shape::new(HL_SIZE, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(INPUT_BUCKETS_NUM);

            let mut l0 = builder.new_affine("l0", 768 * INPUT_BUCKETS_NUM, HL_SIZE);
            l0.weights = l0.weights + expanded_factoriser;

            let l1 = builder.new_affine("l1", HL_SIZE, OUTPUT_BUCKETS_NUM);

            let stm_hl = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_hl = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let hl = stm_hl.concat(ntm_hl);
            l1.forward(hl).select(output_buckets)
        });

    let stricter_clipping = RangerParams { min_weight: -0.99, max_weight: 0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);
    trainer.optimiser.set_params_for_weight("l0f", stricter_clipping);

    let lr_scheduler = OneCycleLr {
        max_lr: 0.001 * BATCH_SIZE as f32 / 16384.0,
        div: 25.0,
        final_div: 10000.0,
        warmup_pct: 0.3,
        anneal_cos: false,
        three_phase: true,
        final_superbatch: SUPERBATCHES,
    };

    let wdl_scheduler = WdlSequence {
        first: LinearWdl { start: 0.125, end: 0.875 },
        first_scheduler_final_superbatch: lr_scheduler.warmup_sb() + lr_scheduler.annealing_sb(),
        second: ConstWdl { value: 1.0 },
    };

    let schedule = TrainingSchedule {
        net_id: format!("{}hl-{}b-{}ob", HL_SIZE, INPUT_BUCKETS_NUM, OUTPUT_BUCKETS_NUM).to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: BATCH_SIZE,
            batches_per_superbatch: BATCHES,
            start_superbatch: CHECKPOINT + 1usize,
            end_superbatch: SUPERBATCHES,
        },
        wdl_scheduler: wdl_scheduler,
        lr_scheduler: lr_scheduler,
        save_rate: SUPERBATCHES / 20,
    };

    let settings = LocalSettings { threads: 2, test_set: None, output_directory: "checkpoints", batch_queue_size: 32 };

    let dataloader = {
        let path = "/k4/oil_data/dfrc.vf";
        let buffer_size = 1024;
        let threads = 16;
        ViriBinpackLoader::new(&path, buffer_size, threads, ViriFilter::Custom(filter))
    };

    if CHECKPOINT > 0 {
        let dir = format!("{}/{}-{}", settings.output_directory.to_string(), schedule.net_id, CHECKPOINT).to_string();
        trainer.load_from_checkpoint(&dir);
    }
    trainer.run(&schedule, &settings, &dataloader);
}
