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
            lr::{CosineDecayLR as CosLr, ExponentialDecayLR as ExpLr, LinearDecayLR as LinearLr, LrScheduler},
            wdl::{ConstantWDL as ConstWdl, LinearWDL as LinearWdl, Sequence as WdlSequence, WdlScheduler},
            TrainingSchedule, TrainingSteps,
        },
        settings::LocalSettings,
    },
    value::{
        loader::{viribinpack::ViriFilter, ViriBinpackLoader},
        ValueTrainerBuilder,
    },
};
use bullet_trainer::run::logger::ansi;
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
pub struct WarmupStableDecayLr {
    pub min_lr: f32,
    pub max_lr: f32,
    pub warmup_pct: f32,
    pub decay_pct: f32,
    pub final_superbatch: usize,
}

impl WarmupStableDecayLr {
    fn warmup_sb(&self) -> usize {
        (self.final_superbatch as f32 * self.warmup_pct) as usize
    }

    fn stable_sb(&self) -> usize {
        self.final_superbatch - self.warmup_sb() - self.decay_sb()
    }

    fn decay_sb(&self) -> usize {
        (self.final_superbatch as f32 * self.decay_pct) as usize
    }
}

impl LrScheduler for WarmupStableDecayLr {
    fn lr(&self, batch: usize, sb: usize) -> f32 {
        let warmup = self.warmup_sb();
        let stable = self.stable_sb();
        let decay = self.decay_sb();

        if sb <= warmup {
            let inner = LinearLr { initial_lr: self.min_lr, final_lr: self.max_lr, final_superbatch: warmup * BATCHES };
            inner.lr(1, (sb - 1) * BATCHES + batch)
        } else if sb <= warmup + stable {
            self.max_lr
        } else {
            let inner = ExpLr { initial_lr: self.max_lr, final_lr: self.min_lr, final_superbatch: decay * BATCHES };
            inner.lr(1, (sb - warmup - stable - 1) * BATCHES + batch)
        }
    }

    fn colourful(&self) -> String {
        format!(
            "wsd(\
            min = {}, max = {}, \
            warmup = {} sb, stable = {} sb, decay = {} sb)",
            ansi(self.min_lr, 31),
            ansi(self.max_lr, 31),
            ansi(self.warmup_sb(), 31),
            ansi(self.stable_sb(), 31),
            ansi(self.decay_sb(), 31),
        )
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

#[derive(Clone)]
pub struct TrainingConfig<'a> {
    name: &'a str,
    superbatches: usize,
}

const SEE_PIECE_VALUES: [i32; 6] = [100, 325, 325, 500, 900, 0];

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

const L1: usize = 1024;
const L2: usize = 16;
const L3: usize = 32;

const L0_SHIFT: u32 = 8;
const L0_SCALE: f32 = QA as f32 / (1 << L0_SHIFT) as f32;
const L1_RANGE: f32 = i8::MAX as f32 / QB as f32 * L0_SCALE * L0_SCALE;

const QA: i16 = 255;
const QB: i16 = 128;
const Q: i32 = 64;
const SCALE: f32 = 400.0;

const BATCH_SIZE: usize = 16384 * 2;
const BATCHES: usize = 16384 * 6144 / BATCH_SIZE;
const MAIN_SUPERBATCHES: usize = 200;
const TUNE_SUPERBATCHES: usize = 50;

const CHECKPOINTS_DIR: &str = "checkpoints";

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

fn train<Wdl: WdlScheduler, Lr: LrScheduler>(
    data_path: &str,
    wdl_scheduler: Wdl,
    lr_scheduler: Lr,
    config: TrainingConfig,
    prev_config: Option<TrainingConfig>,
) {
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(Ranger)
        .inputs(ChessBucketsMirrored::new(INPUT_BUCKETS))
        .output_buckets(CjBuckets)
        .save_format(&[
            SavedFormat::id("l0w")
                .transform(|builder, mut weights| {
                    let factoriser = builder.get("l0f").values;
                    let expanded = factoriser.f32().repeat(INPUT_BUCKETS_NUM);

                    for (i, &j) in weights.iter_mut().zip(expanded.iter()) {
                        *i += j;
                    }

                    weights
                })
                .round()
                .quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w")
                .transform(|_, mut weights| {
                    for i in weights.iter_mut() {
                        *i /= L0_SCALE * L0_SCALE;
                    }
                    weights
                })
                .round()
                .quantise::<i8>(QB),
            SavedFormat::id("l1b").round().quantise::<i32>(Q),
            SavedFormat::id("l2w").round().quantise::<i32>(Q),
            SavedFormat::id("l2b").round().quantise::<i32>(Q.pow(3)),
            SavedFormat::id("l3w").round().quantise::<i32>(Q),
            SavedFormat::id("l3b").round().quantise::<i32>(Q.pow(4)),
        ])
        .build_custom(|builder, (stm_inputs, ntm_inputs, output_buckets), targets| {
            let l0f = builder.new_weights("l0f", Shape::new(L1, 768), InitSettings::Zeroed);
            let expanded_factoriser = l0f.repeat(INPUT_BUCKETS_NUM);

            let mut l0 = builder.new_affine("l0", 768 * INPUT_BUCKETS_NUM, L1);
            l0.weights = l0.weights + expanded_factoriser;

            let l1 = builder.new_affine("l1", L1, OUTPUT_BUCKETS_NUM * L2);
            let l2 = builder.new_affine("l2", L2 * 2, OUTPUT_BUCKETS_NUM * L3);
            let l3 = builder.new_affine("l3", L3, OUTPUT_BUCKETS_NUM);

            let stm_out = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_out = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let l0_out = stm_out.concat(ntm_out);

            let l1_mean = builder.new_constant(Shape::new(1, L1), &[1.0 / L1 as f32; L1]);
            let l0_norm = l1_mean.matmul(l0_out);

            let l1_raw = l1.forward(l0_out).select(output_buckets);
            let l1_out = l1_raw.concat(l1_raw * l1_raw).crelu();

            let l2_out = l2.forward(l1_out).select(output_buckets).screlu() + l1_raw.repeat(2);
            let l3_out = l3.forward(l2_out).select(output_buckets);

            let loss = l3_out.sigmoid().squared_error(targets) + 0.005 * l0_norm;
            (l3_out, loss)
        });

    let l0_clip = RangerParams { min_weight: -0.99, max_weight: 0.99, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l0w", l0_clip);
    trainer.optimiser.set_params_for_weight("l0f", l0_clip);

    let l1_clip = RangerParams { min_weight: -L1_RANGE, max_weight: L1_RANGE, ..Default::default() };
    trainer.optimiser.set_params_for_weight("l1w", l1_clip);

    if let Some(checkpoint) = prev_config.clone() {
        if checkpoint.name != config.name {
            trainer
                .optimiser
                .load_weights_from_file(
                    format!(
                        "{}/{}/{}-{}/optimiser_state/weights.bin",
                        CHECKPOINTS_DIR, checkpoint.name, checkpoint.name, checkpoint.superbatches,
                    )
                    .as_str(),
                )
                .unwrap();
        } else {
            trainer.load_from_checkpoint(
                format!("{}/{}/{}-{}", CHECKPOINTS_DIR, checkpoint.name, checkpoint.name, checkpoint.superbatches,)
                    .as_str(),
            );
        }
    }

    let start = if prev_config.clone().map(|checkpoint| checkpoint.name == config.name).unwrap_or(false) {
        prev_config.unwrap().superbatches + 1
    } else {
        1
    };

    let schedule = TrainingSchedule {
        net_id: config.name.to_string(),
        eval_scale: SCALE,
        steps: TrainingSteps {
            batch_size: BATCH_SIZE,
            batches_per_superbatch: BATCHES,
            start_superbatch: start,
            end_superbatch: config.superbatches,
        },
        wdl_scheduler: wdl_scheduler,
        lr_scheduler: lr_scheduler,
        save_rate: 10,
    };

    let checkpoints_dir = format!("{}/{}", CHECKPOINTS_DIR, config.name);
    let settings =
        LocalSettings { threads: 2, output_directory: &checkpoints_dir, test_set: None, batch_queue_size: 512 };

    let dataloader = ViriBinpackLoader::new(data_path, 1024, 2, ViriFilter::Custom(filter));

    trainer.run(&schedule, &settings, &dataloader);

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2QBp/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2QBRK1 w kq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/P2P2PP/q2QBR1K w kq - 0 2",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1",
        "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rn1qkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQka - 0 1",
    ] {
        let eval = trainer.eval(fen);
        println!("fen: {}", fen);
        println!("eval: {}", eval * SCALE);
    }
}

fn main() {
    let lr_scheduler = WarmupStableDecayLr {
        min_lr: 0.001 * (BATCH_SIZE as f32 / 16384.0).powf(0.5) / 25.0,
        max_lr: 0.001 * (BATCH_SIZE as f32 / 16384.0).powf(0.5),
        warmup_pct: 0.15,
        decay_pct: 0.1,
        final_superbatch: MAIN_SUPERBATCHES,
    };
    let wdl_scheduler = WdlSequence {
        first: WdlSequence {
            first: ConstWdl { value: 0.25 },
            first_scheduler_final_superbatch: lr_scheduler.warmup_sb(),
            second: LinearWdl { start: 0.25, end: 1.0 },
        },
        first_scheduler_final_superbatch: lr_scheduler.warmup_sb() + lr_scheduler.stable_sb(),
        second: ConstWdl { value: 1.0 },
    };
    train(
        "data/main.vf",
        wdl_scheduler,
        lr_scheduler,
        TrainingConfig { name: "main", superbatches: MAIN_SUPERBATCHES },
        None,
    );

    train(
        "data/tune.vf",
        ConstWdl { value: 1.0 },
        OneCycleLr {
            max_lr: 0.001 * (BATCH_SIZE as f32 / 16384.0).powf(0.5),
            div: 25.0,
            final_div: 1e5,
            warmup_pct: 0.4,
            anneal_cos: true,
            three_phase: false,
            final_superbatch: TUNE_SUPERBATCHES,
        },
        TrainingConfig { name: "tune", superbatches: TUNE_SUPERBATCHES },
        None,
    );
}
