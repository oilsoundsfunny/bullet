use acyclib::trainer::logger::ansi;
use bulletformat::ChessBoard;
use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, get_num_buckets},
        outputs::OutputBuckets,
    },
    nn::{
        InitSettings, Shape,
        optimiser::{Ranger, RangerParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{
            TrainingSchedule,
            TrainingSteps,
            lr::{CosineDecayLR as CosLr, LinearDecayLR as LinearLr, LrScheduler},
            wdl::{ConstantWDL as ConstWdl, LinearWDL as LinearWdl, Sequence as WdlSequence},
        },
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::{ViriBinpackLoader, viribinpack::ViriFilter}},
};
use rand::{Rng, rng};
use std::{mem::MaybeUninit, ops::Div, sync::atomic::{AtomicU64, Ordering}};
use viriformat::{chess::{board::Board, chessmove::Move}, dataformat::{Filter, WDL as Result}};

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
        if self.three_phase { self.final_superbatch - self.warmup_sb() * 2 } else { 0 }
    }
}

impl LrScheduler for OneCycleLr {
    fn lr(&self, batch: usize, sb: usize) -> f32 {
        let warmup = self.warmup_sb();
        let anneal = self.annealing_sb();

        if sb <= warmup {
            let inner = CosLr {
                initial_lr: self.initial_lr(),
                final_lr: self.max_lr,
                final_superbatch: warmup * BATCHES,
            };
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

const BATCH_SIZE: usize = 16384 * 4;
const BATCHES: usize = 16384 * 6144 / BATCH_SIZE;
const SUPERBATCHES: usize = 200;
const CHECKPOINT: usize = 0;

const QA: i16 = 255;
const QB: i16 = 64;
const SCALE: f32 = 400.0;

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
        random_fen_skipping: true,
        random_fen_skip_probability: 0.05,
        ..Default::default()
    };
    let wdl = match wdl {
        1.0 => Result::Win,
        0.5 => Result::Draw,
        0.0 => Result::Loss,
        _ => unreachable!(),
    };
    let mut rng = rng();

    !default_filter.should_filter(mv, eval as i32, board, wdl, &mut rng)
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

    let stricter_clipping = RangerParams {
        min_weight: -0.99,
        max_weight: 0.99,
        ..Default::default()
    };
    trainer.optimiser.set_params_for_weight("l0w", stricter_clipping);
    trainer.optimiser.set_params_for_weight("l0f", stricter_clipping);

    let lr_scheduler = OneCycleLr {
        max_lr: 0.001 * BATCH_SIZE as f32 / 16384.0,
        div: 25.0,
        final_div: 10000.0,
        warmup_pct: 0.3,
        anneal_cos: true,
        three_phase: true,
        final_superbatch: SUPERBATCHES,
    };

    let wdl_scheduler = WdlSequence {
        first: WdlSequence {
            first: ConstWdl { value: 0.5 },
            first_scheduler_final_superbatch: lr_scheduler.warmup_sb(),
            second: LinearWdl { start: 0.625, end: 0.875 },
        },
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

    let settings = LocalSettings {
        threads: 2,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 32,
    };

    let dataloader = {
        let path = "data/dfrc.vf";
        let buffer_size = 1024;
        let threads = 2;
        ViriBinpackLoader::new(&path, buffer_size, threads, ViriFilter::Custom(filter))
    };

    if CHECKPOINT > 0 {
        let dir = format!("{}/{}-{}", settings.output_directory.to_string(), schedule.net_id, CHECKPOINT)
            .to_string();
        trainer.load_from_checkpoint(&dir);
    }
    trainer.run(&schedule, &settings, &dataloader);
}
