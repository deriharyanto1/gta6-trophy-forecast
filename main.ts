// GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).
//
// Model: earners / players decomposition with log-time power-law trends.
//
// An Exophase earn rate is a ratio: rate = earners / players_tracked. On launch
// night the tracked-player pool jumps in big refreshes (638 tracked at 23:48,
// 3,868 after the 04:30 refresh), so a rate falls mostly because the denominator
// grows with new players who have barely played, not because earners shrink.
// Forecasting the rate directly (persistence, or a straight logit trend) misses
// that. Instead this program forecasts the two counts separately, in log time:
//
//   ln E_i(t) = a_i + b_i * ln(1 + t)      earners of trophy i
//   ln N(t)   = a_N + b_N * ln(1 + t)      players tracked
//   rate_i(T) = 100 * E_i(T) / N(T)
//
// with t in hours since the release (2026-11-19T00:00:00Z). Slopes are fitted
// per trophy with a Theil-Sen estimator over the last 24 hours of reads and
// shrunk toward a per-trophy prior (hierarchical shrinkage): the prior exponents
// were calibrated on the public Exophase page at launch, from the earners at
// launch night (638 tracked) to the earners at midday (about 6,100 tracked):
// fast-earned trophies (fishing, chapter 1) grow near 0.5-0.6, slow ones near
// 0.15. When a case's history is flat (the launch-night practice cases, where
// the page had not refreshed yet), no trend can be fitted, so the priors carry
// the forecast and the players are assumed to jump once at the next refresh by
// the first refresh's factor (3,868 / 638 = 6.06, stated in the rules).
//
// Spreads are honest and widen with the horizon: a fit residual, a random-walk
// term (0.5 logit per sqrt day, as the house baseline), and a refresh term for
// the jump a refresh can bring. Quantiles are logit-normal around the median.
//
// Node 22, standard library only, no network, deterministic output.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Scored quantile levels, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Release instant: the origin of log time. */
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');
/** Hours of history a trend is fitted on (the recent reads carry the dynamics). */
const FIT_WINDOW_HOURS = 24;
/** First refresh's player-growth factor, from the rules (3,868 / 638). */
const REFRESH_FACTOR = 6.06;
/** Shrinkage pseudo-count: fitted slope weight is n / (n + SHRINK_K). */
const SHRINK_K = 3;
/** Fallback prior exponent where a trophy has none calibrated. */
const B_FALLBACK = 0.25;
/** Spread pieces, in logit units (see the header for what each covers). */
const SIG_FIT_FLOOR = 0.1;
const SIG_FIT_FLAT = 0.25;
const SIG_REFRESH_FLAT = 0.35;
const SIG_REFRESH_FIT = 0.15;
const SIG_RW_PER_SQRT_DAY = 0.5;

/**
 * Per-trophy prior exponents b_i for ln E_i(t) = a_i + b_i * ln(1 + t),
 * calibrated on the public Exophase page: earners at launch night (638 players
 * tracked) against earners at midday 19 Nov (about 6,100 tracked). Index i is
 * trophy id - 1. Trophy 2 sits at 100% and never moves; its entry is unused.
 */
const B_PRIOR: number[] = [
  0.15, 0.25, 0.48, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
  0.25, 0.15, 0.15, 0.17, 0.15, 0.15, 0.17, 0.62, 0.51, 0.17, 0.17, 0.25,
  0.15, 0.15, 0.17, 0.15, 0.17, 0.15, 0.15, 0.15, 0.22, 0.25, 0.4, 0.17,
  0.31, 0.23, 0.22, 0.17, 0.19, 0.25, 0.17, 0.28, 0.17, 0.25, 0.25, 0.29,
  0.25, 0.29, 0.29, 0.17, 0.22, 0.25, 0.25,
];

interface Trophy {
  id: number;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

const clampRate = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
};
const sigmoidPct = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
/** Hours from the release to an ISO timestamp. */
const hoursSinceRelease = (iso: string): number =>
  (Date.parse(iso) - RELEASE_MS) / 3_600_000;
/** Log-time coordinate, safe for reads just before the release. */
const logTime = (iso: string): number => Math.log(1 + Math.max(hoursSinceRelease(iso), -0.99));
const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
};

/** Theil-Sen line (median pairwise slope, median intercept); robust to jumps. */
function theilSen(xs: number[], ys: number[]): [number, number] {
  const n = xs.length;
  const slopes: number[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (xs[j] !== xs[i]) slopes.push((ys[j] - ys[i]) / (xs[j] - xs[i]));
    }
  }
  const slope = slopes.length > 0 ? median(slopes) : 0;
  const intercept = median(ys.map((y, i) => y - slope * xs[i]));
  return [slope, intercept];
}

/** Residual standard deviation of a line fit. */
function residualSd(xs: number[], ys: number[], slope: number, intercept: number): number {
  if (xs.length <= 2) return 0;
  let ss = 0;
  for (let i = 0; i < xs.length; i++) {
    const r = ys[i] - (intercept + slope * xs[i]);
    ss += r * r;
  }
  return Math.sqrt(ss / (xs.length - 2));
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${c.id} asks for levels ${JSON.stringify(c.levels)}`);
  }
  const hist = c.history;
  if (hist.length === 0) throw new Error(`${c.id} has no history`);
  const checkpointMs = Date.parse(c.at);
  const windowStart = checkpointMs - FIT_WINDOW_HOURS * 3_600_000;
  const inWindow = hist.map((_, j) => Date.parse(hist[j].at) >= windowStart);
  const xs = hist.map((r) => logTime(r.at));
  const xTarget = logTime(c.target);

  // Anchor: the last read where anything changed. The page is stale after it,
  // so the observed rates describe the state at the anchor, not at the last
  // read. With a flat history the anchor is the first read.
  let anchor = 0;
  for (let j = 1; j < hist.length; j++) {
    const changed =
      hist[j].players_tracked !== hist[j - 1].players_tracked ||
      hist[j].rates.some((r, i) => r !== hist[j - 1].rates[i]);
    if (changed) anchor = j;
  }
  const xAnchor = xs[anchor];

  // ---- Players tracked at the target. ----
  // Fit on distinct states only: between refreshes the page repeats itself,
  // and those repeated reads would overweight the plateaus. A distinct state
  // is a read whose players or rates differ from the previous read's.
  const distinctIdx: number[] = [];
  for (let j = 0; j < hist.length; j++) {
    if (
      j === 0 ||
      hist[j].players_tracked !== hist[j - 1].players_tracked ||
      hist[j].rates.some((r, i) => r !== hist[j - 1].rates[i])
    ) {
      distinctIdx.push(j);
    }
  }
  const fitIdx = distinctIdx.filter((j) => inWindow[j]);
  const nVals = fitIdx.map((j) => hist[j].players_tracked);
  const nVaries = new Set(nVals).size > 1;
  let lnNTarget: number;
  let sigN = 0;
  if (nVaries && fitIdx.length >= 3) {
    const [bN, aN] = theilSen(
      fitIdx.map((j) => xs[j]),
      nVals.map((n) => Math.log(n)),
    );
    lnNTarget = aN + bN * xTarget;
    sigN = Math.min(
      residualSd(
        fitIdx.map((j) => xs[j]),
        nVals.map((n) => Math.log(n)),
        bN,
        aN,
      ),
      0.5,
    );
  } else {
    // Flat history: the page has not refreshed yet; assume one refresh by the
    // target, growing the tracked pool by the first refresh's factor.
    lnNTarget = Math.log(hist[anchor].players_tracked) + Math.log(REFRESH_FACTOR);
  }

  // ---- Each trophy's earners at the target. ----
  return c.trophies.map((t, i) => {
    const lastRate = hist[hist.length - 1].rates[i] ?? 0;
    if (lastRate >= 99.99) return { id: t.id, q: [100, 100, 100, 100, 100] };

    const pts: Array<[number, number]> = [];
    for (const j of fitIdx) {
      const rate = hist[j].rates[i];
      if (rate === null || rate === undefined) continue;
      pts.push([xs[j], Math.log(Math.max((rate * hist[j].players_tracked) / 100, 0.05))]);
    }
    const bPrior = B_PRIOR[i] ?? B_FALLBACK;
    let lnETarget: number;
    let sigFit: number;
    let sigRefresh: number;
    const earnersVary = new Set(pts.map(([, y]) => y.toFixed(6))).size > 2;
    if (earnersVary && pts.length >= 3) {
      const [bFit] = theilSen(
        pts.map(([x]) => x),
        pts.map(([, y]) => y),
      );
      const w = pts.length / (pts.length + SHRINK_K);
      const b = w * bFit + (1 - w) * bPrior;
      const medX = median(pts.map(([x]) => x));
      const medY = median(pts.map(([, y]) => y));
      const a = medY - b * medX;
      lnETarget = a + b * xTarget;
      sigFit = Math.min(
        Math.max(
          residualSd(
            pts.map(([x]) => x),
            pts.map(([, y]) => y),
            b,
            a,
          ),
          SIG_FIT_FLOOR,
        ),
        0.6,
      );
      sigRefresh = SIG_REFRESH_FIT;
    } else {
      // Flat history: grow the anchor read's earners by the prior exponent.
      const anchorEarners = Math.max(
        ((hist[anchor].rates[i] ?? 0) * hist[anchor].players_tracked) / 100,
        0.05,
      );
      lnETarget = Math.log(anchorEarners) + bPrior * (xTarget - xAnchor);
      sigFit = SIG_FIT_FLAT;
      sigRefresh = SIG_REFRESH_FLAT;
    }

    const medianRate = clampRate(100 * Math.exp(lnETarget - lnNTarget));
    const sigma = Math.sqrt(
      sigFit ** 2 +
        sigN ** 2 +
        sigRefresh ** 2 +
        (SIG_RW_PER_SQRT_DAY * Math.sqrt(c.horizon_minutes / 1440)) ** 2,
    );
    const l = logit(medianRate);
    const q = Z.map((z) => round4(clampRate(sigmoidPct(l + z * sigma))));
    for (let k = 1; k < q.length; k++) if (q[k] < q[k - 1]) q[k] = q[k - 1];
    return { id: t.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
