# arb-bot

Crypto market-data pipeline and research harness. **Analysis only — no order
placement or execution**, by design: the goal is data you can trust and results
you can reproduce before any strategy sits on top of it. See
[Model 3](#model-3-cross-exchange-arbitrage--results),
[Model 4](#model-4-btceth-pairs--feasibility),
[Model 5](#model-5-cointegration-screen--multiple-testing-correction), and
[Model 6](#model-6-out-of-sample-validation--a-lead-then-correctly-disproven) for
completed studies — a series of tested negative results, including one promising
lead found and then correctly rejected.

## Layout

```
data/feed.py      MarketFeed — exchange-agnostic async market-data client
data/history.py   load collected CSVs; align series across exchanges
scripts/          runnable tools: collection, comparison, backtest
tests/            automated tests (no network)
data_store/       collected OHLCV CSVs (git-ignored)
.env.example      API-key placeholders (.env is git-ignored)
requirements.txt      runtime deps: ccxt, pandas, numpy, statsmodels
requirements-dev.txt  dev deps: pytest (+ runtime)
```

## Setup

```bash
python -m venv venv
./venv/bin/pip install -r requirements-dev.txt
```

## Run

```bash
./venv/bin/python -m pytest                    # automated tests (no network)
./venv/bin/python -m scripts.test_feed         # live BTC/USD ticker (Coinbase)
./venv/bin/python -m scripts.compare_feed      # Coinbase vs Kraken, side by side
./venv/bin/python -m scripts.collect_history   # 90d 1h OHLCV -> data_store/
./venv/bin/python -m scripts.show_spread       # spread over the shared window
./venv/bin/python -m scripts.backtest_arbitrage --fee-threshold 0.8
```

## Model 3: Cross-Exchange Arbitrage — Results

**Question:** Do BTC/USD or ETH/USD show a price spread between Coinbase and
Kraken large enough to beat round-trip trading fees?

**Answer (completed):** No. Over every hour the two exchanges share, the widest
spread observed was **0.134%** — far under a realistic **0.8% round-trip** fee
threshold. This is a deliberate, tested **negative result**: on these liquid
pairs and this window, there is no fee-beating arbitrage to capture.

### What was built

- **Exchange-agnostic data client** (`data/feed.py`) — `MarketFeed` wraps a ccxt
  async exchange; the same class serves both Coinbase and Kraken unchanged.
  `fetch_ohlcv` paginates past each exchange's per-request candle cap to assemble
  a full range, deduping by timestamp.
- **History collection** (`scripts/collect_history.py`) — pulls 90 days of 1-hour
  OHLCV for both symbols from both exchanges into per-source CSVs in `data_store/`.
  Runs the two exchanges concurrently (`asyncio.gather`), each with its own
  session and rate limiter.
- **Shared-window alignment** (`data/history.align`) — inner-joins the per-exchange
  series on their timestamp index, so only hours present on *both* venues are ever
  compared.
- **Backtest** (`scripts/backtest_arbitrage.py`) — for each shared hour, measures
  the spread against the lower (buy-side) price and flags a signal when it clears
  a configurable fee threshold. Analysis only: no order placement, no execution.

### Methodology

- **Data:** 1-hour candle closes, BTC/USD and ETH/USD, Coinbase and Kraken.
- **Shared window:** the timestamp intersection of the two exchanges — **721 hours
  per symbol, 1,442 total.** The window is bounded by Kraken's public OHLC endpoint,
  which serves only ~720 of the most recent candles per interval (~30 days at 1h),
  regardless of how far back you ask. Coinbase reaches ~90 days; the comparison is
  restricted to the overlap rather than padded with unmatched data.
- **Signal rule:** flag an hour when `(high − low) / low` exceeds the round-trip
  fee threshold. Measuring against the lower price maps the spread directly onto
  the fee rate — the gross gap must beat the fee to be worth anything.
- **Correctness guards:** symbols are validated against each exchange's live
  markets before any fetch (a mismatched pair fails loudly rather than returning a
  plausible-but-wrong number); the inner join drops any hour missing on either
  side, so a gap is never paired against a real quote. The test suite runs offline.

### Finding

| Threshold | Hours checked | Signals | Largest spread seen |
|-----------|--------------:|--------:|---------------------|
| 0.8% (round-trip) | 1,442 | **0** | 0.134% ($2.10, ETH/USD) |
| 0.1% (sanity check) | 1,442 | 3 | 0.134% ($2.10, ETH/USD) |

Zero signals at the realistic threshold is the expected outcome for two of the
deepest USD venues: their hourly closes stay tightly coupled. Lowering the
threshold to 0.1% surfaces three ETH hours, which confirms the signal path works —
the 0.8% result is a real absence of opportunity, not a silent bug.

### Scope and honest caveats

- **Hourly closes are not tradable quotes.** A real fill needs same-instant
  bid/ask and order-book depth; this models neither slippage, depth, latency, nor
  withdrawal times. A flagged hour is a research signal, not an executable edge.
- **The window is the ~30 shared days** allowed by Kraken's API, and BTC/ETH are
  the tightest pairs on the most liquid venues — i.e. the least likely place to
  find an edge. A wider search (thinner pairs, more distant venues, live order
  books instead of candle closes) would be the direction to take it further; that
  is out of scope for this model.

The value here is a clean, reproducible pipeline and a result you can trust in
either direction — including trusting it when it says *no*.

## Model 4: BTC/ETH Pairs — Feasibility

**Question:** BTC and ETH move together — is that relationship stable enough to
build a pairs (mean-reversion) strategy on?

**Answer (completed):** No. Across three escalating tests, the two assets clearly
*co-move* (0.89 log-return correlation) but their price relationship does not
*mean-revert*: the ratio is non-stationary, and even an optimally-hedged spread
is not cointegrated (p ≈ 0.50). Another deliberate, tested **negative result** —
strong correlation is not a tradable pairs relationship, and here the difference
is measurable rather than a matter of opinion.

### What was built

- **Correlation and ratio analysis** (`scripts/check_correlation.py`) — loads
  Coinbase's BTC/USD and ETH/USD 1h closes via `data.history.load_candles`,
  inner-joins them on their shared hours, and reports price-level correlation,
  log-return correlation, and price-ratio summary stats. Plots the ratio over
  time as a PNG when matplotlib is available, else an ASCII sparkline.
- **Stationarity test** — an Augmented Dickey-Fuller (ADF) test on the ratio plus
  a mean-reversion half-life from an AR(1) fit, quantifying whether the ratio
  actually reverts or merely wanders.
- **Cointegration test** — an Engle-Granger procedure: estimate the hedge ratio β
  (OLS of BTC on ETH), then test the β-weighted spread for stationarity with
  statsmodels' `coint`, which applies the correct critical values for a spread
  whose β was estimated from the data. Exploratory analysis only — no strategy or
  order logic.

### Methodology

- **Data:** 1-hour candle closes, BTC/USD and ETH/USD, Coinbase, over the full
  ~90-day collected window — **2,155 shared hours** after inner-joining on
  timestamp (which drops the known Coinbase candle gaps rather than pairing a hole
  with a price).
- **Why three tests, not one:** price-level correlation is inflated by common
  trend (two rising series always look correlated), so it is reported but not
  trusted; log-return correlation measures genuine co-movement; and stationarity
  / cointegration test the property a pairs strategy actually needs — that the
  spread reverts to a stable level. Each test rules out a way the previous one
  could mislead.
- **Correct critical values:** the Engle-Granger step uses `coint` rather than a
  plain ADF on the residuals. Because β is estimated, residuals look artificially
  stationary, so the rejection bar is correctly *raised* (5% critical value
  −3.339 vs −2.863). Using the right test is the difference between a real result
  and a flattering one.

### Finding

| Test | Metric | Result | Reading |
|------|--------|--------|---------|
| Log-return corr | Pearson | **+0.889** | strong genuine co-movement |
| ADF on ratio | p-value | **0.273** | not stationary — ratio wanders |
| Ratio half-life | AR(1) | **~237 h (~10 d)** | reversion too slow to trade |
| Engle-Granger | p-value | **0.495** | **not cointegrated** — hedged spread drifts too |

The estimated hedge ratio (β ≈ 27, with a large +$16,886 intercept) is itself a
tell: a relationship that needs a big constant offset to fit is two assets that
drifted together, not a tight structural link. Giving the idea its best shot — an
optimally-hedged spread — still fails the cointegration test decisively.

### Scope and honest caveats

- **Correlation ≠ cointegration.** This model exists to make that distinction
  concrete: 0.89 co-movement alongside a p ≈ 0.50 cointegration result is exactly
  the "high correlation, no reversion" trap that sinks naive pairs strategies.
- **90 days is one market regime.** Cointegration is regime-dependent, so a longer,
  multi-regime history could reach a different verdict — that is the single open
  door, and it needs data beyond what is collected here. On the evidence in hand,
  the answer is a well-tested no.

The value, as in Model 3, is a reproducible test that fails honestly: the pairs
idea is set aside on measurement, not on a hunch.

## Model 5: Cointegration Screen — Multiple-Testing Correction

**Question:** Model 4 tested one pair. Widen the net to all 21 pairs across seven
Coinbase assets (BTC, ETH, SOL, ADA, LINK, LTC, XRP) — does *any* pair cointegrate?

**Answer (completed):** No — and the interesting part is *why the raw numbers say
yes and the honest answer is still no*. Two pairs clear raw p < 0.05, but that is
exactly the number of false positives you expect from 21 simultaneous tests. After
a Benjamini-Hochberg false-discovery-rate correction, **zero pairs survive.** The
headline result of this model is the correction itself: searching many pairs
manufactures significance, and controlling for it is what separates a real finding
from a lucky one.

### What was built

- **Shared statistics module** (`data/stats.py`) — the Engle-Granger routine and
  mean-reversion half-life from Model 4, refactored out of the single-pair script
  into one place (`engle_granger`, `half_life`) so the batch screen and the
  detailed single-pair report compute identical maths. No duplicated test logic.
- **Pair screen** (`scripts/screen_pairs.py`) — loads all seven assets' closes,
  inner-joins them onto one common-timestamp sample, runs Engle-Granger on every
  pair, applies the Benjamini-Hochberg correction (`statsmodels.multipletests`,
  `fdr_bh`), and prints a table sorted by adjusted p-value. Exploratory analysis
  only — no strategy or order logic.

### Methodology

- **Data:** 1-hour closes for all seven assets on Coinbase, inner-joined to a
  single **2,155-hour** sample so every pair is tested over identical hours.
- **Why the correction is the point:** at raw p < 0.05, each test has a 5% chance
  of a false positive under the null. Run 21 of them and you expect ~1 by chance —
  so a raw "hit" from a wide search is uninformative on its own. Benjamini-Hochberg
  rescales the p-values to control the *false discovery rate* — the expected
  fraction of claimed discoveries that are false — across the whole family of tests.
- **Exactly 21 tests, honestly counted:** Engle-Granger is direction-dependent, so
  each unordered pair is tested once in a fixed order. That keeps the family at the
  21 tests the correction is applied over — inflating the count (e.g. testing both
  directions) would itself be a form of the bias this model is about.

### Finding

| Pair | Raw p | BH-adjusted p | Half-life | Survives |
|------|------:|--------------:|----------:|:--------:|
| BTC–LTC | **0.010** | 0.216 | ~2.0 d | no |
| BTC–ADA | **0.038** | 0.403 | ~3.1 d | no |
| … 18 more … | > 0.05 | > 0.4 | — | no |
| ETH–XRP | 0.934 | 0.934 | ~13.3 d | no |

**2 of 21 pairs at raw p < 0.05; 0 survive correction.** BTC–LTC is the tell:
a raw p of 0.010 looks compelling in isolation, but adjusted to 0.216 it is
nowhere near significant once the 21-test search is accounted for. Reported without
the correction, it would have been a spurious "cointegrated pair."

### Scope and honest caveats

- **This is the whole lesson:** a wide search plus an uncorrected threshold is a
  machine for producing false positives. The negative result here is only
  trustworthy *because* of the correction — without it, this model would have
  reported a discovery that the data does not support.
- **Direction and regime.** Engle-Granger asymmetry means a fuller screen could
  test both orderings (with a correspondingly larger correction); and, as in
  Model 4, this is one ~90-day regime. Neither changes the finding on the evidence
  in hand: no cointegrated pair in this basket.

The value, once more, is a result you can trust — here specifically *because* the
method refuses to be fooled by the two pairs that raw significance would have
waved through.

## Model 6: Out-of-Sample Validation — a lead, then correctly disproven

**Question:** The crypto pairs never cointegrated at all. Extending the method to
equities (via yfinance, 2 years of daily adjusted closes), the gold pair GLD/GDX
*did* pass an in-sample cointegration test — p = 0.016, a tradable ~14-day
half-life. Is it real, or an artifact of testing the hedge ratio on the same data
that estimated it?

**Answer (completed):** An artifact. Estimating the hedge ratio on the first half
of the window and testing the second half, the relationship falls apart — the
train half alone does not cointegrate, and the hedge ratio drifts 30% between
halves. The in-sample p = 0.016 was a regime artifact, not a durable link. This
is the model that most earns its place: **it found the project's one promising
lead and then disproved it with the appropriate test** — which is exactly what an
in-sample-only workflow would have failed to do.

### What was built

- **Stock collector** (`scripts/collect_stock_history.py`) — pulls 2 years of
  daily dividend/split-adjusted closes via yfinance for KO, PEP, GLD, GDX into
  `data_store/stock_<TICKER>_1d.csv`. Deliberately *not* built on the crypto
  `MarketFeed`: yfinance's interface is nothing like ccxt's, so the two collectors
  share a naming convention and an output folder, not a code path. Downloads are
  verified on the way in (row count, date span, price-level sanity, unexpected
  gaps), which caught a trailing NaN placeholder row and confirmed the large
  GLD/GDX price move was a real gold rally, not bad data.
- **Configurable pair analysis** (`scripts/check_stock_pair.py`) — the stock analog
  of Model 4's script, reusing the shared `data.stats` routines so the maths is
  identical; the pair is a command-line argument.
- **Out-of-sample validator** (`scripts/validate_stock_pair.py`) — splits the
  window, estimates the hedge ratio β on the train half only, freezes it, forms
  the spread on the test half with that frozen β, and tests *that* spread for
  stationarity. Reports a hedge-ratio-drift diagnostic across the two halves.

### Methodology

- **The split test is the point.** An in-sample Engle-Granger test estimates β on
  the same data it tests, so a low p-value can just mean "a line fit this window."
  Real validation freezes a relationship learned on the past and checks it against
  unseen data: estimate β on 2024-07 → 2025-07, then test 2025-07 → 2026-07.
- **The right test out-of-sample is a plain ADF.** Because β is *not* re-estimated
  on the test half, the out-of-sample spread is a fixed linear combination with no
  parameters fitted to the test data — so standard ADF critical values apply, not
  the Engle-Granger ones (which exist precisely to penalize estimating β on the
  data under test). Using the ordinary ADF is what makes this a genuine hold-out
  rather than the in-sample test wearing a disguise.
- **A passing hold-out is only meaningful if the training fit was real.** When the
  train half itself fails to cointegrate, a passing out-of-sample ADF is luck, not
  validation — the script flags that case as moot rather than let a green result
  mislead.

### Finding

| Window | Hedge ratio β | Cointegration |
|--------|--------------:|---------------|
| Full 500 days (in-sample) | 3.22 | p = **0.016** — passes |
| Train half (2024-07 → 2025-07) | 4.59 | p = **0.260** — fails |
| Test half, re-estimated | 3.22 | — |

The tell is in the last two rows: the test-half β (3.22) matches the *full-window*
β exactly, while the train half's β is 4.59. The entire signal lives in the second
half — the accelerating-gold-rally period — and is absent in the first. The hedge
ratio drifts 30% between halves. A relationship this unstable through time is not a
structural link; it is two instruments riding the same trend.

### Scope and honest caveats

- **This is the capstone lesson of the project.** In-sample significance is not
  evidence until it survives a hold-out. The GLD/GDX lead was the strongest signal
  found across crypto and equities, and the correct test dissolved it — which is a
  feature, not a disappointment. An in-sample-only pipeline would have reported a
  tradable pair that the data does not support.
- **One split, one regime.** A single mid-point split of one 2-year window is the
  simplest hold-out; walk-forward validation across multiple windows would be the
  next step. It would not rescue GLD/GDX here, but it is how this would be done at
  scale.

Across six models — arbitrage, single-pair cointegration, a corrected multi-pair
screen, and now an out-of-sample hold-out — every apparent edge, tested properly,
has failed honestly. The deliverable was never a winning strategy; it is a
pipeline whose *no* you can trust, and whose one *maybe* it knew how to reject.
