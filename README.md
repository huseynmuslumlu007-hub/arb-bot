# arb-bot

Crypto market-data pipeline and research harness. **Analysis only — no order
placement or execution**, by design: the goal is data you can trust and results
you can reproduce before any strategy sits on top of it. See
[Model 3](#model-3-cross-exchange-arbitrage--results) for the first completed
study.

## Layout

```
data/feed.py      MarketFeed — exchange-agnostic async market-data client
data/history.py   load collected CSVs; align series across exchanges
scripts/          runnable tools: collection, comparison, backtest
tests/            automated tests (no network)
data_store/       collected OHLCV CSVs (git-ignored)
.env.example      API-key placeholders (.env is git-ignored)
requirements.txt      runtime deps: ccxt, pandas, numpy
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
