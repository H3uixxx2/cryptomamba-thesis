# Frozen reference artifacts

Copies of two files from `output/evaluation/`, which is gitignored. They live here so a
fresh clone — a Colab runtime, in practice — can verify the trading scorer against
numbers that were settled long before this round.

| file | what it holds |
|---|---|
| `forecast_predictions.csv` | per-day predictions of the official and retrained CryptoMamba-v checkpoints across train/val/test |
| `trading_metrics.csv` | the recorded trading results: 2 checkpoints x 2 splits x 4 strategies x 3 cost levels |

`trading_metrics.csv` is where the targets of the current round come from: at 0% cost on
the test split, holding returns 222.51 and the authors' checkpoint reaches 262.78 with
`smart_w_short`. It also records `risk_pct = 2.0`, which is *not* the risk=5 default in
`utils/trade.py` — scoring at 5 collapses `smart` and `smart_w_short` onto identical
decisions and misses 262.78 by about 90.

`tests/test_trading_objective.py` replays these predictions through the scorer and
requires all eight recorded balances back to within $0.05. Without these files that test
skips, and a Colab run would train against a scorer nobody checked.

Treat as fixed input. Regenerating them is not part of any run.
