# polymarket-faa-attribution-02 entry: forecast each market's Yes price for the
# 240 minutes after the checkpoint.
#
# Step 1: the house persistence baseline (fixed 0.0025 spread) scored 0.363589
# mean pinball loss on the six practice cases. Step 2: the spread is not fixed,
# it is each market's own realised one-minute volatility, measured on its last
# 240 minutes: q_tau(h) = clamp(x + z_tau * sigma * sqrt(h), 0, 1), rounded to
# 6 decimals, where x is the market's last price at the checkpoint and sigma is
# the sample stdev of those 240 one-minute returns. That alone scored 0.328475.
import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
OUTPUT_DIR = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0, 0.674490, 1.644854]
WINDOW = 240


def price6(x):
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0


def forecast(market, horizon):
    prices = market['prices']
    returns = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    seg = returns[-WINDOW:] if len(returns) >= WINDOW else returns
    mu = sum(seg) / len(seg) if seg else 0.0
    var = sum((r - mu) ** 2 for r in seg) / (len(seg) - 1) if len(seg) > 1 else 0.0
    sigma = math.sqrt(max(var, 0.0))
    x = prices[-1]
    rows = []
    for h in range(1, horizon + 1):
        s = math.sqrt(h)
        rows.append([price6(x + z * sigma * s) for z in Z])
    return rows


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(in_dir.glob('*.in')):
        case = json.loads(f.read_text())
        forecasts = [{'id': m['id'], 'q': forecast(m, case['horizon_minutes'])} for m in case['markets']]
        (out_dir / (f.stem + '.out')).write_text(json.dumps({'forecasts': forecasts}) + '\n')


if __name__ == '__main__':
    main()
