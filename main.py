# polymarket-faa-attribution-02 entry: forecast each market's Yes price for the
# 240 minutes after the checkpoint.
#
# Step 1: the house persistence baseline, which the practice cases score at
# 0.363589 mean pinball loss, prices the median at the checkpoint's last price
# and spreads the quantiles by a fixed 0.0025 per sqrt-minute random walk.
import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
OUTPUT_DIR = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0, 0.674490, 1.644854]
SIGMA = 0.0025


def price6(x):
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0


def forecast(market, horizon):
    x = market['prices'][-1]
    rows = []
    for h in range(1, horizon + 1):
        s = math.sqrt(h)
        rows.append([price6(x + z * SIGMA * s) for z in Z])
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
