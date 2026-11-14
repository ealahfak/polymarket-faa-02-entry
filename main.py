# polymarket-faa-attribution-02 entry: forecast each market's Yes price for the
# 240 minutes after the checkpoint.
#
# Step 1: the house persistence baseline (fixed 0.0025 spread) scored 0.363589
# mean pinball loss on the six practice cases. Step 2: the spread is not fixed,
# it is each market's own realised one-minute volatility over its last 240
# minutes; that scored 0.328475. Step 3, this file: a market that jumps leaves
# fatter tails than a Gaussian of the same volatility, and a market that sits
# still leaves narrower ones, so the quantiles are the empirical quantiles of
# those same 240 one-minute returns, blended 40/60 with the Gaussian ones
# (0.4 * q_emp + 0.6 * z * sigma). The blend keeps the empirical tails where the
# market jumps and the smooth body where it does not; it scored 0.300295, and it
# improved on the baseline in every one of the six practice cases and in 95 of
# its 128 markets.
#
# q_tau(h) = clamp(x + q_tau * sqrt(h), 0, 1), rounded to 6 decimals, where x is
# the market's last price at the checkpoint and q_tau is that blend. Standard
# library only, no randomness, no network, no clock: byte-identical every run.
import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
OUTPUT_DIR = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0, 0.674490, 1.644854]
WINDOW = 240
BLEND = 0.4


def price6(x):
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0


def emp_q(sorted_returns, tau):
    n = len(sorted_returns)
    if n == 0:
        return 0.0
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(n - 1, lo + 1)
    frac = pos - lo
    return sorted_returns[lo] * (1 - frac) + sorted_returns[hi] * frac


def forecast(market, horizon):
    prices = market['prices']
    returns = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    seg = returns[-WINDOW:] if len(returns) >= WINDOW else returns
    seg_sorted = sorted(seg)
    mu = sum(seg) / len(seg) if seg else 0.0
    var = sum((r - mu) ** 2 for r in seg) / (len(seg) - 1) if len(seg) > 1 else 0.0
    sigma = math.sqrt(max(var, 0.0))
    qs = [BLEND * emp_q(seg_sorted, tau) + (1 - BLEND) * z * sigma
          for tau, z in zip(QUANTILES, Z)]
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    x = prices[-1]
    rows = []
    for h in range(1, horizon + 1):
        s = math.sqrt(h)
        rows.append([price6(x + q * s) for q in qs])
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
