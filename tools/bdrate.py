#!/usr/bin/env python3
"""BD-rate (Bjontegaard) between the GPU-ME modes and stock x265 from the
bench720.ps1 summary CSV.

Usage:
    py -3 tools/bdrate.py bench-results/comparison/repro_summary.csv

Every CSV row is one rate/quality point of both encoders (same source, same
frame count).  The script reports, for PSNR and SSIM separately, the average
bitrate difference of the GPU-ME encoder at matched quality over the range
where the two quality curves overlap.

Sign convention: negative means the GPU-ME encoder needs *fewer* bits than
stock x265 for the same quality, i.e. negative is better.  Samples outside the
overlap are skipped, and the coverage is printed so that a partial overlap
cannot silently distort the number (the caveat matters when one curve stops
short of the other).
"""

import csv
import math
import sys


def bd_rate(curve, ref):
    """curve/ref: lists of (kbps, quality).  Returns (rate_diff, cover, span).

    rate_diff is the Bjontegaard delta rate of `curve` against `ref`
    (average over 21 samples of the log-rate difference at matched quality);
    cover is the fraction of the largest quality span that both curves share.
    """

    def interp(pts, q):
        pts = sorted(pts, key=lambda p: p[1])
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if y0 <= q <= y1:
                if y1 == y0:
                    return math.log(x0)
                t = (q - y0) / (y1 - y0)
                return math.log(x0) + t * (math.log(x1) - math.log(x0))
        return None

    lo = max(min(q for _, q in curve), min(q for _, q in ref))
    hi = min(max(q for _, q in curve), max(q for _, q in ref))
    if hi <= lo:
        return float("nan"), 0.0, (lo, hi)

    diffs = []
    for i in range(21):
        q = lo + (hi - lo) * i / 20
        a, b = interp(curve, q), interp(ref, q)
        if a is not None and b is not None:
            diffs.append(a - b)
    if not diffs:
        return float("nan"), 0.0, (lo, hi)

    span_ours = max(q for _, q in curve) - min(q for _, q in curve)
    span_ref = max(q for _, q in ref) - min(q for _, q in ref)
    cover = (hi - lo) / max(span_ours, span_ref)
    return math.exp(sum(diffs) / len(diffs)) - 1, cover, (lo, hi)


def main():
    if len(sys.argv) != 2:
        print(__doc__.strip())
        return 1
    with open(sys.argv[1], newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("no data rows in %s" % sys.argv[1])
        return 1

    pairs = (
        ("h_kbps", "h_psnr", "n_kbps", "n_psnr", "PSNR"),
        ("h_kbps", "h_ssim", "n_kbps", "n_ssim", "SSIM"),
    )
    for hx, hy, nx, ny, name in pairs:
        ours = [(float(r[hx]), float(r[hy])) for r in rows]
        ref = [(float(r[nx]), float(r[ny])) for r in rows]
        d, cover, (lo, hi) = bd_rate(ours, ref)
        note = "" if cover > 0.98 else "  (partial overlap, %.0f%% of the span)" % (100 * cover)
        print(
            "BD-rate %s: GPU-ME vs stock %+.2f%% bits at matched quality "
            "(overlap %.4f..%.4f, %d points)%s"
            % (name, 100 * d, lo, hi, len(rows), note)
        )
    print("(negative = the GPU-ME encoder needs fewer bits; stock x265 is the reference)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
