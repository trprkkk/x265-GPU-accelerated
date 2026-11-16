#!/usr/bin/env python3
"""Generate a deterministic synthetic test clip (YUV420p) for benchmarks.

The clip has moving boxes (large and small MVs), a panning background
texture, random-noise blocks (hard-to-predict), a diagonal ramp and a
scene cut in the middle -- a mix that exercises coarse + fine ME.

Usage:
    py -3 tools/make_testclip.py [outfile] [width] [height] [frames]

Defaults: testdata/motion720p.yuv 1280x720 240 frames.
"""
import math
import os
import random
import struct
import sys


def gen(path, w, h, frames, seed=1234):
    rnd = random.Random(seed)
    # background: tileable noise texture, fixed for the whole clip
    bg = [[rnd.randint(16, 235) for _ in range(w)] for _ in range(h)]

    boxes = []
    for _ in range(14):
        boxes.append((
            rnd.randint(0, w - 1), rnd.randint(0, h - 1),
            rnd.randint(20, 160), rnd.randint(20, 160),
            rnd.uniform(-8, 8), rnd.uniform(-8, 8),
            rnd.randint(60, 240)))
    noise_blocks = []
    for _ in range(6):
        noise_blocks.append((
            rnd.randint(0, w - 200), rnd.randint(0, h - 200),
            rnd.randint(40, 120), rnd.randint(40, 120),
            rnd.uniform(-4, 4), rnd.uniform(-4, 4)))
    noise_frame = [[rnd.randint(0, 255) for _ in range(w)] for _ in range(h)]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        for t in range(frames):
            if t == frames // 2:
                # scene cut: reshuffle boxes
                for i in range(len(boxes)):
                    x, y, bw, bh, vx, vy, v = boxes[i]
                    boxes[i] = (rnd.randint(0, w - 1), rnd.randint(0, h - 1), bw, bh, vx, vy, v)
            # luma
            yplane = bytearray(w * h)
            for yy in range(h):
                row = bg[yy]
                base = yy * w
                for xx in range(w):
                    v = row[(xx + t * 2) % w]
                    v += int(30 * math.sin((xx + yy) / 40.0 + t * 0.1))
                    yplane[base + xx] = max(0, min(255, v))
            sx = int(t * 0.7) % w
            nb = []
            for (x0, y0, bw, bh, vx, vy) in noise_blocks:
                nx = int(x0 + vx * t) % w
                ny = int(y0 + vy * t) % h
                nb.append((nx, ny, bw, bh))
                for yy in range(ny, min(h, ny + bh)):
                    for xx in range(nx, min(w, nx + bw)):
                        yplane[yy * w + xx] = noise_frame[yy][xx]
            for (x0, y0, bw, bh, vx, vy, v) in boxes:
                nx = int(x0 + vx * t) % w
                ny = int(y0 + vy * t) % h
                for yy in range(ny, min(h, ny + bh)):
                    for xx in range(nx, min(w, nx + bw)):
                        yplane[yy * w + xx] = v
            f.write(bytes(yplane))
            # chroma: flat with a moving stripe (kept simple; ME is luma)
            u = bytearray(w // 2 * h // 2)
            vplane = bytearray(w // 2 * h // 2)
            for yy in range(h // 2):
                for xx in range(w // 2):
                    p = yy * (w // 2) + xx
                    u[p] = 128 + int(40 * math.cos((xx + t * 3) / 30.0))
                    vplane[p] = 128 + int(40 * math.sin((yy + t) / 30.0))
            f.write(bytes(u))
            f.write(bytes(vplane))
    print("wrote %s (%dx%d %d frames)" % (path, w, h, frames))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "testdata/motion720p.yuv"
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
    h = int(sys.argv[3]) if len(sys.argv) > 3 else 720
    frames = int(sys.argv[4]) if len(sys.argv) > 4 else 240
    gen(path, w, h, frames)
