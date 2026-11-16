# Hybrid CPU/GPU x265 encoder (this fork)

**日本語: [README.md](README.md)** | **English (this page)**

---

A fork of **x265 (3.4, releasetag 3.4)** in which parts of motion estimation
run on the GPU's **general-purpose compute** (CUDA / OpenCL) instead of
fixed-function hardware encoding (NVENC/QSV are *not* used).  The GPU
pre-analysis supplies integer-MV **candidates**; the CPU still performs
subpel refinement, mode decision, RDO, transform/quant sparks, SAO/deblock,
rate control and entropy coding with the stock x265 code.  The intent is
x265-equal (or better) quality with a faster encoder — modelling on
[trprkkk/x264-GPU-accelerated](https://github.com/trprkkk/x264-GPU-accelerated).

## Design invariants

* **Candidates only.** The GPU never writes coding decisions. It runs an
  open-loop, coarse-to-fine search per 32x32 block:
  1. quarter-resolution coarse search (8x8 SAD, stride-2 grid + neighbour
     pass), then
  2. full-resolution refinement (stride-2 subsampled 32x32 SAD).
  Up to 4 candidates per reference (max 4 references per list) are handed to
  the CPU, which re-evaluates each with the real x265 cost (SAD/SATD + MV
  cost via `Search`), exactly like the x264-GPU reference implementation.
* **`--gpu-me 0` is bit-identical to stock x265** (the GPU context is never
  created; the CPU search path is untouched).
* **Graceful fallback everywhere.** Any GPU failure (no device, kernel
  failure, simulated failure) switches backend CUDA→OpenCL once, else falls
  back to the stock CPU search permanently for that encode. Missing GPU is
  never fatal.
* **Reference completeness.** Reference planes are read only after the
  reference frame's loop filter has published the last row
  (`m_reconRowFlag[m_numRows-1]`, the same signal the row encoder gates
  on).  Reading the planes without that wait made the three backends
  diverge, because frames encoded in parallel see partially reconstructed
  references — an input the CPU search (which gates per row) never sees.
  The cost is lost frame-parallel overlap; see "Known limitation" below.
* **Backends are bit-identical by construction** (same iteration order and
  comparisons in CPU/CUDA/OpenCL; verified: `selftest.ps1` #6).
* Works in **8-bit and 10/12-bit builds** (`sizeof(pixel)` selects the
  kernel variant: `uint8_t` / `uint16_t`). Analysis save/load still forces
  the CPU fallback.
* The new parameters are **appended to the end of `x265_param`**, so every
  pre-existing field keeps its offset; `X265_BUILD` stays at upstream's 198
  so the user-data SEI (which carries the build number) stays identical and
  `--gpu-me 0` remains byte-for-byte comparable with stock.  Code that sets
  the new fields must be rebuilt against this fork's headers.
* Dispatcher serialization: a process-wide lock (SRWLOCK on Windows), since
  x265 encodes frames in parallel and backends are not internally
  thread-safe. Candidates are keyed by reference luma-plane pointer + POC.

## Options

| Option | Values | Default | Meaning |
|---|---|---|---|
| `--gpu-me` | 0..4 | 0 | 0 off (stock), 1 seed CPU search with GPU candidates, 2 seed + CPU merange 4, 3 seed + skip CPU integer search, 4 seed + CPU merange 1 |
| `--gpu-me-backend` | 0..3 | 0 | 0 auto (CUDA→OpenCL), 1 CUDA, 2 OpenCL, 3 CPU reference (testing, never auto-selected) |
| `--gpu-me-range` | 4..128 | 64 | full-resolution range of the coarse search (pixels) |
| `--gpu-me-radius` | 0..8 | 2 | refinement radius around coarse candidates |
| `--gpu-me-refs` | 0..4 | 4 | max reference frames analysed per list (0 = all) |
| `--gpu-me-device` | int | 0 | GPU device index |
| `--gpu-lookahead` | 0,1,2 | 0 | GPU lowres lookahead (8-bit build only, see below): 0 off (stock, default), 1 fast GPU lowres MV field (coarse+refine, output differs from stock, quality comparable), 2 bitperfect GPU SAD table + stock CPU HEX on top (output byte-identical to stock) |

Debug environment variables: `X265_GPU_DUMP=1` (per-call candidate dump +
reference checksums), `X265_GPU_TIMING=1` (backend kernel timings),
`X265_GPU_NOCACHE=1` (disable reference-plane cache), `X265_GPU_FATBIN=legacy|modern`
(CUDA kernel image), `X265_GPU_SIMULATE_FAIL=cpu|cuda` (fault injection for
fallback tests). All are test/diagnostic hooks only.

## Build (this machine)

Windows + Visual Studio + CUDA Toolkit (nvcc) is enough; OpenCL comes from
the system `OpenCL.dll` (no SDK install).  x265 is configured with the
Visual Studio generator (`x265` subdir is the fork):

```bat
cmake -S x265\source -B build-vs -G "Visual Studio 18 2026" -A x64 ^
  -DENABLE_ASSEMBLY=OFF -DENABLE_GPU_ME=ON
cmake --build build-vs --config Release
```

Notes:

* `ENABLE_ASSEMBLY=OFF` is required on machines without NASM; results are
  comparable only against stock built the same way.
* The CUDA kernels are compiled to a fatbin **at configure time** (VS
  generators don't order build-time custom commands before `.rc`
  compilation); re-running the configure step rebuilds them. nvcc is
  pointed at the MSVC directory (`-ccbin`) from `CMAKE_CXX_COMPILER`, so no
  vcvars environment is needed.
* Legacy CUDA (≤11) uses resource id 101; modern (CUDA 12+) uses 102 with
  matching `-gencode` arch lists. `X265_GPU_FATBIN` overrides the choice.
* `-DENABLE_GPU_ME=OFF` builds the fork with the feature compiled out.
* `-DHIGH_BIT_DEPTH=ON` (10/12-bit) builds compile; full-res GPU ME logs
  "GPU ME: only supported in the 8-bit build" in the 8-bit sense (it is
  supported in both — see the 8/10/12-bit note above) and the output is
  identical to `--gpu-me 0` in 10-bit (verified).  The fork targets
  `sizeof(pixel)`-selected kernels.
* GPU lookahead (`--gpu-lookahead`) is **8-bit only**: in a 10/12-bit
  build the encoder logs a warning and uses the stock CPU lookahead
  (so mode 2 there is still stock-identical, just not GPU).  Full-res
  GPU ME remains 8/10/12-bit.

## GPU lookahead (`--gpu-lookahead`, 8-bit build)

The lookahead (`slicetype.cpp` `CostEstimateGroup::estimateCUCost`) runs
the lowres HEX search per 8x8 block.  Two GPU modes are provided:

* **Mode 2 — bitperfect (bonus, `OpenCLlookaheadbitperfect`-style).**
  The GPU computes the exact 8x8 SAD for every MV in `[-24,24]²` once per
  `(fenc, ref)` pair into a table.  The CPU then runs the *unmodified*
  stock HEX search, with `MotionEstimate::sad/sad_x3/sad_x4` answered
  from the table (out-of-range / out-of-plane requests fall back to the
  exact CPU primitive, so results are identical by construction).
  Verified byte-identical to stock (`--gpu-lookahead 0`) for
  medium/ultrafast and 848x480, on all three backends; frame types
  identical.  Measured hit rate at 720p: 34 % of all SAD calls, but
  **99.5 % of the integer-search SADs** (the remaining misses are the
  half-pel planes used by the subpel refinement, which the table does
  not cover by design, plus 0.5 % MVs beyond ±24).  It is slower than
  stock (720p medium 60f: ~7.6 fps vs 13.9 fps) because the table costs
  49×49 SADs per block — a correctness demonstration, not a speed
  feature.  The table is 35 MB per reference at 720p and 78 MB at
  1080p; above `X265_GPU_LA_TABLE_MAX_BYTES` (128 MB, i.e. ~1440p and
  up) the encoder warns and keeps the stock CPU lookahead instead.
* **Mode 1 — fast (main line, x264 OpenCL-lookahead style).**  The GPU
  computes a coarse (step-4) + refine (±4) integer best-MV field in
  parallel.  The CPU picks the better of (neighbour MVP, GPU MV) by
  SATD (preserving spatial coherence), then subpel-refines with the
  stock code (`gpuReplaceSearch`, same as `--gpu-me 3`).  Slice-type,
  scenecut, cu-tree, AQ and rate control stay on the CPU.  Outputs
  differ from stock by design.  Measured (720p CRF23 60f, RTX 2060
  SUPER, best of 4): stock 14.00 fps / 7514 kb/s / 40.049 dB global
  PSNR; mode 1 13.96 fps / 7444 kb/s (−0.9 %) / 39.952 dB (−0.10 dB).
  In a lookahead-bound run (`--pass 1 --no-slow-firstpass`, slower
  preset) the two are level: 18.18 fps stock vs 17.68 fps mode 1.
  That is the honest headline: **the GPU lookahead is neutral, not a
  speed win**, because the 8x8 lowres SAD search is cheap on the CPU —
  the same conclusion as for the full-resolution ME, which only pays
  off for exhaustive search.  What it buys is a wider integer search
  (the GPU covers the whole ±24 window, the stock HEX search walks a
  path from the MVP) at no measurable cost; quality is comparable.
  All three backends produce bit-identical mode-1 streams.  HME (4x4)
  and weighted-prediction references always use the stock CPU search.
  Analysis save/load disables GPU lookahead (so analysis files keep
  their meaning), and any GPU failure falls back to stock CPU.

## Verify

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File selftest.ps1
```

Checks: `--gpu-me 0` determinism/no-GPU (and byte-identity against
`build-baseline/` when present), modes 1-4 run cleanly, CUDA engagement,
analysis-save and 10-bit fallbacks, determinism of every mode,
**CUDA/OpenCL/CPU-reference backend bit-identity**, `X265_GPU_NOCACHE`
and broken-fatbin fallback identity, an 848x480 (non-multiple-of-32)
clip, CUDA-failure → OpenCL fallback identity, plus a compatibility smoke
pass (2-pass stats, qpfile, zones, `--frame-threads 4`).

## What has been verified

### GPU lookahead audit (this fork, `--gpu-lookahead`)

Three defects were found in a review pass and fixed:

1. **Leak of the GPU field struct.** The per-estimate cleanup was guarded
   by `gpuLaActive && gpuLaAll`, so an estimate that allocated the struct
   but produced no valid field (weighted-prediction reference in list 0
   only, geometry mismatch) never freed it.
2. **Half-pel plane confusion in the bitperfect lookup.** The SAD
   function pointers are also fed half-pel planes by
   `Lowres::lowresQPelCost`, whose addresses are unrelated to the table
   base. A pointer difference that happened to land inside the ±24 window
   (possible for small frames / `maxCUSize 16`, where a plane is smaller
   than the window) would have returned a wrong SAD silently. The lookup
   now requires the 8x8 block to lie inside the integer-pel plane
   allocation, so such a request always takes the exact CPU primitive.
3. **Unbounded table memory.** Mode 2 allocates
   `cuCount × 49 × 49 × 4` bytes per reference (78 MB at 1080p, 311 MB at
   4K) and several estimates can be in flight; the encoder now warns and
   keeps the stock CPU lookahead above `X265_GPU_LA_TABLE_MAX_BYTES`
   (128 MB, ~1440p).
4. **Fallback policy was weaker than documented.** A lookahead backend
   failure went straight to the CPU, skipping the CUDA→OpenCL switch the
   full-resolution ME performs. Both modes now retire the failing backend
   once and retry on the other one (`X265_GPU_SIMULATE_FAIL=cuda` →
   "switched to opencl", output still byte-identical), and only fall back
   to the stock CPU lookahead if the retry fails as well.

Verified after the fixes: mode 2 still byte-identical to stock on all
three backends (720p/848x480, `--lookahead-slices 1` and 4, `-F 1` and
`-F 4`, P-only `--bframes 0`, ultrafast), mode 1 still bit-identical
across backends and deterministic, `--gpu-me 0..4` behaviour unchanged,
10-bit fallback unchanged, and mode 1+mode 3 combined works.

Against a pristine upstream build of the same commit (built with the same
`ENABLE_ASSEMBLY=OFF`):

* `--gpu-me 0` is **byte-identical** to stock x265 for `preset
  ultrafast/fast/medium`, CRF and ABR, and for an 848x480
  (non-multiple-of-32) clip; likewise in a **10-bit build** (against a
  pristine 10-bit build).
* CUDA / OpenCL / CPU-reference backends produce **bit-identical** streams
  (fast, slow, 848x480, `--me full`) in **both 8-bit and 10-bit** builds.
* Every mode is deterministic across runs, including `-F 4`
  (frame-parallel).  Note: stock x265 itself changes output with `-F`,
  so only equal-`-F` runs are comparable.
* Frame-type decisions are identical between mode 0 and modes 1-4 for
  every preset (x265 `--csv-log-level 2` type column).
* Rate/distortion parity at CRF 23 (40 frames): PSNR within ±0.01 dB and
  bitrate within ±0.4 % of mode 0 for all modes and presets.
* Fallbacks: `--analysis-save` (warning, CPU), `--gpu-me-device 99`
  (warns, clamps to 0 on both backends), broken CUDA fatbin (switches to
  OpenCL, identical output), `X265_GPU_SIMULATE_FAIL=cuda` (switches,
  identical output).
* `X265_GPU_NOCACHE=1` does not change the output (reference cache is
  transparent).
* Stress: 240-frame 720p run, no warnings/errors, peak RSS plateaus
  (~395 MB) and drops at flush — no unbounded growth.

## Benchmarks (RTX 2060 SUPER, synthetic 1280x720 clip, CRF 23)

`bench720.ps1` produces a BD-rate summary CSV (`tools/bdrate.py`).
Best-of-2, 40 frames (8 frames for the exhaustive rows):

| preset | `-F` | mode 0 | mode 3 | size Δ |
|---|---|---|---|---|
| medium | default | 13.82 fps | 11.67 fps | +0.2 % |
| medium | 1 | 8.93 fps | 8.98 fps | +0.2 % |
| slow | default | 6.54 fps | 5.53 fps | +0.3 % |
| slow | 1 | 4.39 fps | 4.36 fps | +0.3 % |
| slow `--me full --subme 7` | default | 0.37 fps | **2.06 fps (5.6×)** | +0.36 % |
| placebo | default | 0.54 fps | 0.48 fps | −0.03 % |

The GPU analysis costs ~3.7 ms/frame at 720p (measured with
`X265_GPU_TIMING=1`: coarse 1.4 ms, refine 1.9 ms, uploads 0.3 ms).  So it
only pays off when the CPU integer search dominates — the
`--me full --subme 7` row is the intended use case (5.6× at +0.36 %
bits), matching the x264-GPU result that gains appear only for
exhaustive search.  At medium/slow the HEX/UMH search is cheap, so the
GPU is break-even at `-F 1` and a net loss at the default `-F`.

### Known limitation: frame-parallel overlap

The analysis needs the **complete** reference planes (a 32x32 block's
search window can reach any neighbouring row), so it waits for the
reference's last `m_reconRowFlag` before running.  Stock x265 instead
starts row *r* as soon as the references published row *r +
m_refLagRows*, so with `-F > 1` the fork gives up some frame-parallel
overlap: that is the difference between the `-F 1` and `-F default`
columns above.  The candidates for a block only depend on a local
reference window, so a row-band paced analysis (dispatch band *r* after
the same row-flag wait stock uses) would recover the overlap; it is not
implemented yet.  Correctness was prioritised because reading
partially-reconstructed references made the three backends diverge
before this wait was added.

### Unrelated upstream issue

`--no-wpp --slices N` segfaults (exit 139) after printing "Multiple-Slices
mode must be enable Wavefront Parallel Processing".  This reproduces on
pristine x265 with `--gpu-me 0`, i.e. it is an upstream error-path bug in
this x265 revision, not caused by the GPU code.


## Layout

```
x265/          the fork (source/… modified; new: source/common/gpu*)
tools/         test clip generator, backend-identity harness, bdrate.py
selftest.ps1 / bench720.ps1   acceptance + benchmark scripts
testdata/      synthetic clip (tools/make_testclip.py) and run artifacts
build-vs/      CMake (VS generator) build tree
build-baseline/  pristine upstream build used for the bit-identity checks
baseline-src/    pristine upstream sources (git archive of the fork's HEAD)
verify-out/      verification logs/outputs from the audit run
x264ref_tmp/   reference x264 GPU-accelerated sources (read-only)
Fable5.2memo*.txt  design memos that shaped this architecture
```

## License

This fork is **GNU General Public License Version 2 (GPLv2)**, the same as
upstream x265 (see [`LICENSE`](LICENSE), identical to upstream `x265/COPYING`).
The added GPU code (`x265/source/common/gpu*` etc.) is distributed under the
same license.
