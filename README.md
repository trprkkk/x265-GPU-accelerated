# hybrid-x265-gpu

**日本語 (このページ)** | **[English: README.en.md](README.en.md)**

---

x265 (3.4, releasetag 3.4) をベースに、**CUDA / OpenCL による動き探索 (ME) の候補生成と
ルックアヘッド**を追加した CPU/GPU ハイブリッド HEVC エンコーダです。
[trprkkk/x264-GPU-accelerated](https://github.com/trprkkk/x264-GPU-accelerated) の x265 版です。

- GPU は NVENC のような固定機能ではなく、汎用 CUDA / OpenCL コアで動き探索の候補を計算します。
- **最終的な符号化判断はすべて CPU 側の x265 コードが行います。** GPU が返すのは
  32x32 ブロックごとの整数精度 MV 候補（最大 4 個/参照/リスト）と、ルックアヘッド用の
  8x8 lowres SAD のみで、x265 本体が本物のコスト関数（SAD/SATD + λ·MV コスト、RDO）で
  評価し直します。
- `--gpu-me 0`（既定）と `--gpu-lookahead 2` は GPU を一切使わず（既定）／
  使っても stock の HEX 探索をそのまま歩くため、**stock x265 とビット単位で同一**の出力を
  生成します（検証済み）。

---

## 1. 特徴

| 項目 | 内容 |
|---|---|
| 対象 | H.265/HEVC (x265 フォーク、8-bit / 10,12-bit ビルドでフル ME 有効) |
| GPU 利用 | 汎用 CUDA / OpenCL カーネル (NVENC/QSV 等の固定機能は不使用) |
| 画質 | GPU は候補を「追加」するだけ。CPU が本物のコストで選び直すため、実測でも PSNR 差は ±0.01 dB 以内 |
| 互換性 | 出力は通常の HEVC ストリーム |
| 対応解像度 | 任意 (x265 準拠。32 画素境界までパディングして GPU 解析。848x480 のような 32 の倍数でない解像度も GPU 加速) |
| 速度 | `--me full --subme 7` で **5.6 倍** (2.06 fps vs 0.37 fps、下記ベンチマーク) |
| 既定動作 | `--gpu-me 0` / `--gpu-lookahead 0`。従来と完全に同一・ビット一致 |
| フォールバック | CUDA 不可の GPU は OpenCL へ自動切替。GPU 自体が無い / 非対応条件では警告を出して自動的に CPU 動作へ |

### フル解像度 ME モード一覧

| `--gpu-me` | 動作 | 用途 |
|---|---|---|
| `0` | GPU オフ (既定) | 従来と同じ。ビット一致 |
| `1` | GPU 候補を追加してから通常探索 | 画質重視。候補が追加されるため画質は同等以上 |
| `2` | 候補を追加し、CPU 探索範囲を 4 に縮小 | バランス重視 |
| `3` | 候補を追加し、CPU の整数ペル探索を省略 | 最速。品質は内容依存 |
| `4` | 候補を追加し、CPU 探索範囲を 1 に縮小 | mode 2 と 3 の中間 |

### GPU ルックアヘッド一覧（8-bit ビルドのみ）

| `--gpu-lookahead` | 動作 | 用途 |
|---|---|---|
| `0` | GPU オフ (既定) | stock とビット一致 |
| `1` | GPU が coarse+refine の best-MV フィールドを並列計算し、CPU が隣接 MVP と比較して subpel 精密化 | 高速版。出力は stock と異なる（画質は同等: -0.10 dB） |
| `2` | GPU が全 MV の exact 8x8 SAD テーブルを計算し、CPU の stock HEX 探索がそのまま読む | **ビット一致版**（ボーナス）。遅い（テーブル 49x49 SAD/ブロック）ため正確性の実証用 |

---

## 2. 必要環境

- Windows x64
- Visual Studio (C++ ツールセット。動作確認: VS 18 Community / cl 19.51)
- **CUDA Toolkit 11.8** (`nvcc`) — Kepler〜Ampere 用カーネル (legacy fatbin: `sm_35,52,60,70,75,86`)
- **CUDA Toolkit 13.x** (`nvcc`) — Turing〜Blackwell 用カーネル (modern fatbin: `sm_75,80,86,89,90,100,120`)
  (13.x は sm_75 未満をコンパイルできないため、旧 GPU 対応に 11.8 も必要です。
  逆に 13.x は sm_100/sm_120 を生成できる唯一の系統です)
- `ENABLE_ASSEMBLY=OFF` が既定（NASM 不要）。NASM がある場合は `-DENABLE_ASSEMBLY=ON` で AVX2 により高速
- NVIDIA GPU (Kepler 以降) + 対応ドライバ (CUDA 利用時: 11.8 は R520 以降、13.x は R590 以降)
- OpenCL 1.2+ ランタイム (任意。`--gpu-me-backend 2` 用。`OpenCL.dll` があれば SDK 不要)

GPU 世代ごとのバックエンド (既定の自動選択。実行時に解析が失敗した場合は
もう一方のバックエンドへ 1 回だけ切り替え、警告を表示します):

| GPU 世代 | CUDA | OpenCL | 備考 |
|---|---|---|---|
| Kepler〜Volta (sm_35〜sm_70) | ✅ | ✅ | CUDA 11.8 カーネル (750Ti = sm_52 含む) |
| Turing〜Hopper (sm_75〜sm_90) | ✅ | ✅ | 両カーネルとも対応 (動作確認: RTX 2060 SUPER / sm_75) |
| Blackwell (sm_100/120) | ✅ | ✅ | CUDA 13.x カーネル (modern fatbin 必須) |
| それ以降の世代 | ✅ | ✅ | compute_90 PTX を JIT |

GPU 対応条件 (これ以外は自動で CPU フォールバック):

- フル ME: 8-bit / 10,12-bit どちらも対応 (`sizeof(pixel)` でカーネル切替)
- GPU ルックアヘッド: **8-bit のみ**。10/12-bit では警告して CPU へ (stock と同一出力)
- プログレッシブ。analysis save/load 時は CPU フォールバック (解析ファイルの意味を保つため)

---

## 3. ビルド手順

```bat
:: modern 版 (CUDA 13.3 優先。Turing〜Blackwell)
cmake -S x265\source -B build-vs -G "Visual Studio 18 2026" -A x64 ^
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ^
  -DENABLE_ASSEMBLY=OFF -DENABLE_GPU_ME=ON
cmake --build build-vs --config Release

:: legacy 版 (CUDA 11.8。Kepler〜Ampere。環境変数 CUDA_PATH を先に)
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
cmake -S x265\source -B build-legacy -G "Visual Studio 18 2026" -A x64 ^
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ^
  -DENABLE_ASSEMBLY=OFF -DENABLE_GPU_ME=ON
```

> 補足:
> CUDA カーネル (`x265/source/common/gpu_kernels.cu`) は configure 時に nvcc の `-fatbin` で
> コンパイルされ、RCDATA リソース (legacy=101、modern=102) として exe に埋め込まれます。
> configure を実行し直すと再生成されます。nvcc には `-ccbin` で MSVC ディレクトリが
> 指定されるため、vcvars 不要です。
> `-DHIGH_BIT_DEPTH=ON` で 10/12-bit ビルドもできます (GPU ルックアヘッドは CPU フォールバック)。
> `-DENABLE_GPU_ME=OFF` で機能ごと無効化できます。

---

## 4. 使い方

```bat
:: 例1: フル GPU ME (exhaustive 探索で最大効果)
x265.exe --preset slow --me full --subme 7 --gpu-me 3 -i input.yuv -o out.hevc

:: 例2: GPU ルックアヘッド (ビット一致)
x265.exe --preset medium --gpu-lookahead 2 -i input.yuv -o out.hevc

:: 例3: GPU ルックアヘッド (高速版)
x265.exe --preset medium --gpu-lookahead 1 -i input.yuv -o out.hevc
```

### オプション

| オプション | 値 | 既定 | 意味 |
|---|---|---|---|
| `--gpu-me` | 0..4 | 0 | フル解像度 GPU ME (上記) |
| `--gpu-me-backend` | 0..3 | 0 | 0=auto(CUDA→OpenCL)、1=CUDA、2=OpenCL、3=CPU 参照 (テスト専用) |
| `--gpu-me-range` | 4..128 | 64 | coarse 探索の全解像度範囲 (画素) |
| `--gpu-me-radius` | 0..8 | 2 | coarse 候補周辺の refine 半径 |
| `--gpu-me-refs` | 0..4 | 4 | 1リストあたり最大参照数 (0=全て) |
| `--gpu-me-device` | int | 0 | GPU デバイス番号 |
| `--gpu-lookahead` | 0,1,2 | 0 | GPU ルックアヘッド (上記。8-bit のみ) |

デバッグ用環境変数 (テスト専用): `X265_GPU_DUMP=1` (候補ダンプ)、
`X265_GPU_TIMING=1` (カーネル timing)、`X265_GPU_NOCACHE=1` (参照キャッシュ無効)、
`X265_GPU_FATBIN=legacy|modern` (画像選択)、`X265_GPU_SIMULATE_FAIL=cpu|cuda` (障害注入)。

### 動作ログ

```
x265 [info]: GPU ME: cuda / NVIDIA GeForce RTX 2060 SUPER (sm_75)
x265 [info]: GPU lookahead: mode 2 / cuda / NVIDIA GeForce RTX 2060 SUPER (sm_75)
```

ログに `GPU ME:` / `GPU lookahead:` が出なければ GPU 未使用 (CPU フォールバック) です。

---

## 5. ベンチマーク

### 測定環境
- NVIDIA GeForce RTX 2060 SUPER (sm_75)、Windows、VS 18、組み込み合成 1280x720 クリップ、`tools/make_testclip.py` 生成
- `bench720.ps1` で BD-rate 概要 CSV (`tools/bdrate.py`) を出力

### 結果 (fps / サイズ変化、CRF 23)

| preset | `-F` | mode 0 | mode 3 | サイズ変化 |
|---|---|---|---|---|
| medium | default | 13.82 fps | 11.67 fps | +0.2 % |
| medium | 1 | 8.93 fps | 8.98 fps | +0.2 % |
| slow | default | 6.54 fps | 5.53 fps | +0.3 % |
| slow | 1 | 4.39 fps | 4.36 fps | +0.3 % |
| slow `--me full --subme 7` | default | 0.37 fps | **2.06 fps (5.6×)** | +0.36 % |
| placebo | default | 0.54 fps | 0.48 fps | −0.03 % |

GPU 解析コストは 720p で約 3.7 ms/frame (coarse 1.4 ms、refine 1.9 ms、転送 0.3 ms)。
CPU の整数探索が支配的な場合にのみ効き、`--me full --subme 7` が本命の用途です
(5.6 倍、+0.36 % ビット)。medium/slow の HEX/UMH は安いため、`-F 1` では互角、
既定 `-F` では参照recon 待ちのフレーム並列喪失で損します (下記「既知の制限」)。

GPU ルックアヘッド (720p、CRF23 60f、最良 4 回):
stock 14.00 fps / 7514 kb/s / 40.049 dB → mode 1 13.96 fps / 7444 kb/s (−0.9 %) / 39.952 dB (−0.10 dB)。
lookahead 律速の pass1 (`--pass 1 --no-slow-firstpass`, slower) でも 18.18 fps stock vs 17.68 fps mode 1 と互角。
8x8 lowres SAD は CPU で十分安いため、フル ME と同じく exhaustive でないと元が取れない、という結論です。

### 既知の制限: フレーム並列との重なり

解析は**完全な**参照プレーンを必要とするため、参照のループフィルタが最終行を
公開する (`m_reconRowFlag[m_numRows-1]`) まで待ちます。stock は参照が
`m_refLagRows` 行前まで進んだ時点で開始するので、`-F > 1` で並列度が落ちます
(`-F 1` と `-F default` の列差)。ブロックの参照窓は局所的なので、stock と同じ
行フラグで行バンド paced に dispatch すれば回復可能です (未実装。先に3バックエンドの
発散を直すことを優先したため)。

---

## 6. 検証済み事項

互いに同じ条件 (`ENABLE_ASSEMBLY=OFF`) でビルドした pristine な upstream との比較:

- `--gpu-me 0` は `ultrafast/fast/medium`、CRF/ABR、848x480 (32 の倍数でない解像度) で
  **バイト同一**。10-bit ビルドでも pristine 10-bit とバイト同一。
- `--gpu-lookahead 2` は medium/ultrafast/848x480 で **バイト同一** (3 バックエンド、`--lookahead-slices 1/4`、`-F 1/4`、`--bframes 0`、
  ultrafast 含む)。整数探索の SAD の 99.5 % はテーブル命中 (残りは設計通り subpel の half-pel プレーンと ±24 超)。
- CUDA / OpenCL / CPU 参照の3バックエンドは **ビット同一** (fast, slow, 848x480, `--me full`、`--gpu-lookahead 1/2`、8-bit/10-bit 全て)。
- 全モードとも実行ごとに決定論的 (`-F 4` 含む)。stock 自体が `-F` で出力が変わるため同値比較のみ有効。
- フレーム種別決定は mode 0 と 1-4 で一致 (`--csv-log-level 2` の種別列)。
- CRF23 (40 フレーム) のレート歪み: 全モード・全プリセットで PSNR ±0.01 dB、ビットレート ±0.4 % 以内。
- フォールバック: `--analysis-save` (警告、CPU)、`--gpu-me-device 99` (警告、0 にクランプ)、
  破損 CUDA fatbin (OpenCL へ切替・同一出力)、`X265_GPU_SIMULATE_FAIL=cuda` (切替・同一出力)、
  mode 2 の `X265_GPU_SIMULATE_FAIL=cuda` も OpenCL へ切替・同一出力。
- `X265_GPU_NOCACHE=1` は出力不変 (参照キャッシュ透過)。
- 240 フレーム 720p ストレステスト: 警告/エラーなし、ピーク RSS は約 395 MB でプラトー (リークなし)。
- GPU ルックアヘッド監査で発見・修正した3件: フィールド struct のリーク、half-pel プレーンへの
  テーブル誤参照 (整数ペル確保範囲内必須に修正)、テーブル・メモリ上限 (`X265_GPU_LA_TABLE_MAX_BYTES` 128 MB、
  約 1440p 以上は警告して stock CPU へ)。

無関係な upstream 問題: `--no-wpp --slices N` は pristine で segfault します
(「Multiple-Slices mode must be enable Wavefront Parallel Processing」後のエラーパス)。
本フォークが原因ではありません。

---

## 7. 内部設計

### 設計上の不変条件

- **候補のみ (Candidates only)**。GPU は符号化判断を一切書きません。32x32 ブロック毎に
  1) 1/4 解像度 coarse 探索 (8x8 SAD、step-2 + 近傍パス)、2) 全解像度 refine
  (step-2 部分サンプル 32x32 SAD) を開ループで実行し、参照毎に最大 4 候補を渡します。
  CPU が本物の x265 コストで再評価します (x264-GPU と同方式)。
- **`--gpu-me 0` は stock とビット同一** (GPU コンテキスト自体を作らない)。
- **全方向の graceful fallback**。失敗時は CUDA→OpenCL へ 1 回だけ切替、無理なら株価 CPU 探索へ。
- **参照完全性 (Reference completeness)**。参照プレーンはループフィルタの最終行公開後にのみ読みます。
  この待ちが無いと並列フレームが部分再構成の参照を見るため、3 バックエンドが発散しました。
- **3 バックエンド bit-identical** (同一の反復順序と比較、CPU/CUDA/OpenCL)。
- 新規パラメータは `x265_param` 末尾に追加済みで、既存オフセット不変。`X265_BUILD` は upstream の 198 のまま
  (user-data SEI が一致し、`--gpu-me 0` が stock とバイト互換)。
- ディスパッチャはプロセス全体の SRWLOCK で直列化 (x265 は並列に符号化するため)。
  候補は参照 luma 平面ポインタ + POC でキー付け。

### バックエンド (CUDA / OpenCL)

- CUDA: ドライバ API を `nvcuda.dll` から動的ロード。fatbin リソース (101/102) からカーネル取得。
  pinned staging リング、参照キャッシュ (平面ポインタ + POC キー)。
- OpenCL: `OpenCL.dll` を動的ロード。カーネルソース内包、`PIX` マクロで 8/16-bit 切替。
- CPU (`gpu_cpu.cpp`): スカラー参照実装。テストと決定的フォールバック。
- CMAKE 4 の `2.8.8` 拒否には `CMAKE_POLICY_VERSION_MINIMUM=3.5` を設定して対応。

### 実装ファイル

```
x265/source/common/gpu.{c,h}            ディスパッチャ + 公開 API
x265/source/common/gpu_kernels.{cu,h}   CUDA カーネル (8/16-bit テンプレート)
x265/source/common/gpu_cuda.cpp         CUDA バックエンド
x265/source/common/gpu_cl.c             OpenCL バックエンド
x265/source/common/gpu_cpu.cpp          CPU 参照バックエンド
x265/source/encoder/slicetype.{cpp,h}   GPU ルックアヘッド (CostEstimateGroup)
x265/source/encoder/motion.{cpp,h}      MotionEstimate の SAD テーブル / gpuReplaceSearch
x265/source/encoder/{encoder.cpp,frameencoder.cpp,search.cpp}
tools/                                  テストクリップ生成、バックエンド同一性ハーネス、bdrate.py
selftest.ps1 / bench720.ps1             承認テスト / ベンチマーク
x265/                                   本フォーク本体
```

---

## 8. About (GitHub)

> x265 フォーク + CUDA/OpenCL による CPU/GPU ハイブリッド HEVC エンコーダ:
> --gpu-me 0 と --gpu-lookahead 2 は本家 x265 とビット一致、exhaustive 探索で 5.6 倍。
> CUDA と OpenCL は同一のビットストリームを生成 (mode 2 の PSNR は stock と完全一致)。

## 9. ライセンス

本フォークは upstream の x265 と同じく **GNU General Public License Version 2 (GPLv2)** です
(全文は [`LICENSE`](LICENSE) を参照。upstream `x265/COPYING` と同一)。
GPU 追加コード (`x265/source/common/gpu*` 等) も同じライセンスの下で配布します。
