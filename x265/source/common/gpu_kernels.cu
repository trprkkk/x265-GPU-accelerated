// Device code for the CUDA candidate generator.
//
// The kernels do a coarse exhaustive search on a quarter-resolution luma copy
// and then refine the best four integer-pel positions at full resolution.
// They are *open-loop*: the host side never writes a coding decision, and
// search.cpp / motion.cpp re-evaluate every returned vector with the real
// x265 cost function.
//
// The OpenCL backend (gpu_cl.c) and the scalar CPU reference (gpu_cpu.cpp)
// implement the same algorithm with the same iteration order and the same
// strict comparisons, so all backends produce bit-identical candidates.
//
// This file contains no host code.  It is compiled with `nvcc -fatbin` and
// the resulting fat binary is embedded in x265.exe (see gpu_cuda.cpp), so
// the kernels can be built with a toolkit that still supports older GPUs
// even though the host code is compiled by the (much newer) MSVC toolchain.
//
// The kernels are declared extern "C" so that cuModuleGetFunction() can find
// them by their plain name.  Every algorithm exists twice, for 8-bit
// (uint8_t) and 10/12-bit (uint16_t) luma samples; the host side picks the
// pair from sizeof(pixel) (see x265_gpu_create()).
//
// The search grid is 32x32 blocks (HEVC): at quarter resolution a block is
// 8x8, so the coarse SAD is 8x8 and the full-resolution refinement is a
// stride-2 subsampled 32x32 SAD (16x16 samples).  With 16-bit samples the
// largest possible SAD is 256 * 4095 = 1,048,320, so uint32_t accumulators
// are always wide enough.

#include "gpu_kernels.h"

/* CUDA's device min()/max() are declared in the runtime headers; provide them
 * here so the kernels compile against the minimal cuda_runtime.h stub (see
 * cuda_stub/) that keeps the C++ standard library out of the device build. */
__device__ __forceinline__ int gpu_min( int a, int b ) { return a < b ? a : b; }
__device__ __forceinline__ int gpu_max( int a, int b ) { return a > b ? a : b; }

template <typename PIX>
__device__ __forceinline__ uint32_t sad8x8( const PIX *a, int as,
                                            const PIX *b, int bs )
{
    uint32_t s = 0;
#pragma unroll
    for( int j = 0; j < 8; j++ )
#pragma unroll
        for( int i = 0; i < 8; i++ )
        {
            int d = (int)a[j*as+i] - (int)b[j*bs+i];
            s += (uint32_t)(d < 0 ? -d : d);
        }
    return s;
}

__device__ __forceinline__ void insert_topk( Cand *best, int x, int y, uint32_t sad )
{
    if( sad >= best[GK-1].sad )
        return;
    int i = GK - 1;
    while( i > 0 && sad < best[i-1].sad )
    {
        best[i] = best[i-1];
        i--;
    }
    best[i].mv[0] = (int16_t)x;
    best[i].mv[1] = (int16_t)y;
    best[i].sad = sad;
}

template <typename PIX>
__device__ __forceinline__ void downsample4_impl( const PIX *__restrict__ src, int sw, int sh,
                                                  PIX *__restrict__ dst, int dw, int dh )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if( x >= dw || y >= dh )
        return;
    int sx = x * 4, sy = y * 4;
    int sum = 0;
#pragma unroll
    for( int j = 0; j < 4; j++ )
#pragma unroll
        for( int i = 0; i < 4; i++ )
        {
            int px = gpu_min( sx + i, sw - 1 );
            int py = gpu_min( sy + j, sh - 1 );
            sum += (int)src[py * sw + px];
        }
    dst[y * dw + x] = (PIX)((sum + 8) >> 4);
}

template <typename PIX>
__device__ __forceinline__ void coarse_impl( const PIX *__restrict__ cur, int cw, int ch,
                                             const PIX *__restrict__ ref, int rw, int rh,
                                             int bw, int bh, int range, Cand *out )
{
    int mb = blockIdx.x * blockDim.x + threadIdx.x;
    if( mb >= bw * bh )
        return;
    int mx = mb % bw, my = mb / bw;
    int bx = mx * 8, by = my * 8;
    Cand best[GK];
#pragma unroll
    for( int i = 0; i < GK; i++ )
    {
        best[i].mv[0] = 0;
        best[i].mv[1] = 0;
        best[i].sad = 0xFFFFFFFFu;
    }
    /* The full-resolution range is expressed in luma pixels; the quarter
     * resolution grid covers it with range/4 steps.  The first pass runs on
     * a two-step grid, the second pass tests the eight neighbours of the
     * winner, so the grid is only ~1/4 as expensive with practically the
     * same result. */
    int r = range / 4;
    if( r < 1 )
        r = 1;
    for( int dy = -r; dy <= r; dy += 2 )
        for( int dx = -r; dx <= r; dx += 2 )
        {
            int x = gpu_max( 0, gpu_min( bx + dx, cw - 8 ) );
            int y = gpu_max( 0, gpu_min( by + dy, ch - 8 ) );
            uint32_t sad = sad8x8<PIX>( cur + by * cw + bx, cw, ref + y * rw + x, rw );
            insert_topk( best, (x - bx) * 4, (y - by) * 4, sad );
        }
    int fx = bx + best[0].mv[0] / 4;
    int fy = by + best[0].mv[1] / 4;
    for( int dy = -1; dy <= 1; dy++ )
        for( int dx = -1; dx <= 1; dx++ )
        {
            if( !dx && !dy )
                continue;
            int x = gpu_max( 0, gpu_min( fx + dx, cw - 8 ) );
            int y = gpu_max( 0, gpu_min( fy + dy, ch - 8 ) );
            uint32_t sad = sad8x8<PIX>( cur + by * cw + bx, cw, ref + y * rw + x, rw );
            insert_topk( best, (x - bx) * 4, (y - by) * 4, sad );
        }
    for( int i = 0; i < GK; i++ )
        out[mb * GK + i] = best[i];
}

template <typename PIX>
__device__ __forceinline__ void refine_impl( const PIX *__restrict__ cur, int cw, int ch,
                                             const PIX *__restrict__ ref, int rw, int rh,
                                             int bw, int bh, int radius,
                                             const Cand *__restrict__ coarse, Cand *out )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bw * bh * GK;
    if( idx >= total )
        return;
    int mb = idx / GK;
    int mx = mb % bw, my = mb / bw;
    int bx = mx * 32, by = my * 32;
    int cx = bx + coarse[idx].mv[0];
    int cy = by + coarse[idx].mv[1];
    const PIX *pc = cur + by * cw + bx;
    Cand best;
    best.mv[0] = 0;
    best.mv[1] = 0;
    best.sad = 0xFFFFFFFFu;
    for( int dy = -radius; dy <= radius; dy++ )
        for( int dx = -radius; dx <= radius; dx++ )
        {
            int x = gpu_max( 0, gpu_min( cx + dx, cw - 32 ) );
            int y = gpu_max( 0, gpu_min( cy + dy, ch - 32 ) );
            const PIX *pr = ref + y * rw + x;
            uint32_t sad = 0;
            /* Subsample the 32x32 block with a stride of two pixels.  The
             * candidates are re-scored on the CPU with the real x265 metric,
             * so this only needs to rank positions, not to be exact. */
#pragma unroll
            for( int j = 0; j < 32; j += 2 )
#pragma unroll
                for( int i = 0; i < 32; i += 2 )
                {
                    int d = (int)pc[j*cw+i] - (int)pr[j*rw+i];
                    sad += (uint32_t)(d < 0 ? -d : d);
                }
            if( sad < best.sad )
            {
                best.sad = sad;
                best.mv[0] = (int16_t)(x - bx);
                best.mv[1] = (int16_t)(y - by);
            }
        }
    out[idx] = best;
}

/* ---- 8-bit entry points ------------------------------------------------- */

extern "C" __global__ void downsample4_8( const uint8_t *__restrict__ src, int sw, int sh,
                                          uint8_t *__restrict__ dst, int dw, int dh )
{
    downsample4_impl<uint8_t>( src, sw, sh, dst, dw, dh );
}

extern "C" __global__ void coarse_search_8( const uint8_t *__restrict__ cur, int cw, int ch,
                                            const uint8_t *__restrict__ ref, int rw, int rh,
                                            int bw, int bh, int range, Cand *out )
{
    coarse_impl<uint8_t>( cur, cw, ch, ref, rw, rh, bw, bh, range, out );
}

extern "C" __global__ void refine_search_8( const uint8_t *__restrict__ cur, int cw, int ch,
                                            const uint8_t *__restrict__ ref, int rw, int rh,
                                            int bw, int bh, int radius,
                                            const Cand *__restrict__ coarse, Cand *out )
{
    refine_impl<uint8_t>( cur, cw, ch, ref, rw, rh, bw, bh, radius, coarse, out );
}

/* ---- 10/12-bit entry points (pixel == uint16_t) ------------------------- */

extern "C" __global__ void downsample4_16( const uint16_t *__restrict__ src, int sw, int sh,
                                           uint16_t *__restrict__ dst, int dw, int dh )
{
    downsample4_impl<uint16_t>( src, sw, sh, dst, dw, dh );
}

extern "C" __global__ void coarse_search_16( const uint16_t *__restrict__ cur, int cw, int ch,
                                             const uint16_t *__restrict__ ref, int rw, int rh,
                                             int bw, int bh, int range, Cand *out )
{
    coarse_impl<uint16_t>( cur, cw, ch, ref, rw, rh, bw, bh, range, out );
}

extern "C" __global__ void refine_search_16( const uint16_t *__restrict__ cur, int cw, int ch,
                                             const uint16_t *__restrict__ ref, int rw, int rh,
                                             int bw, int bh, int radius,
                                             const Cand *__restrict__ coarse, Cand *out )
{
    refine_impl<uint16_t>( cur, cw, ch, ref, rw, rh, bw, bh, radius, coarse, out );
}

/* ---- GPU lookahead kernels (8x8 lowres blocks) ---------------------------
 *
 * Both kernels share the exact 8x8 SAD (sad8x8 above) with the full-res
 * search, so SAD values are identical on every backend.
 *
 * lowres_best: one thread per 8x8 block, exhaustive integer search over
 *   [-range,range]^2 clipped to the per-block frame bounds
 *   mvmin=(-bx-8,-by-8), mvmax=(Wl-bx,Hl-by) (identical in the
 *   OpenCL/CPU backends).  Best = min SAD, ties
 *   keep the earliest position in (my,mx) scan order (strict <).
 * lowres_table: one thread per (block, MV) pair, exact SAD for every MV
 *   in [-tr,tr]^2 with no frame clipping (the caller in slicetype falls
 *   back to the CPU SAD for MVs outside the table, so clipping here
 *   would only corrupt entries the CPU walk may still query).
 * Plane geometry: cur/ref are the full lowres allocations (padded to the
 * context size); the visible (0,0) pixel is at (pad_x,pad_y) and block
 * (cuX,cuY) starts at (pad_x+cuX*8, pad_y+cuY*8).  cw/ch below are the
 * device-buffer width/height (stride in samples). */

template <typename PIX>
__device__ __forceinline__ void lowres_best_impl( const PIX *__restrict__ cur, int cw, int ch,
                                                  const PIX *__restrict__ ref, int rw, int rh,
                                                  int pad_x, int pad_y, int wcu, int hcu,
                                                  int range, Cand *out )
{
    int mb = blockIdx.x * blockDim.x + threadIdx.x;
    if( mb >= wcu * hcu )
        return;
    int cuX = mb % wcu, cuY = mb / wcu;
    int bx = cuX * 8, by = cuY * 8;
    int wl = wcu * 8, hl = hcu * 8;
    int mvmin_x = -bx - 8, mvmin_y = -by - 8;
    int mvmax_x = wl - bx, mvmax_y = hl - by;
    const PIX *pc = cur + (pad_y + by) * cw + (pad_x + bx);
    Cand best;
    best.mv[0] = 0;
    best.mv[1] = 0;
    best.sad = 0xFFFFFFFFu;
    /* Coarse grid (step 4) then refine (+/-4) around the coarse winner.
     * The scan order and strict-"<" update rule are part of the
     * algorithm and are mirrored exactly in the OpenCL/CPU backends,
     * so all backends return the same best MV.  This is ~10x fewer
     * SADs than the exhaustive window at comparable lowres quality. */
    for( int my = -range; my <= range; my += 4 )
        for( int mx = -range; mx <= range; mx += 4 )
        {
            if( mx < mvmin_x || mx > mvmax_x || my < mvmin_y || my > mvmax_y )
                continue;
            const PIX *pr = ref + (pad_y + by + my) * rw + (pad_x + bx + mx);
            uint32_t sad = sad8x8<PIX>( pc, cw, pr, rw );
            if( sad < best.sad )
            {
                best.sad = sad;
                best.mv[0] = (int16_t)mx;
                best.mv[1] = (int16_t)my;
            }
        }
    int cx = best.mv[0], cy = best.mv[1];
    for( int my = cy - 4; my <= cy + 4; my++ )
        for( int mx = cx - 4; mx <= cx + 4; mx++ )
        {
            if( mx < mvmin_x || mx > mvmax_x || my < mvmin_y || my > mvmax_y )
                continue;
            const PIX *pr = ref + (pad_y + by + my) * rw + (pad_x + bx + mx);
            uint32_t sad = sad8x8<PIX>( pc, cw, pr, rw );
            if( sad < best.sad )
            {
                best.sad = sad;
                best.mv[0] = (int16_t)mx;
                best.mv[1] = (int16_t)my;
            }
        }
    out[mb] = best;
}

template <typename PIX>
__device__ __forceinline__ void lowres_table_impl( const PIX *__restrict__ cur, int cw, int ch,
                                                   const PIX *__restrict__ ref, int rw, int rh,
                                                   int pad_x, int pad_y, int wcu, int hcu,
                                                   int tr, uint32_t *out )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tw = 2 * tr + 1;
    int total = wcu * hcu * tw * tw;
    if( idx >= total )
        return;
    int mx = idx % tw - tr;
    int my = (idx / tw) % tw - tr;
    int mb = idx / (tw * tw);
    int cuX = mb % wcu, cuY = mb / wcu;
    int bx = cuX * 8, by = cuY * 8;
    const PIX *pc = cur + (pad_y + by) * cw + (pad_x + bx);
    const PIX *pr = ref + (pad_y + by + my) * rw + (pad_x + bx + mx);
    out[idx] = sad8x8<PIX>( pc, cw, pr, rw );
}

extern "C" __global__ void lowres_best_8( const uint8_t *__restrict__ cur, int cw, int ch,
                                          const uint8_t *__restrict__ ref, int rw, int rh,
                                          int pad_x, int pad_y, int wcu, int hcu,
                                          int range, Cand *out )
{
    lowres_best_impl<uint8_t>( cur, cw, ch, ref, rw, rh, pad_x, pad_y, wcu, hcu, range, out );
}

extern "C" __global__ void lowres_best_16( const uint16_t *__restrict__ cur, int cw, int ch,
                                           const uint16_t *__restrict__ ref, int rw, int rh,
                                           int pad_x, int pad_y, int wcu, int hcu,
                                           int range, Cand *out )
{
    lowres_best_impl<uint16_t>( cur, cw, ch, ref, rw, rh, pad_x, pad_y, wcu, hcu, range, out );
}

extern "C" __global__ void lowres_table_8( const uint8_t *__restrict__ cur, int cw, int ch,
                                           const uint8_t *__restrict__ ref, int rw, int rh,
                                           int pad_x, int pad_y, int wcu, int hcu,
                                           int tr, uint32_t *out )
{
    lowres_table_impl<uint8_t>( cur, cw, ch, ref, rw, rh, pad_x, pad_y, wcu, hcu, tr, out );
}

extern "C" __global__ void lowres_table_16( const uint16_t *__restrict__ cur, int cw, int ch,
                                            const uint16_t *__restrict__ ref, int rw, int rh,
                                            int pad_x, int pad_y, int wcu, int hcu,
                                            int tr, uint32_t *out )
{
    lowres_table_impl<uint16_t>( cur, cw, ch, ref, rw, rh, pad_x, pad_y, wcu, hcu, tr, out );
}
