/*****************************************************************************
 * gpu_cpu.cpp: scalar CPU reference backend for the GPU candidate generator
 *****************************************************************************
 * This backend implements exactly the same integer algorithm as the CUDA
 * kernels (gpu_kernels.cu) and the OpenCL kernels (gpu_cl.c): same
 * downsample, same coarse search order, same refinement order, same strict
 * comparisons.  It therefore produces bit-identical candidates to the GPU
 * backends, which makes it usable as a test oracle and as a deterministic
 * fallback on machines without a GPU.
 *
 * The search code is templated on the luma sample type (uint8_t for 8-bit,
 * uint16_t for 10/12-bit builds) and selected from the sample_bytes passed
 * to the backend's create(), mirroring the CUDA/OpenCL backends.
 *
 * It is not expected to be fast (it does the same exhaustive work on the
 * CPU); selecting it explicitly with --gpu-me-backend 3 is for testing.
 *****************************************************************************/

#define __STDC_LIMIT_MACROS
#define __STDC_CONSTANT_MACROS

#include "gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GK X265_GPU_CANDS

struct x265_gpu_frame_t
{
    x265_gpu_frame_header_t hdr;   /* must be first: dispatcher reads it */
    int bw, bh;
    int nref[2];
    int nfilled[2];
    int ncand;
    const void *slot_ptr[2][X265_GPU_MAX_REFS];
    int64_t     slot_id[2][X265_GPU_MAX_REFS];
    x265_gpu_cand_t *cand[2][X265_GPU_MAX_REFS];
};

struct gpu_cpu_ctx_t
{
    int w, h, lw, lh, bw, bh, max_refs, device;   /* w/h: padded dims */
    int sb;                                       /* bytes per luma sample */
    uint8_t *cur, *cur_low;
    uint8_t *ref, *ref_low;      /* scratch for one reference at a time */
    x265_gpu_cand_t *coarse, *out;
    int disabled;
    int timing;
    long long nframes;
    double t_total;
};

static char gpu_cpu_name[256] = "CPU reference backend";

/* Forward declaration (used by frame_alloc below). */
extern "C" void x265_gpu_cpu_frame_free( x265_gpu_frame_t *f );

static int gpu_min_i( int a, int b ) { return a < b ? a : b; }
static int gpu_max_i( int a, int b ) { return a > b ? a : b; }

template <typename PIX>
static uint32_t sad8x8( const PIX *a, int as, const PIX *b, int bs )
{
    uint32_t s = 0;
    for( int j = 0; j < 8; j++ )
        for( int i = 0; i < 8; i++ )
        {
            int d = (int)a[j*as+i] - (int)b[j*bs+i];
            s += (uint32_t)(d < 0 ? -d : d);
        }
    return s;
}

static void insert_topk( x265_gpu_cand_t *best, int x, int y, uint32_t sad )
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

/* Copy a real (unpadded) plane into a padded buffer, replicating the
 * right/bottom edge pixels into the padding.  This defines the image the
 * GPU backends see, so the candidates match bit for bit. */
template <typename PIX>
static int upload_padded( PIX *dst, int pw, int ph,
                          const PIX *src, int stride, int w, int h )
{
    if( !src || stride < w || w <= 0 || h <= 0 )
        return -1;
    for( int y = 0; y < h; y++ )
    {
        PIX *row = dst + (size_t)y * pw;
        memcpy( row, src + (size_t)y * stride, (size_t)w * sizeof(PIX) );
        for( int x = w; x < pw; x++ )
            row[x] = row[w-1];
    }
    for( int y = h; y < ph; y++ )
        memcpy( dst + (size_t)y * pw, dst + (size_t)(h-1) * pw,
                (size_t)pw * sizeof(PIX) );
    return 0;
}

template <typename PIX>
static void downsample4( const PIX *src, int sw, int sh,
                         PIX *dst, int dw, int dh )
{
    for( int y = 0; y < dh; y++ )
        for( int x = 0; x < dw; x++ )
        {
            int sx = x * 4, sy = y * 4;
            int sum = 0;
            for( int j = 0; j < 4; j++ )
                for( int i = 0; i < 4; i++ )
                {
                    int px = gpu_min_i( sx + i, sw - 1 );
                    int py = gpu_min_i( sy + j, sh - 1 );
                    sum += (int)src[py * sw + px];
                }
            dst[y * dw + x] = (PIX)((sum + 8) >> 4);
        }
}

extern "C" void *x265_gpu_cpu_create( int width, int height, int max_refs, int device,
                                     int sample_bytes )
{
    if( width <= 0 || height <= 0 || (width & 31) || (height & 31) )
        return NULL;
    if( sample_bytes != 1 && sample_bytes != 2 )
        return NULL;
    gpu_cpu_ctx_t *c = (gpu_cpu_ctx_t*)calloc( 1, sizeof(*c) );
    if( !c )
        return NULL;
    c->w = width;
    c->h = height;
    c->lw = (width + 3) / 4;
    c->lh = (height + 3) / 4;
    c->bw = width / X265_GPU_BLOCK;
    c->bh = height / X265_GPU_BLOCK;
    c->max_refs = max_refs < 1 ? 1 : (max_refs > X265_GPU_MAX_REFS ? X265_GPU_MAX_REFS : max_refs);
    c->device = device;
    c->sb = sample_bytes;
    c->timing = getenv( "X265_GPU_TIMING" ) != NULL;
    size_t fsz = (size_t)width * height * sample_bytes;
    size_t lsz = (size_t)c->lw * c->lh * sample_bytes;
    size_t csz = (size_t)c->bw * c->bh * GK * sizeof(x265_gpu_cand_t);
    if( !( c->cur = (uint8_t*)malloc( fsz ) ) ||
        !( c->cur_low = (uint8_t*)malloc( lsz ) ) ||
        !( c->ref = (uint8_t*)malloc( fsz ) ) ||
        !( c->ref_low = (uint8_t*)malloc( lsz ) ) ||
        !( c->coarse = (x265_gpu_cand_t*)malloc( csz ) ) ||
        !( c->out = (x265_gpu_cand_t*)malloc( csz ) ) )
    {
        free( c->cur ); free( c->cur_low ); free( c->ref ); free( c->ref_low );
        free( c->coarse ); free( c->out ); free( c );
        return NULL;
    }
    snprintf( gpu_cpu_name, sizeof(gpu_cpu_name), "CPU reference" );
    return c;
}

extern "C" void x265_gpu_cpu_destroy( void *ctx )
{
    gpu_cpu_ctx_t *c = (gpu_cpu_ctx_t*)ctx;
    if( !c )
        return;
    if( c->timing && c->nframes )
        printf( "x265 CPU ME timing: %.3f ms/frame over %lld frames\n",
                c->t_total / (double)c->nframes, c->nframes );
    free( c->cur ); free( c->cur_low ); free( c->ref ); free( c->ref_low );
    free( c->coarse ); free( c->out );
    free( c );
}

extern "C" const char *x265_gpu_cpu_name( void )
{
    return gpu_cpu_name;
}

extern "C" x265_gpu_frame_t *x265_gpu_cpu_frame_alloc( int bw, int bh, int nref0, int nref1 )
{
    x265_gpu_frame_t *f = (x265_gpu_frame_t*)calloc( 1, sizeof(*f) );
    if( !f )
        return NULL;
    f->hdr.backend = X265_GPU_BACKEND_CPU;
    f->bw = bw;
    f->bh = bh;
    f->ncand = GK;
    f->nref[0] = nref0 < 0 ? 0 : (nref0 > X265_GPU_MAX_REFS ? X265_GPU_MAX_REFS : nref0);
    f->nref[1] = nref1 < 0 ? 0 : (nref1 > X265_GPU_MAX_REFS ? X265_GPU_MAX_REFS : nref1);
    size_t n = (size_t)bw * bh * GK * sizeof(x265_gpu_cand_t);
    for( int l = 0; l < 2; l++ )
        for( int r = 0; r < f->nref[l]; r++ )
        {
            f->cand[l][r] = (x265_gpu_cand_t*)malloc( n );
            if( !f->cand[l][r] )
            {
                x265_gpu_cpu_frame_free( f );
                return NULL;
            }
            memset( f->cand[l][r], 0, n );
        }
    return f;
}

extern "C" void x265_gpu_cpu_frame_free( x265_gpu_frame_t *f )
{
    if( !f )
        return;
    for( int l = 0; l < 2; l++ )
        for( int r = 0; r < f->nref[l]; r++ )
            free( f->cand[l][r] );
    free( f );
}

template <typename PIX>
static int analyze_typed( gpu_cpu_ctx_t *c, x265_gpu_frame_t *frame,
                          const uint8_t *cur, int cur_stride,
                          uint8_t *const refs0[], const int64_t ref_ids0[], const int stride0[], int nref0,
                          uint8_t *const refs1[], const int64_t ref_ids1[], const int stride1[], int nref1,
                          int width, int height, int range, int radius )
{
    PIX *curBuf = (PIX*)c->cur;
    PIX *curLow = (PIX*)c->cur_low;
    PIX *refBuf = (PIX*)c->ref;
    PIX *refLow = (PIX*)c->ref_low;

    if( upload_padded<PIX>( curBuf, c->w, c->h, (const PIX*)cur, cur_stride,
                            width, height ) < 0 )
        return -1;
    downsample4<PIX>( curBuf, c->w, c->h, curLow, c->lw, c->lh );

    int mbcount = c->bw * c->bh;
    for( int list = 0; list < 2; list++ )
    {
        uint8_t *const *refs = list ? refs1 : refs0;
        const int64_t *ids = list ? ref_ids1 : ref_ids0;
        const int *strides = list ? stride1 : stride0;
        int nref = list ? nref1 : nref0;
        if( nref > frame->nref[list] )
            nref = frame->nref[list];
        for( int r = 0; r < nref; r++ )
        {
            frame->slot_ptr[list][r] = refs[r];
            frame->slot_id[list][r] = ids[r];
            if( upload_padded<PIX>( refBuf, c->w, c->h, (const PIX*)refs[r], strides[r],
                                    width, height ) < 0 )
                return -1;
            downsample4<PIX>( refBuf, c->w, c->h, refLow, c->lw, c->lh );

            /* coarse search: identical order to the GPU kernels */
            for( int mb = 0; mb < mbcount; mb++ )
            {
                int mx = mb % c->bw, my = mb / c->bw;
                int bx = mx * 8, by = my * 8;
                x265_gpu_cand_t best[GK];
                for( int i = 0; i < GK; i++ )
                {
                    best[i].mv[0] = 0;
                    best[i].mv[1] = 0;
                    best[i].sad = 0xFFFFFFFFu;
                }
                int rr = range / 4;
                if( rr < 1 )
                    rr = 1;
                for( int dy = -rr; dy <= rr; dy += 2 )
                    for( int dx = -rr; dx <= rr; dx += 2 )
                    {
                        int x = gpu_max_i( 0, gpu_min_i( bx + dx, c->lw - 8 ) );
                        int y = gpu_max_i( 0, gpu_min_i( by + dy, c->lh - 8 ) );
                        uint32_t sad = sad8x8<PIX>( curLow + by * c->lw + bx, c->lw,
                                                    refLow + y * c->lw + x, c->lw );
                        insert_topk( best, (x - bx) * 4, (y - by) * 4, sad );
                    }
                int fx = bx + best[0].mv[0] / 4;
                int fy = by + best[0].mv[1] / 4;
                for( int dy = -1; dy <= 1; dy++ )
                    for( int dx = -1; dx <= 1; dx++ )
                    {
                        if( !dx && !dy )
                            continue;
                        int x = gpu_max_i( 0, gpu_min_i( fx + dx, c->lw - 8 ) );
                        int y = gpu_max_i( 0, gpu_min_i( fy + dy, c->lh - 8 ) );
                        uint32_t sad = sad8x8<PIX>( curLow + by * c->lw + bx, c->lw,
                                                    refLow + y * c->lw + x, c->lw );
                        insert_topk( best, (x - bx) * 4, (y - by) * 4, sad );
                    }
                for( int i = 0; i < GK; i++ )
                    c->coarse[mb * GK + i] = best[i];
            }

            /* refinement: identical order to the GPU kernels */
            int total = mbcount * GK;
            for( int idx = 0; idx < total; idx++ )
            {
                int mb = idx / GK;
                int mx = mb % c->bw, my = mb / c->bw;
                int bx = mx * 32, by = my * 32;
                int cx = bx + c->coarse[idx].mv[0];
                int cy = by + c->coarse[idx].mv[1];
                const PIX *pc = curBuf + by * c->w + bx;
                x265_gpu_cand_t best;
                best.mv[0] = 0;
                best.mv[1] = 0;
                best.sad = 0xFFFFFFFFu;
                for( int dy = -radius; dy <= radius; dy++ )
                    for( int dx = -radius; dx <= radius; dx++ )
                    {
                        int x = gpu_max_i( 0, gpu_min_i( cx + dx, c->w - 32 ) );
                        int y = gpu_max_i( 0, gpu_min_i( cy + dy, c->h - 32 ) );
                        const PIX *pr = refBuf + y * c->w + x;
                        uint32_t sad = 0;
                        for( int j = 0; j < 32; j += 2 )
                            for( int i = 0; i < 32; i += 2 )
                            {
                                int d = (int)pc[j*c->w+i] - (int)pr[j*c->w+i];
                                sad += (uint32_t)(d < 0 ? -d : d);
                            }
                        if( sad < best.sad )
                        {
                            best.sad = sad;
                            best.mv[0] = (int16_t)(x - bx);
                            best.mv[1] = (int16_t)(y - by);
                        }
                    }
                c->out[idx] = best;
            }
            memcpy( frame->cand[list][r], c->out, (size_t)total * sizeof(x265_gpu_cand_t) );
        }
        frame->nfilled[list] = nref;
    }
    c->nframes++;
    return 0;
}


/* ---- GPU lookahead (scalar reference, same order/clips as GPU) ---------- */

template <typename PIX>
static int lowres_best_typed( const uint8_t *fenc, int fenc_stride, int fenc_bufw, int fenc_bufh,
                              const uint8_t *ref, int ref_stride, int ref_bufw, int ref_bufh,
                              int pad_x, int pad_y, int wcu, int hcu, int range,
                              x265_gpu_cand_t *out )
{
    (void)fenc_bufw; (void)fenc_bufh; (void)ref_bufw; (void)ref_bufh;
    const PIX *cur = (const PIX*)fenc;
    const PIX *rf = (const PIX*)ref;
    int wl = wcu * 8, hl = hcu * 8;
    for( int mb = 0; mb < wcu * hcu; mb++ )
    {
        int cuX = mb % wcu, cuY = mb / wcu;
        int bx = cuX * 8, by = cuY * 8;
        int mvmin_x = -bx - 8, mvmin_y = -by - 8;
        int mvmax_x = wl - bx, mvmax_y = hl - by;
        const PIX *pc = cur + (size_t)(pad_y + by) * fenc_stride + (pad_x + bx);
        x265_gpu_cand_t best;
        best.mv[0] = 0; best.mv[1] = 0; best.sad = 0xFFFFFFFFu;
        for( int my = -range; my <= range; my += 4 )
            for( int mx = -range; mx <= range; mx += 4 )
            {
                if( mx < mvmin_x || mx > mvmax_x || my < mvmin_y || my > mvmax_y )
                    continue;
                const PIX *pr = rf + (size_t)(pad_y + by + my) * ref_stride + (pad_x + bx + mx);
                uint32_t sad = sad8x8<PIX>( pc, fenc_stride, pr, ref_stride );
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
                const PIX *pr = rf + (size_t)(pad_y + by + my) * ref_stride + (pad_x + bx + mx);
                uint32_t sad = sad8x8<PIX>( pc, fenc_stride, pr, ref_stride );
                if( sad < best.sad )
                {
                    best.sad = sad;
                    best.mv[0] = (int16_t)mx;
                    best.mv[1] = (int16_t)my;
                }
            }
        out[mb] = best;
    }
    return 0;
}

template <typename PIX>
static int lowres_table_typed( const uint8_t *fenc, int fenc_stride, int fenc_bufw, int fenc_bufh,
                               const uint8_t *ref, int ref_stride, int ref_bufw, int ref_bufh,
                               int pad_x, int pad_y, int wcu, int hcu, int tr,
                               uint32_t *out )
{
    (void)fenc_bufw; (void)fenc_bufh; (void)ref_bufw; (void)ref_bufh;
    const PIX *cur = (const PIX*)fenc;
    const PIX *rf = (const PIX*)ref;
    int tw = 2 * tr + 1;
    for( int mb = 0; mb < wcu * hcu; mb++ )
    {
        int cuX = mb % wcu, cuY = mb / wcu;
        int bx = cuX * 8, by = cuY * 8;
        const PIX *pc = cur + (size_t)(pad_y + by) * fenc_stride + (pad_x + bx);
        for( int my = -tr; my <= tr; my++ )
            for( int mx = -tr; mx <= tr; mx++ )
            {
                const PIX *pr = rf + (size_t)(pad_y + by + my) * ref_stride + (pad_x + bx + mx);
                out[(size_t)mb * tw * tw + (size_t)(my + tr) * tw + (mx + tr)] =
                    sad8x8<PIX>( pc, fenc_stride, pr, ref_stride );
            }
    }
    return 0;
}

static int cpu_lowres_best( void *ctx, x265_gpu_cand_t *out,
                            const uint8_t *fenc, int fenc_stride, int fenc_bufw, int fenc_bufh,
                            const uint8_t *ref, int ref_stride, int ref_bufw, int ref_bufh,
                            int pad_x, int pad_y, int wcu, int hcu, int range )
{
    gpu_cpu_ctx_t *c = (gpu_cpu_ctx_t*)ctx;
    if( !c || !out || !fenc || !ref || c->disabled )
        return -1;
    if( wcu <= 0 || hcu <= 0 || range < 1 || range > 64 )
        return -1;
    if( fenc_bufw > c->w || fenc_bufh > c->h || ref_bufw > c->w || ref_bufh > c->h )
        return -1;
    return c->sb == 2
        ? lowres_best_typed<uint16_t>( fenc, fenc_stride, fenc_bufw, fenc_bufh,
                                       ref, ref_stride, ref_bufw, ref_bufh,
                                       pad_x, pad_y, wcu, hcu, range, out )
        : lowres_best_typed<uint8_t>( fenc, fenc_stride, fenc_bufw, fenc_bufh,
                                      ref, ref_stride, ref_bufw, ref_bufh,
                                      pad_x, pad_y, wcu, hcu, range, out );
}

static int cpu_lowres_table( void *ctx, uint32_t *out,
                             const uint8_t *fenc, int fenc_stride, int fenc_bufw, int fenc_bufh,
                             const uint8_t *ref, int ref_stride, int ref_bufw, int ref_bufh,
                             int pad_x, int pad_y, int wcu, int hcu, int tr )
{
    gpu_cpu_ctx_t *c = (gpu_cpu_ctx_t*)ctx;
    if( !c || !out || !fenc || !ref || c->disabled )
        return -1;
    if( wcu <= 0 || hcu <= 0 || tr < 1 || tr > 64 )
        return -1;
    if( fenc_bufw > c->w || fenc_bufh > c->h || ref_bufw > c->w || ref_bufh > c->h )
        return -1;
    return c->sb == 2
        ? lowres_table_typed<uint16_t>( fenc, fenc_stride, fenc_bufw, fenc_bufh,
                                        ref, ref_stride, ref_bufw, ref_bufh,
                                        pad_x, pad_y, wcu, hcu, tr, out )
        : lowres_table_typed<uint8_t>( fenc, fenc_stride, fenc_bufw, fenc_bufh,
                                       ref, ref_stride, ref_bufw, ref_bufh,
                                       pad_x, pad_y, wcu, hcu, tr, out );
}

static int analyze_common( void *ctx, x265_gpu_frame_t *frame,
                           const uint8_t *cur, int cur_stride,
                           uint8_t *const refs0[], const int64_t ref_ids0[], const int stride0[], int nref0,
                           uint8_t *const refs1[], const int64_t ref_ids1[], const int stride1[], int nref1,
                           int width, int height, int range, int radius )
{
    gpu_cpu_ctx_t *c = (gpu_cpu_ctx_t*)ctx;
    if( !c || !frame || c->disabled )
        return -1;
    if( frame->bw != c->bw || frame->bh != c->bh )
        return -1;
    if( width <= 0 || height <= 0 || width > c->w || height > c->h )
        return -1;
    if( range < 4 ) range = 4;
    if( range > 128 ) range = 128;
    if( radius < 0 ) radius = 0;
    if( radius > 8 ) radius = 8;

    /* Testing hook: X265_GPU_SIMULATE_FAIL=cpu makes this backend fail at
     * the first analysis (used to exercise the GPU fallback). */
    {
        static int sim = -1;
        if( sim < 0 )
        {
            const char *ev = getenv( "X265_GPU_SIMULATE_FAIL" );
            sim = ev && !strcmp( ev, "cpu" );
        }
        if( sim )
        {
            c->disabled = 1;
            return -1;
        }
    }

    frame->nfilled[0] = 0;
    frame->nfilled[1] = 0;

    if( c->sb == 2 )
        return analyze_typed<uint16_t>( c, frame, cur, cur_stride,
                                        refs0, ref_ids0, stride0, nref0,
                                        refs1, ref_ids1, stride1, nref1,
                                        width, height, range, radius );
    return analyze_typed<uint8_t>( c, frame, cur, cur_stride,
                                   refs0, ref_ids0, stride0, nref0,
                                   refs1, ref_ids1, stride1, nref1,
                                   width, height, range, radius );
}

static int cpu_available( void )
{
    return 1;
}

static const x265_gpu_cand_t *cpu_frame_get( const x265_gpu_frame_t *frame,
                                             int list, const void *ref_plane, int64_t ref_id,
                                             int bx, int by, int *count )
{
    const x265_gpu_frame_t *f = frame;
    if( !f || !count || list < 0 || list > 1 )
        return NULL;
    if( bx < 0 || bx >= f->bw || by < 0 || by >= f->bh )
        return NULL;
    for( int r = 0; r < f->nfilled[list]; r++ )
        if( f->slot_ptr[list][r] == ref_plane && f->slot_id[list][r] == ref_id )
        {
            *count = f->ncand;
            return (const x265_gpu_cand_t *)( f->cand[list][r] +
                                             ((size_t)by * f->bw + bx) * GK );
        }
    return NULL;
}

static int cpu_frame_covers( const x265_gpu_frame_t *frame,
                             int list, const void *ref_plane, int64_t ref_id )
{
    const x265_gpu_frame_t *f = frame;
    if( !f || list < 0 || list > 1 )
        return 0;
    for( int r = 0; r < f->nfilled[list]; r++ )
        if( f->slot_ptr[list][r] == ref_plane && f->slot_id[list][r] == ref_id )
            return 1;
    return 0;
}

extern "C" const x265_gpu_backend_t x265_gpu_backend_cpu =
{
    X265_GPU_BACKEND_CPU,
    "cpu",
    cpu_available,
    x265_gpu_cpu_create,
    x265_gpu_cpu_destroy,
    x265_gpu_cpu_name,
    x265_gpu_cpu_frame_alloc,
    x265_gpu_cpu_frame_free,
    analyze_common,
    cpu_frame_get,
    cpu_frame_covers,
    cpu_lowres_best,
    cpu_lowres_table,
};
