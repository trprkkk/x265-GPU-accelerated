/*****************************************************************************
 * gpu_cuda.cpp: CUDA host implementation of the hierarchical candidate
 * generator.
 *****************************************************************************
 * The GPU does a coarse exhaustive search on a quarter-resolution luma copy
 * and then refines the best four integer-pel positions at full resolution.
 * It is deliberately an *open-loop* pre-analysis: it uses the reference
 * planes that x265 already reconstructed, but never writes any coding
 * decision.  search.cpp / motion.cpp evaluate every returned vector with
 * the real x265 cost function, so rate control, RDO, entropy coding and
 * the bitstream syntax stay exactly in the original CPU code.
 *
 * This file is the CUDA backend host side; the public API (dispatcher) is
 * gpu.c and the symbols here carry the x265_gpu_cuda_ prefix.
 *
 * Device code lives in gpu_kernels.cu and is compiled to a fat binary by nvcc
 * (CUDA 11.8 for sm_35..sm_90, CUDA 13.x for the newer architectures).  The
 * fat binary is embedded in x265.exe as RCDATA resources and loaded here
 * through the CUDA driver API from nvcuda.dll, which is loaded dynamically:
 * no CUDA runtime library has to be present, the kernels can come from two
 * different toolkits, and the host code stays independent of the (much newer)
 * nvcc that could no longer build for older GPUs.  The host API (memory,
 * streams, copies) uses the CUDA runtime (cudart, statically linked).
 *
 * The fat binary is picked from the device's compute capability:
 *   * cc < 10  -> legacy image (CUDA 11.8: sm_35..sm_90 + compute_75 PTX)
 *   * cc >= 10 -> modern image (CUDA 13.x: sm_100/sm_120 + compute_90 PTX)
 * When the preferred image cannot be loaded (missing resource, unsupported
 * device, ...) the other one is tried; if that fails too, x265_gpu_create()
 * returns NULL and the dispatcher falls back to OpenCL or the CPU.
 *
 * The analysis is synchronous: x265_gpu_analyze() waits for the stream
 * before returning, so the output is deterministic by construction.
 *****************************************************************************/

#define __STDC_LIMIT_MACROS
#define __STDC_CONSTANT_MACROS

#include "gpu.h"
#include "gpu_kernels.h"

/* Rename the backend entry points (definitions below only; the public
 * declarations from gpu.h keep their names and are implemented by gpu.c). */
#define x265_gpu_create       x265_gpu_cuda_create
#define x265_gpu_destroy      x265_gpu_cuda_destroy
#define x265_gpu_name         x265_gpu_cuda_name
#define x265_gpu_frame_alloc  x265_gpu_cuda_frame_alloc
#define x265_gpu_frame_free   x265_gpu_cuda_frame_free
#define x265_gpu_analyze      x265_gpu_cuda_analyze
#define x265_gpu_frame_get    x265_gpu_cuda_frame_get
#define x265_gpu_frame_covers x265_gpu_cuda_frame_covers

/* Forward declarations for calls that appear before the definitions. */
extern "C" x265_gpu_frame_t *x265_gpu_frame_alloc( int bw, int bh, int nref0, int nref1 );
extern "C" void x265_gpu_frame_free( x265_gpu_frame_t *frame );

/* CUDA 11.8 predates the MSVC toolchain used here; the runtime API and the
 * headers work fine, only the version gates in crt/host_config.h refuse the
 * newer compiler. */
#define __NV_NO_HOST_COMPILER_CHECK
#include <cuda.h>
#include <cuda_runtime.h>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mutex>

/* Resource ids of the embedded kernel images (kept in sync with the generated
 * resource script, see the CMake build). */
#define GPU_RES_LEGACY 101
#define GPU_RES_MODERN 102

#define GK X265_GPU_CANDS

/* --- CUDA driver API, loaded dynamically from nvcuda.dll ------------------ */

namespace
{
struct GpuDriverApi
{
    decltype(&cuModuleLoadData)          cuModuleLoadData;
    decltype(&cuModuleGetFunction)       cuModuleGetFunction;
    decltype(&cuModuleUnload)            cuModuleUnload;
    decltype(&cuLaunchKernel)            cuLaunchKernel;
    decltype(&cuCtxSetCurrent)           cuCtxSetCurrent;
    decltype(&cuDeviceGet)               cuDeviceGet;
    decltype(&cuDevicePrimaryCtxRetain)  cuDevicePrimaryCtxRetain;
    decltype(&cuDevicePrimaryCtxRelease) cuDevicePrimaryCtxRelease;
};
}

static GpuDriverApi g_drv;
static int g_drv_state = -1;   /* -1: unknown, 0: unavailable, 1: ready */

static int gpu_drv_load( void )
{
    if( g_drv_state >= 0 )
        return g_drv_state;
    HMODULE lib = LoadLibraryW( L"nvcuda.dll" );
    if( !lib )
    {
        g_drv_state = 0;
        return 0;
    }
    #define GPU_DRV_LOAD( name ) (FARPROC&)g_drv.name = GetProcAddress( lib, #name )
    GPU_DRV_LOAD( cuModuleLoadData );
    GPU_DRV_LOAD( cuModuleGetFunction );
    GPU_DRV_LOAD( cuModuleUnload );
    GPU_DRV_LOAD( cuLaunchKernel );
    GPU_DRV_LOAD( cuCtxSetCurrent );
    GPU_DRV_LOAD( cuDeviceGet );
    GPU_DRV_LOAD( cuDevicePrimaryCtxRetain );
    GPU_DRV_LOAD( cuDevicePrimaryCtxRelease );
    #undef GPU_DRV_LOAD
    g_drv_state = g_drv.cuModuleLoadData && g_drv.cuModuleGetFunction &&
                  g_drv.cuModuleUnload && g_drv.cuLaunchKernel &&
                  g_drv.cuCtxSetCurrent &&
                  g_drv.cuDeviceGet && g_drv.cuDevicePrimaryCtxRetain &&
                  g_drv.cuDevicePrimaryCtxRelease;
    return g_drv_state;
}

/* Kernel images embedded as RCDATA resources in the executable. */
static const void *gpu_res_blob( int id, size_t *size )
{
    HRSRC res = FindResourceW( NULL, MAKEINTRESOURCEW( id ), (LPCWSTR)RT_RCDATA );
    if( !res )
        return NULL;
    DWORD sz = SizeofResource( NULL, res );
    if( !sz )
        return NULL;
    HGLOBAL dat = LoadResource( NULL, res );
    if( !dat )
        return NULL;
    const void *p = LockResource( dat );
    if( !p )
        return NULL;
    *size = (size_t)sz;
    return p;
}

static double gpu_now_ms()
{
    static LARGE_INTEGER freq = {};
    LARGE_INTEGER c;
    if( !freq.QuadPart )
        QueryPerformanceFrequency( &freq );
    QueryPerformanceCounter( &c );
    return (double)c.QuadPart * 1000.0 / (double)freq.QuadPart;
}

struct x265_gpu_ctx_t;

struct x265_gpu_frame_t
{
    x265_gpu_frame_header_t hdr;   /* must be first: dispatcher reads it */
    int bw, bh;
    int nref[2];       /* allocated candidate slots per list */
    int nfilled[2];    /* references actually analysed and published */
    int ncand;
    /* Reference identity of every analysed slot. */
    const void *slot_ptr[2][X265_GPU_MAX_REFS];
    int64_t     slot_id[2][X265_GPU_MAX_REFS];
    Cand *cand[2][X265_GPU_MAX_REFS];
};

/* Pinned staging ring for reference uploads.  Two buffers are alternated and
 * each async copy out of a buffer is fenced with an event, so an upload that
 * starts while the previous copy is still reading its staging buffer cannot
 * corrupt it (the two are of comparable speed on old, slow PCIe links). */
#define X265_GPU_STAGE 2

struct x265_gpu_ctx_t
{
    int w, h, lw, lh, bw, bh, max_refs, device;   /* w/h: padded dims */
    int sb;                 /* bytes per luma sample: 1 (8-bit) or 2 (10/12-bit) */
    uint8_t *d_cur;
    uint8_t *d_cur_low;
    Cand    *d_coarse;
    Cand    *d_out;
    cudaStream_t stream;
    std::mutex lock;
    int disabled;
    uint8_t *h_ref[X265_GPU_STAGE];    /* pinned staging ring for plane uploads */
    cudaEvent_t stage_done[X265_GPU_STAGE];
    int stage_next;
    Cand    *h_out;            /* pinned staging for candidate D2H copies */
    /* Kernel module (loaded from an embedded fat binary). */
    CUmodule module;
    CUfunction fn_ds;          /* downsample4 */
    CUfunction fn_coarse;      /* coarse_search */
    CUfunction fn_refine;      /* refine_search */
    CUfunction fn_la_best;     /* lowres_best (GPU lookahead) */
    CUfunction fn_la_table;    /* lowres_table (GPU lookahead) */
    /* Lazily allocated lowres outputs (sized on first use). */
    uint8_t *d_la_best;
    size_t   d_la_best_sz;
    uint8_t *d_la_table;
    size_t   d_la_table_sz;
    CUcontext cu_ctx;          /* primary context (retained when we created it) */
    CUdevice  cu_dev;
    int cu_retained;
    /* Optional per-stage timing (enabled with X265_GPU_TIMING=1). */
    int timing;
    long long nframes;
    double t_upload_cur, t_upload_ref, t_down, t_coarse, t_refine, t_d2h, t_sync;
    /* Reference plane cache.  Keyed by the source frame pointer and its
     * POC, because x265 recycles frame objects. */
    struct CacheEntry
    {
        const void *ptr;
        int64_t id;
        uint8_t *full;
        uint8_t *low;
    };
    CacheEntry cache[X265_GPU_CACHE];
    int cache_next;
};

static char gpu_name[256] = "CUDA device";

/* Launch one of the module's kernels on the analysis stream. */
static int gpu_launch( x265_gpu_ctx_t *c, CUfunction fn, dim3 grid, dim3 block, void **args )
{
    if( !fn || g_drv.cuLaunchKernel( fn, grid.x, grid.y, grid.z,
                                     block.x, block.y, block.z,
                                     0, (CUstream)c->stream, args, NULL ) != CUDA_SUCCESS )
        return -1;
    return 0;
}

/* Copy a real (unpadded) plane into a padded staging buffer, replicating
 * the right/bottom edge pixels into the padding.  This defines the image
 * the kernels see, so all backends produce identical candidates. */
static void fill_padded( uint8_t *dst, int pw, int ph,
                         const uint8_t *src, int stride, int w, int h, int sb )
{
    /* Replicate the right/bottom edge pixels into the padding.  sb is the
     * sample size in bytes, so the same code serves 8-bit and 10/12-bit. */
    if( sb == 1 )
    {
        for( int y = 0; y < h; y++ )
        {
            uint8_t *row = dst + (size_t)y * pw;
            memcpy( row, src + (size_t)y * stride, (size_t)w );
            for( int x = w; x < pw; x++ )
                row[x] = row[w-1];
        }
        for( int y = h; y < ph; y++ )
            memcpy( dst + (size_t)y * pw, dst + (size_t)(h-1) * pw, (size_t)pw );
    }
    else
    {
        /* stride is in samples (x265's m_stride), pw/w in samples too. */
        for( int y = 0; y < h; y++ )
        {
            uint16_t *row = (uint16_t*)dst + (size_t)y * pw;
            const uint16_t *srow = (const uint16_t*)src + (size_t)y * stride;
            memcpy( row, srow, (size_t)w * 2 );
            for( int x = w; x < pw; x++ )
                row[x] = row[w-1];
        }
        for( int y = h; y < ph; y++ )
            memcpy( (uint16_t*)dst + (size_t)y * pw, (uint16_t*)dst + (size_t)(h-1) * pw,
                    (size_t)pw * 2 );
    }
}

extern "C" void *x265_gpu_create( int width, int height, int max_refs, int device,
                                 int sample_bytes )
{
    /* The kernels work on whole 32x32 blocks; the caller passes the frame
     * size rounded up to the block grid (planes are padded by the caller). */
    if( width <= 0 || height <= 0 || (width & 31) || (height & 31) )
        return NULL;

    int ndev = 0;
    cudaError_t ce = cudaGetDeviceCount( &ndev );
    if( ce != cudaSuccess || ndev <= 0 )
        return NULL;
    if( device < 0 || device >= ndev )
    {
        if( device != 0 )
            fprintf( stderr, "x265 GPU ME: CUDA device %d unavailable (%d found), using 0\n",
                     device, ndev );
        device = 0;
    }
    ce = cudaSetDevice( device );
    if( ce != cudaSuccess )
        return NULL;

    if( !gpu_drv_load() )
        return NULL;

    cudaDeviceProp prop;
    memset( &prop, 0, sizeof(prop) );
    if( cudaGetDeviceProperties( &prop, device ) == cudaSuccess )
        snprintf( gpu_name, sizeof(gpu_name), "%s (sm_%d%d)", prop.name,
                  prop.major, prop.minor );

    /* Retain the primary context explicitly instead of trusting whatever the
     * runtime left current: some cudart versions expose a destroyed handle
     * after their device queries, and the driver API needs a live current
     * context.  cudart shares the same primary context, so this only adds a
     * reference. */
    CUcontext cu_ctx = NULL;
    CUdevice  cu_dev = 0;
    int cu_retained = 0;
    if( g_drv.cuDeviceGet( &cu_dev, device ) != CUDA_SUCCESS ||
        g_drv.cuDevicePrimaryCtxRetain( &cu_ctx, cu_dev ) != CUDA_SUCCESS ||
        g_drv.cuCtxSetCurrent( cu_ctx ) != CUDA_SUCCESS )
    {
        fprintf( stderr, "x265 CUDA ME: cannot establish a CUDA context (device %d)\n", device );
        return NULL;
    }
    cu_retained = 1;

    x265_gpu_ctx_t *c = new (std::nothrow) x265_gpu_ctx_t();
    if( !c )
    {
        if( cu_retained )
            g_drv.cuDevicePrimaryCtxRelease( cu_dev );
        return NULL;
    }
    c->w = width;
    c->h = height;
    c->lw = (width + 3) / 4;
    c->lh = (height + 3) / 4;
    c->bw = width / X265_GPU_BLOCK;
    c->bh = height / X265_GPU_BLOCK;
    c->max_refs = max_refs < 1 ? 1 : (max_refs > X265_GPU_MAX_REFS ? X265_GPU_MAX_REFS : max_refs);
    c->device = device;
    c->sb = sample_bytes;
    c->disabled = 0;
    c->cu_ctx = cu_ctx;
    c->cu_dev = cu_dev;
    c->cu_retained = cu_retained;
    c->timing = getenv( "X265_GPU_TIMING" ) != NULL;
    if( c->timing )
        printf( "x265 CUDA ME timing: %dx%d (padded %dx%d), %d refs, device %d\n",
                c->w, c->h, c->w, c->h, c->max_refs, device );

    /* Pick and load the kernel image.  cc >= 10 (Blackwell and newer) uses
     * the image built by the modern toolkit, everything else the CUDA 11.8
     * image; both carry a PTX fallback for unknown devices. */
    {
        static const int ids[2] = { GPU_RES_LEGACY, GPU_RES_MODERN };
        int first = prop.major >= 10 ? 1 : 0;
        int ntry = 2;
        int order[2];
        order[0] = ids[first];
        order[1] = ids[1-first];
        const char *force = getenv( "X265_GPU_FATBIN" );   /* test hook */
        if( force )
        {
            if( !strcmp( force, "legacy" ) )
                order[0] = GPU_RES_LEGACY;
            else if( !strcmp( force, "modern" ) )
                order[0] = GPU_RES_MODERN;
            ntry = 1;   /* forced: keep the outcome observable */
        }
        for( int i = 0; i < ntry && !c->module; i++ )
        {
            size_t bsz = 0;
            const void *blob = gpu_res_blob( order[i], &bsz );
            if( !blob )
                continue;
            if( g_drv.cuModuleLoadData( &c->module, blob ) == CUDA_SUCCESS )
            {
                if( force )
                    fprintf( stderr, "x265 CUDA ME: using %s kernel image\n",
                             order[i] == GPU_RES_LEGACY ? "legacy (CUDA 11.8)"
                                                        : "modern (CUDA 13.x)" );
            }
            else
                c->module = NULL;
        }
    }
    int look_fail = 0;
    if( c->module )
    {
        /* Every kernel exists for 8-bit and 16-bit samples; pick the pair
         * that matches the caller's pixel type. */
        const char *k_ds = sample_bytes == 2 ? "downsample4_16" : "downsample4_8";
        const char *k_coarse = sample_bytes == 2 ? "coarse_search_16" : "coarse_search_8";
        const char *k_refine = sample_bytes == 2 ? "refine_search_16" : "refine_search_8";
        const char *k_la_best = sample_bytes == 2 ? "lowres_best_16" : "lowres_best_8";
        const char *k_la_table = sample_bytes == 2 ? "lowres_table_16" : "lowres_table_8";
        if( g_drv.cuModuleGetFunction( &c->fn_ds, c->module, k_ds ) != CUDA_SUCCESS ||
            g_drv.cuModuleGetFunction( &c->fn_coarse, c->module, k_coarse ) != CUDA_SUCCESS ||
            g_drv.cuModuleGetFunction( &c->fn_refine, c->module, k_refine ) != CUDA_SUCCESS ||
            g_drv.cuModuleGetFunction( &c->fn_la_best, c->module, k_la_best ) != CUDA_SUCCESS ||
            g_drv.cuModuleGetFunction( &c->fn_la_table, c->module, k_la_table ) != CUDA_SUCCESS )
        {
            fprintf( stderr, "x265 CUDA ME: kernel entry points missing from the image\n" );
            look_fail = 1;
        }
    }
    else
        fprintf( stderr, "x265 CUDA ME: no kernel image for this device (sm_%d%d)\n",
                 prop.major, prop.minor );
    if( !c->module || look_fail )
    {
        if( c->module )
            g_drv.cuModuleUnload( c->module );
        if( cu_retained )
            g_drv.cuDevicePrimaryCtxRelease( cu_dev );
        delete c;
        return NULL;
    }

    size_t fsz = (size_t)width * height * sample_bytes;
    size_t lsz = (size_t)c->lw * c->lh * sample_bytes;
    size_t csz = (size_t)c->bw * c->bh * GK * sizeof(Cand);
    int fail = 0;
    if( cudaStreamCreate( &c->stream ) != cudaSuccess ||
        cudaMalloc( (void**)&c->d_cur, fsz ) != cudaSuccess ||
        cudaMalloc( (void**)&c->d_cur_low, lsz ) != cudaSuccess ||
        cudaMalloc( (void**)&c->d_coarse, csz ) != cudaSuccess ||
        cudaMalloc( (void**)&c->d_out, csz ) != cudaSuccess ||
        cudaMallocHost( (void**)&c->h_out, 2 * X265_GPU_MAX_REFS * csz ) != cudaSuccess )
        fail = 1;
    for( int i = 0; i < X265_GPU_STAGE && !fail; i++ )
    {
        if( cudaMallocHost( (void**)&c->h_ref[i], fsz ) != cudaSuccess ||
            cudaEventCreateWithFlags( &c->stage_done[i], cudaEventDisableTiming ) != cudaSuccess )
            fail = 1;
        else
            cudaEventRecord( c->stage_done[i], c->stream );   /* start "done" */
    }
    for( int i = 0; i < X265_GPU_CACHE && !fail; i++ )
        if( cudaMalloc( (void**)&c->cache[i].full, fsz ) != cudaSuccess ||
            cudaMalloc( (void**)&c->cache[i].low, lsz ) != cudaSuccess )
            fail = 1;
    if( !fail )
    {
        /* Probe launch: verify that the loaded image can actually run on this
         * device (unsupported architecture, broken driver, ...).  On failure
         * this backend is rejected and the dispatcher can fall back. */
        const uint8_t *a_src = c->d_cur;   int a_sw = 32, a_sh = 32;
        uint8_t *a_dst = c->d_cur_low;     int a_dw = 8, a_dh = 8;
        void *args[] = { &a_src, &a_sw, &a_sh, &a_dst, &a_dw, &a_dh };
        if( gpu_launch( c, c->fn_ds, dim3( 1 ), dim3( 1 ), args ) < 0 ||
            cudaStreamSynchronize( c->stream ) != cudaSuccess )
        {
            cudaError_t pe = cudaGetLastError();   /* clear the sticky error */
            fprintf( stderr, "x265 CUDA ME: kernel launch failed on this device (%d: %s)\n",
                     (int)pe, cudaGetErrorString( pe ) );
            fail = 1;
        }
    }
    if( fail )
    {
        if( c->stream ) cudaStreamDestroy( c->stream );
        if( c->d_cur ) cudaFree( c->d_cur );
        if( c->d_cur_low ) cudaFree( c->d_cur_low );
        if( c->d_coarse ) cudaFree( c->d_coarse );
        if( c->d_out ) cudaFree( c->d_out );
        for( int i = 0; i < X265_GPU_STAGE; i++ )
        {
            if( c->h_ref[i] ) cudaFreeHost( c->h_ref[i] );
            if( c->stage_done[i] ) cudaEventDestroy( c->stage_done[i] );
        }
        if( c->h_out ) cudaFreeHost( c->h_out );
        for( int i = 0; i < X265_GPU_CACHE; i++ )
        {
            if( c->cache[i].full ) cudaFree( c->cache[i].full );
            if( c->cache[i].low ) cudaFree( c->cache[i].low );
        }
        if( c->module ) g_drv.cuModuleUnload( c->module );
        if( cu_retained ) g_drv.cuDevicePrimaryCtxRelease( cu_dev );
        delete c;
        return NULL;
    }
    return c;
}

extern "C" void x265_gpu_destroy( void *ctx )
{
    x265_gpu_ctx_t *c = (x265_gpu_ctx_t*)ctx;
    if( !c )
        return;
    if( c->timing && c->nframes )
    {
        double n = (double)c->nframes;
        printf( "x265 CUDA ME timing (%lld frames analysed, ms/frame): "
                "cur-upload %.3f  ref-upload %.3f  downsample %.3f  coarse %.3f  "
                "refine %.3f  d2h %.3f  sync %.3f  total %.3f\n",
                c->nframes,
                c->t_upload_cur / n, c->t_upload_ref / n, c->t_down / n,
                c->t_coarse / n, c->t_refine / n, c->t_d2h / n,
                c->t_sync / n,
                (c->t_upload_cur + c->t_upload_ref + c->t_down + c->t_coarse
                 + c->t_refine + c->t_d2h + c->t_sync) / n );
    }
    if( c->stream ) cudaStreamSynchronize( c->stream );
    if( c->d_la_best ) cudaFree( c->d_la_best );
    if( c->d_la_table ) cudaFree( c->d_la_table );
    if( c->stream ) cudaStreamDestroy( c->stream );
    if( c->d_cur ) cudaFree( c->d_cur );
    if( c->d_cur_low ) cudaFree( c->d_cur_low );
    if( c->d_coarse ) cudaFree( c->d_coarse );
    if( c->d_out ) cudaFree( c->d_out );
    for( int i = 0; i < X265_GPU_STAGE; i++ )
    {
        if( c->h_ref[i] ) cudaFreeHost( c->h_ref[i] );
        if( c->stage_done[i] ) cudaEventDestroy( c->stage_done[i] );
    }
    if( c->h_out ) cudaFreeHost( c->h_out );
    for( int i = 0; i < X265_GPU_CACHE; i++ )
    {
        if( c->cache[i].full ) cudaFree( c->cache[i].full );
        if( c->cache[i].low ) cudaFree( c->cache[i].low );
    }
    if( c->module ) g_drv.cuModuleUnload( c->module );
    if( c->cu_retained ) g_drv.cuDevicePrimaryCtxRelease( c->cu_dev );
    delete c;
}

extern "C" const char *x265_gpu_name( void )
{
    return gpu_name;
}

extern "C" x265_gpu_frame_t *x265_gpu_frame_alloc( int bw, int bh, int nref0, int nref1 )
{
    x265_gpu_frame_t *f = (x265_gpu_frame_t*)calloc( 1, sizeof(*f) );
    if( !f )
        return NULL;
    f->hdr.backend = X265_GPU_BACKEND_CUDA;
    f->bw = bw;
    f->bh = bh;
    f->ncand = GK;
    f->nref[0] = nref0 < 0 ? 0 : (nref0 > X265_GPU_MAX_REFS ? X265_GPU_MAX_REFS : nref0);
    f->nref[1] = nref1 < 0 ? 0 : (nref1 > X265_GPU_MAX_REFS ? X265_GPU_MAX_REFS : nref1);
    f->nfilled[0] = 0;
    f->nfilled[1] = 0;
    size_t n = (size_t)bw * bh * GK * sizeof(Cand);
    for( int l = 0; l < 2; l++ )
        for( int r = 0; r < f->nref[l]; r++ )
        {
            f->cand[l][r] = (Cand*)malloc( n );
            if( !f->cand[l][r] )
            {
                x265_gpu_frame_free( f );
                return NULL;
            }
            memset( f->cand[l][r], 0, n );
        }
    return f;
}

extern "C" void x265_gpu_frame_free( x265_gpu_frame_t *f )
{
    if( !f )
        return;
    for( int l = 0; l < 2; l++ )
        for( int r = 0; r < f->nref[l]; r++ )
            free( f->cand[l][r] );
    free( f );
}

/* Upload through the pinned staging ring: pageable sources would make the
 * asynchronous copy synchronise the stream, which serialises the whole
 * analysis onto the calling thread.  Each buffer is reused only after the
 * copy that last read it has completed (see the ring comment).  The real
 * (unpadded) plane is edge-replicated into the padding first. */
static int upload_plane_pinned( x265_gpu_ctx_t *c, uint8_t *dst, const uint8_t *src,
                                int stride, int width, int height )
{
    if( !src || stride < width || width <= 0 || height <= 0 || width > c->w || height > c->h )
        return -1;
    int slot = c->stage_next++ % X265_GPU_STAGE;
    if( cudaEventQuery( c->stage_done[slot] ) == cudaErrorNotReady )
        cudaEventSynchronize( c->stage_done[slot] );
    uint8_t *stage = c->h_ref[slot];
    fill_padded( stage, c->w, c->h, src, stride, width, height, c->sb );
    if( cudaMemcpyAsync( dst, stage, (size_t)c->w * c->h * c->sb,
                         cudaMemcpyHostToDevice, c->stream ) != cudaSuccess )
        return -1;
    if( cudaEventRecord( c->stage_done[slot], c->stream ) != cudaSuccess )
        return -1;
    return 0;
}

/* Copy the staged candidates into the frame after the stream has been
 * synchronised. */
static void publish_candidates( x265_gpu_ctx_t *c, x265_gpu_frame_t *frame, int list, int nref )
{
    size_t n = (size_t)c->bw * c->bh * GK * sizeof(Cand);
    for( int r = 0; r < nref; r++ )
        memcpy( frame->cand[list][r],
                c->h_out + (size_t)(list * X265_GPU_MAX_REFS + r) * (c->bw * c->bh * GK),
                n );
}

extern "C" int x265_gpu_analyze( void *ctx, x265_gpu_frame_t *frame,
                                 const uint8_t *cur, int cur_stride,
                                 uint8_t *const refs0[], const int64_t ref_ids0[], const int stride0[], int nref0,
                                 uint8_t *const refs1[], const int64_t ref_ids1[], const int stride1[], int nref1,
                                 int width, int height, int range, int radius )
{
    x265_gpu_ctx_t *c = (x265_gpu_ctx_t*)ctx;
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

    /* Testing hook: X265_GPU_SIMULATE_FAIL=cuda makes this backend fail at
     * the first analysis (used to exercise the OpenCL fallback). */
    {
        static int sim = -1;
        if( sim < 0 )
        {
            const char *ev = getenv( "X265_GPU_SIMULATE_FAIL" );
            sim = ev && !strcmp( ev, "cuda" );
        }
        if( sim )
        {
            c->disabled = 1;
            return -1;
        }
    }

    std::lock_guard<std::mutex> guard( c->lock );

    /* Candidates are only valid for the references analysed in this call;
     * anything else must fall back to the CPU search. */
    frame->nfilled[0] = 0;
    frame->nfilled[1] = 0;

    int lw = c->lw, lh = c->lh;
    dim3 blk( 16, 16 );
    dim3 grd( (lw + 15) / 16, (lh + 15) / 16 );
    int threads = 256;
    int mbcount = c->bw * c->bh;
    int total = mbcount * GK;

    double t_last = c->timing ? gpu_now_ms() : 0;
#define TIM_MARK(member) \
    if( c->timing ) { cudaStreamSynchronize( c->stream ); \
                      c->member += gpu_now_ms() - t_last; t_last = gpu_now_ms(); }

    if( upload_plane_pinned( c, c->d_cur, cur, cur_stride, width, height ) < 0 )
        goto fail;
    TIM_MARK( t_upload_cur );
    {
        const uint8_t *a_src = c->d_cur;   int a_sw = c->w, a_sh = c->h;
        uint8_t *a_dst = c->d_cur_low;     int a_dw = lw, a_dh = lh;
        void *args[] = { &a_src, &a_sw, &a_sh, &a_dst, &a_dw, &a_dh };
        if( gpu_launch( c, c->fn_ds, grd, blk, args ) < 0 )
            goto fail;
    }
    TIM_MARK( t_down );

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
            /* Remember which reference this slot belongs to. */
            frame->slot_ptr[list][r] = refs[r];
            frame->slot_id[list][r] = ids[r];
            /* Reuse the device copy of a reference plane while its content
             * stays the same.  References are typically re-analysed every
             * frame, so this removes most of the PCIe traffic. */
            uint8_t *d_ref = NULL, *d_ref_low = NULL;
            /* Testing hook: X265_GPU_NOCACHE=1 disables the reference
             * plane cache (used to compare backends bit-for-bit). */
            static int nocache = -1;
            if( nocache < 0 )
                nocache = getenv( "X265_GPU_NOCACHE" ) != NULL;
            if( !nocache )
                for( int i = 0; i < X265_GPU_CACHE; i++ )
                    if( c->cache[i].ptr == refs[r] && c->cache[i].id == ids[r] )
                    {
                        d_ref = c->cache[i].full;
                        d_ref_low = c->cache[i].low;
                        break;
                    }
            if( !d_ref )
            {
                int slot = c->cache_next++ % X265_GPU_CACHE;
                d_ref = c->cache[slot].full;
                d_ref_low = c->cache[slot].low;
                if( upload_plane_pinned( c, d_ref, refs[r], strides[r], width, height ) < 0 )
                    goto fail;
                TIM_MARK( t_upload_ref );
                {
                    const uint8_t *a_src = d_ref;   int a_sw = c->w, a_sh = c->h;
                    uint8_t *a_dst = d_ref_low;     int a_dw = lw, a_dh = lh;
                    void *args[] = { &a_src, &a_sw, &a_sh, &a_dst, &a_dw, &a_dh };
                    if( gpu_launch( c, c->fn_ds, grd, blk, args ) < 0 )
                        goto fail;
                }
                TIM_MARK( t_down );
                c->cache[slot].ptr = refs[r];
                c->cache[slot].id = ids[r];
            }
            {
                const uint8_t *a_cur = c->d_cur_low; int a_cw = lw, a_ch = lh;
                const uint8_t *a_ref = d_ref_low;    int a_rw = lw, a_rh = lh;
                int a_bw = c->bw, a_bh = c->bh, a_range = range;
                Cand *a_out = c->d_coarse;
                void *args[] = { &a_cur, &a_cw, &a_ch, &a_ref, &a_rw, &a_rh,
                                 &a_bw, &a_bh, &a_range, &a_out };
                if( gpu_launch( c, c->fn_coarse, dim3( (mbcount + threads - 1) / threads ),
                                dim3( threads ), args ) < 0 )
                    goto fail;
            }
            TIM_MARK( t_coarse );
            {
                const uint8_t *a_cur = c->d_cur; int a_cw = c->w, a_ch = c->h;
                const uint8_t *a_ref = d_ref;    int a_rw = c->w, a_rh = c->h;
                int a_bw = c->bw, a_bh = c->bh, a_radius = radius;
                const Cand *a_coarse = c->d_coarse;
                Cand *a_out = c->d_out;
                void *args[] = { &a_cur, &a_cw, &a_ch, &a_ref, &a_rw, &a_rh,
                                 &a_bw, &a_bh, &a_radius, &a_coarse, &a_out };
                if( gpu_launch( c, c->fn_refine, dim3( (total + threads - 1) / threads ),
                                dim3( threads ), args ) < 0 )
                    goto fail;
            }
            TIM_MARK( t_refine );
            /* Stage the D2H through pinned memory: writing directly into the
             * frame's pageable candidate buffer would synchronise the stream
             * and serialise the analysis onto the calling thread. */
            if( cudaMemcpyAsync( c->h_out + (size_t)(list * X265_GPU_MAX_REFS + r) * total,
                                 c->d_out, (size_t)total * sizeof(Cand),
                                 cudaMemcpyDeviceToHost, c->stream ) != cudaSuccess )
                goto fail;
            TIM_MARK( t_d2h );
        }
        frame->nfilled[list] = nref;
    }
    if( cudaStreamSynchronize( c->stream ) != cudaSuccess )
        goto fail;
    TIM_MARK( t_sync );
    publish_candidates( c, frame, 0, frame->nfilled[0] );
    publish_candidates( c, frame, 1, frame->nfilled[1] );
    c->nframes++;
#undef TIM_MARK
    return 0;

fail:
    cudaStreamSynchronize( c->stream );
    c->disabled = 1;
    frame->nfilled[0] = frame->nfilled[1] = 0;
    return -1;
}

extern "C" const x265_gpu_cand_t *x265_gpu_frame_get( const x265_gpu_frame_t *f,
                                                      int list, const void *ref_plane, int64_t ref_id,
                                                      int bx, int by, int *count )
{
    /* Only references published by the most recent analysis have valid
     * candidates; everything else falls back to the normal CPU search.
     * Slots are matched by identity so candidates stay usable when the
     * analysis ran for a predicted reference set. */
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

extern "C" int x265_gpu_frame_covers( const x265_gpu_frame_t *f,
                                      int list, const void *ref_plane, int64_t ref_id )
{
    if( !f || list < 0 || list > 1 )
        return 0;
    for( int r = 0; r < f->nfilled[list]; r++ )
        if( f->slot_ptr[list][r] == ref_plane && f->slot_id[list][r] == ref_id )
            return 1;
    return 0;
}

/* ---- GPU lookahead -------------------------------------------------------
 * The lowres planes are uploaded into the context's device buffers
 * (d_cur for fenc, the first cache slot's full buffer for ref, without
 * caching semantics: refs change every call).  Outputs are staged
 * through lazily allocated device buffers and copied back synchronously. */

static int cuda_la_ensure( x265_gpu_ctx_t *c, uint8_t **buf, size_t *sz, size_t need )
{
    if( *buf && *sz >= need )
        return 0;
    if( *buf )
    {
        cudaFree( *buf );
        *buf = NULL;
        *sz = 0;
    }
    if( cudaMalloc( (void**)buf, need ) != cudaSuccess )
        return -1;
    *sz = need;
    return 0;
}

static int cuda_lowres_common( x265_gpu_ctx_t *c, int is_table,
                               const uint8_t *fenc, int fenc_stride, int fenc_bufw, int fenc_bufh,
                               const uint8_t *ref, int ref_stride, int ref_bufw, int ref_bufh,
                               int pad_x, int pad_y, int wcu, int hcu, int rt,
                               void *out )
{
    if( !c || !out || !fenc || !ref || c->disabled )
        return -1;
    if( wcu <= 0 || hcu <= 0 || rt < 1 || rt > 64 )
        return -1;
    if( fenc_bufw <= 0 || fenc_bufh <= 0 || fenc_bufw > c->w || fenc_bufh > c->h ||
        ref_bufw <= 0 || ref_bufh <= 0 || ref_bufw > c->w || ref_bufh > c->h )
        return -1;
    if( pad_x < 0 || pad_y < 0 || pad_x + wcu * 8 > fenc_bufw || pad_y + hcu * 8 > fenc_bufh )
        return -1;
    {
        static int sim = -1;
        if( sim < 0 )
        {
            const char *ev = getenv( "X265_GPU_SIMULATE_FAIL" );
            sim = ev && !strcmp( ev, "cuda" );
        }
        if( sim )
        {
            c->disabled = 1;
            return -1;
        }
    }
    std::lock_guard<std::mutex> guard( c->lock );
    if( upload_plane_pinned( c, c->d_cur, fenc, fenc_stride, fenc_bufw, fenc_bufh ) < 0 )
        goto fail;
    if( upload_plane_pinned( c, c->cache[0].full, ref, ref_stride, ref_bufw, ref_bufh ) < 0 )
        goto fail;
    {
        int nthreads = 256;
        if( is_table )
        {
            int tw = 2 * rt + 1;
            size_t total_sz = (size_t)wcu * hcu * tw * tw * sizeof(uint32_t);
            int total = wcu * hcu * tw * tw;
            if( cuda_la_ensure( c, &c->d_la_table, &c->d_la_table_sz, total_sz ) < 0 )
                goto fail;
            const uint8_t *a_cur = c->d_cur;   int a_cw = c->w, a_ch = c->h;
            const uint8_t *a_ref = c->cache[0].full; int a_rw = c->w, a_rh = c->h;
            int a_px = pad_x, a_py = pad_y, a_wcu = wcu, a_hcu = hcu, a_tr = rt;
            uint32_t *a_out = (uint32_t*)c->d_la_table;
            void *args[] = { &a_cur, &a_cw, &a_ch, &a_ref, &a_rw, &a_rh,
                             &a_px, &a_py, &a_wcu, &a_hcu, &a_tr, &a_out };
            if( gpu_launch( c, c->fn_la_table,
                            dim3( (total + nthreads - 1) / nthreads ), dim3( nthreads ), args ) < 0 )
                goto fail;
            if( cudaMemcpy( out, c->d_la_table, total_sz, cudaMemcpyDeviceToHost ) != cudaSuccess )
                goto fail;
        }
        else
        {
            size_t total_sz = (size_t)wcu * hcu * sizeof(Cand);
            int total = wcu * hcu;
            if( cuda_la_ensure( c, &c->d_la_best, &c->d_la_best_sz, total_sz ) < 0 )
                goto fail;
            const uint8_t *a_cur = c->d_cur;   int a_cw = c->w, a_ch = c->h;
            const uint8_t *a_ref = c->cache[0].full; int a_rw = c->w, a_rh = c->h;
            int a_px = pad_x, a_py = pad_y, a_wcu = wcu, a_hcu = hcu, a_range = rt;
            Cand *a_out = (Cand*)c->d_la_best;
            void *args[] = { &a_cur, &a_cw, &a_ch, &a_ref, &a_rw, &a_rh,
                             &a_px, &a_py, &a_wcu, &a_hcu, &a_range, &a_out };
            if( gpu_launch( c, c->fn_la_best,
                            dim3( (total + nthreads - 1) / nthreads ), dim3( nthreads ), args ) < 0 )
                goto fail;
            if( cudaMemcpy( out, c->d_la_best, total_sz, cudaMemcpyDeviceToHost ) != cudaSuccess )
                goto fail;
        }
    }
    if( cudaStreamSynchronize( c->stream ) != cudaSuccess )
        goto fail;
    return 0;

fail:
    cudaStreamSynchronize( c->stream );
    c->disabled = 1;
    return -1;
}

extern "C" int x265_gpu_cuda_lowres_best( void *ctx, x265_gpu_cand_t *out,
                                          const uint8_t *fenc, int fenc_stride, int fenc_bufw, int fenc_bufh,
                                          const uint8_t *ref, int ref_stride, int ref_bufw, int ref_bufh,
                                          int pad_x, int pad_y, int wcu, int hcu, int range )
{
    return cuda_lowres_common( (x265_gpu_ctx_t*)ctx, 0, fenc, fenc_stride, fenc_bufw, fenc_bufh,
                               ref, ref_stride, ref_bufw, ref_bufh, pad_x, pad_y, wcu, hcu, range, out );
}

extern "C" int x265_gpu_cuda_lowres_table( void *ctx, uint32_t *out,
                                           const uint8_t *fenc, int fenc_stride, int fenc_bufw, int fenc_bufh,
                                           const uint8_t *ref, int ref_stride, int ref_bufw, int ref_bufh,
                                           int pad_x, int pad_y, int wcu, int hcu, int tr )
{
    return cuda_lowres_common( (x265_gpu_ctx_t*)ctx, 1, fenc, fenc_stride, fenc_bufw, fenc_bufh,
                               ref, ref_stride, ref_bufw, ref_bufh, pad_x, pad_y, wcu, hcu, tr, out );
}

static int gpu_cuda_available( void )
{
    int ndev = 0;
    return cudaGetDeviceCount( &ndev ) == cudaSuccess && ndev > 0;
}

/* --- backend registration ------------------------------------------------- */

extern "C" const x265_gpu_backend_t x265_gpu_backend_cuda =
{
    X265_GPU_BACKEND_CUDA,
    "cuda",
    gpu_cuda_available,
    x265_gpu_cuda_create,
    x265_gpu_cuda_destroy,
    x265_gpu_cuda_name,
    x265_gpu_cuda_frame_alloc,
    x265_gpu_cuda_frame_free,
    x265_gpu_cuda_analyze,
    x265_gpu_cuda_frame_get,
    x265_gpu_cuda_frame_covers,
    x265_gpu_cuda_lowres_best,
    x265_gpu_cuda_lowres_table,
};
