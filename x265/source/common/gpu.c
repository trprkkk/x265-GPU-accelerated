/*****************************************************************************
 * gpu.c: dispatcher for the GPU candidate generator
 *****************************************************************************
 * The public API declared in gpu.h is implemented here by forwarding to the
 * selected backend: CUDA (gpu_cuda.cpp with the kernels in gpu_kernels.cu),
 * OpenCL (gpu_cl.c) or the scalar CPU reference (gpu_cpu.cpp).  Frames
 * start with a backend header so lookups can be routed without extra state.
 *
 * When an analysis fails at runtime, x265_gpu_fallback() retires the failing
 * backend (kept alive so that frames still holding its candidate buffers can
 * be freed safely) and continues on the other backend.
 *
 * x265 encodes frames in parallel (several FrameEncoder threads), so all
 * analyses are serialised through one process-wide mutex: there is only
 * one GPU, and the backends are not required to be internally thread-safe.
 *****************************************************************************/

#include "gpu.h"

#include <stdio.h>
#include <stdlib.h>

/* Set by the CMake build (see source/CMakeLists.txt); 0 = backend not
 * compiled in. */
#ifndef X265_GPU_HAS_CUDA
#define X265_GPU_HAS_CUDA 0
#endif
#ifndef X265_GPU_HAS_OPENCL
#define X265_GPU_HAS_OPENCL 0
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

typedef struct
{
    const x265_gpu_backend_t *be;
    void *impl;
    const x265_gpu_backend_t *old_be;   /* retired backend, kept for frame frees */
    void *old_impl;
    /* creation parameters, needed to build a fallback backend */
    int w, h, max_refs, device, sb;
} gpu_ctx_pub_t;

#if defined(_WIN32)
/* NOTE: SRWLOCK has static storage (SRWLOCK_STATIC_INIT), which avoids
 * the classic lazy-init CRITICAL_SECTION bug: entering a CRITICAL_SECTION
 * before InitializeCriticalSection is undefined behaviour. */
static SRWLOCK g_gpu_lock;
static void gpu_lock( void )   { AcquireSRWLockExclusive( &g_gpu_lock ); }
static void gpu_unlock( void ) { ReleaseSRWLockExclusive( &g_gpu_lock ); }
#else
static pthread_mutex_t g_gpu_lock = PTHREAD_MUTEX_INITIALIZER;
static void gpu_lock( void )   { pthread_mutex_lock( &g_gpu_lock ); }
static void gpu_unlock( void ) { pthread_mutex_unlock( &g_gpu_lock ); }
#endif

static const x265_gpu_backend_t *gpu_backend_by_id( int id )
{
    switch( id )
    {
#if X265_GPU_HAS_CUDA
        case X265_GPU_BACKEND_CUDA:   return &x265_gpu_backend_cuda;
#endif
#if X265_GPU_HAS_OPENCL
        case X265_GPU_BACKEND_OPENCL: return &x265_gpu_backend_cl;
#endif
        case X265_GPU_BACKEND_CPU:    return &x265_gpu_backend_cpu;
    }
    return NULL;
}

static const x265_gpu_backend_t *gpu_backend_of_frame( const x265_gpu_frame_t *frame )
{
    if( !frame )
        return NULL;
    return gpu_backend_by_id( ((const x265_gpu_frame_header_t*)frame)->backend );
}

void *x265_gpu_create( int width, int height, int max_refs, int device, int backend,
                       int sample_bytes )
{
    const x265_gpu_backend_t *be = NULL;
    void *impl = NULL;
    if( sample_bytes < X265_GPU_SB_MIN || sample_bytes > X265_GPU_SB_MAX )
        return NULL;
    if( backend == X265_GPU_BACKEND_AUTO )
    {
        static const int order[2] = { X265_GPU_BACKEND_CUDA, X265_GPU_BACKEND_OPENCL };
        for( int i = 0; i < 2 && !impl; i++ )
        {
            be = gpu_backend_by_id( order[i] );
            if( be && be->available() )
                impl = be->create( width, height, max_refs, device, sample_bytes );
        }
    }
    else
    {
        be = gpu_backend_by_id( backend );
        if( be && be->available() )
            impl = be->create( width, height, max_refs, device, sample_bytes );
    }
    if( !impl )
        return NULL;
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)calloc( 1, sizeof(*pub) );
    if( !pub )
    {
        be->destroy( impl );
        return NULL;
    }
    pub->be = be;
    pub->impl = impl;
    pub->w = width;
    pub->h = height;
    pub->max_refs = max_refs;
    pub->device = device;
    pub->sb = sample_bytes;
    return pub;
}

int x265_gpu_sample_bytes( void *ctx )
{
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)ctx;
    return pub ? pub->sb : 0;
}

void x265_gpu_destroy( void *ctx )
{
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)ctx;
    if( !pub )
        return;
    gpu_lock();
    pub->be->destroy( pub->impl );
    if( pub->old_be )
        pub->old_be->destroy( pub->old_impl );
    gpu_unlock();
    free( pub );
}

/* After an analysis failure: drain and retire the active backend and start
 * the other one.  At most one switch is allowed, so a double failure ends in
 * the regular CPU fallback.  The caller must have released the frames it
 * still holds (their buffers belong to the failing backend). */
int x265_gpu_fallback( void *ctx )
{
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)ctx;
    if( !pub )
        return -1;
    gpu_lock();
    if( pub->old_be )
    {
        gpu_unlock();
        return -1;
    }
    static const int order[2] = { X265_GPU_BACKEND_CUDA, X265_GPU_BACKEND_OPENCL };
    int ret = -1;
    for( int i = 0; i < 2; i++ )
    {
        const x265_gpu_backend_t *be = gpu_backend_by_id( order[i] );
        if( !be || be == pub->be || !be->available() )
            continue;
        void *impl = be->create( pub->w, pub->h, pub->max_refs, pub->device, pub->sb );
        if( !impl )
            continue;
        pub->old_be = pub->be;
        pub->old_impl = pub->impl;
        pub->be = be;
        pub->impl = impl;
        ret = 0;
        break;
    }
    gpu_unlock();
    return ret;
}

const char *x265_gpu_name( void *ctx )
{
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)ctx;
    return pub ? pub->be->name() : "";
}

const char *x265_gpu_backend_id( void *ctx )
{
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)ctx;
    return pub ? pub->be->name_id : "";
}

x265_gpu_frame_t *x265_gpu_frame_alloc( void *ctx, int bw, int bh, int nref0, int nref1 )
{
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)ctx;
    return pub ? pub->be->frame_alloc( bw, bh, nref0, nref1 ) : NULL;
}

void x265_gpu_frame_free( x265_gpu_frame_t *frame )
{
    const x265_gpu_backend_t *be = gpu_backend_of_frame( frame );
    if( be )
        be->frame_free( frame );
}

/* The analysis can only run on frames allocated by the active backend. */
static int gpu_frame_is_current( const gpu_ctx_pub_t *pub, const x265_gpu_frame_t *frame )
{
    const x265_gpu_backend_t *fbe = gpu_backend_of_frame( frame );
    return fbe && fbe == pub->be;
}

#define GPU_ANALYZE_ARGS \
    x265_gpu_frame_t *frame, const uint8_t *cur, int cur_stride, \
    uint8_t *const refs0[], const int64_t ref_ids0[], const int stride0[], int nref0, \
    uint8_t *const refs1[], const int64_t ref_ids1[], const int stride1[], int nref1, \
    int width, int height, int range, int radius

int x265_gpu_analyze( void *ctx, GPU_ANALYZE_ARGS )
{
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)ctx;
    if( !pub )
        return -1;
    gpu_lock();
    int ok = gpu_frame_is_current( pub, frame ) ?
        pub->be->analyze( pub->impl, frame, cur, cur_stride,
                          refs0, ref_ids0, stride0, nref0,
                          refs1, ref_ids1, stride1, nref1,
                          width, height, range, radius ) : -1;
    /* Debug: X265_GPU_DUMP=1 prints candidates for a few blocks so that
     * backends can be compared bit-for-bit outside the x265 pipeline. */
    if( ok == 0 && nref0 > 0 )
    {
        const char *dump = getenv( "X265_GPU_DUMP" );
        if( dump && dump[0] == '1' )
        {
            static int callno = 0;
            int call = callno++;
            {
                /* checksum of the first reference plane region (what the
                 * search reads) to detect input divergence early */
            static int nchecksum = 0;
            if( nchecksum < 8 )
            {
                unsigned sum = 0;
                const uint8_t *rp = refs0[0];
                int rs = stride0[0];
                for( int y = 0; y < height; y++ )
                    for( int x = 0; x < width; x++ )
                        sum += rp[y * rs + x];
                fprintf( stderr, "CHECKSUM %s call=%d ref0 ptr=%p id=%lld stride=%d sum=%u cur ptr=%p region=%u\n",
                         pub->be->name(), call, (void*)refs0[0], (long long)ref_ids0[0], stride0[0], sum, (void*)cur, cur[0] + cur[1] );
                nchecksum++;
            }
            }
            for( int by = 0; by < 3; by++ )
                for( int bx = 0; bx < 5; bx++ )
                {
                    int n = 0;
                    const x265_gpu_cand_t *cc = pub->be->frame_get( frame, 0, refs0[0], ref_ids0[0], bx, by, &n );
                    fprintf( stderr, "DUMP call=%d b%d,%d n=%d:", call, bx, by, n );
                    if( cc )
                        for( int i = 0; i < n; i++ )
                            fprintf( stderr, " %d,%d|%u", cc[i].mv[0], cc[i].mv[1], cc[i].sad );
                    fprintf( stderr, "\n" );
                }
        }
    }
    gpu_unlock();
    return ok;
}

const x265_gpu_cand_t *x265_gpu_frame_get( const x265_gpu_frame_t *frame,
                                           int list, const void *ref_plane, int64_t ref_id,
                                           int bx, int by, int *count )
{
    const x265_gpu_backend_t *be = gpu_backend_of_frame( frame );
    return be ? be->frame_get( frame, list, ref_plane, ref_id, bx, by, count ) : NULL;
}

int x265_gpu_frame_covers( const x265_gpu_frame_t *frame,
                           int list, const void *ref_plane, int64_t ref_id )
{
    const x265_gpu_backend_t *be = gpu_backend_of_frame( frame );
    return be ? be->frame_covers( frame, list, ref_plane, ref_id ) : 0;
}

int x265_gpu_lowres_best( void *ctx, x265_gpu_cand_t *out,
                          const uint8_t *fenc, int fenc_stride, int fenc_bufw, int fenc_bufh,
                          const uint8_t *ref, int ref_stride, int ref_bufw, int ref_bufh,
                          int pad_x, int pad_y, int wcu, int hcu, int range )
{
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)ctx;
    if( !pub || !pub->be->lowres_best )
        return -1;
    gpu_lock();
    int ret = pub->be->lowres_best( pub->impl, out, fenc, fenc_stride, fenc_bufw, fenc_bufh,
                                    ref, ref_stride, ref_bufw, ref_bufh,
                                    pad_x, pad_y, wcu, hcu, range );
    gpu_unlock();
    return ret;
}

int x265_gpu_lowres_table( void *ctx, uint32_t *out,
                           const uint8_t *fenc, int fenc_stride, int fenc_bufw, int fenc_bufh,
                           const uint8_t *ref, int ref_stride, int ref_bufw, int ref_bufh,
                           int pad_x, int pad_y, int wcu, int hcu, int tr )
{
    gpu_ctx_pub_t *pub = (gpu_ctx_pub_t*)ctx;
    if( !pub || !pub->be->lowres_table )
        return -1;
    gpu_lock();
    int ret = pub->be->lowres_table( pub->impl, out, fenc, fenc_stride, fenc_bufw, fenc_bufh,
                                     ref, ref_stride, ref_bufw, ref_bufh,
                                     pad_x, pad_y, wcu, hcu, tr );
    gpu_unlock();
    return ret;
}
