/*****************************************************************************
 * gpu_cl.c: OpenCL backend for the GPU candidate generator
 *****************************************************************************
 * Same algorithm and public behaviour as the CUDA backend (gpu_cuda.cpp with
 * the kernels in gpu_kernels.cu): a quarter-resolution coarse search followed
 * by a full-resolution refinement returns up to X265_GPU_CANDS integer-pel
 * candidates per 32x32 block and reference.  The candidates are matched by
 * reference identity (plane pointer + POC) and published only after the
 * queue has been finished, so the result is identical to the synchronous
 * CUDA path (and to the scalar CPU reference in gpu_cpu.cpp).
 *
 * The analysis is serialised through the dispatcher mutex (gpu.c), so the
 * command queue is only touched from one encoder thread at a time.
 *****************************************************************************/

#include "gpu.h"

#include <CL/cl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

/* OpenCL is loaded dynamically (like x265's own OpenCL path does) so the
 * executable starts on machines without an OpenCL ICD; the backend simply
 * reports itself unavailable in that case. */
static cl_int (CL_API_CALL *p_clGetPlatformIDs)( cl_uint, cl_platform_id *, cl_uint * );
static cl_int (CL_API_CALL *p_clGetDeviceIDs)( cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint * );
static cl_int (CL_API_CALL *p_clGetDeviceInfo)( cl_device_id, cl_device_info, size_t, void *, size_t * );
static cl_context (CL_API_CALL *p_clCreateContext)( const cl_context_properties *, cl_uint, const cl_device_id *,
                                                    void (CL_CALLBACK *)( const char *, const void *, size_t, void * ),
                                                    void *, cl_int * );
static cl_command_queue (CL_API_CALL *p_clCreateCommandQueue)( cl_context, cl_device_id,
                                                               cl_command_queue_properties, cl_int * );
static cl_program (CL_API_CALL *p_clCreateProgramWithSource)( cl_context, cl_uint, const char **,
                                                              const size_t *, cl_int * );
static cl_int (CL_API_CALL *p_clBuildProgram)( cl_program, cl_uint, cl_device_id *, const char *,
                                               void (CL_CALLBACK *)( cl_program, void * ), void * );
static cl_int (CL_API_CALL *p_clGetProgramBuildInfo)( cl_program, cl_device_id, cl_program_build_info,
                                                      size_t, void *, size_t * );
static cl_kernel (CL_API_CALL *p_clCreateKernel)( cl_program, const char *, cl_int * );
static cl_int (CL_API_CALL *p_clSetKernelArg)( cl_kernel, cl_uint, size_t, const void * );
static cl_int (CL_API_CALL *p_clEnqueueNDRangeKernel)( cl_command_queue, cl_kernel, cl_uint, const size_t *,
                                                       const size_t *, const size_t *, cl_uint,
                                                       const cl_event *, cl_event * );
static cl_mem (CL_API_CALL *p_clCreateBuffer)( cl_context, cl_mem_flags, size_t, void *, cl_int * );
static cl_int (CL_API_CALL *p_clEnqueueWriteBuffer)( cl_command_queue, cl_mem, cl_bool, size_t, size_t,
                                                     const void *, cl_uint, const cl_event *, cl_event * );
static cl_int (CL_API_CALL *p_clEnqueueReadBuffer)( cl_command_queue, cl_mem, cl_bool, size_t, size_t,
                                                    void *, cl_uint, const cl_event *, cl_event * );
static cl_int (CL_API_CALL *p_clFinish)( cl_command_queue );
static void * (CL_API_CALL *p_clEnqueueMapBuffer)( cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, size_t,
                                                   cl_uint, const cl_event *, cl_event *, cl_int * );
static cl_int (CL_API_CALL *p_clEnqueueUnmapMemObject)( cl_command_queue, cl_mem, void *, cl_uint,
                                                        const cl_event *, cl_event * );
static cl_int (CL_API_CALL *p_clReleaseMemObject)( cl_mem );
static cl_int (CL_API_CALL *p_clReleaseKernel)( cl_kernel );
static cl_int (CL_API_CALL *p_clReleaseProgram)( cl_program );
static cl_int (CL_API_CALL *p_clReleaseCommandQueue)( cl_command_queue );
static cl_int (CL_API_CALL *p_clReleaseContext)( cl_context );

#define clGetPlatformIDs       p_clGetPlatformIDs
#define clGetDeviceIDs         p_clGetDeviceIDs
#define clGetDeviceInfo        p_clGetDeviceInfo
#define clCreateContext        p_clCreateContext
#define clCreateCommandQueue   p_clCreateCommandQueue
#define clCreateProgramWithSource p_clCreateProgramWithSource
#define clBuildProgram         p_clBuildProgram
#define clGetProgramBuildInfo  p_clGetProgramBuildInfo
#define clCreateKernel         p_clCreateKernel
#define clSetKernelArg         p_clSetKernelArg
#define clEnqueueNDRangeKernel p_clEnqueueNDRangeKernel
#define clCreateBuffer         p_clCreateBuffer
#define clEnqueueWriteBuffer   p_clEnqueueWriteBuffer
#define clEnqueueReadBuffer    p_clEnqueueReadBuffer
#define clFinish               p_clFinish
#define clEnqueueMapBuffer     p_clEnqueueMapBuffer
#define clEnqueueUnmapMemObject p_clEnqueueUnmapMemObject
#define clReleaseMemObject     p_clReleaseMemObject
#define clReleaseKernel        p_clReleaseKernel
#define clReleaseProgram       p_clReleaseProgram
#define clReleaseCommandQueue  p_clReleaseCommandQueue
#define clReleaseContext       p_clReleaseContext

static void *gpu_cl_lib;
static int gpu_cl_load_state = -1;

#ifdef _WIN32
#define GPU_CL_LOAD( name ) p_##name = (void*)GetProcAddress( (HMODULE)gpu_cl_lib, #name );
#else
#define GPU_CL_LOAD( name ) p_##name = (void*)dlsym( gpu_cl_lib, #name );
#endif

static int gpu_cl_load( void )
{
    if( gpu_cl_load_state >= 0 )
        return gpu_cl_load_state;
#ifdef _WIN32
    gpu_cl_lib = (void*)LoadLibraryW( L"OpenCL.dll" );
#else
    gpu_cl_lib = dlopen( "libOpenCL.so", RTLD_NOW );
#endif
    if( !gpu_cl_lib )
    {
        gpu_cl_load_state = 0;
        return 0;
    }
    GPU_CL_LOAD( clGetPlatformIDs )
    GPU_CL_LOAD( clGetDeviceIDs )
    GPU_CL_LOAD( clGetDeviceInfo )
    GPU_CL_LOAD( clCreateContext )
    GPU_CL_LOAD( clCreateCommandQueue )
    GPU_CL_LOAD( clCreateProgramWithSource )
    GPU_CL_LOAD( clBuildProgram )
    GPU_CL_LOAD( clGetProgramBuildInfo )
    GPU_CL_LOAD( clCreateKernel )
    GPU_CL_LOAD( clSetKernelArg )
    GPU_CL_LOAD( clEnqueueNDRangeKernel )
    GPU_CL_LOAD( clCreateBuffer )
    GPU_CL_LOAD( clEnqueueWriteBuffer )
    GPU_CL_LOAD( clEnqueueReadBuffer )
    GPU_CL_LOAD( clFinish )
    GPU_CL_LOAD( clEnqueueMapBuffer )
    GPU_CL_LOAD( clEnqueueUnmapMemObject )
    GPU_CL_LOAD( clReleaseMemObject )
    GPU_CL_LOAD( clReleaseKernel )
    GPU_CL_LOAD( clReleaseProgram )
    GPU_CL_LOAD( clReleaseCommandQueue )
    GPU_CL_LOAD( clReleaseContext )
#undef GPU_CL_LOAD
    gpu_cl_load_state = 1;
#define GPU_CL_CHECK( name ) \
    if( !p_##name ) { fprintf( stderr, "x265 OpenCL ME: missing OpenCL symbol %s\n", #name ); \
                      gpu_cl_load_state = 0; }
    GPU_CL_CHECK( clGetPlatformIDs )
    GPU_CL_CHECK( clGetDeviceIDs )
    GPU_CL_CHECK( clGetDeviceInfo )
    GPU_CL_CHECK( clCreateContext )
    GPU_CL_CHECK( clCreateCommandQueue )
    GPU_CL_CHECK( clCreateProgramWithSource )
    GPU_CL_CHECK( clBuildProgram )
    GPU_CL_CHECK( clGetProgramBuildInfo )
    GPU_CL_CHECK( clCreateKernel )
    GPU_CL_CHECK( clSetKernelArg )
    GPU_CL_CHECK( clEnqueueNDRangeKernel )
    GPU_CL_CHECK( clCreateBuffer )
    GPU_CL_CHECK( clEnqueueWriteBuffer )
    GPU_CL_CHECK( clEnqueueReadBuffer )
    GPU_CL_CHECK( clFinish )
    GPU_CL_CHECK( clEnqueueMapBuffer )
    GPU_CL_CHECK( clEnqueueUnmapMemObject )
    GPU_CL_CHECK( clReleaseMemObject )
    GPU_CL_CHECK( clReleaseKernel )
    GPU_CL_CHECK( clReleaseProgram )
    GPU_CL_CHECK( clReleaseCommandQueue )
    GPU_CL_CHECK( clReleaseContext )
#undef GPU_CL_CHECK
    if( !gpu_cl_load_state )
        fprintf( stderr, "x265 OpenCL ME: OpenCL runtime is not usable\n" );
    return gpu_cl_load_state;
}

#define GK X265_GPU_CANDS

#if X265_GPU_CANDS != 4
#error "gpu_cl.c kernel source assumes X265_GPU_CANDS == 4"
#endif

typedef struct x265_gpu_frame_t gpu_cl_frame_t;
typedef struct gpu_cl_ctx_t gpu_cl_ctx_t;

/* Forward declarations (used before their definitions below). */
static void gpu_cl_destroy( void *ctx );
static void gpu_cl_frame_free( x265_gpu_frame_t *frame );

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

struct gpu_cl_ctx_t
{
    int w, h, lw, lh, bw, bh, max_refs, device;   /* w/h: padded dims */
    int sb;                 /* bytes per luma sample: 1 (8-bit) or 2 (10/12-bit) */
    const char *pix_type;   /* OpenCL sample type used to build the kernels */
    cl_context cl_ctx;
    cl_command_queue queue;
    cl_program program;
    cl_kernel k_down, k_coarse, k_refine;
    cl_kernel k_la_best, k_la_table;
    cl_mem d_la_best, d_la_table;
    size_t d_la_best_sz, d_la_table_sz;
    cl_mem d_cur, d_cur_low, d_coarse, d_out;
    cl_mem stage_ref;      /* pinned staging for plane uploads */
    cl_mem stage_out;      /* pinned staging for candidate readbacks */
    uint8_t *stage_ref_map;
    uint8_t *stage_out_map;
    struct
    {
        const void *ptr;
        int64_t id;
        cl_mem full, low;
    } cache[X265_GPU_CACHE];
    int cache_next;
    int disabled;
    int timing;
};

static char gpu_cl_name[256] = "OpenCL device";

/* The kernels mirror the CUDA kernels (gpu_kernels.cu) line for line (same
 * iteration order and strict comparisons), so both backends produce the same
 * candidates. */
static const char *gpu_cl_source =
"typedef struct { short2 mv; uint sad; } Cand;\n"
"#define GK 4\n"
"inline uint sad8x8( __global const PIX *a, int as, __global const PIX *b, int bs )\n"
"{\n"
"    uint s = 0;\n"
"    for( int j = 0; j < 8; j++ )\n"
"        for( int i = 0; i < 8; i++ )\n"
"        {\n"
"            int d = (int)a[j*as+i] - (int)b[j*bs+i];\n"
"            s += (uint)(d < 0 ? -d : d);\n"
"        }\n"
"    return s;\n"
"}\n"
"inline void insert_topk( Cand *best, int x, int y, uint sad )\n"
"{\n"
"    if( sad >= best[GK-1].sad )\n"
"        return;\n"
"    int i = GK - 1;\n"
"    while( i > 0 && sad < best[i-1].sad )\n"
"    {\n"
"        best[i] = best[i-1];\n"
"        i--;\n"
"    }\n"
"    best[i].mv = (short2)( x, y );\n"
"    best[i].sad = sad;\n"
"}\n"
"__kernel void downsample4( __global const PIX *src, int sw, int sh,\n"
"                           __global PIX *dst, int dw, int dh )\n"
"{\n"
"    int x = get_global_id(0);\n"
"    int y = get_global_id(1);\n"
"    if( x >= dw || y >= dh )\n"
"        return;\n"
"    int sx = x * 4, sy = y * 4;\n"
"    int sum = 0;\n"
"    for( int j = 0; j < 4; j++ )\n"
"        for( int i = 0; i < 4; i++ )\n"
"        {\n"
"            int px = min( sx + i, sw - 1 );\n"
"            int py = min( sy + j, sh - 1 );\n"
"            sum += src[py * sw + px];\n"
"        }\n"
"    dst[y * dw + x] = (PIX)((sum + 8) >> 4);\n"
"}\n"
"__kernel void coarse_search( __global const PIX *cur, int cw, int ch,\n"
"                             __global const PIX *ref, int rw, int rh,\n"
"                             int bw, int bh, int range, __global Cand *out )\n"
"{\n"
"    int mb = get_global_id(0);\n"
"    if( mb >= bw * bh )\n"
"        return;\n"
"    int mx = mb % bw, my = mb / bw;\n"
"    int bx = mx * 8, by = my * 8;\n"
"    Cand best[GK];\n"
"    for( int i = 0; i < GK; i++ )\n"
"    {\n"
"        best[i].mv = (short2)( 0, 0 );\n"
"        best[i].sad = 0xFFFFFFFFu;\n"
"    }\n"
"    int r = range / 4;\n"
"    if( r < 1 )\n"
"        r = 1;\n"
"    for( int dy = -r; dy <= r; dy += 2 )\n"
"        for( int dx = -r; dx <= r; dx += 2 )\n"
"        {\n"
"            int x = max( 0, min( bx + dx, cw - 8 ) );\n"
"            int y = max( 0, min( by + dy, ch - 8 ) );\n"
"            uint sad = sad8x8( cur + by * cw + bx, cw, ref + y * rw + x, rw );\n"
"            insert_topk( best, (x - bx) * 4, (y - by) * 4, sad );\n"
"        }\n"
"    int fx = bx + best[0].mv.x / 4;\n"
"    int fy = by + best[0].mv.y / 4;\n"
"    for( int dy = -1; dy <= 1; dy++ )\n"
"        for( int dx = -1; dx <= 1; dx++ )\n"
"        {\n"
"            if( !dx && !dy )\n"
"                continue;\n"
"            int x = max( 0, min( fx + dx, cw - 8 ) );\n"
"            int y = max( 0, min( fy + dy, ch - 8 ) );\n"
"            uint sad = sad8x8( cur + by * cw + bx, cw, ref + y * rw + x, rw );\n"
"            insert_topk( best, (x - bx) * 4, (y - by) * 4, sad );\n"
"        }\n"
"    for( int i = 0; i < GK; i++ )\n"
"        out[mb * GK + i] = best[i];\n"
"}\n"
"__kernel void refine_search( __global const PIX *cur, int cw, int ch,\n"
"                             __global const PIX *ref, int rw, int rh,\n"
"                             int bw, int bh, int radius,\n"
"                             __global const Cand *coarse, __global Cand *out )\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    int total = bw * bh * GK;\n"
"    if( idx >= total )\n"
"        return;\n"
"    int mb = idx / GK;\n"
"    int mx = mb % bw, my = mb / bw;\n"
"    int bx = mx * 32, by = my * 32;\n"
"    int cx = bx + coarse[idx].mv.x;\n"
"    int cy = by + coarse[idx].mv.y;\n"
"    __global const PIX *pc = cur + by * cw + bx;\n"
"    Cand best;\n"
"    best.mv = (short2)( 0, 0 );\n"
"    best.sad = 0xFFFFFFFFu;\n"
"    for( int dy = -radius; dy <= radius; dy++ )\n"
"        for( int dx = -radius; dx <= radius; dx++ )\n"
"        {\n"
"            int x = max( 0, min( cx + dx, cw - 32 ) );\n"
"            int y = max( 0, min( cy + dy, ch - 32 ) );\n"
"            __global const PIX *pr = ref + y * rw + x;\n"
"            uint sad = 0;\n"
"            for( int j = 0; j < 32; j += 2 )\n"
"                for( int i = 0; i < 32; i += 2 )\n"
"                {\n"
"                    int d = (int)pc[j * cw + i] - (int)pr[j * rw + i];\n"
"                    sad += (uint)(d < 0 ? -d : d);\n"
"                }\n"
"            if( sad < best.sad )\n"
"            {\n"
"                best.sad = sad;\n"
"                best.mv = (short2)( x - bx, y - by );\n"
"            }\n"
"        }\n"
"    out[idx] = best;\n"
"}\n"
"__kernel void lowres_best( __global const PIX *cur, int cw, int ch,\n"
"                             __global const PIX *ref, int rw, int rh,\n"
"                             int pad_x, int pad_y, int wcu, int hcu,\n"
"                             int range, __global Cand *out )\n"
"{\n"
"    int mb = get_global_id(0);\n"
"    if( mb >= wcu * hcu ) return;\n"
"    int cuX = mb % wcu, cuY = mb / wcu;\n"
"    int bx = cuX * 8, by = cuY * 8;\n"
"    int wl = wcu * 8, hl = hcu * 8;\n"
"    int mvmin_x = -bx - 8, mvmin_y = -by - 8;\n"
"    int mvmax_x = wl - bx, mvmax_y = hl - by;\n"
"    __global const PIX *pc = cur + (pad_y + by) * cw + (pad_x + bx);\n"
"    Cand best;\n"
"    best.mv = (short2)( 0, 0 );\n"
"    best.sad = 0xFFFFFFFFu;\n"
"    for( int my = -range; my <= range; my += 4 )\n"
"        for( int mx = -range; mx <= range; mx += 4 )\n"
"        {\n"
"            if( mx < mvmin_x || mx > mvmax_x || my < mvmin_y || my > mvmax_y ) continue;\n"
"            __global const PIX *pr = ref + (pad_y + by + my) * rw + (pad_x + bx + mx);\n"
"            uint sad = sad8x8( pc, cw, pr, rw );\n"
"            if( sad < best.sad ) { best.sad = sad; best.mv = (short2)( mx, my ); }\n"
"        }\n"
"    int cx = best.mv.x, cy = best.mv.y;\n"
"    for( int my = cy - 4; my <= cy + 4; my++ )\n"
"        for( int mx = cx - 4; mx <= cx + 4; mx++ )\n"
"        {\n"
"            if( mx < mvmin_x || mx > mvmax_x || my < mvmin_y || my > mvmax_y ) continue;\n"
"            __global const PIX *pr = ref + (pad_y + by + my) * rw + (pad_x + bx + mx);\n"
"            uint sad = sad8x8( pc, cw, pr, rw );\n"
"            if( sad < best.sad ) { best.sad = sad; best.mv = (short2)( mx, my ); }\n"
"        }\n"
"    out[mb] = best;\n"
"}\n"
"__kernel void lowres_table( __global const PIX *cur, int cw, int ch,\n"
"                              __global const PIX *ref, int rw, int rh,\n"
"                              int pad_x, int pad_y, int wcu, int hcu,\n"
"                              int tr, __global uint *out )\n"
"{\n"
"    int idx = get_global_id(0);\n"
"    int tw = 2 * tr + 1;\n"
"    if( idx >= wcu * hcu * tw * tw ) return;\n"
"    int mx = idx % tw - tr;\n"
"    int my = (idx / tw) % tw - tr;\n"
"    int mb = idx / (tw * tw);\n"
"    int cuX = mb % wcu, cuY = mb / wcu;\n"
"    int bx = cuX * 8, by = cuY * 8;\n"
"    __global const PIX *pc = cur + (pad_y + by) * cw + (pad_x + bx);\n"
"    __global const PIX *pr = ref + (pad_y + by + my) * rw + (pad_x + bx + mx);\n"
"    out[idx] = sad8x8( pc, cw, pr, rw );\n"
"}\n";

static int gpu_cl_available( void )
{
    cl_uint np = 0, nd = 0;
    if( !gpu_cl_load() )
        return 0;
    if( clGetPlatformIDs( 1, NULL, &np ) != CL_SUCCESS || !np )
        return 0;
    cl_platform_id plats[8];
    if( clGetPlatformIDs( 8, plats, NULL ) != CL_SUCCESS )
        return 0;
    for( cl_uint p = 0; p < np && p < 8; p++ )
        if( clGetDeviceIDs( plats[p], CL_DEVICE_TYPE_GPU, 1, NULL, &nd ) == CL_SUCCESS && nd )
            return 1;
    return 0;
}

/* Pick the platform with the most GPU devices (NVIDIA first, then generic). */
static cl_device_id gpu_cl_pick_device( int device )
{
    cl_uint np = 0;
    cl_platform_id plats[8];
    if( clGetPlatformIDs( 8, plats, &np ) != CL_SUCCESS )
        return NULL;
    cl_device_id best = NULL;
    int seen = 0;
    for( cl_uint p = 0; p < np; p++ )
    {
        cl_uint nd = 0;
        if( clGetDeviceIDs( plats[p], CL_DEVICE_TYPE_GPU, 0, NULL, &nd ) != CL_SUCCESS || !nd )
            continue;
        cl_device_id devs[8];
        if( clGetDeviceIDs( plats[p], CL_DEVICE_TYPE_GPU, 8, devs, &nd ) != CL_SUCCESS )
            continue;
        for( cl_uint d = 0; d < nd; d++ )
        {
            if( seen == device )
                return devs[d];
            if( !best )
                best = devs[d];
            seen++;
        }
    }
    /* Out-of-range device index: behave like the CUDA backend and use the
     * first GPU instead of failing the backend outright. */
    if( device > 0 && best )
        fprintf( stderr, "x265 GPU ME: OpenCL device %d unavailable (%d found), using 0\n",
                 device, seen );
    return best;
}

/* Copy a real (unpadded) plane into the padded staging buffer, replicating
 * the right/bottom edge pixels into the padding (same image as the CUDA
 * backend sees). */
static int gpu_cl_upload( gpu_cl_ctx_t *c, cl_mem dst, const uint8_t *src,
                          int stride, int width, int height )
{
    if( !src || stride < width || width <= 0 || height <= 0 || width > c->w || height > c->h )
        return -1;
    /* Replicate the right/bottom edge pixels into the padding, per sample
     * so that the same code serves 8-bit and 10/12-bit luma. */
    if( c->sb == 1 )
    {
        for( int y = 0; y < height; y++ )
        {
            uint8_t *row = c->stage_ref_map + (size_t)y * c->w;
            memcpy( row, src + (size_t)y * stride, (size_t)width );
            for( int x = width; x < c->w; x++ )
                row[x] = row[width-1];
        }
        for( int y = height; y < c->h; y++ )
            memcpy( c->stage_ref_map + (size_t)y * c->w,
                    c->stage_ref_map + (size_t)(height-1) * c->w, (size_t)c->w );
    }
    else
    {
        uint16_t *stage = (uint16_t*)c->stage_ref_map;
        const uint16_t *src16 = (const uint16_t*)src;
        for( int y = 0; y < height; y++ )
        {
            uint16_t *row = stage + (size_t)y * c->w;
            memcpy( row, src16 + (size_t)y * stride, (size_t)width * 2 );
            for( int x = width; x < c->w; x++ )
                row[x] = row[width-1];
        }
        for( int y = height; y < c->h; y++ )
            memcpy( stage + (size_t)y * c->w, stage + (size_t)(height-1) * c->w,
                    (size_t)c->w * 2 );
    }
    /* Blocking write: reference uploads only happen on cache misses. */
    if( clEnqueueWriteBuffer( c->queue, dst, CL_TRUE, 0,
                              (size_t)c->w * c->h * c->sb, c->stage_ref_map,
                              0, NULL, NULL ) != CL_SUCCESS )
        return -1;
    return 0;
}

static int gpu_cl_downsample( gpu_cl_ctx_t *c, cl_mem src, cl_mem dst )
{
    if( clSetKernelArg( c->k_down, 0, sizeof(cl_mem), &src ) ||
        clSetKernelArg( c->k_down, 1, sizeof(int), &c->w ) ||
        clSetKernelArg( c->k_down, 2, sizeof(int), &c->h ) ||
        clSetKernelArg( c->k_down, 3, sizeof(cl_mem), &dst ) ||
        clSetKernelArg( c->k_down, 4, sizeof(int), &c->lw ) ||
        clSetKernelArg( c->k_down, 5, sizeof(int), &c->lh ) )
        return -1;
    {
        size_t g[2] = { (size_t)c->lw, (size_t)c->lh };
        if( clEnqueueNDRangeKernel( c->queue, c->k_down, 2, NULL, g, NULL, 0, NULL, NULL ) != CL_SUCCESS )
            return -1;
    }
    return 0;
}

static int gpu_cl_fail( gpu_cl_ctx_t *c )
{
    if( c->queue )
        clFinish( c->queue );
    c->disabled = 1;
    return -1;
}

static void gpu_cl_publish( gpu_cl_ctx_t *c, gpu_cl_frame_t *frame, int list, int nref )
{
    size_t n = (size_t)c->bw * c->bh * GK * sizeof(x265_gpu_cand_t);
    for( int r = 0; r < nref; r++ )
        memcpy( frame->cand[list][r],
                c->stage_out_map + (size_t)(list * X265_GPU_MAX_REFS + r) * n,
                n );
}

static int gpu_cl_analyze( void *ctx, gpu_cl_frame_t *frame,
                           const uint8_t *cur, int cur_stride,
                           uint8_t *const refs0[], const int64_t ref_ids0[], const int stride0[], int nref0,
                           uint8_t *const refs1[], const int64_t ref_ids1[], const int stride1[], int nref1,
                           int width, int height, int range, int radius )
{
    gpu_cl_ctx_t *c = (gpu_cl_ctx_t*)ctx;
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

    /* Testing hook: X265_GPU_SIMULATE_FAIL=opencl makes this backend fail at
     * the first analysis (used to exercise the CUDA fallback). */
    {
        static int sim = -1;
        if( sim < 0 )
        {
            const char *ev = getenv( "X265_GPU_SIMULATE_FAIL" );
            sim = ev && !strcmp( ev, "opencl" );
        }
        if( sim )
        {
            c->disabled = 1;
            return -1;
        }
    }

    frame->nfilled[0] = 0;
    frame->nfilled[1] = 0;

    int mbcount = c->bw * c->bh;
    int total = mbcount * GK;
    size_t sz = (size_t)total * sizeof(x265_gpu_cand_t);

    if( gpu_cl_upload( c, c->d_cur, cur, cur_stride, width, height ) < 0 )
        return gpu_cl_fail( c );
    if( gpu_cl_downsample( c, c->d_cur, c->d_cur_low ) < 0 )
        return gpu_cl_fail( c );

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
            cl_mem d_ref = NULL, d_ref_low = NULL;
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
                if( gpu_cl_upload( c, d_ref, refs[r], strides[r], width, height ) < 0 )
                    return gpu_cl_fail( c );
                if( gpu_cl_downsample( c, d_ref, d_ref_low ) < 0 )
                    return gpu_cl_fail( c );
                c->cache[slot].ptr = refs[r];
                c->cache[slot].id = ids[r];
            }
            if( clSetKernelArg( c->k_coarse, 0, sizeof(cl_mem), &c->d_cur_low ) ||
                clSetKernelArg( c->k_coarse, 1, sizeof(int), &c->lw ) ||
                clSetKernelArg( c->k_coarse, 2, sizeof(int), &c->lh ) ||
                clSetKernelArg( c->k_coarse, 3, sizeof(cl_mem), &d_ref_low ) ||
                clSetKernelArg( c->k_coarse, 4, sizeof(int), &c->lw ) ||
                clSetKernelArg( c->k_coarse, 5, sizeof(int), &c->lh ) ||
                clSetKernelArg( c->k_coarse, 6, sizeof(int), &c->bw ) ||
                clSetKernelArg( c->k_coarse, 7, sizeof(int), &c->bh ) ||
                clSetKernelArg( c->k_coarse, 8, sizeof(int), &range ) ||
                clSetKernelArg( c->k_coarse, 9, sizeof(cl_mem), &c->d_coarse ) )
                return gpu_cl_fail( c );
            {
                size_t g = (size_t)mbcount;
                if( clEnqueueNDRangeKernel( c->queue, c->k_coarse, 1, NULL, &g, NULL, 0, NULL, NULL ) != CL_SUCCESS )
                    return gpu_cl_fail( c );
            }
            if( clSetKernelArg( c->k_refine, 0, sizeof(cl_mem), &c->d_cur ) ||
                clSetKernelArg( c->k_refine, 1, sizeof(int), &c->w ) ||
                clSetKernelArg( c->k_refine, 2, sizeof(int), &c->h ) ||
                clSetKernelArg( c->k_refine, 3, sizeof(cl_mem), &d_ref ) ||
                clSetKernelArg( c->k_refine, 4, sizeof(int), &c->w ) ||
                clSetKernelArg( c->k_refine, 5, sizeof(int), &c->h ) ||
                clSetKernelArg( c->k_refine, 6, sizeof(int), &c->bw ) ||
                clSetKernelArg( c->k_refine, 7, sizeof(int), &c->bh ) ||
                clSetKernelArg( c->k_refine, 8, sizeof(int), &radius ) ||
                clSetKernelArg( c->k_refine, 9, sizeof(cl_mem), &c->d_coarse ) ||
                clSetKernelArg( c->k_refine, 10, sizeof(cl_mem), &c->d_out ) )
                return gpu_cl_fail( c );
            {
                size_t g = (size_t)total;
                if( clEnqueueNDRangeKernel( c->queue, c->k_refine, 1, NULL, &g, NULL, 0, NULL, NULL ) != CL_SUCCESS )
                    return gpu_cl_fail( c );
            }
            if( clEnqueueReadBuffer( c->queue, c->d_out, CL_FALSE, 0, sz,
                                     c->stage_out_map + (size_t)(list * X265_GPU_MAX_REFS + r) * sz,
                                     0, NULL, NULL ) != CL_SUCCESS )
                return gpu_cl_fail( c );
        }
        frame->nfilled[list] = nref;
    }
    if( clFinish( c->queue ) != CL_SUCCESS )
        return gpu_cl_fail( c );
    gpu_cl_publish( c, frame, 0, frame->nfilled[0] );
    gpu_cl_publish( c, frame, 1, frame->nfilled[1] );
    return 0;
}

/* --- GPU lookahead ------------------------------------------------------- */

static int gpu_cl_la_ensure( gpu_cl_ctx_t *c, cl_mem *buf, size_t *sz, size_t need )
{
    cl_int err;
    if( *buf && *sz >= need )
        return 0;
    if( *buf )
    {
        clReleaseMemObject( *buf );
        *buf = NULL;
        *sz = 0;
    }
    *buf = clCreateBuffer( c->cl_ctx, CL_MEM_READ_WRITE, need, NULL, &err );
    if( err != CL_SUCCESS )
    {
        *buf = NULL;
        return -1;
    }
    *sz = need;
    return 0;
}

static int gpu_cl_lowres_common( gpu_cl_ctx_t *c, int is_table,
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
            sim = ev && !strcmp( ev, "opencl" );
        }
        if( sim )
        {
            c->disabled = 1;
            return -1;
        }
    }
    if( gpu_cl_upload( c, c->d_cur, fenc, fenc_stride, fenc_bufw, fenc_bufh ) < 0 )
        return gpu_cl_fail( c );
    if( gpu_cl_upload( c, c->cache[0].full, ref, ref_stride, ref_bufw, ref_bufh ) < 0 )
        return gpu_cl_fail( c );
    {
        cl_mem d_out;
        size_t total_sz;
        cl_kernel kern;
        if( is_table )
        {
            int tw = 2 * rt + 1;
            total_sz = (size_t)wcu * hcu * tw * tw * sizeof(uint32_t);
            if( gpu_cl_la_ensure( c, &c->d_la_table, &c->d_la_table_sz, total_sz ) < 0 )
                return gpu_cl_fail( c );
            d_out = c->d_la_table;
            kern = c->k_la_table;
        }
        else
        {
            total_sz = (size_t)wcu * hcu * sizeof(x265_gpu_cand_t);
            if( gpu_cl_la_ensure( c, &c->d_la_best, &c->d_la_best_sz, total_sz ) < 0 )
                return gpu_cl_fail( c );
            d_out = c->d_la_best;
            kern = c->k_la_best;
        }
        {
            int a_cw = c->w, a_ch = c->h, a_rw = c->w, a_rh = c->h;
            int a_px = pad_x, a_py = pad_y, a_wcu = wcu, a_hcu = hcu, a_rt = rt;
            if( clSetKernelArg( kern, 0, sizeof(cl_mem), &c->d_cur ) ||
                clSetKernelArg( kern, 1, sizeof(int), &a_cw ) ||
                clSetKernelArg( kern, 2, sizeof(int), &a_ch ) ||
                clSetKernelArg( kern, 3, sizeof(cl_mem), &c->cache[0].full ) ||
                clSetKernelArg( kern, 4, sizeof(int), &a_rw ) ||
                clSetKernelArg( kern, 5, sizeof(int), &a_rh ) ||
                clSetKernelArg( kern, 6, sizeof(int), &a_px ) ||
                clSetKernelArg( kern, 7, sizeof(int), &a_py ) ||
                clSetKernelArg( kern, 8, sizeof(int), &a_wcu ) ||
                clSetKernelArg( kern, 9, sizeof(int), &a_hcu ) ||
                clSetKernelArg( kern, 10, sizeof(int), &a_rt ) ||
                clSetKernelArg( kern, 11, sizeof(cl_mem), &d_out ) )
                return gpu_cl_fail( c );
        }
        {
            size_t g;
            if( is_table )
            {
                int tw = 2 * rt + 1;
                g = (size_t)wcu * hcu * tw * tw;
            }
            else
                g = (size_t)wcu * hcu;
            if( clEnqueueNDRangeKernel( c->queue, kern, 1, NULL, &g, NULL, 0, NULL, NULL ) != CL_SUCCESS )
                return gpu_cl_fail( c );
        }
        if( clEnqueueReadBuffer( c->queue, d_out, CL_TRUE, 0, total_sz, out, 0, NULL, NULL ) != CL_SUCCESS )
            return gpu_cl_fail( c );
    }
    if( clFinish( c->queue ) != CL_SUCCESS )
        return gpu_cl_fail( c );
    return 0;
}

static int gpu_cl_lowres_best( void *ctx, x265_gpu_cand_t *out,
                               const uint8_t *fenc, int fenc_stride, int fenc_bufw, int fenc_bufh,
                               const uint8_t *ref, int ref_stride, int ref_bufw, int ref_bufh,
                               int pad_x, int pad_y, int wcu, int hcu, int range )
{
    return gpu_cl_lowres_common( (gpu_cl_ctx_t*)ctx, 0, fenc, fenc_stride, fenc_bufw, fenc_bufh,
                                 ref, ref_stride, ref_bufw, ref_bufh, pad_x, pad_y, wcu, hcu, range, out );
}

static int gpu_cl_lowres_table( void *ctx, uint32_t *out,
                                const uint8_t *fenc, int fenc_stride, int fenc_bufw, int fenc_bufh,
                                const uint8_t *ref, int ref_stride, int ref_bufw, int ref_bufh,
                                int pad_x, int pad_y, int wcu, int hcu, int tr )
{
    return gpu_cl_lowres_common( (gpu_cl_ctx_t*)ctx, 1, fenc, fenc_stride, fenc_bufw, fenc_bufh,
                                 ref, ref_stride, ref_bufw, ref_bufh, pad_x, pad_y, wcu, hcu, tr, out );
}

/* --- backend entry points ------------------------------------------------- */

static void *gpu_cl_create( int width, int height, int max_refs, int device,
                           int sample_bytes )
{
    if( width <= 0 || height <= 0 || (width & 31) || (height & 31) )
        return NULL;
    if( !gpu_cl_load() )
        return NULL;
    cl_device_id dev = gpu_cl_pick_device( device );
    if( !dev )
        return NULL;

    {
        char dname[128] = "", dver[64] = "";
        clGetDeviceInfo( dev, CL_DEVICE_NAME, sizeof(dname), dname, NULL );
        clGetDeviceInfo( dev, CL_DEVICE_VERSION, sizeof(dver), dver, NULL );
        snprintf( gpu_cl_name, sizeof(gpu_cl_name), "%s (%s)", dname, dver );
    }

    gpu_cl_ctx_t *c = (gpu_cl_ctx_t*)calloc( 1, sizeof(*c) );
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
    c->pix_type = sample_bytes == 2 ? "ushort" : "uchar";
    c->timing = getenv( "X265_GPU_TIMING" ) != NULL;

    cl_int err;
    c->cl_ctx = clCreateContext( NULL, 1, &dev, NULL, NULL, &err );
    if( err != CL_SUCCESS )
        goto fail;
    c->queue = clCreateCommandQueue( c->cl_ctx, dev, 0, &err );
    if( err != CL_SUCCESS )
        goto fail;
    c->program = clCreateProgramWithSource( c->cl_ctx, 1, &gpu_cl_source, NULL, &err );
    if( err != CL_SUCCESS )
        goto fail;
    /* The kernels are written for the PIX sample type so that the same
     * source serves 8-bit and 10/12-bit luma. */
    char buildopts[64];
    snprintf( buildopts, sizeof(buildopts), "-DPIX=%s", c->pix_type );
    if( clBuildProgram( c->program, 1, &dev, buildopts, NULL, NULL ) != CL_SUCCESS )
    {
        char log[4096] = "";
        clGetProgramBuildInfo( c->program, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL );
        fprintf( stderr, "x265 OpenCL ME: kernel build failed:\n%s\n", log );
        goto fail;
    }
    c->k_down = clCreateKernel( c->program, "downsample4", &err );
    if( err != CL_SUCCESS )
        goto fail;
    c->k_coarse = clCreateKernel( c->program, "coarse_search", &err );
    if( err != CL_SUCCESS )
        goto fail;
    c->k_refine = clCreateKernel( c->program, "refine_search", &err );
    if( err != CL_SUCCESS )
        goto fail;
    c->k_la_best = clCreateKernel( c->program, "lowres_best", &err );
    if( err != CL_SUCCESS )
        goto fail;
    c->k_la_table = clCreateKernel( c->program, "lowres_table", &err );
    if( err != CL_SUCCESS )
        goto fail;

    {
        size_t fsz = (size_t)width * height * sample_bytes;
        size_t lsz = (size_t)c->lw * c->lh * sample_bytes;
        size_t csz = (size_t)c->bw * c->bh * GK * sizeof(x265_gpu_cand_t);
        c->d_cur = clCreateBuffer( c->cl_ctx, CL_MEM_READ_WRITE, fsz, NULL, &err );
        if( err != CL_SUCCESS ) goto fail;
        c->d_cur_low = clCreateBuffer( c->cl_ctx, CL_MEM_READ_WRITE, lsz, NULL, &err );
        if( err != CL_SUCCESS ) goto fail;
        c->d_coarse = clCreateBuffer( c->cl_ctx, CL_MEM_READ_WRITE, csz, NULL, &err );
        if( err != CL_SUCCESS ) goto fail;
        c->d_out = clCreateBuffer( c->cl_ctx, CL_MEM_READ_WRITE, csz, NULL, &err );
        if( err != CL_SUCCESS ) goto fail;
        c->stage_ref = clCreateBuffer( c->cl_ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, fsz, NULL, &err );
        if( err != CL_SUCCESS ) goto fail;
        c->stage_out = clCreateBuffer( c->cl_ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                       2 * X265_GPU_MAX_REFS * csz, NULL, &err );
        if( err != CL_SUCCESS ) goto fail;
        c->stage_ref_map = (uint8_t*)clEnqueueMapBuffer( c->queue, c->stage_ref, CL_TRUE, CL_MAP_WRITE,
                                                         0, fsz, 0, NULL, NULL, &err );
        if( err != CL_SUCCESS ) goto fail;
        c->stage_out_map = (uint8_t*)clEnqueueMapBuffer( c->queue, c->stage_out, CL_TRUE,
                                                         CL_MAP_READ | CL_MAP_WRITE,
                                                         0, 2 * X265_GPU_MAX_REFS * csz, 0, NULL, NULL, &err );
        if( err != CL_SUCCESS ) goto fail;
        for( int i = 0; i < X265_GPU_CACHE; i++ )
        {
            c->cache[i].full = clCreateBuffer( c->cl_ctx, CL_MEM_READ_WRITE, fsz, NULL, &err );
            if( err != CL_SUCCESS ) goto fail;
            c->cache[i].low = clCreateBuffer( c->cl_ctx, CL_MEM_READ_WRITE, lsz, NULL, &err );
            if( err != CL_SUCCESS ) goto fail;
        }
    }
    if( c->timing )
        printf( "x265 OpenCL ME timing: %dx%d, device %s\n", width, height, gpu_cl_name );
    return c;

fail:
    gpu_cl_destroy( c );
    return NULL;
}

static void gpu_cl_destroy( void *ctx )
{
    gpu_cl_ctx_t *c = (gpu_cl_ctx_t*)ctx;
    if( !c )
        return;
    if( c->queue )
        clFinish( c->queue );
    if( c->stage_ref && c->queue )
        clEnqueueUnmapMemObject( c->queue, c->stage_ref, c->stage_ref_map, 0, NULL, NULL );
    if( c->stage_out && c->queue )
        clEnqueueUnmapMemObject( c->queue, c->stage_out, c->stage_out_map, 0, NULL, NULL );
    if( c->stage_ref ) clReleaseMemObject( c->stage_ref );
    if( c->stage_out ) clReleaseMemObject( c->stage_out );
    if( c->d_cur ) clReleaseMemObject( c->d_cur );
    if( c->d_cur_low ) clReleaseMemObject( c->d_cur_low );
    if( c->d_coarse ) clReleaseMemObject( c->d_coarse );
    if( c->d_out ) clReleaseMemObject( c->d_out );
    for( int i = 0; i < X265_GPU_CACHE; i++ )
    {
        if( c->cache[i].full ) clReleaseMemObject( c->cache[i].full );
        if( c->cache[i].low ) clReleaseMemObject( c->cache[i].low );
    }
    if( c->k_down ) clReleaseKernel( c->k_down );
    if( c->k_coarse ) clReleaseKernel( c->k_coarse );
    if( c->k_refine ) clReleaseKernel( c->k_refine );
    if( c->k_la_best ) clReleaseKernel( c->k_la_best );
    if( c->k_la_table ) clReleaseKernel( c->k_la_table );
    if( c->d_la_best ) clReleaseMemObject( c->d_la_best );
    if( c->d_la_table ) clReleaseMemObject( c->d_la_table );
    if( c->program ) clReleaseProgram( c->program );
    if( c->queue ) clReleaseCommandQueue( c->queue );
    if( c->cl_ctx ) clReleaseContext( c->cl_ctx );
    free( c );
}

static const char *gpu_cl_get_name( void )
{
    return gpu_cl_name;
}

static gpu_cl_frame_t *gpu_cl_frame_alloc( int bw, int bh, int nref0, int nref1 )
{
    gpu_cl_frame_t *f = (gpu_cl_frame_t*)calloc( 1, sizeof(*f) );
    if( !f )
        return NULL;
    f->hdr.backend = X265_GPU_BACKEND_OPENCL;
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
                gpu_cl_frame_free( (x265_gpu_frame_t*)f );
                return NULL;
            }
            memset( f->cand[l][r], 0, n );
        }
    return f;
}

static void gpu_cl_frame_free( x265_gpu_frame_t *frame )
{
    gpu_cl_frame_t *f = (gpu_cl_frame_t*)frame;
    if( !f )
        return;
    for( int l = 0; l < 2; l++ )
        for( int r = 0; r < f->nref[l]; r++ )
            free( f->cand[l][r] );
    free( f );
}

static const x265_gpu_cand_t *gpu_cl_frame_get( const x265_gpu_frame_t *frame,
                                                int list, const void *ref_plane, int64_t ref_id,
                                                int bx, int by, int *count )
{
    const gpu_cl_frame_t *f = (const gpu_cl_frame_t*)frame;
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

static int gpu_cl_frame_covers( const x265_gpu_frame_t *frame,
                                int list, const void *ref_plane, int64_t ref_id )
{
    const gpu_cl_frame_t *f = (const gpu_cl_frame_t*)frame;
    if( !f || list < 0 || list > 1 )
        return 0;
    for( int r = 0; r < f->nfilled[list]; r++ )
        if( f->slot_ptr[list][r] == ref_plane && f->slot_id[list][r] == ref_id )
            return 1;
    return 0;
}

const x265_gpu_backend_t x265_gpu_backend_cl =
{
    X265_GPU_BACKEND_OPENCL,
    "opencl",
    gpu_cl_available,
    gpu_cl_create,
    gpu_cl_destroy,
    gpu_cl_get_name,
    gpu_cl_frame_alloc,
    gpu_cl_frame_free,
    gpu_cl_analyze,
    gpu_cl_frame_get,
    gpu_cl_frame_covers,
    gpu_cl_lowres_best,
    gpu_cl_lowres_table,
};
