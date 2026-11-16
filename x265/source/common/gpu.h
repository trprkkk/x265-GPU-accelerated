/*****************************************************************************
 * gpu.h: GPU candidate generation for the hybrid x265 fork
 *****************************************************************************
 * This module never makes the final coding decision.  It returns a small
 * list of integer-pel motion vectors per 32x32 block; the unmodified x265
 * rate-distortion code (Search::singleMotionEstimation /
 * MotionEstimate::motionEstimate) evaluates them against the real
 * reconstructed reference and remains the only authority on the output
 * bitstream.
 *
 * common/gpu.c is the dispatcher for the public API below; the actual
 * implementations live in common/gpu_cuda.cpp + common/gpu_kernels.cu (CUDA),
 * common/gpu_cl.c (OpenCL) and common/gpu_cpu.cpp (scalar CPU reference,
 * bit-identical algorithm, used for testing and as a deterministic
 * fallback).  Selected by x265_gpu_create()'s backend argument.
 *
 * The analysis is synchronous per frame: x265_gpu_analyze() runs the
 * coarse-to-fine search on the GPU and returns only when the candidates
 * are published.  There is deliberately no asynchronous prefetch in this
 * version, so the output is deterministic by construction.
 *
 * Frame dimensions passed to x265_gpu_create() are padded up to the
 * 32x32 block grid by the caller.  x265_gpu_analyze() receives the real
 * (unpadded) width/height and replicates the right/bottom edge pixels
 * into the padding, so all backends see exactly the same image.
 *****************************************************************************/

#ifndef X265_GPU_H
#define X265_GPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define X265_GPU_MAX_REFS  4
#define X265_GPU_CANDS     4
#define X265_GPU_CACHE     8
#define X265_GPU_BLOCK     32
/* Bytes per luma sample the backends support.  The analyse/search kernels are
 * compiled for both sizes and selected from the value passed to
 * x265_gpu_create(), so 8-bit and 10/12-bit (uint16_t pixel) builds share the
 * same backend code. */
#define X265_GPU_SB_MIN    1
#define X265_GPU_SB_MAX    2
/* GPU lookahead (see x265/source/encoder/slicetype.cpp): the lowres
 * planes are searched on a grid of 8x8 blocks (X265_LOWRES_CU_SIZE).
 * X265_GPU_LA_BEST_RANGE is the integer-pel search radius (lowres
 * pixels) used by the fast mode (--gpu-lookahead 1): a coarse grid
 * (step 4) plus a +/-4 refine around the coarse winner, computed
 * identically by every backend.  The stock lowres search range is
 * s_merange == 16.  X265_GPU_LA_TABLE_RADIUS is
 * the half-width of the per-block SAD table built by the bitperfect
 * mode (--gpu-lookahead 2): every MV in [-R,R]x[-R,R] gets one entry.
 * MVs outside the table fall back to the exact CPU SAD in the caller,
 * so the table size never affects correctness, only speed. */
#define X265_GPU_LA_BEST_RANGE    24
#define X265_GPU_LA_TABLE_RADIUS  24
/* Upper bound on the mode-2 SAD table (bytes per reference).  The table
 * holds one uint32 per 8x8 block per MV in [-(R),R]^2, so it grows with
 * the frame area (720p/R=24: 35 MB, 1080p: 78 MB, 4K: 311 MB) and the
 * lookahead can have several estimates in flight.  Above this limit the
 * lookahead keeps the stock CPU search instead of allocating the table. */
#define X265_GPU_LA_TABLE_MAX_BYTES (128u * 1024u * 1024u)

/* Backend selection for x265_gpu_create(). */
#define X265_GPU_BACKEND_AUTO   0   /* CUDA first, then OpenCL */
#define X265_GPU_BACKEND_CUDA   1
#define X265_GPU_BACKEND_OPENCL 2
#define X265_GPU_BACKEND_CPU    3   /* scalar reference (explicit only) */

typedef struct
{
    int16_t  mv[2];   /* full-pel displacement, relative to the block origin */
    uint32_t sad;     /* integer SAD that produced the candidate */
} x265_gpu_cand_t;

typedef struct x265_gpu_frame_t x265_gpu_frame_t;

/* Every backend's frame object starts with this header so that the
 * dispatcher can find the owning backend from an opaque frame pointer. */
typedef struct
{
    int backend;      /* X265_GPU_BACKEND_* */
} x265_gpu_frame_header_t;

typedef struct x265_gpu_backend_t
{
    int id;              /* X265_GPU_BACKEND_* */
    const char *name_id; /* "cuda" / "opencl" / "cpu" */
    int      (*available)( void );
    void *   (*create)( int width, int height, int max_refs, int device, int sample_bytes );
    void     (*destroy)( void *ctx );
    const char *(*name)( void );
    x265_gpu_frame_t *(*frame_alloc)( int bw, int bh, int nref0, int nref1 );
    void     (*frame_free)( x265_gpu_frame_t *frame );
    int      (*analyze)( void *ctx, x265_gpu_frame_t *frame,
                         const uint8_t *cur, int cur_stride,
                         uint8_t *const refs0[], const int64_t ref_ids0[], const int stride0[], int nref0,
                         uint8_t *const refs1[], const int64_t ref_ids1[], const int stride1[], int nref1,
                         int width, int height, int range, int radius );
    const x265_gpu_cand_t *(*frame_get)( const x265_gpu_frame_t *frame,
                                         int list, const void *ref_plane, int64_t ref_id,
                                         int bx, int by, int *count );
    int      (*frame_covers)( const x265_gpu_frame_t *frame,
                              int list, const void *ref_plane, int64_t ref_id );
    /* GPU lookahead primitives.  All planes are lowres luma planes passed
     * together with the full allocation rectangle (bufW x bufH, stride in
     * samples) and the (padX,padY) offset of the visible (0,0) pixel
     * inside it, so the margin pixels the CPU search reads are the ones
     * the GPU sees.  wcu/hcu is the 8x8 block grid; the visible plane is
     * wcu*8 x hcu*8.  fenc/ref must have the same geometry.
     * lowres_best: exhaustive search for the single best integer MV per
     *   block (min SAD, ties keep the earliest in (my,mx) scan order),
     *   written to out[cuXY] (wcu*hcu entries).  Caller clips to its own
     *   mvmin/mvmax again.
     * lowres_table: exact 8x8 SAD for every MV in [-tr,tr]x[-tr,tr] per
     *   block, written to out[cuXY*tw*tw + (my+tr)*tw + (mx+tr)].
     * Both return 0 on success, <0 on failure (caller falls back to the
     * stock CPU lookahead search). */
    int      (*lowres_best)( void *ctx, x265_gpu_cand_t *out,
                             const uint8_t *fenc, int fenc_stride, int fenc_bufw, int fenc_bufh,
                             const uint8_t *ref, int ref_stride, int ref_bufw, int ref_bufh,
                             int pad_x, int pad_y, int wcu, int hcu, int range );
    int      (*lowres_table)( void *ctx, uint32_t *out,
                              const uint8_t *fenc, int fenc_stride, int fenc_bufw, int fenc_bufh,
                              const uint8_t *ref, int ref_stride, int ref_bufw, int ref_bufh,
                              int pad_x, int pad_y, int wcu, int hcu, int tr );
} x265_gpu_backend_t;

/* Backend instances (defined in gpu_cuda.cpp / gpu_cl.c / gpu_cpu.cpp). */
extern const x265_gpu_backend_t x265_gpu_backend_cuda;
extern const x265_gpu_backend_t x265_gpu_backend_cl;
extern const x265_gpu_backend_t x265_gpu_backend_cpu;

/* Create a candidate generator on the requested backend (0 = auto).
 * Returns NULL when the selected backend is unavailable or init fails.
 * width/height are the padded frame dimensions (multiples of 32) and
 * sample_bytes is sizeof(pixel) of the caller (1 for 8-bit, 2 for 10/12-bit);
 * other values are rejected. */
void *x265_gpu_create( int width, int height, int max_refs, int device, int backend,
                       int sample_bytes );
void  x265_gpu_destroy( void *ctx );
/* After an analysis failure, try to continue on the other backend (retires
 * the failing backend; at most one switch).  Returns 0 on success.  The
 * caller must have released frames it still holds before switching. */
int   x265_gpu_fallback( void *ctx );
/* Human-readable name of the active backend/device ("" when none). */
const char *x265_gpu_name( void *ctx );
/* Bytes per luma sample the context was created for. */
int x265_gpu_sample_bytes( void *ctx );
/* Backend id of a completed create() (for logging). */
const char *x265_gpu_backend_id( void *ctx );

x265_gpu_frame_t *x265_gpu_frame_alloc( void *ctx, int bw, int bh, int nref0, int nref1 );
void              x265_gpu_frame_free( x265_gpu_frame_t *frame );

/* Upload the current picture and up to X265_GPU_MAX_REFS references per
 * list, run the coarse-to-fine search, and fill the frame's candidate
 * arrays.  width/height are the real (unpadded) picture dimensions;
 * ref_ids are opaque keys (identifying the reference content together
 * with the plane pointer) used to cache reference planes on the device.
 * Returns 0 on success. */
int x265_gpu_analyze( void *ctx, x265_gpu_frame_t *frame,
                      const uint8_t *cur, int cur_stride,
                      uint8_t *const refs0[], const int64_t ref_ids0[], const int stride0[], int nref0,
                      uint8_t *const refs1[], const int64_t ref_ids1[], const int stride1[], int nref1,
                      int width, int height, int range, int radius );

/* Candidate lookup used by search.cpp.  The reference is identified by the
 * reference frame's luma plane pointer and its POC, so candidates remain
 * usable even when the analysis ran for a predicted reference set.
 * Returns NULL when this list/reference/block has no candidates (unsupported
 * size, reference not analysed, ...). */
const x265_gpu_cand_t *x265_gpu_frame_get( const x265_gpu_frame_t *frame,
                                           int list, const void *ref_plane, int64_t ref_id,
                                           int bx, int by, int *count );

/* 1 when a candidate slot for (list, ref_plane, ref_id) exists. */
int x265_gpu_frame_covers( const x265_gpu_frame_t *frame,
                           int list, const void *ref_plane, int64_t ref_id );

/* GPU lookahead (dispatcher wrappers, see backend vtable above). */
int x265_gpu_lowres_best( void *ctx, x265_gpu_cand_t *out,
                          const uint8_t *fenc, int fenc_stride, int fenc_bufw, int fenc_bufh,
                          const uint8_t *ref, int ref_stride, int ref_bufw, int ref_bufh,
                          int pad_x, int pad_y, int wcu, int hcu, int range );
int x265_gpu_lowres_table( void *ctx, uint32_t *out,
                           const uint8_t *fenc, int fenc_stride, int fenc_bufw, int fenc_bufh,
                           const uint8_t *ref, int ref_stride, int ref_bufw, int ref_bufh,
                           int pad_x, int pad_y, int wcu, int hcu, int tr );

#ifdef __cplusplus
}
#endif

#endif
