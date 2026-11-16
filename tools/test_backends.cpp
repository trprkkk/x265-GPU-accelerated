/* Standalone backend comparison harness (not shipped in the library).
 * Feeds identical cur/ref planes to the CPU and OpenCL backends and
 * compares every candidate bit-for-bit.  Usage: test_backends [range radius]
 */
#include "common/gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t g_rng = 123456789;
static uint8_t rnd8( void )
{
    g_rng = g_rng * 1664525u + 1013904223u;
    return (uint8_t)(g_rng >> 24);
}

int main( int argc, char **argv )
{
    int range = argc > 1 ? atoi( argv[1] ) : 64;
    int radius = argc > 2 ? atoi( argv[2] ) : 2;
    const int W = 1280, H = 720;
    const int PW = 1280, PH = 736;   /* padded: (W+31)/32*32 = 1280, h -> 736 */

    uint8_t *cur = (uint8_t*)malloc( (size_t)W * H );
    uint8_t *ref = (uint8_t*)malloc( (size_t)W * H );
    for( int y = 0; y < H; y++ )
        for( int x = 0; x < W; x++ )
        {
            int v = rnd8();
            if( (x / 64 + y / 64) % 3 )
                v = (x * 31 + y * 17 + (x ^ y)) & 255;
            cur[y * W + x] = (uint8_t)v;
            ref[y * W + x] = (uint8_t)((v + 7) & 255);
        }
    /* a textured box so that some blocks have non-trivial matches */
    for( int y = 200; y < 280; y++ )
        for( int x = 300; x < 380; x++ )
        {
            cur[y * W + x] = (uint8_t)(x * 3 + y * 5);
            ref[(y + 4) * W + x] = (uint8_t)(x * 3 + (y + 4) * 5);
        }

    int bw = (PW + 31) / 32, bh = (PH + 31) / 32;
    int backends[3] = { X265_GPU_BACKEND_CPU, X265_GPU_BACKEND_CUDA, X265_GPU_BACKEND_OPENCL };
    const char *names[3] = { "cpu", "cuda", "opencl" };
    x265_gpu_frame_t *frames[3];
    void *ctxs[3];
    for( int b = 0; b < 3; b++ )
    {
        ctxs[b] = x265_gpu_create( PW, PH, 2, 0, backends[b] );
        if( !ctxs[b] )
        {
            printf( "%s: unavailable\n", names[b] );
            frames[b] = NULL;
            continue;
        }
        frames[b] = x265_gpu_frame_alloc( ctxs[b], bw, bh, 1, 0 );
        uint8_t *refs0[1] = { ref };
        const int64_t ids0[1] = { 42 };
        int st0[1] = { W };
        if( x265_gpu_analyze( ctxs[b], frames[b], cur, W, refs0, ids0, st0, 1, NULL, NULL, NULL, 0, W, H, range, radius ) != 0 )
        {
            printf( "%s: analyze failed\n", names[b] );
            return 1;
        }
    }
    if( !frames[0] || !frames[1] || !frames[2] )
    {
        printf( "SKIP: not all backends available\n" );
        return 2;
    }

    int ndiff = 0, nblocks = 0;
    for( int by = 0; by < bh; by++ )
        for( int bx = 0; bx < bw; bx++ )
        {
            nblocks++;
            int n[3];
            const x265_gpu_cand_t *c[3];
            for( int b = 0; b < 3; b++ )
                c[b] = x265_gpu_frame_get( frames[b], 0, ref, 42, bx, by, &n[b] );
            if( !c[0] || !c[1] || !c[2] )
            {
                printf( "block %d,%d: missing candidates (cpu=%p cl=%p cu=%p)\n", bx, by, c[0], c[1], c[2] );
                return 1;
            }
            int bad = n[1] != n[0] || n[2] != n[0];
            if( !bad )
                for( int i = 0; i < n[0]; i++ )
                    if( memcmp( &c[0][i], &c[1][i], sizeof( x265_gpu_cand_t ) ) ||
                        memcmp( &c[0][i], &c[2][i], sizeof( x265_gpu_cand_t ) ) )
                    {
                        bad = 1;
                        break;
                    }
            if( bad )
            {
                if( ndiff < 8 )
                {
                    printf( "block %d,%d DIFF (n=%d):\n  cpu:", bx, by, n[0] );
                    for( int i = 0; i < n[0]; i++ ) printf( " %d,%d|%u", c[0][i].mv[0], c[0][i].mv[1], c[0][i].sad );
                    printf( "\n  cu: ");
                    for( int i = 0; i < n[1]; i++ ) printf( " %d,%d|%u", c[1][i].mv[0], c[1][i].mv[1], c[1][i].sad );
                    printf( "\n  cl: ");
                    for( int i = 0; i < n[2]; i++ ) printf( " %d,%d|%u", c[2][i].mv[0], c[2][i].mv[1], c[2][i].sad );
                    printf( "\n" );
                }
                ndiff++;
            }
        }
    printf( "%s: %d blocks, %d differing\n", "done", nblocks, ndiff );
    return ndiff ? 1 : 0;
}
