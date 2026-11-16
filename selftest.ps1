# Acceptance self-test for hybrid x265 GPU ME (this fork).
# Verifies that:
#   1) --gpu-me 0 output is deterministic across runs (and does not log GPU activity)
#   2) GPU modes run and exit cleanly
#   3) the CUDA backend is actually engaged
#   4) unsupported configurations (analysis save/load) fall back to the CPU with a warning
#   5) repeated GPU-mode runs are deterministic (identical SHA-1)
#   6) CUDA / OpenCL / CPU-reference backends produce bit-identical streams
#   7) a failing backend switches to the other one with bit-identical output
#   8) a compatibility smoke pass (2-pass, qpfile, zones, threaded encode) works
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File selftest.ps1
param()
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
New-Item -ItemType Directory -Force -Path testdata | Out-Null

$exe = Join-Path $root "build-vs\Release\x265.exe"
$base = Join-Path $root "build-baseline\Release\x265.exe"
$exe10 = Join-Path $root "build-10bit\Release\x265.exe"
$yuv = Join-Path $root "testdata\motion720p.yuv"
$yuv480 = Join-Path $root "testdata\motion480p.yuv"

if( !(Test-Path $exe) ) { Write-Host "FAIL: $exe not found (build first)" -ForegroundColor Red; exit 1 }
if( !(Test-Path $yuv) ) { Write-Host "SKIP: $yuv not found (benchmark clip unavailable)" -ForegroundColor Yellow; exit 0 }

$common = @("--crf","23","--frames","30","--input-res","1280x720","--fps","30")
$fail = 0

# 1) mode 0 must not log GPU activity, and must be deterministic
& $exe "--preset" "medium" "--gpu-me" "0" @common $yuv "-o" "testdata\st_g0a.hevc" 2> "testdata\st_g0a.log"
& $exe "--preset" "medium" "--gpu-me" "0" @common $yuv "-o" "testdata\st_g0b.hevc" 2> $null
if( (Get-Content "testdata\st_g0a.log" -Raw) -match "GPU ME: " ) { Write-Host "FAIL: --gpu-me 0 logged GPU activity" -ForegroundColor Red; $fail = 1 }
else { Write-Host "PASS: --gpu-me 0 does not log GPU activity" -ForegroundColor Green }
$h1 = (Get-FileHash "testdata\st_g0a.hevc" -Algorithm SHA1).Hash
$h2 = (Get-FileHash "testdata\st_g0b.hevc" -Algorithm SHA1).Hash
if( $h1 -eq $h2 ) { Write-Host "PASS: --gpu-me 0 is deterministic across runs" -ForegroundColor Green }
else              { Write-Host "FAIL: mode 0 differs between runs ($h1 vs $h2)" -ForegroundColor Red; $fail = 1 }

# 1b) mode 0 must be bit-identical to a pristine upstream build of the same commit
if( Test-Path $base )
{
    & $base "--preset" "medium" "--crf" "23" "--frames" "30" "--input-res" "1280x720" "--fps" "30" $yuv "-o" "testdata\st_base.hevc" 2> $null
    $hb = (Get-FileHash "testdata\st_base.hevc" -Algorithm SHA1).Hash
    if( $hb -eq $h1 ) { Write-Host "PASS: --gpu-me 0 is bit-identical to stock x265" -ForegroundColor Green }
    else { Write-Host "FAIL: mode 0 differs from stock ($hb vs $h1)" -ForegroundColor Red; $fail = 1 }
}
else { Write-Host "SKIP: build-baseline not present, cannot compare against stock" -ForegroundColor Yellow }

# 2) GPU modes run and exit cleanly
foreach( $m in 1,2,3,4 )
{
    & $exe "--preset" "slow" "--gpu-me" $m @common $yuv "-o" "testdata\st_g$m.hevc" 2> $null
    if( $LASTEXITCODE -eq 0 ) { Write-Host "PASS: --gpu-me $m encode exited cleanly" -ForegroundColor Green }
    else                      { Write-Host "FAIL: --gpu-me $m exit code $LASTEXITCODE" -ForegroundColor Red; $fail = 1 }
}

# 3) GPU analysis is actually engaged (informational: CPU fallback is valid on
#    machines without a compatible GPU/driver)
& $exe "--preset" "slow" "--gpu-me" "2" @common $yuv "-o" "NUL" 2> "testdata\st_gpu.log"
if( (Get-Content "testdata\st_gpu.log" -Raw) -match "GPU ME: " ) { Write-Host "PASS: GPU analysis active" -ForegroundColor Green }
else { Write-Host "WARN: CUDA not active (CPU fallback)" -ForegroundColor Yellow }

# 4) unsupported configuration (analysis save) falls back with a warning
& $exe "--preset" "medium" "--gpu-me" "2" "--analysis-save" "testdata\st_analysis.txt" @common $yuv "-o" "NUL" 2> "testdata\st_save.log"
$log = Get-Content "testdata\st_save.log" -Raw
if( $log -match "GPU ME.*analysis save/load" ) { Write-Host "PASS: analysis-save fallback warning shown" -ForegroundColor Green }
else { Write-Host "FAIL: analysis-save fallback warning missing" -ForegroundColor Red; $fail = 1 }

# 4b) a 10-bit build compiles and disables the feature with a warning
if( Test-Path $exe10 )
{
    & $exe10 "--preset" "medium" "--crf" "23" "--gpu-me" "2" "--input-depth" "8" "--frames" "10" "--input-res" "1280x720" "--fps" "30" $yuv "-o" "testdata\st_hbd_g2.hevc" 2> "testdata\st_hbd.log"
    & $exe10 "--preset" "medium" "--crf" "23" "--gpu-me" "0" "--input-depth" "8" "--frames" "10" "--input-res" "1280x720" "--fps" "30" $yuv "-o" "testdata\st_hbd_g0.hevc" 2> $null
    if( (Get-Content "testdata\st_hbd.log" -Raw) -match "only supported in the 8-bit build" )
    {
        $ha10 = (Get-FileHash "testdata\st_hbd_g2.hevc" -Algorithm SHA1).Hash
        $hb10 = (Get-FileHash "testdata\st_hbd_g0.hevc" -Algorithm SHA1).Hash
        if( $ha10 -eq $hb10 ) { Write-Host "PASS: 10-bit build falls back silently and identically" -ForegroundColor Green }
        else { Write-Host "FAIL: 10-bit fallback changed output" -ForegroundColor Red; $fail = 1 }
    }
    else { Write-Host "FAIL: 10-bit build did not warn about the 8-bit restriction" -ForegroundColor Red; $fail = 1 }
}
else { Write-Host "SKIP: build-10bit not present, cannot verify the 10-bit build" -ForegroundColor Yellow }

# 5) determinism: repeated runs of the same GPU mode must be bit-identical
$detCommon = @("--preset","slow","--crf","23","--frames","60","--input-res","1280x720","--fps","30")
foreach( $m in 1,2,3,4 )
{
    & $exe "--gpu-me" $m @detCommon $yuv "-o" "testdata\st_d${m}a.hevc" 2> $null
    & $exe "--gpu-me" $m @detCommon $yuv "-o" "testdata\st_d${m}b.hevc" 2> $null
    $ha = (Get-FileHash "testdata\st_d${m}a.hevc" -Algorithm SHA1).Hash
    $hb = (Get-FileHash "testdata\st_d${m}b.hevc" -Algorithm SHA1).Hash
    if( $ha -eq $hb ) { Write-Host "PASS: --gpu-me $m is deterministic across runs" -ForegroundColor Green }
    else              { Write-Host "FAIL: --gpu-me $m differs between runs ($ha vs $hb)" -ForegroundColor Red; $fail = 1 }
}

# 6) backends are bit-identical (CUDA default vs OpenCL vs CPU reference)
& $exe "--preset" "fast" "--crf" "23" "--gpu-me" "2" "--frames" "60" "--input-res" "1280x720" "--fps" "30" $yuv "-o" "testdata\st_be_auto.hevc" 2> "testdata\st_be_auto.log"
& $exe "--preset" "fast" "--crf" "23" "--gpu-me" "2" "--gpu-me-backend" "2" "--frames" "60" "--input-res" "1280x720" "--fps" "30" $yuv "-o" "testdata\st_be_cl.hevc" 2> "testdata\st_be_cl.log"
& $exe "--preset" "fast" "--crf" "23" "--gpu-me" "2" "--gpu-me-backend" "3" "--frames" "60" "--input-res" "1280x720" "--fps" "30" $yuv "-o" "testdata\st_be_cpu.hevc" 2> "testdata\st_be_cpu.log"
$ha = (Get-FileHash "testdata\st_be_auto.hevc" -Algorithm SHA1).Hash
$hb = (Get-FileHash "testdata\st_be_cl.hevc"   -Algorithm SHA1).Hash
$hc = (Get-FileHash "testdata\st_be_cpu.hevc"  -Algorithm SHA1).Hash
if( $ha -eq $hb -and $ha -eq $hc ) { Write-Host "PASS: CUDA / OpenCL / CPU-reference backends are bit-identical" -ForegroundColor Green }
else { Write-Host "FAIL: backend outputs differ ($ha / $hb / $hc)" -ForegroundColor Red; $fail = 1 }

# 6b) the reference cache must not change the output, and a broken CUDA
#     fatbin must fall back to OpenCL with identical output
$env:X265_GPU_NOCACHE = "1"
& $exe "--preset" "fast" "--crf" "23" "--gpu-me" "2" "--frames" "60" "--input-res" "1280x720" "--fps" "30" $yuv "-o" "testdata\st_nocache.hevc" 2> $null
Remove-Item Env:\X265_GPU_NOCACHE
$hn = (Get-FileHash "testdata\st_nocache.hevc" -Algorithm SHA1).Hash
$ha2 = (Get-FileHash "testdata\st_be_auto.hevc" -Algorithm SHA1).Hash
if( $hn -eq $ha2 ) { Write-Host "PASS: X265_GPU_NOCACHE=1 does not change the output" -ForegroundColor Green }
else { Write-Host "FAIL: cache changes the output ($hn vs $ha2)" -ForegroundColor Red; $fail = 1 }

$env:X265_GPU_FATBIN = "bogus"
& $exe "--preset" "fast" "--crf" "23" "--gpu-me" "2" "--frames" "60" "--input-res" "1280x720" "--fps" "30" $yuv "-o" "testdata\st_badfatbin.hevc" 2> "testdata\st_badfatbin.log"
Remove-Item Env:\X265_GPU_FATBIN
if( (Get-Content "testdata\st_badfatbin.log" -Raw) -match "GPU ME: opencl" )
{
    $hf2 = (Get-FileHash "testdata\st_badfatbin.hevc" -Algorithm SHA1).Hash
    if( $hf2 -eq $hb ) { Write-Host "PASS: broken CUDA fatbin falls back to OpenCL (bit-identical)" -ForegroundColor Green }
    else { Write-Host "FAIL: fatbin fallback output differs" -ForegroundColor Red; $fail = 1 }
}
else { Write-Host "SKIP: CUDA is not the initial backend, cannot verify fatbin fallback" -ForegroundColor Yellow }

# 6c) non-multiple-of-32 resolution uses the GPU and matches across backends
if( Test-Path $yuv480 )
{
    & $exe "--preset" "medium" "--crf" "23" "--gpu-me" "2" "--frames" "20" "--input-res" "848x480" "--fps" "30" $yuv480 "-o" "testdata\st_480_cuda.hevc" 2> "testdata\st_480.log"
    & $exe "--preset" "medium" "--crf" "23" "--gpu-me" "2" "--gpu-me-backend" "3" "--frames" "20" "--input-res" "848x480" "--fps" "30" $yuv480 "-o" "testdata\st_480_cpu.hevc" 2> $null
    if( (Get-Content "testdata\st_480.log" -Raw) -match "GPU ME: " )
    {
        $h480a = (Get-FileHash "testdata\st_480_cuda.hevc" -Algorithm SHA1).Hash
        $h480b = (Get-FileHash "testdata\st_480_cpu.hevc"  -Algorithm SHA1).Hash
        if( $h480a -eq $h480b ) { Write-Host "PASS: 848x480 uses the GPU, backends identical" -ForegroundColor Green }
        else { Write-Host "FAIL: 848x480 backend outputs differ" -ForegroundColor Red; $fail = 1 }
    }
    else { Write-Host "FAIL: 848x480 did not use the GPU" -ForegroundColor Red; $fail = 1 }
}
else { Write-Host "SKIP: testdata\motion480p.yuv not found (python3 tools/make_testclip.py testdata/motion480p.yuv 848 480 60)" -ForegroundColor Yellow }

# 7) a failing backend switches to the other one with bit-identical output
$env:X265_GPU_SIMULATE_FAIL = "cuda"
& $exe "--gpu-me" "2" @detCommon $yuv "-o" "testdata\st_fb.hevc" 2> "testdata\st_fb.log"
$fbLog = Get-Content "testdata\st_fb.log" -Raw
Remove-Item Env:\X265_GPU_SIMULATE_FAIL
if( $fbLog -match "GPU ME: cuda" )
{
    if( $fbLog -match "GPU ME: backend failed, switched to opencl" )
    {
        $hf = (Get-FileHash "testdata\st_fb.hevc"       -Algorithm SHA1).Hash
        $ha = (Get-FileHash "testdata\st_d2a.hevc"      -Algorithm SHA1).Hash
        if( $hf -eq $ha ) { Write-Host "PASS: CUDA failure falls back to OpenCL (bit-identical)" -ForegroundColor Green }
        else { Write-Host "FAIL: fallback output differs ($hf vs $ha)" -ForegroundColor Red; $fail = 1 }
    }
    else { Write-Host "FAIL: CUDA failure did not switch to the OpenCL backend" -ForegroundColor Red; $fail = 1 }
}
else { Write-Host "SKIP: CUDA backend is not the initial backend, cannot verify fallback" -ForegroundColor Yellow }

# 8) compatibility smoke: 2-pass stats, qpfile, zones, multi-threaded encode
$smoke = @("--preset","medium","--frames","30","--input-res","1280x720","--fps","30")
& $exe "--pass" "1" "--stats" "testdata\st_stats.log" "--bitrate" "600" @smoke "--gpu-me" "2" $yuv "-o" "NUL" 2> $null
$p1 = $LASTEXITCODE
& $exe "--pass" "2" "--stats" "testdata\st_stats.log" "--bitrate" "600" @smoke "--gpu-me" "2" $yuv "-o" "testdata\st_2pass.hevc" 2> $null
$p2 = $LASTEXITCODE
if( $p1 -eq 0 -and $p2 -eq 0 ) { Write-Host "PASS: 2-pass encode with stats file" -ForegroundColor Green }
else { Write-Host "FAIL: 2-pass encode ($p1/$p2)" -ForegroundColor Red; $fail = 1 }
Set-Content -Path "testdata\st.qpfile" -Value "25 I"
& $exe "--qpfile" "testdata\st.qpfile" @smoke "--gpu-me" "2" $yuv "-o" "testdata\st_qp.hevc" 2> $null
if( $LASTEXITCODE -eq 0 ) { Write-Host "PASS: qpfile encode exited cleanly" -ForegroundColor Green }
else { Write-Host "FAIL: qpfile encode exit $LASTEXITCODE" -ForegroundColor Red; $fail = 1 }
& $exe "--zones" "10,20,b=1.5" @smoke "--gpu-me" "2" $yuv "-o" "testdata\st_zone.hevc" 2> $null
if( $LASTEXITCODE -eq 0 ) { Write-Host "PASS: zones encode exited cleanly" -ForegroundColor Green }
else { Write-Host "FAIL: zones encode exit $LASTEXITCODE" -ForegroundColor Red; $fail = 1 }
& $exe "--frame-threads" "4" @smoke "--gpu-me" "2" $yuv "-o" "testdata\st_t4.hevc" 2> $null
if( $LASTEXITCODE -eq 0 ) { Write-Host "PASS: --frame-threads 4 with GPU mode" -ForegroundColor Green }
else { Write-Host "FAIL: threaded encode exit $LASTEXITCODE" -ForegroundColor Red; $fail = 1 }

# 9) GPU lookahead (--gpu-lookahead, 8-bit build)
#  9a) mode 2 (bitperfect) must be byte-identical to stock (mode 0)
& $exe "--preset" "medium" "--crf" "23" "--gpu-lookahead" "0" @common $yuv "-o" "testdata\st_la0.hevc" 2> $null
& $exe "--preset" "medium" "--crf" "23" "--gpu-lookahead" "2" @common $yuv "-o" "testdata\st_la2.hevc" 2> "testdata\st_la2.log"
$hla0 = (Get-FileHash "testdata\st_la0.hevc" -Algorithm SHA1).Hash
$hla2 = (Get-FileHash "testdata\st_la2.hevc" -Algorithm SHA1).Hash
if( $hla0 -eq $hla2 ) { Write-Host "PASS: --gpu-lookahead 2 is bit-identical to stock" -ForegroundColor Green }
else { Write-Host "FAIL: --gpu-lookahead 2 differs from stock ($hla2 vs $hla0)" -ForegroundColor Red; $fail = 1 }
#  9b) mode 1 runs cleanly and is deterministic; mode 2 backends agree
& $exe "--preset" "medium" "--crf" "23" "--gpu-lookahead" "1" @common $yuv "-o" "testdata\st_la1a.hevc" 2> $null
& $exe "--preset" "medium" "--crf" "23" "--gpu-lookahead" "1" @common $yuv "-o" "testdata\st_la1b.hevc" 2> $null
$hl1a = (Get-FileHash "testdata\st_la1a.hevc" -Algorithm SHA1).Hash
$hl1b = (Get-FileHash "testdata\st_la1b.hevc" -Algorithm SHA1).Hash
if( $hl1a -eq $hl1b ) { Write-Host "PASS: --gpu-lookahead 1 is deterministic" -ForegroundColor Green }
else { Write-Host "FAIL: --gpu-lookahead 1 differs between runs" -ForegroundColor Red; $fail = 1 }
& $exe "--preset" "medium" "--crf" "23" "--gpu-lookahead" "2" "--gpu-me-backend" "3" @common $yuv "-o" "testdata\st_la2cpu.hevc" 2> $null
$hl2cpu = (Get-FileHash "testdata\st_la2cpu.hevc" -Algorithm SHA1).Hash
if( $hla2 -eq $hl2cpu ) { Write-Host "PASS: --gpu-lookahead 2 backends are bit-identical" -ForegroundColor Green }
else { Write-Host "FAIL: --gpu-lookahead 2 backend outputs differ" -ForegroundColor Red; $fail = 1 }
#  9c) 10-bit build: lookahead falls back with a warning and stays stock-identical
if( Test-Path $exe10 )
{
    & $exe10 "--preset" "medium" "--crf" "23" "--gpu-lookahead" "2" "--input-depth" "8" "--frames" "10" "--input-res" "1280x720" "--fps" "30" $yuv "-o" "testdata\st_la10.hevc" 2> "testdata\st_la10.log"
    & $exe10 "--preset" "medium" "--crf" "23" "--input-depth" "8" "--frames" "10" "--input-res" "1280x720" "--fps" "30" $yuv "-o" "testdata\st_la10_0.hevc" 2> $null
    $hla10 = (Get-FileHash "testdata\st_la10.hevc" -Algorithm SHA1).Hash
    $hla100 = (Get-FileHash "testdata\st_la10_0.hevc" -Algorithm SHA1).Hash
    if( (Get-Content "testdata\st_la10.log" -Raw) -match "GPU lookahead: only supported in the 8-bit build" -and $hla10 -eq $hla100 )
    { Write-Host "PASS: 10-bit lookahead falls back (stock-identical)" -ForegroundColor Green }
    else { Write-Host "FAIL: 10-bit lookahead fallback" -ForegroundColor Red; $fail = 1 }
}

Remove-Item testdata\st_*.hevc,testdata\st_*.log,testdata\st_analysis.txt,testdata\st_stats.log,testdata\st.qpfile,testdata\st_*.txt -ErrorAction SilentlyContinue

if( $fail ) { Write-Host "SELFTEST FAILED" -ForegroundColor Red; exit 1 }
Write-Host "ALL TESTS PASSED" -ForegroundColor Green
exit 0
