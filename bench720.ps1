# 720p motion-search benchmark for the hybrid x265 GPU-ME fork.
# x265 has no esa/tesa primitives; --me full is the exhaustive search,
# which is the closest analogue.  Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File bench720.ps1
# Results are written to testdata\*.hevc + testdata\*.log and
# bench-results\*.log (with a repro_summary.csv for tools/bdrate.py).
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$exe = "build-vs\Release\x265.exe"
$yuv = "testdata\motion720p.yuv"
New-Item -ItemType Directory -Force -Path "bench-results\comparison" | Out-Null

$summary = New-Object System.Collections.Generic.List[string]
$summary.Add("h_kbps,h_psnr,h_ssim,n_kbps,n_psnr,n_ssim")

function Run-One($tag, $opts) {
    $out = "testdata\$tag.hevc"
    $log = "testdata\$tag.log"
    cmd /c "`"$exe`" $opts --input-res 1280x720 --fps 30 --psnr --ssim --frames 60 `"$yuv`" -o `"$out`" 1> `"$log`" 2>&1" | Out-Null
    $text = Get-Content $log -Raw
    $enc  = [regex]::Match($text, "encoded (\d+) frames, ([\d\.]+) fps, ([\d\.]+) kb/s")
    $ssim = [regex]::Match($text, "SSIM Mean Y:([\d\.]+)")
    $psnr = [regex]::Match($text, "PSNR Mean Y:([\d\.]+).*Global:([\d\.]+)")
    $line = "{0,-14} fps={1,7}  kb/s={2,8}  ssim={3}  psnr={4}  size={5}" -f `
        $tag, $enc.Groups[2].Value, $enc.Groups[3].Value, $ssim.Groups[1].Value,
        $psnr.Groups[2].Value, (Get-Item $out).Length
    $line
    Copy-Item $log "bench-results\comparison\$tag.log" -Force
    return ,@($enc.Groups[3].Value, $psnr.Groups[2].Value, $ssim.Groups[1].Value)
}

function Run-Pair($tag, $opts, $crfList) {
    $h = New-Object System.Collections.Generic.List[string]
    foreach( $crf in $crfList )
    {
        $t = $tag + "_crf" + $crf
        $o = $opts + " --crf " + $crf
        $g = Run-One ($t + "_g") $o
        $n = Run-One ($t + "_s") ($o -replace " --gpu-me \d+"," --gpu-me 0" -replace " --gpu-lookahead \d+"," --gpu-lookahead 0")
        $summary.Add(("{0},{1},{2},{3},{4},{5}" -f $g[0],$g[1],$g[2],$n[0],$n[1],$n[2]))
    }
}

$crfs = @(17, 20, 23, 26, 29)
"=== medium (hex) ==="
Run-Pair "med" "--preset medium --gpu-me 2" $crfs
"=== slow (umh) ==="
Run-Pair "slow" "--preset slow --gpu-me 2" $crfs
"=== slow --me full --subme 7 ==="
Run-Pair "full" "--preset slow --me full --subme 7 --gpu-me 3" $crfs
"=== placebo ==="
Run-Pair "plc" "--preset placebo --gpu-me 3" $crfs
"=== lookahead medium (stock vs --gpu-lookahead 1, fast) ==="
Run-Pair "lam" "--preset medium --gpu-lookahead 1" $crfs

$summary | Set-Content "bench-results\comparison\repro_summary.csv"
"summary CSV: bench-results\comparison\repro_summary.csv"
