# run_ablation2.ps1 —— 震荡消融 第二轮（快照每 5 万步，使"最近5个"与"全部快照"真正不同）
# 固定：25x25 duel gap=3 黑先，50万步，步罚 0.002，种子 1..10，启发式比例 0.65
# 变量：A 组 最近5个快照；B 组 全部快照（约10个）均匀采样
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$arms = @(
    @{ name = 'A'; window = 5; ratio = 0.65 },
    @{ name = 'B'; window = 0; ratio = 0.65 }
)
$maxPar = 8
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'

$jobs = @()
foreach ($arm in $arms) {
    foreach ($seed in 1..10) {
        $jobs += [pscustomobject]@{ arm = $arm; seed = $seed }
    }
}

$running = New-Object System.Collections.ArrayList
foreach ($job in $jobs) {
    $exp = "abl2_$($job.arm.name)_s$($job.seed)"
    while (@($running | Where-Object { -not $_.HasExited }).Count -ge $maxPar) {
        Start-Sleep -Seconds 20
    }
    $argList = @(
        'train_ppo.py',
        '--steps', '500000',
        '--size', '25',
        '--seed', "$($job.seed)",
        '--step-penalty', '0.002',
        '--heuristic-ratio', "$($job.arm.ratio)",
        '--pool-window', "$($job.arm.window)",
        '--snapshot-every', '50000',
        '--tb=',
        '--exp', $exp,
        '--out', "$exp.zip"
    )
    $p = Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory $PSScriptRoot `
        -RedirectStandardOutput "logs_$exp.out.txt" -RedirectStandardError "logs_$exp.err.txt" `
        -WindowStyle Hidden -PassThru
    [void]$running.Add($p)
    Add-Content -Path 'run_ablation2_status.txt' -Value "$(Get-Date -Format s) started $exp pid=$($p.Id)"
}

$running | Wait-Process
Add-Content -Path 'run_ablation2_status.txt' -Value "$(Get-Date -Format s) ALL DONE"