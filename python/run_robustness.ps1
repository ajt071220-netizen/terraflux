# 实验④：多种子复现 + 静止棋盘对照（报告 §4 / §5 / §6 的稳健性检验）
# 在仓库根目录运行：
#   powershell -ExecutionPolicy Bypass -File python\run_robustness.ps1                          # 第一批（四条并行流）
#   powershell -ExecutionPolicy Bypass -File python\run_robustness.ps1 -Batch2                  # 第二批：补到每组约 6 个种子
#   powershell -ExecutionPolicy Bypass -File python\run_robustness.ps1 -Only sparse_s1,sparse_s2 # 指定任务
# 每个训练 50 万步（25×25 duel，黑先，gap=3，与主实验相同的对手池）。
# 输出写入 experiments\robust_logs\。
#
# 勘误：早期版本的 SelfPlayEnv.step 硬编码了 reward -= 0.002，导致 --step-penalty 在训练中无效。
# 因此首次运行的 sparse_s2 / sparse_s3 与 shaped 完全相同（逐行一致），已作废并删除，
# 修复后（train_ppo.py 读取 stepPenalty）重新运行。shaped / static 的步罚即为 0.002，不受影响。

param([string[]]$Only, [switch]$Batch2, [switch]$Batch3)

$py  = '.\.venv-py312\Scripts\python.exe'
$log = 'experiments\robust_logs'
New-Item -ItemType Directory -Force $log | Out-Null

# 任务：exp 标签 -> 随机种子, 步罚, 是否静止棋盘
$jobs = @{
  shaped_s2 = @{ seed = 2; penalty = 0.002; static = $false }
  shaped_s3 = @{ seed = 3; penalty = 0.002; static = $false }
  sparse_s1 = @{ seed = 1; penalty = 0;     static = $false }
  sparse_s2 = @{ seed = 2; penalty = 0;     static = $false }
  sparse_s3 = @{ seed = 3; penalty = 0;     static = $false }
  static_s1 = @{ seed = 1; penalty = 0.002; static = $true }
  static_s2 = @{ seed = 2; penalty = 0.002; static = $true }
  static_s3 = @{ seed = 3; penalty = 0.002; static = $true }
  # 第二批
  shaped_s4 = @{ seed = 4; penalty = 0.002; static = $false }
  shaped_s5 = @{ seed = 5; penalty = 0.002; static = $false }
  sparse_s4 = @{ seed = 4; penalty = 0;     static = $false }
  sparse_s5 = @{ seed = 5; penalty = 0;     static = $false }
  sparse_s6 = @{ seed = 6; penalty = 0;     static = $false }
  static_s4 = @{ seed = 4; penalty = 0.002; static = $true }
  static_s5 = @{ seed = 5; penalty = 0.002; static = $true }
  static_s6 = @{ seed = 6; penalty = 0.002; static = $true }
  # 第三批：步罚组补到每组 10 个种子（shaped 2–11，sparse 1–10）
  shaped_s6  = @{ seed = 6;  penalty = 0.002; static = $false }
  shaped_s7  = @{ seed = 7;  penalty = 0.002; static = $false }
  shaped_s8  = @{ seed = 8;  penalty = 0.002; static = $false }
  shaped_s9  = @{ seed = 9;  penalty = 0.002; static = $false }
  shaped_s10 = @{ seed = 10; penalty = 0.002; static = $false }
  shaped_s11 = @{ seed = 11; penalty = 0.002; static = $false }
  sparse_s7  = @{ seed = 7;  penalty = 0;     static = $false }
  sparse_s8  = @{ seed = 8;  penalty = 0;     static = $false }
  sparse_s9  = @{ seed = 9;  penalty = 0;     static = $false }
  sparse_s10 = @{ seed = 10; penalty = 0;     static = $false }
}

function Build-Cmd([string]$exp) {
  $j = $jobs[$exp]
  if ($null -eq $j) { throw "unknown job: $exp" }
  $flags = "--steps 500000 --size 25 --seed $($j.seed) --step-penalty $($j.penalty)"
  if ($j.static) { $flags += ' --static-terrain' }
  $out = "models\ppo25_$exp.zip"
  return "& '$py' python\train_ppo.py $flags --exp $exp --out $out --tb tb_robust *> '$log\$exp.log'"
}

if ($Only -and $Only.Count -gt 0) {
  # 参数可能以单个逗号分隔字符串传入（-File 模式），统一拆分；每个任务一条并行流
  $names   = ($Only -join ',') -split ',' | Where-Object { $_ }
  $streams = $names | ForEach-Object { , @($_) }
} elseif ($Batch2) {
  # 第二批：四条并行流，每流两个任务
  $streams = @(
    @('shaped_s4', 'static_s4'),
    @('shaped_s5', 'static_s5'),
    @('sparse_s4', 'static_s6'),
    @('sparse_s5', 'sparse_s6')
  )
} elseif ($Batch3) {
  # 第三批：五条并行流，每流两个任务（共 10 个）
  $streams = @(
    @('sparse_s7', 'shaped_s6'),
    @('sparse_s8', 'shaped_s7'),
    @('sparse_s9', 'shaped_s8'),
    @('sparse_s10', 'shaped_s9'),
    @('shaped_s10', 'shaped_s11')
  )
} else {
  # 第一批：四条并行流，每流顺序执行若干任务
  $streams = @(
    @('shaped_s2', 'static_s1'),
    @('sparse_s1', 'static_s2'),
    @('shaped_s3', 'static_s3'),
    @('sparse_s2', 'sparse_s3')
  )
}

foreach ($s in $streams) {
  $cmds = ($s | ForEach-Object { Build-Cmd $_ }) -join '; '
  $full = "`$env:OMP_NUM_THREADS='6'; $cmds"
  Start-Process powershell -ArgumentList '-NoProfile', '-Command', $full -WindowStyle Hidden
  Write-Host "started stream: $($s -join ' -> ')"
}
