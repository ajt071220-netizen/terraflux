# 第二批流水线：等待批次二训练全部完成（每个日志 ≥ 500k 步且模型已保存），
# 然后并行做新增模型的自对弈与循环赛（4 个分片，--new-only），合并结果，重算统计与图。
# 用法（仓库根目录）：powershell -ExecutionPolicy Bypass -File python\run_batch2_pipeline.ps1
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = '.\.venv-py312\Scripts\python.exe'
$exps = 'shaped_s4', 'shaped_s5', 'sparse_s4', 'sparse_s5', 'sparse_s6', 'static_s4', 'static_s5', 'static_s6'
New-Item -ItemType Directory -Force 'experiments\replicate_shards' | Out-Null

function Finished($e) {
  $f = "training_log_$e.jsonl"
  if (-not (Test-Path $f)) { return $false }
  $r = Get-Content $f -Encoding UTF8 | Select-Object -Last 1 | ConvertFrom-Json
  return ($r.steps -ge 500000) -and (Test-Path "models\ppo25_$e.zip")
}

# 1) 等待训练完成
while ($true) {
  $pending = @($exps | Where-Object { -not (Finished $_) })
  if ($pending.Count -eq 0) { break }
  Start-Sleep -Seconds 30
}
Write-Output 'training finished; starting evaluation shards'

# 2) 并行评估（只算新增的模型与对局）
$procs = @()
foreach ($i in 0..3) {
  $procs += Start-Process -FilePath $py -ArgumentList '-X', 'utf8', '-B', 'python\replicate_eval.py', '--new-only', '--shard', "$i", '--nshards', '4', '--games', '300', '--rr-games', '100' -PassThru -WindowStyle Hidden -RedirectStandardOutput "experiments\replicate_shards\log_$i.txt" -RedirectStandardError "experiments\replicate_shards\err_$i.txt"
}
$procs | Wait-Process
Write-Output 'shards finished'

# 3) 合并、统计、画图
& $py -X utf8 -B python\replicate_eval.py --merge
& $py -X utf8 -B python\analyze_robustness.py | Out-File -Encoding utf8 'experiments\robustness_stdout.txt'
& $py -X utf8 -B python\fig_replicates.py
Write-Output 'PIPELINE_DONE'
