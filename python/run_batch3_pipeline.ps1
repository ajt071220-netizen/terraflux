# 第三批流水线：等待批次三训练全部完成（每个日志 ≥ 500k 步且模型已保存），
# 然后并行做新增模型的自对弈（4 个分片，--new-only --skip-rr），合并，重算统计与图。
# 循环赛不扩展（仍为 13 个动态模型），报告中注明。
# 用法（仓库根目录）：powershell -ExecutionPolicy Bypass -File python\run_batch3_pipeline.ps1
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = '.\.venv-py312\Scripts\python.exe'
$exps = 'shaped_s6', 'shaped_s7', 'shaped_s8', 'shaped_s9', 'shaped_s10', 'shaped_s11', 'sparse_s7', 'sparse_s8', 'sparse_s9', 'sparse_s10'
New-Item -ItemType Directory -Force 'experiments\replicate_shards' | Out-Null

function Finished($e) {
  $f = "training_log_$e.jsonl"
  if (-not (Test-Path $f)) { return $false }
  $r = Get-Content $f -Encoding UTF8 | Select-Object -Last 1 | ConvertFrom-Json
  return ($r.steps -ge 500000) -and (Test-Path "models\ppo25_$e.zip")
}

while ($true) {
  $pending = @($exps | Where-Object { -not (Finished $_) })
  if ($pending.Count -eq 0) { break }
  Start-Sleep -Seconds 30
}
Write-Output 'training finished; starting evaluation shards'

$procs = @()
foreach ($i in 0..3) {
  $procs += Start-Process -FilePath $py -ArgumentList '-X', 'utf8', '-B', 'python\replicate_eval.py', '--new-only', '--skip-rr', '--shard', "$i", '--nshards', '4', '--games', '300' -PassThru -WindowStyle Hidden -RedirectStandardOutput "experiments\replicate_shards\log3_$i.txt" -RedirectStandardError "experiments\replicate_shards\err3_$i.txt"
}
$procs | Wait-Process
Write-Output 'shards finished'

& $py -X utf8 -B python\replicate_eval.py --merge
& $py -X utf8 -B python\analyze_robustness.py | Out-File -Encoding utf8 'experiments\robustness_stdout.txt'
& $py -X utf8 -B python\fig_replicates.py
Write-Output 'PIPELINE_DONE'
