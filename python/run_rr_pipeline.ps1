# 循环赛扩展流水线：把新增的动态模型（批次二、三）与全部已有动态模型两两对打（只算缺失的对局），
# 4 个分片并行，合并，统计，重画图。完成时输出 PIPELINE_DONE。
# 用法（仓库根目录）：powershell -ExecutionPolicy Bypass -File python\run_rr_pipeline.ps1
$ErrorActionPreference = 'Continue'
Set-Location (Split-Path -Parent $PSScriptRoot)
$py = '.\.venv-py312\Scripts\python.exe'
New-Item -ItemType Directory -Force 'experiments\replicate_shards' | Out-Null
Remove-Item 'experiments\replicate_shards\shard_*.json' -Force -ErrorAction SilentlyContinue

$procs = @()
foreach ($i in 0..3) {
  $procs += Start-Process -FilePath $py -ArgumentList '-X', 'utf8', '-B', 'python\replicate_eval.py', '--new-only', '--shard', "$i", '--nshards', '4', '--games', '300', '--rr-games', '100' -PassThru -WindowStyle Hidden -RedirectStandardOutput "experiments\replicate_shards\logrr_$i.txt" -RedirectStandardError "experiments\replicate_shards\errrr_$i.txt"
}
$procs | Wait-Process
Write-Output 'rr shards finished'

& $py -X utf8 -B python\replicate_eval.py --merge
& $py -X utf8 -B python\analyze_roundrobin.py | Out-File -Encoding utf8 'experiments\roundrobin_stdout.txt'
& $py -X utf8 -B python\fig_replicates.py
Write-Output 'PIPELINE_DONE'
