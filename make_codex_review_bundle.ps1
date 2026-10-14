$ErrorActionPreference = 'Stop'
$root = 'E:\LLM GATE\local-llm-reliability-gate'
if (!(Test-Path $root -PathType Container)) { throw "Project root not found: $root" }
$root = (Resolve-Path $root).Path
$stage = Join-Path $env:TEMP 'llm_gate_review_bundle_stage'
$zip = Join-Path $root 'local_llm_gate_review_bundle.zip'
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $stage | Out-Null
$exts = @('.py','.md','.txt','.json','.toml','.yaml','.yml','.ini','.cfg','.csv','.diff','.log')
$skipDirs = '[\\/](\.venv|venv|env|__pycache__|\.pytest_cache|\.mypy_cache|node_modules|\.git|models?|hf_cache|transformers_cache)[\\/]'
$skipFiles = '(^|[\\/])\.env($|\.)|secrets\.toml|credentials|api[_-]?key|token[_-]?secret'
foreach ($dir in @('src','scripts','tests')) {
  $full = Join-Path $root $dir
  if (!(Test-Path $full -PathType Container)) { continue }
  Get-ChildItem $full -Recurse -File | Where-Object {
    $rel = $_.FullName.Substring($root.Length).TrimStart('\','/')
    ($exts -contains $_.Extension.ToLowerInvariant()) -and $_.Length -lt 30MB -and
      $rel -notmatch $skipDirs -and $rel -notmatch $skipFiles
  } | ForEach-Object {
    $rel = $_.FullName.Substring($root.Length).TrimStart('\','/')
    $dest = Join-Path $stage $rel
    New-Item -ItemType Directory -Force -Path (Split-Path $dest -Parent) | Out-Null
    Copy-Item $_.FullName $dest -Force
  }
}
foreach ($name in @('app.py','README.md','requirements.txt','requirements-dev.txt','pyproject.toml','uv.lock','pytest.ini','.gitignore','_gitignore')) {
  $src = Join-Path $root $name
  if (Test-Path $src -PathType Leaf) { Copy-Item $src (Join-Path $stage $name) -Force }
}
$extras = @('artifacts\results\smoke_v5.json','artifacts\results\smoke_v5_diagnostic.json','artifacts\results\benchmark_results_300_v3.csv','artifacts\results\benchmark_results_300_v4.csv','artifacts\results\live_validation_10_v4.csv','artifacts\classifier\selected_threshold_v4.json','artifacts\classifier\risk_coverage_summary_v4.txt','artifacts\classifier\risk_coverage_thresholds_v4.csv','artifacts\classifier\risk_coverage_split_summaries_v4.csv')
foreach ($rel in $extras) {
  $src = Join-Path $root $rel
  if ((Test-Path $src -PathType Leaf) -and (Get-Item $src).Length -lt 30MB) {
    $dest = Join-Path $stage $rel
    New-Item -ItemType Directory -Force -Path (Split-Path $dest -Parent) | Out-Null
    Copy-Item $src $dest -Force
  }
}
$git = Get-Command git -ErrorAction SilentlyContinue
if ($git) {
  Push-Location $root
  try {
    git status --short 2>&1 | Set-Content (Join-Path $stage 'git_status.txt') -Encoding UTF8
    git diff -- app.py src scripts tests 2>&1 | Set-Content (Join-Path $stage 'working_tree.diff') -Encoding UTF8
  } catch {} finally { Pop-Location }
}
$manifest = @("Created: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')", "Project root: $root", 'Excluded virtual environments, caches, model weights, secrets, .git, and files above 30 MB.', '', 'Included files:')
$manifest += Get-ChildItem $stage -Recurse -File | ForEach-Object { $_.FullName.Substring($stage.Length).TrimStart('\','/') }
$manifest | Set-Content (Join-Path $stage 'BUNDLE_MANIFEST.txt') -Encoding UTF8
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -CompressionLevel Optimal
$item = Get-Item $zip
Write-Host "REVIEW BUNDLE CREATED: $($item.FullName)" -ForegroundColor Green
Write-Host ("Size: {0:N2} MB" -f ($item.Length / 1MB))
Write-Host 'Check BUNDLE_MANIFEST.txt includes src/gate, tests, scripts, app.py and smoke_v5_diagnostic.json.'
