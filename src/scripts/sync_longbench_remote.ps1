$ErrorActionPreference = 'Stop'

$tasks = @('gov_report','multifieldqa_en','narrativeqa','qasper')
foreach ($t in $tasks) {
  $localPath = "C:\source\COREY_Transformer\src\data\longbench_subset\$t\test.jsonl"
  $localSize = (Get-Item $localPath).Length
  $attempt = 0
  $remoteSize = 0

  while (($remoteSize -lt $localSize) -and ($attempt -lt 20)) {
    $attempt++
    wsl bash -lc "ssh -o BatchMode=yes mabo1215@10.147.20.176 'mkdir -p /home1/mabo1215/COREY_Transformer/src/data/longbench_subset/$t'" | Out-Null
    wsl bash -lc "rsync -av --partial --append --timeout=120 /mnt/c/source/COREY_Transformer/src/data/longbench_subset/$t/test.jsonl mabo1215@10.147.20.176:/home1/mabo1215/COREY_Transformer/src/data/longbench_subset/$t/test.jsonl"

    $remoteSizeStr = wsl bash -lc "ssh -o BatchMode=yes mabo1215@10.147.20.176 'stat -c%s /home1/mabo1215/COREY_Transformer/src/data/longbench_subset/$t/test.jsonl 2>/dev/null || echo 0'"
    $remoteSize = [int64]($remoteSizeStr.Trim())
    Write-Host "[$t] attempt=$attempt local=$localSize remote=$remoteSize"
  }

  if ($remoteSize -lt $localSize) {
    throw "Failed to fully sync $t after $attempt attempts (remote=$remoteSize, local=$localSize)"
  }
}

wsl bash -lc "ssh -o BatchMode=yes mabo1215@10.147.20.176 'for t in gov_report multifieldqa_en narrativeqa qasper; do ls -lh /home1/mabo1215/COREY_Transformer/src/data/longbench_subset/$t/test.jsonl; done'"
