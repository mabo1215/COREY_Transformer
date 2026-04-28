param(
    [string]$EnvPath = 'C:\source\.env',
    [string]$RemoteRoot = '/root/Corey_Transformer',
    [string]$RemotePaths = 'src/outputs fa3_h800_run.log',
    [string]$LocalBackupRoot = '',
    [int]$IntervalMinutes = 10,
    [int]$MaxIterations = 0
)

$ErrorActionPreference = 'Stop'

$workspaceRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if ($LocalBackupRoot -eq '') {
    $LocalBackupRoot = Join-Path $workspaceRoot 'src\outputs\h800_watch_backup'
}
New-Item -ItemType Directory -Force -Path $LocalBackupRoot | Out-Null

$helper = @'
import json
import os
import posixpath
import re
import stat
import sys
from pathlib import Path

import paramiko


def mkdir_p_sftp(sftp, path):
    parts = [p for p in path.split("/") if p]
    cur = ""
    for part in parts:
        cur += "/" + part
        try:
            sftp.stat(cur)
        except IOError:
            sftp.mkdir(cur)


def read_env(path):
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    ssh_line = next((ln for ln in lines if ln.startswith("ssh ")), None)
    if not ssh_line:
        raise RuntimeError(f"No ssh command found in {path}")
    idx = lines.index(ssh_line)
    if idx + 1 >= len(lines):
        raise RuntimeError(f"No password line after ssh command in {path}")
    password = lines[idx + 1]
    m = re.search(r"ssh\s+-p\s+(\d+)\s+([^@\s]+)@([^\s]+)", ssh_line)
    if not m:
        raise RuntimeError(f"Could not parse ssh command: {ssh_line}")
    return m.group(3), int(m.group(1)), m.group(2), password


def download_path(sftp, remote_path, local_path):
    try:
        info = sftp.stat(remote_path)
    except IOError:
        print(f"[watch] missing remote path: {remote_path}")
        return 0

    mode = info.st_mode
    if stat.S_ISDIR(mode):
        local_path.mkdir(parents=True, exist_ok=True)
        count = 0
        for entry in sftp.listdir_attr(remote_path):
            if entry.filename in (".", ".."):
                continue
            count += download_path(
                sftp,
                posixpath.join(remote_path, entry.filename),
                local_path / entry.filename,
            )
        return count

    if stat.S_ISREG(mode):
        local_path.parent.mkdir(parents=True, exist_ok=True)
        # Skip unchanged files by size only; good enough for frequent experiment
        # artifact sync without needing remote checksums.
        if local_path.exists() and local_path.stat().st_size == info.st_size:
            return 0
        tmp = local_path.with_suffix(local_path.suffix + ".tmp")
        sftp.get(remote_path, str(tmp))
        tmp.replace(local_path)
        print(f"[watch] downloaded {remote_path} -> {local_path}")
        return 1

    return 0


def main():
    cfg = json.loads(os.environ["H800_WATCH_CONFIG"])
    host, port, user, password = read_env(cfg["env_path"])
    remote_root = cfg["remote_root"].rstrip("/")
    remote_paths = cfg["remote_paths"]
    local_root = Path(cfg["local_root"])
    local_root.mkdir(parents=True, exist_ok=True)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        port=port,
        username=user,
        password=password,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
    )
    sftp = client.open_sftp()
    changed = 0
    for rel in remote_paths:
        rel = rel.strip("/")
        if not rel:
            continue
        changed += download_path(
            sftp,
            posixpath.join(remote_root, rel),
            local_root / rel.replace("/", os.sep),
        )
    sftp.close()

    # Also save a small remote status snapshot.
    status_cmd = f"cd {remote_root} 2>/dev/null && date -Is && hostname && (nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true) && find src/outputs -maxdepth 3 -type f 2>/dev/null | sort | tail -100"
    _, stdout, stderr = client.exec_command(status_cmd, timeout=60)
    status = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    (local_root / "remote_status.txt").write_text(status + ("\nSTDERR:\n" + err if err else ""), encoding="utf-8")
    client.close()
    print(f"[watch] sync complete, changed_files={changed}")


if __name__ == "__main__":
    main()
'@

$helperPath = Join-Path $LocalBackupRoot '_h800_watch_helper.py'
Set-Content -Path $helperPath -Value $helper -Encoding UTF8

$remotePathList = $RemotePaths -split '\s+' | Where-Object { $_ -and $_.Trim() -ne '' }
$config = @{
    env_path = $EnvPath
    remote_root = $RemoteRoot
    remote_paths = @($remotePathList)
    local_root = $LocalBackupRoot
} | ConvertTo-Json -Compress

$iteration = 0
while ($true) {
    $iteration += 1
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Write-Host "[$stamp] H800 result sync iteration $iteration -> $LocalBackupRoot"
    try {
        $env:H800_WATCH_CONFIG = $config
        python $helperPath
    }
    catch {
        Write-Warning "Sync failed; keeping existing local backup. $($_.Exception.Message)"
    }
    finally {
        Remove-Item Env:\H800_WATCH_CONFIG -ErrorAction SilentlyContinue
    }

    if ($MaxIterations -gt 0 -and $iteration -ge $MaxIterations) {
        break
    }
    Start-Sleep -Seconds ($IntervalMinutes * 60)
}
