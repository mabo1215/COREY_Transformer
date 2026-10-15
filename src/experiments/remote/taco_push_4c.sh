#!/usr/bin/env bash
# Copy the tracked files of HEAD to the 4c host, which has no GitHub credentials (its checkout is a git-archive
# tarball, not a git repository). Keeps the venv, logs and results already on the host. Run from WSL (or any
# machine with the ssh alias "4c" and access to the lab network):
#   wsl bash src/experiments/remote/taco_push_4c.sh
# Results come back with:  wsl bash src/revloop/remote.sh fetch 4c E28
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
R=/home1/mabo1215/COREY_Transformer_taco
TMP=$(mktemp -d)
git -C "$ROOT" archive --format=tar.gz -o "$TMP/corey_head.tar.gz" HEAD || exit 1
scp -q "$TMP/corey_head.tar.gz" 4c:$R/corey_head.tar.gz || exit 1
ssh -o BatchMode=yes 4c "cd $R && tar xzf corey_head.tar.gz && rm -f corey_head.tar.gz && echo \"4c code updated to $(git -C "$ROOT" log --oneline -1 | cut -c1-70)\""
rm -rf "$TMP"
