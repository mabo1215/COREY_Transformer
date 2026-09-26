#!/usr/bin/env bash
# git credential helper for git.codeocean.com. It holds no secret itself: on "get" it reads
# the "CODE OCEAN" block of the local credentials file (first line: account e-mail, second
# line: access token) and hands them to git. Nothing is written anywhere.
#
# Use:  git -c credential.https://git.codeocean.com.helper="$PWD/src/revloop/codeocean_credential.sh" ...
# The file defaults to /mnt/c/source/.env (WSL view of C:\source\.env); override with CODEOCEAN_ENV.
set -euo pipefail
[ "${1:-}" = get ] || exit 0
ENV_FILE=${CODEOCEAN_ENV:-/mnt/c/source/.env}
[ -r "$ENV_FILE" ] || exit 0
awk 'BEGIN{b=0}
     { line=$0; sub(/\r$/, "", line); t=line; gsub(/^[ \t]+|[ \t]+$/, "", t) }
     b==0 && toupper(t) ~ /^CODE[ _]?OCEAN:?$/ { b=1; next }
     b==1 && t != "" { print "username=" t; b=2; next }
     b==2 && t != "" { print "password=" t; exit }' "$ENV_FILE"
