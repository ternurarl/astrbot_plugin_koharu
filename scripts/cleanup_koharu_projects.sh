#!/usr/bin/env bash
set -u

KEEP_COUNT=5
INTERVAL_SECONDS=600
PREFIX="astrbot-koharu-*"

echo "[koharu-cleanup] watching $(pwd)"
echo "[koharu-cleanup] keep newest ${KEEP_COUNT}, check every ${INTERVAL_SECONDS}s"

while true; do
  mapfile -d '' entries < <(
    find . -maxdepth 1 -mindepth 1 -type d -name "${PREFIX}" -printf '%T@ %p\0' |
      sort -z -n
  )

  count=${#entries[@]}
  if (( count > KEEP_COUNT )); then
    delete_count=$((count - KEEP_COUNT))
    echo "[koharu-cleanup] found ${count} projects; deleting oldest ${delete_count}"

    for ((i = 0; i < delete_count; i++)); do
      entry="${entries[$i]}"
      path="${entry#* }"
      if [[ -n "${path}" && "${path}" == ./"astrbot-koharu-"* ]]; then
        echo "[koharu-cleanup] deleting ${path}"
        rm -rf -- "${path}"
      else
        echo "[koharu-cleanup] skip suspicious path: ${path}"
      fi
    done
  else
    echo "[koharu-cleanup] found ${count} projects; nothing to delete"
  fi

  sleep "${INTERVAL_SECONDS}"
done
