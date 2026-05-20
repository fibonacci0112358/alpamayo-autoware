#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/mnt/nvme/alpamayo_data"
FORCE=0
DRY_RUN=0
RCLONE_COPY=1
RCLONE_SRC="gdrive:ver1.0"

usage() {
  cat <<'EOF'
Usage:
  extract_rosbag2_archives.sh [ROOT_DIR] [--force] [--dry-run] [--no-rclone-copy] [--rclone-src SRC]

Description:
  Search ROOT_DIR for archives matching rosbag2*.tar.xz under 3-digit directories
  (e.g. 000, 019, 123) and extract each archive in place.

Options:
  --force    Re-extract even if extracted directory already exists.
  --dry-run  Only show matching archives without extracting.
  --rclone-copy
             Run: rclone copy SRC ROOT_DIR -P -vv before extraction (default).
  --no-rclone-copy
             Skip rclone copy step.
  --rclone-src SRC
             Source remote/path for rclone copy (default: gdrive:ver1.0).

Examples:
  extract_rosbag2_archives.sh
  extract_rosbag2_archives.sh /mnt/nvme/alpamayo_data --dry-run
  extract_rosbag2_archives.sh /mnt/nvme/alpamayo_data --force
  extract_rosbag2_archives.sh /mnt/nvme/alpamayo_data --no-rclone-copy
  extract_rosbag2_archives.sh /mnt/nvme/alpamayo_data --rclone-copy --rclone-src gdrive:ver1.0
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --rclone-copy)
      RCLONE_COPY=1
      shift
      ;;
    --no-rclone-copy)
      RCLONE_COPY=0
      shift
      ;;
    --rclone-src)
      if [[ $# -lt 2 ]]; then
        echo "--rclone-src requires a value" >&2
        usage
        exit 1
      fi
      RCLONE_SRC="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      if [[ "$ROOT_DIR" != "/mnt/nvme/alpamayo_data" ]]; then
        echo "Multiple root directories specified: $ROOT_DIR and $1" >&2
        usage
        exit 1
      fi
      ROOT_DIR="$1"
      shift
      ;;
  esac
done

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Root directory not found: $ROOT_DIR" >&2
  exit 1
fi

echo "Root directory: $ROOT_DIR"

if [[ $RCLONE_COPY -eq 1 ]]; then
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] rclone copy $RCLONE_SRC $ROOT_DIR -P -vv"
  else
    if ! command -v rclone >/dev/null 2>&1; then
      echo "rclone command not found in PATH." >&2
      exit 1
    fi

    echo "[RCLONE] rclone copy $RCLONE_SRC $ROOT_DIR -P -vv"
    rclone copy "$RCLONE_SRC" "$ROOT_DIR" -P -vv
  fi
fi

mapfile -d '' archives < <(
  find "$ROOT_DIR" \
    -regextype posix-extended \
    -type f \
    -regex '.*/[0-9]{3}/rosbag2[^/]*\.tar\.xz' \
    -print0 | sort -z
)

if [[ ${#archives[@]} -eq 0 ]]; then
  echo "No matching rosbag2*.tar.xz archives found under 3-digit directories."
  exit 0
fi

echo "Found ${#archives[@]} archive(s)."

ok_count=0
skip_count=0
fail_count=0

for archive in "${archives[@]}"; do
  archive_dir="$(dirname "$archive")"
  extracted_dir="${archive%.tar.xz}"

  if [[ $FORCE -eq 0 && -d "$extracted_dir" ]]; then
    echo "[SKIP] $archive (directory exists: $extracted_dir)"
    ((skip_count+=1))
    continue
  fi

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] $archive"
    continue
  fi

  echo "[EXTRACT] $archive"
  if tar -xJf "$archive" -C "$archive_dir"; then
    ((ok_count+=1))
  else
    echo "[FAIL] $archive" >&2
    ((fail_count+=1))
  fi
done

if [[ $DRY_RUN -eq 1 ]]; then
  echo "Dry run finished."
  exit 0
fi

echo "Summary: ok=$ok_count skip=$skip_count fail=$fail_count"

if [[ $fail_count -gt 0 ]]; then
  exit 2
fi
