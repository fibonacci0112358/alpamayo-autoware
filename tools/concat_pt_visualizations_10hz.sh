#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/concat_pt_visualizations_10hz.sh [input_dir] [output_mp4]

Defaults:
  input_dir   /home/takeuchi/alpamayo-autoware/outputs/pt_visualizations
  output_mp4  /home/takeuchi/alpamayo-autoware/outputs/pt_visualizations/all_samples.mp4

The script concatenates sample_*.png images at 10 Hz into a single MP4.
EOF
}

input_dir="${1:-/home/takeuchi/alpamayo-autoware/outputs/pt_visualizations}"
output_mp4="${2:-$input_dir/all_samples.mp4}"

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "$input_dir" ]]; then
  echo "Error: input directory not found: $input_dir" >&2
  exit 1
fi

mapfile -t png_files < <(find "$input_dir" -maxdepth 1 -type f -name 'sample_*.png' | sort)

if [[ ${#png_files[@]} -eq 0 ]]; then
  echo "Error: no sample_*.png files found in $input_dir" >&2
  exit 1
fi

tmp_list=$(mktemp)
trap 'rm -f "$tmp_list"' EXIT

for png in "${png_files[@]}"; do
  printf "file '%s'\n" "$png" >> "$tmp_list"
  printf "duration 0.1\n" >> "$tmp_list"
done
# Repeat the last frame so the final duration is preserved.
last_index=$(( ${#png_files[@]} - 1 ))
printf "file '%s'\n" "${png_files[$last_index]}" >> "$tmp_list"

ffmpeg -y \
  -f concat \
  -safe 0 \
  -i "$tmp_list" \
  -vsync vfr \
  -pix_fmt yuv420p \
  "$output_mp4"

echo "Saved: $output_mp4"