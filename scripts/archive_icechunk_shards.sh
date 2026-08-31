#!/usr/bin/env bash
# Tar one icechunk repository into fixed-size shards for cloud transfer, and prove the
# shards reconstruct it.
#
#   ./scripts/archive_icechunk_shards.sh <repo_path> <out_dir> [shard_size]
#
# An icechunk repo is thousands of small chunk files. Uploading them individually is dominated
# by per-object latency; a handful of large sequential objects is not. Shards are plain
# uncompressed tar: the zarr chunks inside are already compressed, so a second pass would burn
# CPU for nothing.
#
# Reconstruction is `cat *.tar.part* | tar -xf -`, which only works if the shards concatenate
# back to the exact byte stream tar produced. That is asserted here rather than assumed --
# the stream is hashed as it is written and the shards are hashed back, and the rebuilt stream
# is listed with `tar -t` to prove it is a well-formed archive. A truncated final shard would
# otherwise surface months later, in the cloud, when someone tries to restore.
set -uo pipefail
REPO="${1:?usage: $0 <repo_path> <out_dir> [shard_size]}"
OUT="${2:?}"
SIZE="${3:-2G}"

[ -d "$REPO" ] || { echo "FATAL: $REPO is not a directory" >&2; exit 2; }
NAME=$(basename "$REPO"); NAME="${NAME%.icechunk}"
PARENT=$(dirname "$REPO"); BASE=$(basename "$REPO")
mkdir -p "$OUT"
rm -f "$OUT/${NAME}.tar.part"* "$OUT/MANIFEST.txt" "$OUT"/*.sha256 "$OUT/contents.txt"

n_src=$(find "$REPO" -type f | wc -l)
b_src=$(du -sb "$REPO" | cut -f1)
echo "=== ${BASE} -> ${OUT} (${SIZE} shards) ==="
echo "  source: ${n_src} files, $(numfmt --to=iec "$b_src")"

echo "  [1/3] writing shards"
set -o pipefail
tar -cf - -C "$PARENT" "$BASE" \
  | tee >(sha256sum | cut -d' ' -f1 > "$OUT/stream.sha256") \
  | split -b "$SIZE" -d -a 3 - "$OUT/${NAME}.tar.part" || {
      echo "FATAL: tar/split failed" >&2; exit 3; }

shopt -s nullglob
parts=("$OUT/${NAME}.tar.part"*)
[ "${#parts[@]}" -gt 0 ] || { echo "FATAL: no shards written" >&2; exit 4; }

echo "  [2/3] hashing ${#parts[@]} shards"
: > "$OUT/shards.sha256"
for p in "${parts[@]}"; do sha256sum "$p" >> "$OUT/shards.sha256"; done

echo "  [3/3] rebuilding from the shards and listing the archive"
cat "${parts[@]}" \
  | tee >(sha256sum | cut -d' ' -f1 > "$OUT/rebuilt.sha256") \
  | tar -tf - > "$OUT/contents.txt" || {
      echo "FATAL: the concatenated shards are not a readable tar archive" >&2; exit 5; }

want=$(cat "$OUT/stream.sha256"); got=$(cat "$OUT/rebuilt.sha256")
n_tar=$(grep -c -v '/$' "$OUT/contents.txt")
b_shards=$(du -cb "${parts[@]}" | tail -1 | cut -f1)

if [ "$want" != "$got" ]; then
  echo "FATAL: the shards do not reconstruct the tar stream" >&2
  echo "  written  ${want}" >&2; echo "  rebuilt  ${got}" >&2; exit 6
fi
if [ "$n_tar" -ne "$n_src" ]; then
  echo "FATAL: archive holds ${n_tar} files, the repo has ${n_src}" >&2; exit 7
fi

{
  echo "archive:      ${BASE}"
  echo "created:      $(date -Is)"
  echo "host:         $(hostname)"
  echo "source:       ${REPO}"
  echo "source_files: ${n_src}"
  echo "source_bytes: ${b_src}"
  echo "shard_size:   ${SIZE}"
  echo "shards:       ${#parts[@]}"
  echo "shard_bytes:  ${b_shards}"
  echo "tar_sha256:   ${want}"
  echo
  echo "restore:"
  echo "  cat ${NAME}.tar.part* | tar -xf -        # recreates ${BASE}/"
  echo "  # verify first:  cat ${NAME}.tar.part* | sha256sum   ->  ${want}"
  echo
  echo "per-shard sha256:"
  sed 's|  .*/|  |' "$OUT/shards.sha256"
} > "$OUT/MANIFEST.txt"

echo "  ✓ ${#parts[@]} shards, $(numfmt --to=iec "$b_shards"), ${n_tar} files"
echo "  ✓ shards reconstruct the tar stream exactly (sha256 ${want:0:16}…)"
echo "  ✓ manifest ${OUT}/MANIFEST.txt"
