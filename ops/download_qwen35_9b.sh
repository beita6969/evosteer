#!/bin/bash
# Resumable parallel download of Qwen/Qwen3.5-9B weights from ModelScope.
D=/workspace/models/Qwen3.5-9B-ms
mkdir -p "$D"
SRC=/workspace/evosteer/models/Qwen3.5-9B
for f in "$SRC"/*; do [ -f "$f" ] && cp -n "$f" "$D/"; done
URL=https://www.modelscope.cn/models/Qwen/Qwen3.5-9B/resolve/master
for i in 1 2 3 4; do
  f="model.safetensors-0000${i}-of-00004.safetensors"
  ( for try in 1 2 3 4 5 6; do curl -sL --retry 3 -C - -o "$D/$f" "$URL/$f" && break; sleep 5; done; echo "$f done $(stat -c %s "$D/$f")" >> "$D/DOWNLOAD.log" ) &
done
wait
echo ALL_DONE >> "$D/DOWNLOAD.log"
