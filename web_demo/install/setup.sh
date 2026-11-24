#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
py=${PYTHON:-python3.10}
"$py" -c 'import sys; assert sys.version_info[:2] == (3,10), "Use Python 3.10"'
git submodule update --init --recursive
for pair in CountGD:countgd CountGDPlusPlus:countgdpp; do
  backend=${pair%:*}
  patch="../patches/${pair#*:}-deformable-attention.patch"
  if ! git -C "$backend" apply --reverse --check "$patch" 2>/dev/null; then
    git -C "$backend" apply "$patch"
  fi
done
for stage in detector semantic sam; do
  bin=".demo-envs/$stage/bin/python"
  if [[ ! -x "$bin" ]]; then "$py" -m venv ".demo-envs/$stage"; fi
  "$bin" -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
  case "$stage" in
    detector) "$bin" -m pip install -r web_demo/install/detector.txt ;;
    semantic) "$bin" -m pip install 'transformers==4.51.3' 'numpy<2' scipy pillow sentencepiece protobuf scikit-learn ;;
    sam) "$bin" -m pip install 'transformers==5.3.0' 'numpy<2' scipy pillow ;;
  esac
done
.demo-envs/detector/bin/python -m web_demo.install.weights
