#!/usr/bin/env bash
set -euo pipefail

csfseg check-manifest \
  --input-paths input_paths.csv \
  --out-dir /absolute/path/to/csfseg_outputs
