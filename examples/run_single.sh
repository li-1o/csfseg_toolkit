#!/usr/bin/env bash
set -euo pipefail

csfseg predict \
  --input /absolute/path/to/sub-001_ses-01_task-rest_run-01_bold.nii.gz \
  --out-dir /absolute/path/to/csfseg_outputs \
  --checkpoint /absolute/path/to/checkpoints/best_3ch.pt

