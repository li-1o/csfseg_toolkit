# CSFSeg Toolkit

CSFSeg Toolkit segments bottom-slice CSF from a preprocessed 4D fMRI NIfTI file
and extracts voxel-level CSF time series.

It is built for one focused workflow:

1. read a preprocessed 4D fMRI NIfTI,
2. compute model input channels: temporal mean, temporal std, and tSNR,
3. run a 3D U-Net CSF segmentation model,
4. save the bottom10 CSF probability map and binary mask,
5. extract L0-L3 mask voxel time series,
6. write QC figures and selected voxel tables.

The toolkit does not prepare raw fMRI data. Motion correction, distortion
correction, registration, and other dataset-specific preprocessing should
already be done before the data comes here.

## Install

Recommended on a research server:

```bash
conda env create -f environment.yml
conda activate csfseg
pip install -e .
```

For an existing Python environment:

```bash
pip install -e .
```

## Smoke Test

Before using real data, you can run a tiny end-to-end demo:

```bash
python examples/run_smoke_test.py --overwrite
```

It creates synthetic 4D fMRI files and a random 3-channel checkpoint, then runs
the real CLI commands from path checking through batch export. This only checks
that the workflow runs and writes outputs. The random checkpoint is not a model
for analysis.

## Quick Start

Run one file:

```bash
csfseg predict \
  --input /data/sub-001_ses-01_task-rest_run-01_bold.nii.gz \
  --out-dir /data/csfseg_outputs \
  --checkpoint /models/best_3ch.pt
```

The output directory will contain:

```text
csfseg_outputs/
├── masks/
├── probabilities/
├── timeseries/
├── selected/
├── qc/
├── reports/
└── logs/
```

The main files are:

```text
probabilities/<output_id>_csf_prob.nii.gz
masks/<output_id>_csf_mask_bottom10.nii.gz
qc/<output_id>_prediction_qc.png
timeseries/<output_id>_voxel_table.csv
selected/auto/<output_id>_<layer>_auto_selected_voxels.csv
selected/auto/<output_id>_<layer>_auto_timeseries_qc.png
reports/selection_template.csv
logs/subjects/<output_id>.log
```

After checking QC, edit `reports/selection_template.csv` if a run should use a
different layer or be excluded, then export a new selected version:

```bash
csfseg select-voxels \
  --out-dir /data/csfseg_outputs \
  --selection-file reports/selection_template.csv \
  --selection-name manual_v1
```

For many files, put paths in a TXT/CSV/TSV file and run:

```bash
csfseg batch \
  --input-paths input_paths.txt \
  --out-dir /data/csfseg_outputs \
  --checkpoint /models/best_3ch.pt \
  --skip-existing
```

The first batch version runs inputs one at a time, but writes batch reports and
detailed per-input logs so large runs are easier to resume and debug.

## Current Commands

```text
csfseg check-manifest       validate an input path file
csfseg check-preprocessing  make QC for model input preparation
csfseg predict              run single-file prediction and extraction
csfseg batch                run prediction for a path file
csfseg select-voxels        re-export selected voxels after QC review
csfseg init-3ch-checkpoint  development helper for finetuning setup
```

## Documentation

Start here:

- [docs/quickstart.md](docs/quickstart.md) for a step-by-step first run
- [docs/input_format.md](docs/input_format.md) for path files, `input_path`, and `output_id`
- [docs/commands.md](docs/commands.md) for command-by-command usage
- [docs/outputs.md](docs/outputs.md) for output files
- [docs/qc.md](docs/qc.md) for QC figures
- [docs/model_checkpoints.md](docs/model_checkpoints.md) for checkpoint files

## License

This project is released under the [MIT License](LICENSE).
