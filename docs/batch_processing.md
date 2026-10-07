# Batch Processing

`csfseg batch` is for the common case where you have many preprocessed fMRI
files and a path file that lists them.

The first batch version is intentionally conservative: it runs one input at a
time. It does not stack several subjects into one GPU batch yet. The goal is to
make large runs easy to resume, inspect, and debug before adding more
complicated parallel scheduling.

## Basic Command

```bash
csfseg batch \
  --input-paths input_paths.csv \
  --out-dir /data/csfseg_outputs \
  --checkpoint /models/best_3ch.pt \
  --device auto \
  --skip-existing
```

`--input-paths` can be TXT, CSV, or TSV. TXT is one path per line. CSV/TSV
needs an `input_path` column and may include `output_id`.

## What Batch Actually Does

For each manifest row, batch mode calls the same single-file pipeline used by
`csfseg predict`:

```text
manifest row
-> SingleRunConfig
-> pipeline.run_single()
-> subject outputs
-> subject log
-> batch report row
```

This means `predict` and `batch` share the same model-input preparation, U-Net
inference, mask saving, voxel-table extraction, layer selection, and QC code.
Batch mode only adds the outer loop: which rows to run, which rows to skip, how
to keep going after a failure, and how to summarize the run.

## Output Files

Batch mode writes run-level files:

```text
input_paths_used.csv
run_config.yaml
reports/batch_report.csv
reports/failed_inputs.csv
logs/run_<run_id>.log
```

It also writes one subject log per input:

```text
logs/subjects/<output_id>.log
```

Successful inputs get the normal prediction outputs:

```text
probabilities/<output_id>_csf_prob.nii.gz
masks/<output_id>_csf_mask_bottom10.nii.gz
qc/<output_id>_prediction_qc.png
timeseries/<output_id>_voxel_table.csv
selected/auto/<output_id>_<layer>_auto_selected_voxels.csv
selected/auto/<output_id>_<layer>_auto_timeseries_qc.png
```

## Reports

`reports/batch_report.csv` is the main table for the whole run. It has one row
per input and includes status, stage, auto layer, voxel counts, elapsed time,
log path, and important output paths.

Example:

```csv
output_id,input_path,status,stage,auto_layer,mask_voxels,selected_voxels,elapsed_seconds,log_path,error
sub001,/data/sub001_bold.nii.gz,ok,done,L0,42,38,8.412,/data/csfseg_outputs/logs/subjects/sub001.log,
sub002,/data/sub002_bold.nii.gz,failed,inference,,,,2.103,/data/csfseg_outputs/logs/subjects/sub002.log,RuntimeError: checkpoint shape mismatch
```

`reports/failed_inputs.csv` is a smaller table with only failed rows. It is the
fastest file to open when a long run finishes with problems.

`run_config.yaml` records the command settings used for this batch run:

```yaml
run_id: 20260603-143012
input_paths: /data/input_paths.csv
out_dir: /data/csfseg_outputs
checkpoint: /models/best_3ch.pt
device: auto
threshold: 0.5
min_valid_voxels: 5
dropout_warning_fraction: 0.2
subject_workers: 1
skip_existing: true
```

## Logs

There are two log levels.

The run log:

```text
logs/run_<run_id>.log
```

This records the identity of the batch run: path file, output directory,
checkpoint, settings, number of inputs, and final summary.

Each subject log:

```text
logs/subjects/<output_id>.log
```

This records the detailed story for one input: input path, output id,
checkpoint, NIfTI shape, model-input shapes, major stages, output paths, and
full traceback if that input fails.

This split is deliberate. The terminal stays readable, the run log tells you
which batch you are looking at, and the subject log tells you exactly what
happened to one file.

## Resume With `--skip-existing`

If a batch run stops halfway through, rerun the same command with
`--skip-existing`.

An input is skipped when these core files already exist:

```text
probabilities/<output_id>_csf_prob.nii.gz
masks/<output_id>_csf_mask_bottom10.nii.gz
qc/<output_id>_prediction_qc.png
timeseries/<output_id>_voxel_table.csv
```

Skipped rows are written to `batch_report.csv` with:

```text
status = skipped_existing
stage = skip_existing
```

## Subject Workers

The command accepts:

```text
--subject-workers auto
--subject-workers 1
```

For this first version, both mean serial execution. Passing a larger value
raises a clear error.

The term means “how many complete input pipelines are running at once.” It does
not mean the PyTorch model batch dimension.

The current model input during inference is still:

```text
(1, 3, 10, 128, 128)
```

That leading `1` is the runtime batch dimension. It is not part of the
checkpoint.

## Future Parallel Version

A future version can add true subject workers or GPU-aware queues. For example,
it could prepare several inputs on CPU and send a stacked tensor to GPU:

```text
(N, 3, 10, 128, 128)
```

That future design needs extra care: per-subject crop/pad metadata, memory
limits, GPU memory checks, filesystem load, and failure handling when one item
inside a group fails. The current version leaves that complexity out on
purpose.
