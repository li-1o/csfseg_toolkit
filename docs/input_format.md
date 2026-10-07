# Input Format

CSFSeg starts from file paths. It does not assume BIDS, HCP folders, or any
specific dataset layout.

The key idea is simple:

```text
input_path = where the 4D fMRI NIfTI file is
output_id  = what prefix to use for this file's outputs
```

`input_path` is required. `output_id` is optional.

## TXT: One Path Per Line

This is the simplest format:

```text
/data/sub-001_ses-01_task-rest_run-01_bold.nii.gz
/data/sub-001_ses-01_task-rest_run-02_bold.nii.gz
/data/sub-002_ses-01_task-rest_run-01_bold.nii.gz
```

Blank lines and lines starting with `#` are ignored:

```text
# resting-state files
/data/sub-001_ses-01_task-rest_run-01_bold.nii.gz
/data/sub-002_ses-01_task-rest_run-01_bold.nii.gz
```

For TXT files, the output id is created from the input filename:

```text
/data/sub-001_ses-01_task-rest_run-01_bold.nii.gz
-> sub-001_ses-01_task-rest_run-01_bold
```

This works well when filenames already contain subject, session, task, and run
information.

## CSV Or TSV: Use When You Need Output IDs

Use CSV or TSV when filenames collide or when you want to control output names.

CSV example:

```csv
input_path,output_id
/data/siteA/sub-001/func/rest_bold.nii.gz,siteA_sub-001_rest
/data/siteB/sub-001/func/rest_bold.nii.gz,siteB_sub-001_rest
```

Required column:

```text
input_path
```

Optional column:

```text
output_id
```

Column names are strict. This will fail because the required column is not
called `input_path`:

```csv
path,output_id
/data/sub-001_bold.nii.gz,sub001
```

CSFSeg does not guess which column contains paths. That strictness prevents the
tool from silently reading the wrong column.

## What `output_id` Means

`output_id` is not a subject id and not a pattern. It is just the filename
prefix for one input row.

Example:

```csv
input_path,output_id
/data/siteA/sub-001/func/rest_bold.nii.gz,siteA_sub001_rest
```

Outputs will start with:

```text
siteA_sub001_rest
```

For example:

```text
masks/siteA_sub001_rest_csf_mask_bottom10.nii.gz
timeseries/siteA_sub001_rest_voxel_table.csv
qc/siteA_sub001_rest_prediction_qc.png
```

If two inputs have the same filename, use `output_id`. Otherwise outputs would
collide:

```text
/data/siteA/sub-001/func/rest_bold.nii.gz
/data/siteB/sub-001/func/rest_bold.nii.gz
```

Both filenames are `rest_bold.nii.gz`, so both would default to:

```text
rest_bold
```

That is ambiguous, and `check-manifest` will reject it.

## `check-manifest`

Run:

```bash
csfseg check-manifest \
  --input-paths input_paths.txt \
  --out-dir /data/csfseg_outputs
```

The command checks:

- the path file can be read
- each `input_path` exists
- CSV/TSV tables contain an `input_path` column
- `output_id` values are not empty when provided
- final output ids are unique
- output ids do not contain `/` or `\`

Then it writes:

```text
/data/csfseg_outputs/input_paths_used.csv
```

This is the resolved table that CSFSeg understood. For a TXT input, it records
the automatically created output ids. For a CSV/TSV input, it records the final
input-path and output-id pairs.

## What This Check Does Not Prove

`check-manifest` does not open NIfTI data. It does not check image shape,
voxel size, TR, preprocessing quality, or model-input QC.

For that, run `csfseg check-preprocessing` on a few representative files before
running prediction.
