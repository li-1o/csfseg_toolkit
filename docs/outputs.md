# Outputs

The user must choose an output directory with `--out-dir`. The toolkit should not write results to a hidden default location.

Recommended output layout:

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

If you run `csfseg check-manifest`, the same output directory also gets:

```text
input_paths_used.csv
```

If you run `csfseg batch`, the output directory also gets:

```text
input_paths_used.csv
run_config.yaml
reports/batch_report.csv
reports/failed_inputs.csv
logs/run_<run_id>.log
```

## Per-Input Outputs

For an input whose output id is:

```text
sub-001_ses-01_task-rest_run-01_bold
```

the outputs are:

```text
masks/sub-001_ses-01_task-rest_run-01_bold_csf_mask_bottom10.nii.gz
probabilities/sub-001_ses-01_task-rest_run-01_bold_csf_prob.nii.gz
timeseries/sub-001_ses-01_task-rest_run-01_bold_voxel_table.csv
qc/sub-001_ses-01_task-rest_run-01_bold_prediction_qc.png
selected/auto/sub-001_ses-01_task-rest_run-01_bold_L0_auto_selected_voxels.csv
selected/auto/sub-001_ses-01_task-rest_run-01_bold_L0_auto_timeseries_qc.png
reports/selection_template.csv
reports/auto_selection_summary.csv
logs/subjects/sub-001_ses-01_task-rest_run-01_bold.log
```

The toolkit does not average the CSF signal. The main data table is voxel-level:
one mask voxel per row, one TR per time column. Downstream averaging or further
cleaning is left to the user.

At the prediction stage, the first files written are:

```text
probabilities/sub-001_ses-01_task-rest_run-01_bold_csf_prob.nii.gz
masks/sub-001_ses-01_task-rest_run-01_bold_csf_mask_bottom10.nii.gz
qc/sub-001_ses-01_task-rest_run-01_bold_prediction_qc.png
```

The probability map is a 3D float image in the input NIfTI space. The mask is a
3D uint8 image made by thresholding that probability map; the default threshold
is `0.5`. The model predicts the bottom `z=0:10` block, so voxels outside that
block are zero. With a contract-bearing ADNI baseline checkpoint, a separate
`*_bottom10_raw.nii.gz` keeps the raw threshold result while the main mask and
prediction QC use `L0-L2`. A checkpoint without a contract uses the legacy
`L0-L3` candidate default.

After prediction, the toolkit also writes:

```text
timeseries/sub-001_ses-01_task-rest_run-01_bold_voxel_table.csv
reports/selection_template.csv
selected/auto/sub-001_ses-01_task-rest_run-01_bold_L0_auto_selected_voxels.csv
selected/auto/sub-001_ses-01_task-rest_run-01_bold_L0_auto_timeseries_qc.png
```

`voxel_table.csv` is the full data source for checkpoint-approved layers: all
mask voxels are kept, including isolated voxels and voxels with zero-dropout warnings. The
`selected/auto` table is the default analysis-ready subset: selected layer,
non-isolated voxels, and no zero-dropout voxels.

Users adjust results by editing `reports/selection_template.csv` and running
`csfseg select-voxels`. New exports go into versioned folders such as:

```text
selected/manual_v1/sub-001_ses-01_task-rest_run-01_bold_L1_manual_v1_selected_voxels.csv
selected/manual_v1/sub-001_ses-01_task-rest_run-01_bold_L1_manual_v1_timeseries_qc.png
reports/manual_v1_selection_summary.csv
```

`input_paths_used.csv` is the resolved input table written by
`csfseg check-manifest` or `csfseg batch`. It records the paths and output ids
that CSFSeg understood from the user's TXT/CSV/TSV file.

For batch runs, `reports/batch_report.csv` is the one-row-per-input summary.
It is the best place to check which inputs finished, failed, or were skipped.
`reports/failed_inputs.csv` is the smaller failure-only table. `run_config.yaml`
records the settings used for that batch run.
