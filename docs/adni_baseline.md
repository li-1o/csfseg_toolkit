# ADNI baseline fine-tuning record

This repository contains code and aggregate results only. Participant images,
labels, predictions, case-level QC, manifests, credentials and trained weights
are excluded. The software license does not grant rights to research data or
third-party pretrained weights.

## Experiment

- 50 sessions from 44 participants; fixed participant-level split, seed 42.
- Training: 41 sessions from 35 participants. Validation: 9 sessions from 9
  other participants. No participant overlaps these groups.
- Fine-tuning starts from a three-channel initialization migrated from an
  existing one-channel network. All model parameters are trainable.
- Input: native-space motion-corrected 4D fMRI, before temporal filtering,
  nuisance regression or spatial smoothing for this baseline.
- Channels: temporal mean, standard deviation and temporal signal-to-noise
  ratio; nonzero z-score normalization; bottom 10 slices; 128x128 XY padding.
- The NIfTI affine must indicate increasing Z toward superior; no automatic
  reorientation or image-label registration is performed.
- Manual positives are confined to L0-L2. All ten real input layers participate
  in BCE + soft Dice loss, including background in L3-L9. XY padding is excluded.
- AdamW, learning rate 0.0001, weight decay 0.0001, batch size 1, 50 epochs,
  gradient clipping 5. No augmentation or learning-rate scheduler.
- Output threshold 0.5; final masks retain only L0-L2. Raw bottom-ten output is
  kept separately for QC. Anatomical holes must not be automatically filled.

## Validation and interpretation

Best checkpoint: epoch 15. Session-macro Dice: 0.830877; median 0.818182;
range 0.777778-0.910053. Six of nine sessions exceed 0.8. Removing the highest
score gives a mean of 0.820981.

These nine validation sessions were used to select the checkpoint. They are
not an independent test set. Training-case predictions describe fit to seen
examples, not generalization. Do not combine training and validation scores
into a performance claim. This is a research segmentation aid requiring QC,
not a validated clinical device or proof of anatomical label correctness.

## Reproduction

Prepare a participant-grouped manifest with the CLI's `prepare-training`
command, then use `train --config ... --out-dir ...` (see `csfseg --help`).
Use `examples/train_baseline.yaml` as a template. Supply authorized local data
and initialization weights separately. The manifest includes checksums and
training rejects changed files or mismatched image-label geometry.

Recorded training runtime: Python 3.10.18 and PyTorch 2.6.0+cu124. This is a
historical experiment record, not a claim these are current recommended versions.
New deployments must use a maintained PyTorch release supporting restricted
`weights_only=True` loading; unrestricted fallback is disabled.

Local logs and generated NIfTI headers can contain source paths or metadata.
They are internal artifacts, not automatically anonymized publication material.
