"""Small, auditable supervised fine-tuning workflow (no data augmentation)."""
from __future__ import annotations

import csv
import hashlib
import json
import itertools
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml

from csfseg.inference.checkpoint import load_checkpoint, extract_state_dict
from csfseg.models.unet3d import UNet3D_NoZDown
from csfseg.preprocessing.features import compute_temporal_features
from csfseg.preprocessing.normalize import normalize_features
from csfseg.preprocessing.bottom import extract_bottom_volume
from csfseg.preprocessing.patch import make_patch


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def contract_from_config(cfg):
    p = cfg['preprocessing']
    policy = cfg['training_policy']
    # Deliberately support only the agreed first baseline, not silent variants.
    expected = {'channels': ['mean', 'std', 'tsnr'], 'time_window': 'all',
                'normalization': 'nonzero_zscore', 'depth': 10, 'out_hw': [128, 128]}
    if p != expected:
        raise ValueError('Unsupported preprocessing contract; expected ' + str(expected))
    if (policy['supervision_depth'] != 10 or policy['target_layers'] != [0, 1, 2]
            or policy['exclude_xy_padding_from_loss'] is not True):
        raise ValueError('Baseline requires ten-layer supervision, target L0-L2, padding excluded')
    return dict(preprocessing=p, target_layers=[0, 1, 2], supervision_depth=10,
                exclude_xy_padding_from_loss=True, z_start=0, require_z_axis='S')


def grid_displacement_vox(image, mask):
    """Maximum corner displacement in mask voxels (tolerates header rounding)."""
    if image.shape[:3] != mask.shape:
        return float('inf')
    corners = np.array([list(c) + [1] for c in itertools.product(
        *[(0, n-1) for n in mask.shape])], dtype=float).T
    transformed = np.linalg.solve(mask.affine, image.affine) @ corners
    return float(np.linalg.norm(transformed[:3] - corners[:3], axis=0).max())


def check_pair(row):
    image, mask = nib.load(row['input']), nib.load(row['mask'])
    if len(image.shape) != 4 or image.shape[2] < 10 or image.shape[3] < 10:
        raise ValueError(f"{row['id']}: expected 4D input, Z>=10, T>=10")
    # Below one thousandth of one voxel: numerical tolerance, never registration.
    displacement = grid_displacement_vox(image, mask)
    if not np.isfinite(displacement) or displacement > 1e-3:
        raise ValueError(f"{row['id']}: input/mask geometry mismatch")
    if not np.isfinite(image.affine).all() or nib.aff2axcodes(image.affine)[2] != 'S':
        raise ValueError(f"{row['id']}: z=0 must be inferior; reorientation is not automatic")
    if any(d > 128 for d in image.shape[:2]):
        raise ValueError(f"{row['id']}: baseline disallows cropping potentially labeled voxels")
    label = mask.get_fdata(dtype=np.float32)
    if not np.isin(label, [0, 1]).all() or not label.any() or label[:, :, 3:].any():
        raise ValueError(f"{row['id']}: need nonempty binary mask limited to L0-L2")
    return image, label


def validate_rows(rows):
    if not rows:
        raise ValueError('Empty manifest')
    for key in ('id', 'input', 'mask'):
        values = [str(Path(r[key]).resolve()) if key != 'id' else r[key] for r in rows]
        if len(set(values)) != len(values):
            raise ValueError('Duplicate ' + key)
    groups = {}
    for row in rows:
        if not row.get('subject_id') or row['split'] not in ('train', 'val'):
            raise ValueError('Verified subject_id and train/val split required')
        groups.setdefault(row['subject_id'], set()).add(row['split'])
    if any(len(v) > 1 for v in groups.values()):
        raise ValueError('Subject leakage between train and validation')
    if {r['split'] for r in rows} != {'train', 'val'}:
        raise ValueError('Both train and validation sets are required')


def prepare_manifest(source, identity_csv, output, seed=42, val_fraction=0.2):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    rows = json.loads(Path(source).read_text())['rows']
    identities = {}
    with open(identity_csv, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            ident, subject = r['fMRI_ImageID'].strip(), r['Subject'].strip()
            if not subject or (ident in identities and identities[ident]['Subject'] != subject):
                raise ValueError('Missing/conflicting identity: ' + ident)
            identities[ident] = r
    for r in rows:
        if r['id'] not in identities:
            raise ValueError('Missing identity: ' + r['id'])
        identity = identities[r['id']]
        r.update(subject_id=identity['Subject'], visit=identity['Visit'],
                 acquisition_date=identity['fMRI_AcqDate'])
        image, _ = check_pair(r)
        r['max_grid_displacement_vox'] = grid_displacement_vox(image, nib.load(r['mask']))
        r['files_sha256'] = {k: digest(r[k]) for k in ('input', 'mask')}
    subjects = sorted({r['subject_id'] for r in rows})
    if len(subjects) < 2 or not 0 < val_fraction < 1:
        raise ValueError('Need >=2 subjects and 0<validation fraction<1')
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(subjects).tolist()
    nval = max(1, min(len(subjects)-1, round(len(subjects)*val_fraction)))
    val = set(shuffled[:nval])
    for r in rows:
        r['split'] = 'val' if r['subject_id'] in val else 'train'
    validate_rows(rows)
    result = dict(rows=rows, seed=seed, val_fraction=val_fraction,
                  identity_source=str(Path(identity_csv).resolve()), identity_sha256=digest(identity_csv),
                  counts={s: sum(r['split'] == s for r in rows) for s in ('train', 'val')},
                  subjects={s: len({r['subject_id'] for r in rows if r['split'] == s}) for s in ('train', 'val')})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return {k: result[k] for k in ('counts', 'subjects', 'seed')}


class PairedDataset:
    """Lazy in-memory feature cache; no transformed public files are written."""
    def __init__(self, rows):
        self.rows, self.cache = rows, {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if index not in self.cache:
            row = self.rows[index]
            for key in ('input', 'mask'):
                if digest(row[key]) != row['files_sha256'][key]:
                    raise ValueError(f"{row['id']}: {key} changed since manifest creation")
            image, label = check_pair(row)
            data = image.get_fdata(dtype=np.float32)
            if not np.isfinite(data).all():
                raise ValueError('Nonfinite fMRI: ' + row['id'])
            features = compute_temporal_features(data)
            norm = normalize_features(features.data, features.channels)
            x = make_patch(extract_bottom_volume(norm.data).data).data
            y = label[:, :, :10].transpose(2, 0, 1)[None]
            valid = make_patch(np.ones_like(y)).data
            y = make_patch(y).data
            if not np.isfinite(x).all():
                raise ValueError('Nonfinite features')
            self.cache[index] = tuple(torch.from_numpy(a)[None] for a in (x, y, valid))
        return self.cache[index]


def masked_loss(logits, target, valid):
    """BCE + soft Dice over real voxels in all ten layers, not padded voxels."""
    bce = (torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction='none') * valid).sum() / valid.sum().clamp_min(1)
    p = logits.sigmoid() * valid
    y = target * valid
    dice = (2 * (p * y).sum() + 1e-6) / (p.sum() + y.sum() + 1e-6)
    return bce + 1 - dice


def hard_dice(logits, target, valid, final=False):
    pred = (logits.sigmoid() >= .5) * valid.bool()
    y = target.bool() * valid.bool()
    if final:
        pred = pred.clone()
        pred[:, :, 3:] = False
    return float((2 * (pred & y).sum() + 1e-6) / (pred.sum() + y.sum() + 1e-6))


def atomic_save(value, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    temp.replace(path)


def train(config_path, out_dir, *, smoke=False, resume=None):
    cfg = yaml.safe_load(Path(config_path).read_text())
    contract = contract_from_config(cfg)
    t = cfg['training']
    seed, epochs, lr = int(t['seed']), int(t['epochs']), float(t['learning_rate'])
    if epochs < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError('Invalid epochs/learning rate')
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(t.get('device', 'cuda'))
    manifest = json.loads(Path(cfg['data']['manifest']).read_text())
    rows = manifest['rows']
    validate_rows(rows)
    train_rows = [r for r in rows if r['split'] == 'train']
    val_rows = [r for r in rows if r['split'] == 'val']
    if smoke:
        train_rows, val_rows = train_rows[:2], val_rows[:1]
    datasets = {s: PairedDataset(rs) for s, rs in [('train', train_rows), ('val', val_rows)]}
    # Freeze scientific/runtime inputs; epochs may increase when resuming.
    signature = dict(contract=contract, seed=seed, learning_rate=lr, smoke=smoke,
                     manifest_sha256=digest(cfg['data']['manifest']),
                     initial_sha256=digest(cfg['initial_checkpoint']),
                     code_sha256={str(p.relative_to(Path(__file__).parent)): digest(p)
                                  for p in sorted(Path(__file__).parent.rglob('*.py'))})
    out = Path(out_dir)
    ckpt = load_checkpoint(resume) if resume else None
    if ckpt and ckpt['signature'] != signature:
        raise ValueError('Resume config/data/code signature mismatch')
    if not resume and out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a fresh output directory or --resume')
    if resume and (not out.exists() or not (out / 'last.pt').exists()
                   or Path(resume).resolve() != (out / 'last.pt').resolve()):
        raise ValueError('Resume must use last.pt in the same run directory')
    start = ckpt['epoch'] + 1 if ckpt else 1
    if start > epochs:
        raise ValueError('Configured epochs must exceed completed epochs')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    model = UNet3D_NoZDown().to(device)
    model.load_state_dict(extract_state_dict(ckpt or load_checkpoint(cfg['initial_checkpoint'])), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = float(ckpt['best_dice']) if ckpt else -1.0
    steps = int(ckpt['steps']) if ckpt else 0
    history = ckpt['history'] if ckpt else []
    if ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
        torch.set_rng_state(ckpt['rng_cpu'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(ckpt['rng_cuda'])
    before = model.enc1[0].weight.detach().clone()
    for epoch in range(start, epochs + 1):
        model.train()
        losses = []
        for i in np.random.default_rng(seed + epoch).permutation(len(datasets['train'])):
            x, y, valid = (a.to(device) for a in datasets['train'][int(i)])
            optimizer.zero_grad(set_to_none=True)
            loss = masked_loss(model(x), y, valid)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach()))
        model.eval()
        scores, raw_scores, val_losses = [], [], []
        with torch.no_grad():
            for i in range(len(datasets['val'])):
                x, y, valid = (a.to(device) for a in datasets['val'][i])
                logits = model(x)
                loss = masked_loss(logits, y, valid)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite validation loss')
                val_losses.append(float(loss))
                scores.append(hard_dice(logits, y, valid, final=True))
                raw_scores.append(hard_dice(logits, y, valid))
        metrics = dict(epoch=epoch, steps=steps, train_loss=float(np.mean(losses)),
                       val_loss=float(np.mean(val_losses)), val_dice=float(np.mean(scores)),
                       val_dice_raw10=float(np.mean(raw_scores)))
        history.append(metrics)
        improved = metrics['val_dice'] > best
        best = max(best, metrics['val_dice'])
        saved = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                     steps=steps, best_dice=best, history=history, signature=signature,
                     rng_cpu=torch.get_rng_state(),
                     rng_cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                     metadata={'segmentation_contract': contract, 'requires_finetuning': bool(smoke),
                               'smoke_only': bool(smoke), 'channel_order': ['mean', 'std', 'tsnr']})
        if improved:
            atomic_save(saved, out / 'best.pt')
        atomic_save(saved, out / 'last.pt')
        (out / 'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(metrics), flush=True)
    delta = float((model.enc1[0].weight.detach() - before).abs().max())
    if delta == 0 or not np.isfinite(delta):
        raise ValueError('Parameters did not update correctly')
    # Verify actual serialization, strict reload and output equivalence.
    reload_model = UNet3D_NoZDown().to(device)
    reload_model.load_state_dict(extract_state_dict(load_checkpoint(out / 'last.pt')), strict=True)
    reload_model.eval()
    with torch.no_grad():
        difference = float((reload_model(x) - model(x)).abs().max())
    if difference > 1e-6:
        raise ValueError('Reload output mismatch')
    result = dict(status='passed', smoke_only=smoke, resumed=bool(resume), completed_epochs=epochs,
                  optimizer_steps=steps, train_cases=len(train_rows), val_cases=len(val_rows),
                  first_layer_max_change=delta, reload_max_error=difference,
                  best_validation_dice=best, device=str(device), torch=str(torch.__version__))
    (out / 'acceptance.json').write_text(json.dumps(result, indent=2))
    return result
