import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

from csfseg import training as tr
from csfseg.models.unet3d import UNet3D_NoZDown
from csfseg.pipeline import SingleRunConfig, run_single


def config():
    return {'preprocessing': {'channels': ['mean', 'std', 'tsnr'], 'time_window': 'all',
            'normalization': 'nonzero_zscore', 'depth': 10, 'out_hw': [128, 128]},
            'training_policy': {'supervision_depth': 10, 'target_layers': [0, 1, 2],
                               'exclude_xy_padding_from_loss': True}}


def pair(tmp, ident):
    rng = np.random.default_rng(4)
    a = rng.normal(100, 10, (8, 8, 12, 10)).astype(np.float32)
    m = np.zeros(a.shape[:3], np.uint8)
    m[3:5, 3:5, :3] = 1
    paths = [tmp / (ident + ext) for ext in ('_image.nii.gz', '_mask.nii.gz')]
    for p, data in zip(paths, (a, m)):
        nib.save(nib.Nifti1Image(data, np.eye(4)), p)
    return dict(id=ident, input=str(paths[0]), mask=str(paths[1]), subject_id=ident,
                split='train', files_sha256={k: tr.digest(p) for k,p in zip(('input','mask'),paths)})


def test_padding_excluded_and_upper_layers_supervised():
    x = torch.zeros((1,1,10,4,4), requires_grad=True)
    y = torch.zeros_like(x); y.data[:,:,0,1,1] = 1
    w = torch.zeros_like(x); w.data[:,:,:,1:3,1:3] = 1
    tr.masked_loss(x,y,w).backward()
    assert x.grad[:,:,:,0,:].abs().sum() == 0
    assert x.grad[:,:,9,1,1] > 0


def test_leakage_rejected():
    rows=[dict(id='a',input='a',mask='am',subject_id='person',split='train'),
          dict(id='b',input='b',mask='bm',subject_id='person',split='val')]
    with pytest.raises(ValueError,match='leakage'):
        tr.validate_rows(rows)


def test_geometry_and_target_gates(tmp_path):
    row=pair(tmp_path,'a')
    tr.check_pair(row)
    m=nib.load(row['mask']); v=m.get_fdata(); v[0,0,4]=1
    nib.save(nib.Nifti1Image(v,m.affine),row['mask'])
    with pytest.raises(ValueError,match='L0-L2'):
        tr.check_pair(row)
    v[0,0,4]=0; aff=m.affine.copy();aff[0,3]=2
    nib.save(nib.Nifti1Image(v,aff),row['mask'])
    with pytest.raises(ValueError,match='geometry'):
        tr.check_pair(row)


def test_manifest_group_split_reproducible(tmp_path):
    rows=[pair(tmp_path,str(i)) for i in range(4)]
    src=tmp_path/'source.json';src.write_text(json.dumps({'rows':rows}))
    identity=tmp_path/'identity.csv'
    identity.write_text('fMRI_ImageID,Subject,Visit,fMRI_AcqDate\n0,A,v1,2020\n1,A,v2,2021\n2,B,v1,2020\n3,C,v1,2020\n')
    for i in range(2):
        tr.prepare_manifest(src,identity,tmp_path/f'out{i}.json')
    a=json.loads((tmp_path/'out0.json').read_text())
    b=json.loads((tmp_path/'out1.json').read_text())
    assert a==b
    assert a['rows'][0]['split']==a['rows'][1]['split']


def test_contract_rejects_silent_changes():
    cfg=config();cfg['preprocessing']['depth']=9
    with pytest.raises(ValueError,match='contract'):
        tr.contract_from_config(cfg)


def test_header_rounding_tolerated_but_subvoxel_shift_rejected(tmp_path):
    row=pair(tmp_path,'a');m=nib.load(row['mask']);v=m.get_fdata()
    affine=m.affine.copy();affine[0,3]=4e-5
    nib.save(nib.Nifti1Image(v,affine),row['mask'])
    tr.check_pair(row)
    affine[0,3]=0.01
    nib.save(nib.Nifti1Image(v,affine),row['mask'])
    with pytest.raises(ValueError,match='geometry'):
        tr.check_pair(row)


def test_train_resume_and_final_mask(tmp_path,monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(tr,'UNet3D_NoZDown',lambda: UNet3D_NoZDown(base=1))
    rows=[pair(tmp_path,str(i)) for i in range(2)];rows[1]['split']='val'
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'rows':rows}))
    initial=tmp_path/'init.pt';torch.save({'model':UNet3D_NoZDown(base=1).state_dict()},initial)
    cfg=config();cfg.update(data={'manifest':str(manifest)},initial_checkpoint=str(initial),
         training={'seed':42,'epochs':1,'learning_rate':0.001,'device':'cpu'})
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(cfg))
    out=tmp_path/'run'
    first=tr.train(path,out,smoke=True)
    assert first['optimizer_steps']==1 and first['first_layer_max_change']>0
    cfg['training']['epochs']=2;path.write_text(yaml.safe_dump(cfg))
    resumed=tr.train(path,out,smoke=True,resume=out/'last.pt')
    assert resumed['optimizer_steps']==2 and resumed['reload_max_error']==0
    result=run_single(SingleRunConfig(input_path=rows[0]['input'],out_dir=tmp_path/'prediction',
            checkpoint=out/'last.pt',device='cpu',model_base=1,threshold=0.,contract_config_path=path))
    mask=nib.load(result.mask_path).get_fdata()
    raw=nib.load(tmp_path/'prediction'/'masks'/'0_image_csf_mask_bottom10_raw.nii.gz').get_fdata()
    assert mask[:,:,:3].all() and not mask[:,:,3:].any() and raw[:,:,:10].all()
    assert result.checkpoint_requires_finetuning
    cfg['training']['learning_rate']=0.01;path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='signature'):
        tr.train(path,out,smoke=True,resume=out/'last.pt')
