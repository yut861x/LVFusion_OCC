#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import mmcv, numpy as np, torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('config'); p.add_argument('checkpoint'); p.add_argument('--out-dir',required=True)
    p.add_argument('--indices',type=int,nargs='+',default=[0]); p.add_argument('--device',default='cuda:0')
    return p.parse_args()

def arr(x):
    if isinstance(x,(list,tuple)): x=x[0]
    return x.detach().float().cpu().numpy()

def main():
    args=parse_args(); cfg=mmcv.Config.fromfile(args.config); cfg.model.pretrained=None
    cfg.data.test.test_mode=True
    dataset=build_dataset(cfg.data.test)
    loader=build_dataloader(dataset,samples_per_gpu=1,workers_per_gpu=0,dist=False,shuffle=False)
    model=build_model(cfg.model,test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model,args.checkpoint,map_location='cpu')
    model=MMDataParallel(model.to(args.device),device_ids=[0]); model.eval(); core=model.module
    cap={}; hooks=[]
    hooks.append(core.img_backbone.register_forward_pre_hook(lambda m,x: cap.__setitem__('img',arr(x[0]))))
    hooks.append(core.pts_middle_encoder.register_forward_hook(lambda m,x,y: cap.__setitem__('pts_spatial_features',arr(y))))
    hooks.append(core.img_neck_3d.register_forward_pre_hook(lambda m,x: cap.__setitem__('img_volume',arr(x[0]))))
    hooks.append(core.pts_neck_fuse.register_forward_hook(lambda m,x,y: cap.__setitem__('pts_bev_feat',arr(y))))
    hooks.append(core.occ_head.register_forward_hook(lambda m,x,y: cap.__setitem__('occ_logits',arr(y))))
    wanted=set(args.indices); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    try:
        for index,data in enumerate(loader):
            if index>max(wanted): break
            if index not in wanted: continue
            cap.clear()
            with torch.no_grad(): model(return_loss=False,rescale=True,**data)
            missing={'img','pts_spatial_features','img_volume','pts_bev_feat','occ_logits'}-cap.keys()
            if missing: raise RuntimeError(f'missing captures: {missing}')
            vol=cap['img_volume']
            packed=np.transpose(vol,(0,2,3,4,1)).reshape(vol.shape[0],vol.shape[2],vol.shape[3],vol.shape[4]*vol.shape[1]).transpose(0,3,1,2)
            tensors={'img':cap['img'],'pts_spatial_features':cap['pts_spatial_features'],
                     'img_volume_packed':packed,'pts_bev_feat':cap['pts_bev_feat'],
                     'occ_logits':cap['occ_logits']}
            sample=out/f'sample_{index:04d}'; sample.mkdir(parents=True,exist_ok=True)
            manifest={'dataset_index':index,'tensors':{}}
            for name,value in tensors.items():
                dtype=np.float32 if name=='occ_logits' else np.float16
                path=sample/f'{name}.{"fp32" if dtype==np.float32 else "fp16"}'
                value.astype(dtype).tofile(path)
                manifest['tensors'][name]={'file':path.name,'shape':list(value.shape),'dtype':np.dtype(dtype).name,
                                           'min':float(value.min()),'max':float(value.max()),
                                           'mean':float(value.mean()),'nonzero':int(np.count_nonzero(value))}
            (sample/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            print(json.dumps(manifest,ensure_ascii=False),flush=True)
    finally:
        for h in hooks: h.remove()
if __name__=='__main__': main()
