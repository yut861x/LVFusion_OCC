#!/usr/bin/env python3
"""Compare two LiDAR streams against common 3D GT boxes."""
import argparse, csv, json, sys
import os
from pathlib import Path
import numpy as np

TOOL = Path(os.environ['LVFUSION_ANNOTATION_ROOT'])
sys.path.insert(0,str(TOOL))
from annotation_tool.label_io import load_labels
from annotation_tool.pcd_io import load_pcd_xyzi

WHITE={'truck','large_vehicle','construction_vehicle','bus','car','motorcycle','bicycle','tricycle','pedestrian'}

def local(xyz,b):
    q=xyz-np.array([b.x,b.y,b.z]); c,s=np.cos(b.theta),np.sin(b.theta)
    return np.c_[c*q[:,0]+s*q[:,1],-s*q[:,0]+c*q[:,1],q[:,2]]

def stats(xyz,b,margin=.6):
    q=local(xyz,b); half=np.array([b.l,b.w,b.h])/2
    near=(np.abs(q)<=half+[margin,margin,.3]).all(1)
    # Remove points clearly below box bottom; these are likely road returns.
    cand=q[near & (q[:,2]>=-b.h/2-.08)]
    if not len(cand): return dict(n_near=0,n_inside=0,hit=np.nan,dx=np.nan,dy=np.nan,dz=np.nan)
    inside=(np.abs(cand)<=half+1e-6).all(1)
    obj=cand[inside] if inside.sum()>=3 else cand
    med=np.median(obj,axis=0)
    return dict(n_near=len(cand),n_inside=int(inside.sum()),hit=float(inside.mean()),dx=float(med[0]),dy=float(med[1]),dz=float(med[2]))

def summarize(rows,sensor):
    valid=[r for r in rows if r[f'{sensor}_n_near']>=3]
    adequate=[r for r in rows if r[f'{sensor}_n_near']>=10]
    def arr(k,src=adequate): return np.array([r[f'{sensor}_{k}'] for r in src],float)
    return dict(boxes_with_3plus=len(valid),boxes_with_10plus=len(adequate),
        median_inside_points=float(np.median(arr('n_inside',valid))) if valid else None,
        median_expanded_hit_ratio=float(np.nanmedian(arr('hit'))) if adequate else None,
        median_local_offset_m=[float(np.nanmedian(arr(k))) if adequate else None for k in ('dx','dy','dz')],
        p90_abs_local_offset_m=[float(np.nanpercentile(np.abs(arr(k)),90)) if adequate else None for k in ('dx','dy','dz')])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); ap.add_argument('--max-range',type=float,default=40); ap.add_argument('--output',type=Path)
    a=ap.parse_args(); out=a.output or a.root/'vis'/'lidar_bbox_alignment'; out.mkdir(parents=True,exist_ok=True)
    stems=sorted(set(x.stem for x in (a.root/'pcds').glob('*.pcd')) & set(x.stem for x in (a.root/'pcds_ouster').glob('*.pcd')) & set(x.stem for x in (a.root/'labels').glob('*.txt')))
    rows=[]
    for fi,stem in enumerate(stems):
        p33,_=load_pcd_xyzi(str(a.root/'pcds'/f'{stem}.pcd')); po,_=load_pcd_xyzi(str(a.root/'pcds_ouster'/f'{stem}.pcd'))
        for b in load_labels(str(a.root/'labels'/f'{stem}.txt')):
            if b.cls.lower() not in WHITE or np.hypot(b.x,b.y)>a.max_range: continue
            row=dict(stem=stem,frame=fi,obj_id=b.obj_id,cls=b.cls.lower(),x=b.x,y=b.y,z=b.z,distance=float(np.hypot(b.x,b.y)))
            for name,pts in [('pcds',p33),('ouster',po)]:
                for k,v in stats(pts,b).items(): row[f'{name}_{k}']=v
            rows.append(row)
    fields=list(rows[0]) if rows else []
    with open(out/'per_box.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    both=[r for r in rows if r['pcds_n_near']>=10 and r['ouster_n_near']>=10]
    delta={k:np.array([r[f'ouster_{k}']-r[f'pcds_{k}'] for r in both],float) for k in ('dx','dy','dz')}
    report=dict(frames=len(stems),boxes=len(rows),max_range_m=a.max_range,pcds=summarize(rows,'pcds'),ouster=summarize(rows,'ouster'),both_10plus=len(both),
        ouster_minus_pcds_median_local_offset_m=[float(np.nanmedian(delta[k])) if len(both) else None for k in ('dx','dy','dz')],
        ouster_minus_pcds_p90_abs_difference_m=[float(np.nanpercentile(np.abs(delta[k]),90)) if len(both) else None for k in ('dx','dy','dz')])
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2)); print(out/'per_box.csv')
if __name__=='__main__': main()
