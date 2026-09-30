#!/usr/bin/env python3
"""Overlay existing 3D boxes on an Ouster PCD without changing source data."""
import argparse
import sys
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

TOOL_ROOT = Path(os.environ["LVFUSION_ANNOTATION_ROOT"])
sys.path.insert(0, str(TOOL_ROOT))
from annotation_tool.geometry import _BOX_EDGES, box_corners_lidar
from annotation_tool.label_io import load_labels
from annotation_tool.pcd_io import load_pcd_xyzi

PALETTE = {
    "truck": "#ff9f1c", "large_vehicle": "#bf6f1d",
    "construction_vehicle": "#f4d21f", "bus": "#c33c54",
    "car": "#1597d4", "motorcycle": "#8247d6", "bicycle": "#35aa8c",
    "tricycle": "#5360c9", "pedestrian": "#e83e8c",
}

def inside_box(xyz, b):
    q = xyz - [b.x, b.y, b.z]
    c, s = np.cos(b.theta), np.sin(b.theta)
    local = np.c_[c*q[:, 0]+s*q[:, 1], -s*q[:, 0]+c*q[:, 1], q[:, 2]]
    return ((np.abs(local[:, 0]) <= b.l/2) & (np.abs(local[:, 1]) <= b.w/2)
            & (np.abs(local[:, 2]) <= b.h/2))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("root",type=Path); ap.add_argument("--stem"); ap.add_argument("--output",type=Path); ap.add_argument("--xmax",type=float,default=120.)
    args=ap.parse_args(); stem=args.stem or sorted(x.stem for x in (args.root/"pcds_ouster").glob("*.pcd"))[0]
    output = args.output or (args.root / "vis" / f"{stem}.png")
    xyz,_=load_pcd_xyzi(str(args.root/"pcds_ouster"/f"{stem}.pcd")); boxes=load_labels(str(args.root/"labels"/f"{stem}.txt"))
    lateral=max(20.,args.xmax*.58)
    roi=(xyz[:,0]>=0)&(xyz[:,0]<=args.xmax)&(np.abs(xyz[:,1])<=lateral)&(xyz[:,2]>=-3)&(xyz[:,2]<=10); p=xyz[roi]
    rng=np.random.default_rng(7); ids=np.arange(len(p)); ids=rng.choice(ids,min(40000,len(ids)),replace=False); show=p[ids]
    fig=plt.figure(figsize=(18,9),facecolor="#171d26"); ax1=fig.add_subplot(121); ax2=fig.add_subplot(122,projection="3d")
    for ax in (ax1,ax2): ax.set_facecolor("#202731")
    ax1.scatter(show[:,0],show[:,1],s=.35,c=show[:,2],cmap="viridis",vmin=-1,vmax=4,alpha=.65,rasterized=True)
    ax2.scatter(show[:,0],show[:,1],show[:,2],s=.25,c=show[:,2],cmap="viridis",vmin=-1,vmax=4,alpha=.45,rasterized=True)
    report=[]
    drawn=0
    for b in boxes:
        if b.x-b.l/2>args.xmax or abs(b.y)-b.w/2>lateral: continue
        drawn+=1
        color=PALETTE.get(b.cls.lower(),"#dddddd"); corners=box_corners_lidar(b)
        n=int(inside_box(xyz,b).sum()); report.append((b.obj_id,b.cls,n))
        bev=corners[[0,2,6,4,0],:2]; ax1.plot(bev[:,0],bev[:,1],color=color,lw=2.2)
        ax1.text(b.x,b.y,f"{b.cls} #{b.obj_id}\n{n} pts",color=color,fontsize=8)
        for i,j in _BOX_EDGES: ax2.plot(corners[[i,j],0],corners[[i,j],1],corners[[i,j],2],color=color,lw=1.8)
    ax1.set(xlim=(0,args.xmax),ylim=(-lateral,lateral),xlabel="X / m",ylabel="Y / m",title="BEV | Ouster point cloud + 3D BBox")
    ax1.set_aspect("equal",adjustable="box"); ax1.grid(color="#52606d",alpha=.35)
    ax2.set(xlim=(0,args.xmax),ylim=(-lateral,lateral),zlim=(-3,10),xlabel="X / m",ylabel="Y / m",zlabel="Z / m",title="3D view | geometric-center boxes")
    ax2.view_init(elev=24,azim=-72); ax2.grid(True,alpha=.25)
    for ax in (ax1,ax2): ax.tick_params(colors="#e5e7eb"); ax.xaxis.label.set_color("#e5e7eb"); ax.yaxis.label.set_color("#e5e7eb"); ax.title.set_color("white")
    ax2.zaxis.label.set_color("#e5e7eb")
    fig.suptitle(f"Frame {stem} | {len(xyz):,} Ouster points | {drawn} GT boxes | X <= {args.xmax:g} m",color="white",fontsize=16,fontweight="bold")
    fig.tight_layout(rect=(0,0,1,.95)); output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(output,dpi=180,facecolor=fig.get_facecolor()); plt.close(fig)
    print("output",output); print("box point counts",report)
if __name__=="__main__": main()
