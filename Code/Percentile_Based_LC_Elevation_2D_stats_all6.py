# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 14:10:21 2025

@author: yujia.yang
"""

import numpy as np
import pandas as pd
import rasterio
import glob
import re
import os

# ================================
# 1. 输入文件 & LC/DEM 文件
# ================================
disk = 'D'
base_dir = f"{disk}:/Mediterranean CDHW"  # 根目录

# 差值栅格所在目录（按你现在路径结构）
diff_dir = f"{base_dir}/Results/HW_baselines/Annual_mean"

lc_file  = f"{base_dir}/RawData/LandCover/LC_mode_2011_2020_clip.tif"
dem_file = f"{base_dir}/RawData/DEM_01deg.tif"

# 网格参数
daily_statistics = ["daily_mean", "daily_maximum"]
percentile_vals = [90, 95, 97, 99]
baseline_pairs = [("1981_2020", "1981_2010"), ("1991_2020", "1981_2010")]

# elevation 分层
ele_bins   = [0, 500, 1000, 1500, 2000, 2500, 3500]
ele_labels = ["0_500","500_1000","1000_1500","1500_2000","2000_2500","2500_3500"]

# IGBP landcover名称（可根据需要删减）
IGBP = {
    0:"NoData",1:"ENF",2:"EBF",3:"DNF",4:"DBF",5:"MF",
    6:"Closed Shrub",7:"Open Shrub",8:"Woody Savanna",9:"Savanna",
    10:"Grass",11:"Wetland",12:"Cropland",13:"Urban",
    14:"Crop/Nat Mosaic",15:"Snow/Ice",16:"Barren",17:"Water"
}

# ================================
# 2. 读取 LC 和 DEM（只读一次）
# ================================
if not os.path.exists(lc_file):
    raise FileNotFoundError(f"LC file not found: {lc_file}")
if not os.path.exists(dem_file):
    raise FileNotFoundError(f"DEM file not found: {dem_file}")

with rasterio.open(lc_file) as src:
    LC = src.read(1)

with rasterio.open(dem_file) as src:
    DEM = src.read(1)

# elevation 分类（1..len(ele_bins)-1）
ele_cat = np.digitize(DEM, bins=ele_bins)
ele_types = range(1, len(ele_bins))

# LC 类型（排除0）
lc_types = sorted(np.unique(LC[LC > 0]))

# ================================
# 3. 主批处理流程（按 daily_statistic 分开输出）
# ================================
for daily_statistic in daily_statistics:

    print("\n" + "=" * 80)
    print(f"Processing daily_statistic: {daily_statistic}")
    print("=" * 80)

    rows = []

    # 三重循环生成期望文件名：p{p}_{A}_minus_{B}_{daily_statistic}_baseline.tif
    for p in percentile_vals:
        p_tag = f"p{p}"

        for (A, B) in baseline_pairs:
            fname = f"{p_tag}_{A}_minus_{B}_{daily_statistic}_baseline.tif"
            fpath = os.path.join(diff_dir, fname)

            if not os.path.exists(fpath):
                print(f"⚠ Missing, skip: {fname}")
                continue

            print(f"Reading: {fname}")

            with rasterio.open(fpath) as src:
                diff = src.read(1).astype(float)

            # 每个文件执行 LC × elevation 统计
            for lc_class in lc_types:
                lc_mask = (LC == lc_class)

                for e_class in ele_types:
                    mask = lc_mask & (ele_cat == e_class)
                    vals = diff[mask]
                    vals = vals[~np.isnan(vals)]

                    if vals.size == 0:
                        continue

                    rows.append({
                        "daily_statistic": daily_statistic,
                        "percentile": p_tag,
                        "baseline_A": A,
                        "baseline_B": B,
                        "filename": fname,
                        "LC_code": int(lc_class),
                        "LC_name": IGBP.get(int(lc_class), "Unknown"),
                        "Elevation": ele_labels[e_class - 1],
                        "count": int(vals.size),
                        "mean": float(np.mean(vals)),
                        "median": float(np.median(vals)),
                        "std": float(np.std(vals)),
                    })

    # ================================
    # 4. 保存结果
    # ================================
    out_csv = f"{diff_dir}/LC_Elevation_2D_stats_{daily_statistic}.csv"

    if len(rows) == 0:
        print(f"⚠ No results for {daily_statistic}. CSV not written.")
        continue

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"✔ Saved: {out_csv}")

print("\n✅ Done for all daily_statistics.")