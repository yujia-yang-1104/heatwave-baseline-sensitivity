# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 11:15:12 2025
@author: yujia.yang

功能：
- 对差值栅格（baseline diff）按 landcover 类别统计（count/mean/median/std）
- 支持 daily_statistic = daily_mean / daily_maximum 的循环嵌套
"""

# -*- coding: utf-8 -*-
import os
import numpy as np
import rasterio
import pandas as pd

# ================================
# 1) 路径与参数网格
# ================================
disk = 'D'
base_dir = f"{disk}:/Mediterranean CDHW"

# 差值 tif 所在目录（按你实际路径修改）
data_dir = f"{base_dir}/Results/HW_baselines/Annual_mean"

# landcover
LC_file = f"{base_dir}/RawData/LandCover/LC_mode_2011_2020_Clip.tif"

# 参数网格
percentile_vals = [90, 95, 97, 99]
daily_statistics = ["daily_mean", "daily_maximum"]

# baseline 差值组合（A - B）
baseline_pairs = [
    ("1981_2020", "1981_2010"),
    ("1991_2020", "1981_2010"),
]

# IGBP landcover 名称
IGBP = {
    0:"NoData",1:"ENF",2:"EBF",3:"DNF",4:"DBF",5:"MF",
    6:"Closed Shrub",7:"Open Shrub",8:"Woody Savanna",9:"Savanna",
    10:"Grass",11:"Wetland",12:"Cropland",13:"Urban",
    14:"Crop/Nat Mosaic",15:"Snow/Ice",16:"Barren",17:"Water"
}

# ================================
# 2) 读取 Landcover（一次即可）
# ================================
if not os.path.exists(LC_file):
    raise FileNotFoundError(f"Landcover file not found: {LC_file}")

with rasterio.open(LC_file) as src_lc:
    LC = src_lc.read(1)

unique_lc = sorted([u for u in np.unique(LC) if u != 0])

# ================================
# 3) 主循环：按 daily_statistic 输出 CSV
# ================================
for daily_statistic in daily_statistics:

    print("\n" + "=" * 80)
    print(f"Processing daily_statistic: {daily_statistic}")
    print("=" * 80)

    all_rows = []

    for p in percentile_vals:
        p_tag = f"p{p}"

        for (A, B) in baseline_pairs:

            fname = f"{p_tag}_{A}_minus_{B}_{daily_statistic}_baseline.tif"
            fpath = os.path.join(data_dir, fname)

            if not os.path.exists(fpath):
                print(f"⚠ Missing, skip: {fname}")
                continue

            print(f"Processing: {fname}")

            with rasterio.open(fpath) as src:
                diff = src.read(1).astype(float)

            # 按 LC 类别统计
            for cls in unique_lc:
                mask = (LC == cls)
                vals = diff[mask]
                vals = vals[~np.isnan(vals)]

                if vals.size == 0:
                    continue

                all_rows.append({
                    "daily_statistic": daily_statistic,
                    "percentile": p_tag,
                    "baseline_A": A,
                    "baseline_B": B,
                    "LC_code": int(cls),
                    "LC_name": IGBP.get(int(cls), "Unknown"),
                    "count": int(vals.size),
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "std": float(np.std(vals)),
                })

    # ================================
    # 4) 保存当前 daily_statistic 的 CSV
    # ================================
    if len(all_rows) == 0:
        print(f"⚠ No results for {daily_statistic}, CSV not written.")
        continue

    df_all = pd.DataFrame(all_rows)

    out_csv = f"{data_dir}/LC_stats_{daily_statistic}.csv"
    df_all.to_csv(out_csv, index=False)

    print(f"✔ Saved: {out_csv}")

print("\n✅ All daily_statistic processed.")
