# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 10:42:37 2025

@author: yujia.yang

Hotspot LC enrichment analysis (GRID MODE)
按参数网格 (daily_statistic × percentile × baseline_pair) 生成差值文件名并读取
文件命名模板：
p95_1981_2020_minus_1981_2010_daily_maximum_baseline.tif

输出（按 daily_statistic 分开）：
LC_hotspots_enrichment_daily_mean.csv
LC_hotspots_enrichment_daily_maximum.csv
"""

import os
import numpy as np
import pandas as pd
import rasterio

# =============== 参数区 ===============
disk = "D"
base_dir = f"{disk}:/Mediterranean CDHW"  # 根目录

diff_dir = f"{base_dir}/Results/HW_baselines/Annual_mean"

LC_file = f"{base_dir}/RawData/LandCover/LC_mode_2011_2020_clip.tif"

daily_statistics = ["daily_mean", "daily_maximum"]
percentile_vals = [90, 95, 97, 99]

# baseline 差值组合（A - B）
baseline_pairs = [
    ("1981_2020", "1981_2010"),
    ("1991_2020", "1981_2010"),
]

# 需要分析的 top X%
top_fracs = [0.05, 0.10]  # top 5%, top 10%

IGBP = {
    0: "NoData", 1: "ENF", 2: "EBF", 3: "DNF", 4: "DBF", 5: "MF",
    6: "Closed Shrub", 7: "Open Shrub", 8: "Woody Savanna", 9: "Savanna",
    10: "Grass", 11: "Wetland", 12: "Cropland", 13: "Urban",
    14: "Crop/Nat Mosaic", 15: "Snow/Ice", 16: "Barren", 17: "Water"
}

# 可靠性阈值
MIN_HOT_COUNT = 30
MIN_GLOBAL_FRAC = 0.01

# =============== 读取 LC 栅格（只读一次） ===============
if not os.path.exists(LC_file):
    raise FileNotFoundError(f"LC file not found: {LC_file}")

with rasterio.open(LC_file) as src_lc:
    LC = src_lc.read(1)

valid_lc_mask = (LC != 0) & (~np.isnan(LC))

# 全区域 LC 占比（enrichment 需要）
valid_lc_values = LC[valid_lc_mask]
unique_all, counts_all = np.unique(valid_lc_values, return_counts=True)
total_all = counts_all.sum()
global_frac = {int(c): cnt / total_all for c, cnt in zip(unique_all, counts_all)}
print("Global landcover fractions computed.")


# =============== 主循环：按 daily_statistic 分开输出 ===============
for daily_statistic in daily_statistics:

    print("\n" + "=" * 80)
    print(f"Daily statistic: {daily_statistic}")
    print("=" * 80)

    rows = []

    # 网格循环：p × baseline_pair
    for p in percentile_vals:
        p_tag = f"p{p}"

        for (A, B) in baseline_pairs:
            fname = f"{p_tag}_{A}_minus_{B}_{daily_statistic}_baseline.tif"
            diff_path = os.path.join(diff_dir, fname)

            if not os.path.exists(diff_path):
                print(f"⚠ Missing, skip: {fname}")
                continue

            print(f"Processing: {fname}")

            # 读取 diff 栅格
            with rasterio.open(diff_path) as src_diff:
                diff = src_diff.read(1).astype(float)

            # shape 检查
            if diff.shape != LC.shape:
                raise ValueError(f"Shape mismatch: diff {diff.shape} vs LC {LC.shape} | file={fname}")

            # diff 有效范围
            valid_mask = (~np.isnan(diff)) & valid_lc_mask
            diff_valid = diff[valid_mask]

            if diff_valid.size == 0:
                print("    !!! No valid diff pixels, skip.")
                continue

            for frac in top_fracs:
                # threshold for hotspot（取 top frac 最大值区域）
                th = np.quantile(diff_valid, 1 - frac)

                # hotspot mask
                hot_mask = valid_mask & (diff >= th)
                hot_lc = LC[hot_mask]

                if hot_lc.size == 0:
                    print(f"    !!! No hotspot pixels for top {frac*100:.1f}%, skip.")
                    continue

                unique_hot, counts_hot = np.unique(hot_lc, return_counts=True)
                total_hot = counts_hot.sum()

                for code, cnt_hot in zip(unique_hot, counts_hot):
                    code = int(code)
                    if code == 0:
                        continue

                    lc_name = IGBP.get(code, "Unknown")

                    frac_hot = cnt_hot / total_hot
                    frac_global = global_frac.get(code, 0.0)

                    enrichment_ratio = frac_hot / frac_global if frac_global > 0 else np.nan
                    diff_ratio = frac_hot - frac_global

                    reliable = (cnt_hot >= MIN_HOT_COUNT) and (frac_global >= MIN_GLOBAL_FRAC)

                    rows.append({
                        "daily_stat": daily_statistic,
                        "percentile": p_tag,
                        "baseline_A": A,
                        "baseline_B": B,
                        "top_frac": frac,
                        "LC_code": code,
                        "LC_name": lc_name,
                        "count_hot": int(cnt_hot),
                        "fraction_hotspot": float(frac_hot),
                        "fraction_global": float(frac_global),
                        "enrichment_ratio": float(enrichment_ratio) if np.isfinite(enrichment_ratio) else np.nan,
                        "difference": float(diff_ratio),
                        "reliable": bool(reliable),
                        "filename": fname,
                    })

    # 保存该 daily_statistic 的结果
    df = pd.DataFrame(rows)
    out_csv = os.path.join(diff_dir, f"LC_hotspots_enrichment_{daily_statistic}.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"Saved hotspot enrichment results to: {out_csv}")

print("\n✅ Done for all daily_statistics.")
