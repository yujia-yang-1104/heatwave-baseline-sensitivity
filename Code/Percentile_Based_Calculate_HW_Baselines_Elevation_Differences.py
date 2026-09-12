# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 10:03:38 2025

@author: yujia.yang
"""

import os
import numpy as np
import rasterio
import pandas as pd

# =========================
# 0. 参数配置（循环网格）
# =========================
disk = 'D'
base_dir = f"{disk}:/Mediterranean CDHW"

# 3种 baseline（注意：生成“差值文件名”中的 A，默认都是 A minus 1981_2010）
baseline_years_list = [
    range(1981, 2021),  # 1981–2020
    range(1991, 2021),  # 1991–2020
]

percentile_vals = [90, 95, 97, 99]
daily_statistics = ["daily_mean", "daily_maximum"]

# 固定参考基线（按你文件名：*_minus_1981_2010_*)
ref_baseline_tag = "1981_2010"

# DEM 与差值栅格所在目录
dem_path = f"{base_dir}/RawData/DEM_01deg.tif"
diff_dir = f"{base_dir}/Results/HW_baselines/Annual_mean"

# 输出目录
out_dir = diff_dir
os.makedirs(out_dir, exist_ok=True)

# =========================
# 1. 读取 DEM
# =========================
with rasterio.open(dem_path) as dem_src:
    dem = dem_src.read(1)
    dem_profile = dem_src.profile

dem = np.where((dem == dem_profile.get('nodata')) | np.isnan(dem), np.nan, dem)

# =========================
# 2. 海拔分区（保持原逻辑：最后区间合并至2500-3500）
# =========================
elev_bins = np.array([0, 500, 1000, 1500, 2000, 2500, 3500])
elev_labels = [f"{int(elev_bins[i])}-{int(elev_bins[i+1])}" for i in range(len(elev_bins)-1)]

max_dem = np.nanmax(dem)
if max_dem > 3500:
    additional_bins = np.arange(4000, max_dem + 500, 500)
    elev_bins = np.concatenate([elev_bins, additional_bins])
    elev_labels = [f"{int(elev_bins[i])}-{int(elev_bins[i+1])}" for i in range(len(elev_bins)-1)]

print(f"海拔分区设置: {elev_bins}")
print(f"分区标签: {elev_labels}")

dem_class = np.digitize(dem, elev_bins) - 1  # 从0开始编号

# =========================
# 3. 三重循环：baseline × percentile × daily_statistic
# =========================
for daily_statistic in daily_statistics:
    for percentile_val in percentile_vals:
        for baseline_years in baseline_years_list:

            a_start = min(baseline_years)
            a_end = max(baseline_years)
            A_tag = f"{a_start}_{a_end}"

            # 差值文件命名模板（按你前面统一的模板）
            # p99_1991_2020_minus_1981_2010_daily_mean_baseline.tif
            attr_path = (
                f"{diff_dir}/"
                f"p{percentile_val}_{A_tag}_minus_{ref_baseline_tag}_{daily_statistic}_baseline.tif"
            )

            if not os.path.exists(attr_path):
                print(f"⚠ Missing diff raster, skip: {os.path.basename(attr_path)}")
                continue

            print("\n" + "=" * 80)
            print(f"Processing: P{percentile_val} | {daily_statistic} | {A_tag} minus {ref_baseline_tag}")
            print("=" * 80)

            # ---------- 3.1 读取差值栅格 ----------
            with rasterio.open(attr_path) as attr_src:
                attr = attr_src.read(1)
                attr_profile = attr_src.profile

            attr = np.where((attr == attr_profile.get('nodata')) | np.isnan(attr), np.nan, attr)

            # ---------- 3.2 按海拔区间分组（像元级 long table） ----------
            data = []
            for i, label in enumerate(elev_labels):
                mask = (dem_class == i)
                values = attr[mask]
                values = values[~np.isnan(values)]
                pixel_count = len(values)

                if pixel_count > 0:
                    data.extend([(label, v) for v in values])
                    print(f"{label}: {pixel_count} 个像元")

            if len(data) == 0:
                print("⚠ No valid pixels after masking; skip outputs.")
                continue

            df = pd.DataFrame(data, columns=["Elevation Zone", "Value"])
            df["Elevation Zone"] = pd.Categorical(
                df["Elevation Zone"],
                categories=elev_labels,
                ordered=True
            )

            # 像元级输出
            out_long = (
                f"{out_dir}/"
                f"Elevation_p{percentile_val}_{A_tag}_minus_{ref_baseline_tag}_{daily_statistic}.csv"
            )
            df.to_csv(out_long, index=False, encoding="utf-8-sig")

            # ---------- 3.3 分区 summary ----------
            summary = (
                df.groupby("Elevation Zone")["Value"]
                .agg(["count", "mean", "std", "median", "min", "max"])
                .reset_index()
            )

            out_sum = (
                f"{out_dir}/"
                f"Elevation_stats_p{percentile_val}_{A_tag}_minus_{ref_baseline_tag}_{daily_statistic}.csv"
            )
            summary.to_csv(out_sum, index=False, encoding="utf-8-sig")

            print(f"✅ Saved long table:   {os.path.basename(out_long)}")
            print(f"✅ Saved summary stats:{os.path.basename(out_sum)}")
            print(summary)

print("\n✅ All combinations finished.")