# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 14:27:54 2026

@author: yujia.yang
"""

import os
import numpy as np
import pandas as pd
import rioxarray as rxr

BASELINE_REF = "1981_2010"
BASELINES = ["1981_2020", "1991_2020"]

metrics = ["hw_count", "hw_max_duration", "hw_magnitude_index"]
percentiles = ["p90", "p95", "p99"]
data_types = ["daily_maximum", "daily_mean"]

CLIM_TIF_DIR = r"D:\Mediterranean CDHW\Results\HW_metrics_climatology"
KOPPEN_TIF = r"D:\Mediterranean CDHW\RawData\Climatezone\koppen_geiger_1991_2020_01deg.tif"

EPS = 1e-8
MIN_REF = 1e-6
MIN_PIXELS = 50

def zonal_mean_np(val2d, zone2d):
    valid = np.isfinite(val2d) & np.isfinite(zone2d)
    v = val2d[valid]
    z = zone2d[valid].astype(int)
    df = pd.DataFrame({"zone": z, "val": v})
    out = df.groupby("zone")["val"].agg(["mean", "count"]).reset_index()
    out = out.rename(columns={"count": "n", "mean": "mean"})
    return out

def open_and_match(path, target):
    da = rxr.open_rasterio(path).squeeze(drop=True)
    if da.rio.crs is None and target.rio.crs is not None:
        da = da.rio.write_crs(target.rio.crs)
    return da.rio.reproject_match(target, resampling=rxr.rio.enums.Resampling.nearest)


kop = rxr.open_rasterio(KOPPEN_TIF).squeeze(drop=True)
for baseline in BASELINES:
    for dtype in data_types:
        rows = []
        for metric in metrics:
            for p in percentiles:
                # 按真实命名修改下面两行 pattern
                ref_path = os.path.join(CLIM_TIF_DIR, f"HW_climatology_{p}_{BASELINE_REF}_{
                                    dtype}\climatology_{metric}_based_on_{p}_{BASELINE_REF}.tif")
                diff_path = os.path.join(CLIM_TIF_DIR, f"HW_climatology_{p}_{BASELINE_REF}_{
                                     dtype}\diff_climatology_{metric}_{p}_{baseline}_minus_{BASELINE_REF}.tif")
                if not (os.path.exists(ref_path) and os.path.exists(diff_path)):
                    print(f"[SKIP] missing: {ref_path} or {diff_path}")
                    continue
                ref = rxr.open_rasterio(ref_path).squeeze(drop=True)
                diff = rxr.open_rasterio(diff_path).squeeze(drop=True)

                ref_stats = zonal_mean_np(ref.values, kop.values).rename(
                    columns={"mean": "ref_mean", "n": "n_ref"})
                diff_stats = zonal_mean_np(diff.values, kop.values).rename(
                    columns={"mean": "diff_mean", "n": "n_diff"})

                out = ref_stats.merge(diff_stats, on="zone", how="outer")
                # 过滤极小样本
                out = out[(out["n_ref"] >= MIN_PIXELS) &
                          (out["n_diff"] >= MIN_PIXELS)]
                out["distortion_ratio"] = np.where(out["ref_mean"] > MIN_REF,
                                                   out["diff_mean"] /
                                                   (out["ref_mean"] + EPS),
                                                   np.nan)
                
                out["metric"] = metric
                out["percentile"] = p
                out["data_type"] = dtype
                out["ref_baseline"] = BASELINE_REF
                out["alt_baseline"] = baseline
                rows.append(out)
        result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        out_csv = f"D:\Mediterranean CDHW\Results\Koppen_stats\koppen_distortion_ratio_{
            dtype}_{baseline}.csv"
        result.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")
