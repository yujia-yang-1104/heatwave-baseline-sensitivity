import os
import xarray as xr
import numpy as np
import rioxarray  # pip install rioxarray

# ===============================
# 参数配置
# ===============================
disk = "D"
base_dir = f"{disk}:/Mediterranean CDHW"

percentile_vals = [90, 95, 97, 99]
daily_statistics = ["daily_mean", "daily_maximum"]
baseline_years_list = [
    range(1981, 2011),   # 1981–2010
    range(1981, 2021),   # 1981–2020
    range(1991, 2021),   # 1991–2020
]
target_years = range(1981, 2026)  # 更新t2m至2025年
variables = [
    "hw_count",
    "hw_days",
    "hw_max_duration",
    "hw_intensity_mean",
    "hw_intensity_max",
    "hw_cumulative_intensity",
    "hw_magnitude_index",
]

# ===============================
# 工具函数：统一经纬度命名
# ===============================
def _normalize_lonlat(obj):
    rename_map = {}
    if "lon" in obj.dims:
        rename_map["lon"] = "longitude"
    if "lat" in obj.dims:
        rename_map["lat"] = "latitude"
    if rename_map:
        obj = obj.rename(rename_map)
    return obj

def compute_and_save_climatology_diffs(
    base_dir: str,
    daily_statistic: str,
    percentile_val: int,
    variables: list[str],
    ref_period: tuple[int, int] = (1981, 2010),
    compare_periods: list[tuple[int, int]] = [(1981, 2020), (1991, 2020)],
):
    """
    读取已生成的 climatology_{var}_based_on_p{p}_{start}_{end}.tif，
    计算 compare - ref 的差值并输出 tif。

    输出目录模板（按你给的风格）：
    D:\\Mediterranean CDHW\\Results\\HW_metrics_climatology\\HW_climatology_p90_1981_2010_daily_maximum\\
    """

    ref_start, ref_end = ref_period

    out_dir = (
        f"{base_dir}/Results/HW_metrics_climatology/"
        f"HW_climatology_p{percentile_val}_{ref_start}_{ref_end}_{daily_statistic}"
    )
    os.makedirs(out_dir, exist_ok=True)

    for var in variables:
        # 参考基线 climatology tif
        ref_tif = f"{out_dir}/climatology_{var}_based_on_p{percentile_val}_{ref_start}_{ref_end}.tif"
        if not os.path.exists(ref_tif):
            print(f"⚠ Missing reference climatology tif, skip var={var}: {ref_tif}")
            continue

        ref_da = rioxarray.open_rasterio(ref_tif).squeeze(drop=True)

        for (c_start, c_end) in compare_periods:
            # 被比较基线所在目录（注意：它的目录名应是 p{p}_{c_start}_{c_end}_{daily_statistic}）
            comp_dir = (
                f"{base_dir}/Results/HW_metrics_climatology/"
                f"HW_climatology_p{percentile_val}_{c_start}_{c_end}_{daily_statistic}"
            )
            comp_tif = f"{comp_dir}/climatology_{var}_based_on_p{percentile_val}_{c_start}_{c_end}.tif"

            if not os.path.exists(comp_tif):
                print(f"⚠ Missing comparison climatology tif, skip: {comp_tif}")
                continue

            comp_da = rioxarray.open_rasterio(comp_tif).squeeze(drop=True)

            # 对齐（确保网格一致）
            comp_da, ref_da_aligned = xr.align(comp_da, ref_da, join="exact")

            diff = comp_da - ref_da_aligned

            # 输出差值到“参考基线目录”（按你的模板风格）
            diff_tif = (
                f"{out_dir}/diff_climatology_{var}_p{percentile_val}_"
                f"{c_start}_{c_end}_minus_{ref_start}_{ref_end}.tif"
            )

            # 保留 CRS 与 transform
            diff = diff.rio.write_crs(ref_da.rio.crs, inplace=False)

            diff.rio.to_raster(diff_tif)
            print(f"✔ Saved diff: {diff_tif}")

# ===============================
# 主循环：p × daily_statistic × baseline
# ===============================
print("=== 开始批量处理：多年序列拼接 + climatology ===")

for percentile_val in percentile_vals:
    for daily_statistic in daily_statistics:
        for baseline_years in baseline_years_list:

            b_start = min(baseline_years)
            b_end = max(baseline_years)
            y_start = min(target_years)
            y_end = max(target_years)

            in_dir = (
                f"{base_dir}/Results/"
                f"HW_p{percentile_val}_{b_start}_{b_end}_{daily_statistic}"
            )

            # climatology 输出目录（推荐按组合分层，避免不同baseline互相覆盖）
            clim_out_dir = (
                f"{base_dir}/Results/HW_metrics_climatology/"
                f"HW_climatology_p{percentile_val}_{b_start}_{b_end}_{daily_statistic}"
            )
            os.makedirs(clim_out_dir, exist_ok=True)

            print(
                f"\n>>> 组合: P{percentile_val} | {daily_statistic} | baseline {b_start}-{b_end}"
            )
            print("Input dir:", in_dir)

            # 逐变量处理：拼多年序列 + climatology
            for var in variables:
                print(f"  - 变量: {var}")

                data_list = []
                used_years = []

                # 1) 拼接多年序列
                for year in target_years:
                    f = (
                        f"{in_dir}/"
                        f"hw_index_for_{year}_based_on_{b_start}_{b_end}.nc"
                    )

                    if not os.path.exists(f):
                        # 缺文件就跳过该年份
                        continue

                    try:
                        da = xr.open_dataset(f)[var]
                        da = _normalize_lonlat(da)

                        # 保证 time 维：每年一层
                        if "time" not in da.dims:
                            da = da.expand_dims(time=[np.datetime64(f"{year}-07-01")])

                        data_list.append(da)
                        used_years.append(year)
                    except KeyError:
                        # 该文件没有这个变量
                        continue

                if len(data_list) == 0:
                    print(f"    ⚠ 没有可用年份数据，跳过 {var}")
                    continue

                HW = xr.concat(data_list, dim="time")

                out_nc = (
                    f"{in_dir}/"
                    f"{var}_for_{y_start}_{y_end}_based_on_{b_start}_{b_end}.nc"
                )
                HW.to_netcdf(out_nc, format="NETCDF4")
                print(f"    ✔ 多年序列NC: {os.path.basename(out_nc)}  (years={len(used_years)})")

                # 2) climatology（多年平均）
                clim = HW.mean(dim="time", skipna=True)
                clim = _normalize_lonlat(clim)

                # rioxarray 输出准备
                if "longitude" in clim.dims and "latitude" in clim.dims:
                    clim = clim.rio.set_spatial_dims(
                        x_dim="longitude", y_dim="latitude", inplace=False
                    )

                if clim.rio.crs is None:
                    clim = clim.rio.write_crs("EPSG:4326")

                # out_tif = (
                #     f"{clim_out_dir}/"
                #     f"climatology_{var}_based_on_p{percentile_val}_{b_start}_{b_end}.tif"
                # )
                # clim.rio.to_raster(out_tif)

                # # （可选）输出 climatology nc，后面做差值/统计会更方便
                # out_clim_nc = (
                #     f"{clim_out_dir}/"
                #     f"climatology_{var}_based_on_p{percentile_val}_{b_start}_{b_end}.nc"
                # )
                # clim.to_netcdf(out_clim_nc, format="NETCDF4")

                # print(f"    ✔ climatology TIF: {os.path.basename(out_tif)}")
                
                compute_and_save_climatology_diffs(
                    base_dir=base_dir,
                    daily_statistic=daily_statistic,
                    percentile_val=percentile_val,
                    variables=variables,
                    ref_period=(1981, 2010),
                    compare_periods=[(1981, 2020), (1991, 2020)]
                    )

print("\n🎉 全部组合处理完成！")
