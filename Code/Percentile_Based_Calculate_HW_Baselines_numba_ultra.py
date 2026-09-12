# -*- coding: utf-8 -*-
"""
Created on Mon May  5 10:20:23 2025
@author: Yujia
Ultra-optimized version using Numba JIT compilation
使用Numba进行JIT编译（并行）计算滑动窗口百分位阈值
并支持：
- 3种 baseline_years
- 4种 percentile_vals
- 2种 daily_statistics
循环嵌套
"""
import xarray as xr
import numpy as np
from numba import jit, prange
import time
import os
import warnings
warnings.filterwarnings('ignore')

# =========================
# 全局配置
# =========================
disk = "D"
base_dir = f"{disk}:/Mediterranean CDHW"
var_name = "t2m"
window_size = 15
half_window = window_size // 2

# === baseline 时间窗（3 种） ===
baseline_years_list = [
    range(1981, 2011),   # 1981–2010
    range(1981, 2021),   # 1981–2020
    range(1991, 2021)    # 1991–2020
]

# === 百分位阈值（4 种） ===
percentile_vals = [90, 95, 97, 99]

# === 日统计量（2 种） ===
daily_statistics = ["daily_mean", "daily_maximum"]


# =========================
# Numba JIT 函数
# =========================
@jit(nopython=True, parallel=True, fastmath=True)
def compute_percentiles_numba(data, doy, window_size, percentile_val):
    """
    使用Numba JIT编译的超快速百分位数计算
    data: (time, lat, lon)
    doy:  (time,)
    """
    n_time, n_lat, n_lon = data.shape
    n_days = 365
    half_window = window_size // 2

    result = np.empty((n_days, n_lat, n_lon), dtype=data.dtype)

    for day in prange(1, n_days + 1):
        # 计算窗口 dayofyear
        window_days = np.empty(window_size, dtype=np.int32)
        for i in range(window_size):
            wd = (day + i - half_window - 1) % n_days + 1
            window_days[i] = wd

        # 找到属于窗口的所有时间索引
        # 注意：Numba 不支持 python list append 的高效方式，这里仍可用 list，但可能慢。
        # 为保持你“最快版本”的结构，沿用原逻辑。
        mask_indices = []
        for t in range(n_time):
            for k in range(window_size):
                if doy[t] == window_days[k]:
                    mask_indices.append(t)
                    break

        n_window = len(mask_indices)

        for i in range(n_lat):
            for j in range(n_lon):
                window_data = np.empty(n_window, dtype=data.dtype)
                for k in range(n_window):
                    t_idx = mask_indices[k]
                    window_data[k] = data[t_idx, i, j]

                result[day - 1, i, j] = np.percentile(window_data, percentile_val)

    return result


# =========================
# 主循环：daily_statistic × baseline × percentile
# =========================
if __name__ == "__main__":

    total_start = time.time()

    for daily_statistic in daily_statistics:
        for baseline_years in baseline_years_list:
            b_start = min(baseline_years)
            b_end = max(baseline_years)

            print("\n" + "=" * 80)
            print("超优化版本 - Numba JIT 编译")
            print(f"处理 daily_statistic: {daily_statistic}")
            print(f"基准期 baseline: {b_start}-{b_end}")
            print(f"percentiles: {percentile_vals}")
            print("=" * 80)

            # === 读取 baseline 数据（该组合下，只读一次，给不同percentile复用）===
            print("\n[1/4] 读取数据...")
            baseline_files = [
                f"{base_dir}/RawData/t2m_{daily_statistic}_era5_land/era5_data_2m_temperature_{year}.nc"
                for year in baseline_years
            ]

            # 若缺文件，直接报出来（你也可以改成跳过）
            missing = [f for f in baseline_files if not os.path.exists(f)]
            if missing:
                print("⚠ 缺少 baseline 文件，跳过该 baseline 组合。缺失示例：")
                print("  ", missing[0])
                continue

            ds_baseline = xr.open_mfdataset(
                baseline_files,
                combine="by_coords",
                parallel=True
            )

            temp = ds_baseline[var_name]
            if "valid_time" in temp.dims:
                temp = temp.rename({"valid_time": "time"})
            temp_c = temp - 273.15

            # 剔除2月29日
            time_index = temp_c["time"].to_index()
            temp_c = temp_c.sel(time=~((time_index.month == 2) & (time_index.day == 29)))

            # 添加 dayofyear
            doy = temp_c["time"].dt.dayofyear.values
            temp_c = temp_c.assign_coords(dayofyear=("time", doy))

            print(f"  数据形状: {temp_c.shape}")
            try:
                print(f"  数据大小: {temp_c.nbytes / 1e9:.2f} GB")
            except Exception:
                pass

            # === 加载数据到内存（该 baseline 组合下，只加载一次）===
            print("\n[2/4] 加载数据到内存...")
            load_start = time.time()

            data_array = temp_c.values  # (time, lat, lon)
            doy_array = temp_c.dayofyear.values

            # 坐标命名兼容
            if "lat" in temp_c.coords:
                lat = temp_c.lat.values
            else:
                lat = temp_c.latitude.values

            if "lon" in temp_c.coords:
                lon = temp_c.lon.values
            else:
                lon = temp_c.longitude.values

            print(f"  耗时: {time.time() - load_start:.2f} 秒")

            # 尽快关闭 dataset 句柄，避免文件打开过多（data_array 已在内存）
            try:
                ds_baseline.close()
            except Exception:
                pass

            # === 对不同 percentile 循环计算并保存 ===
            for percentile_val in percentile_vals:
                print("\n" + "-" * 80)
                print(f"[3/4] 计算百分位数: P{percentile_val} （Numba JIT + 并行）")
                print("  首次运行会触发JIT编译，可能更慢；后续会明显加速。")
                start_time = time.time()

                percentiles_array = compute_percentiles_numba(
                    data_array,
                    doy_array,
                    window_size,
                    percentile_val
                )

                elapsed_time = time.time() - start_time
                print(f"  计算完成！耗时: {elapsed_time:.2f} 秒")
                print(f"  平均每天: {elapsed_time / 365:.3f} 秒")

                # === 构建输出 DataArray ===
                print("[4/4] 保存结果...")
                p_all_days = xr.DataArray(
                    percentiles_array,
                    coords={
                        "dayofyear": np.arange(1, 366),
                        "lat": lat,
                        "lon": lon
                    },
                    dims=["dayofyear", "lat", "lon"],
                    name=f"{var_name}_p{percentile_val}"
                )

                # 属性
                p_all_days.attrs["units"] = "degC"
                p_all_days.attrs["description"] = (
                    f"{percentile_val}th percentile of {window_size}-day window {daily_statistic} "
                    f"temperature from {b_start} to {b_end}"
                )
                p_all_days.attrs["window_size"] = window_size
                p_all_days.attrs["baseline_years"] = f"{b_start}-{b_end}"
                p_all_days.attrs["daily_statistic"] = daily_statistic
                p_all_days.attrs["computation_method"] = "Numba JIT parallel"

                # 输出路径（与之前保持一致）
                output_dir = f"{base_dir}/Results/HW_baselines/"
                os.makedirs(output_dir, exist_ok=True)
                output_file = f"{output_dir}/p{percentile_val}_{b_start}_{b_end}_{daily_statistic}_baseline.nc"

                encoding = {
                    f"{var_name}_p{percentile_val}": {
                        "zlib": True,
                        "complevel": 4,
                        "dtype": "float32"
                    }
                }

                # 写 netcdf（单变量）
                p_all_days.to_netcdf(output_file, encoding=encoding)

                file_size = os.path.getsize(output_file) / 1e6
                print(f"  ✔ 保存完成: {output_file}")
                print(f"  文件大小: {file_size:.2f} MB")

            print("\n✅ 本 baseline 组合完成："
                  f"{daily_statistic} | {b_start}-{b_end} | {percentile_vals}")

    print("\n" + "=" * 80)
    print("全部组合完成！")
    print(f"总耗时: {time.time() - total_start:.2f} 秒")
    print("=" * 80)