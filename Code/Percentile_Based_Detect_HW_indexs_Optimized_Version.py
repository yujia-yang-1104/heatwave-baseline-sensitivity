# -*- coding: utf-8 -*-
"""
优化版热浪检测代码
主要优化：
1. 向量化计算替代逐像元循环
2. 多进程并行处理年份
3. 优化数据I/O和内存使用
4. 使用numba加速关键计算

@author: Yujia (Optimized)
"""

import xarray as xr
import numpy as np
from scipy.ndimage import label
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: numba not available. Install it for better performance: pip install numba")


# 使用numba加速像元级别的热浪统计
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def compute_pixel_stats_vectorized(exceed_array, exceed_temp_array, threshold_days, n_lat, n_lon):
        """向量化计算所有像元的热浪统计指标"""
        hw_count = np.zeros((n_lat, n_lon))
        hw_days = np.zeros((n_lat, n_lon))
        hw_max_duration = np.zeros((n_lat, n_lon))
        hw_intensity_mean = np.zeros((n_lat, n_lon))
        hw_intensity_max = np.zeros((n_lat, n_lon))
        hw_cumulative_intensity = np.zeros((n_lat, n_lon))
        hw_magnitude_index = np.zeros((n_lat, n_lon))
        
        for i in prange(n_lat):
            for j in range(n_lon):
                e_bin = exceed_array[:, i, j]
                e_temp = exceed_temp_array[:, i, j]
                
                # 使用numpy数组替代列表
                max_events = 100  # 假设最多100个事件
                durations = np.zeros(max_events, dtype=np.int32)
                magnitudes = np.zeros(max_events, dtype=np.float64)
                event_count = 0
                
                all_excess = np.zeros(365, dtype=np.float64)  # 最多365天
                excess_count = 0
                
                cumulative = 0.0
                
                in_event = False
                current_duration = 0
                current_sum = 0.0
                
                for day in range(len(e_bin)):
                    if e_bin[day]:
                        if not in_event:
                            in_event = True
                            current_duration = 1
                            current_sum = e_temp[day]
                        else:
                            current_duration += 1
                            current_sum += e_temp[day]
                    else:
                        if in_event:
                            # 事件结束
                            if current_duration >= threshold_days:
                                if event_count < max_events:
                                    durations[event_count] = current_duration
                                    magnitudes[event_count] = current_sum
                                    event_count += 1
                                cumulative += current_sum
                                # 记录所有超出温度值
                                start_day = day - current_duration
                                for k in range(current_duration):
                                    if excess_count < 365:
                                        all_excess[excess_count] = e_temp[start_day + k]
                                        excess_count += 1
                            in_event = False
                            current_duration = 0
                            current_sum = 0.0
                
                # 处理最后一个事件
                if in_event and current_duration >= threshold_days:
                    if event_count < max_events:
                        durations[event_count] = current_duration
                        magnitudes[event_count] = current_sum
                        event_count += 1
                    cumulative += current_sum
                    start_day = len(e_bin) - current_duration
                    for k in range(current_duration):
                        if excess_count < 365:
                            all_excess[excess_count] = e_temp[start_day + k]
                            excess_count += 1
                
                if event_count > 0:
                    hw_count[i, j] = event_count
                    hw_days[i, j] = np.sum(durations[:event_count])
                    hw_max_duration[i, j] = np.max(durations[:event_count])
                    hw_cumulative_intensity[i, j] = cumulative
                    hw_intensity_mean[i, j] = np.sum(all_excess[:excess_count]) / excess_count
                    hw_intensity_max[i, j] = np.max(all_excess[:excess_count])
                    hw_magnitude_index[i, j] = np.max(magnitudes[:event_count])
        
        return (hw_count, hw_days, hw_max_duration, hw_intensity_mean, 
                hw_intensity_max, hw_cumulative_intensity, hw_magnitude_index)
else:
    def compute_pixel_stats_vectorized(exceed_array, exceed_temp_array, threshold_days, n_lat, n_lon):
        """不使用numba的备用版本"""
        hw_count = np.zeros((n_lat, n_lon))
        hw_days = np.zeros((n_lat, n_lon))
        hw_max_duration = np.zeros((n_lat, n_lon))
        hw_intensity_mean = np.zeros((n_lat, n_lon))
        hw_intensity_max = np.zeros((n_lat, n_lon))
        hw_cumulative_intensity = np.zeros((n_lat, n_lon))
        hw_magnitude_index = np.zeros((n_lat, n_lon))
        
        for i in tqdm(range(n_lat), desc="Processing pixels", leave=False):
            for j in range(n_lon):
                e_bin = exceed_array[:, i, j]
                e_temp = exceed_temp_array[:, i, j]
                
                labeled, n_events = label(e_bin)
                durations = []
                magnitudes = []
                cumulative = 0
                all_excess = []
                
                for k in range(1, n_events + 1):
                    idx = (labeled == k)
                    duration = np.sum(idx)
                    if duration >= threshold_days:
                        excess = e_temp[idx]
                        durations.append(duration)
                        magnitudes.append(np.sum(excess))
                        cumulative += np.sum(excess)
                        all_excess.extend(excess)
                
                if durations:
                    hw_count[i, j] = len(durations)
                    hw_days[i, j] = sum(durations)
                    hw_max_duration[i, j] = max(durations)
                    hw_cumulative_intensity[i, j] = cumulative
                    hw_intensity_mean[i, j] = np.mean(all_excess)
                    hw_intensity_max[i, j] = np.max(all_excess)
                    hw_magnitude_index[i, j] = np.max(magnitudes)
        
        return (hw_count, hw_days, hw_max_duration, hw_intensity_mean, 
                hw_intensity_max, hw_cumulative_intensity, hw_magnitude_index)


def process_single_year(args):
    """处理单个年份的热浪检测（用于并行处理）
    
    使用args元组避免传递大型xarray对象
    """
    y, temp_files, p_file, percentage, threshold_days, baseline_years, base_dir, daily_statistic, var_name = args
    
    # 在子进程中重新加载数据（避免序列化大对象）
    p = xr.open_dataset(p_file)
    p_baseline = p[f"t2m_p{percentage}"]
    
    # 只加载当前年份的数据
    temp_file = [f for f in temp_files if f"{y}.nc" in f][0]
    ds_temp = xr.open_dataset(temp_file)
    temp = ds_temp[var_name]
    
    if 'valid_time' in temp.dims:
        temp = temp.rename({'valid_time': 'time'})
    
    # 转换为摄氏度
    temp_c = temp - 273.15
    
    # 剔除2月29日
    time_idx = temp_c['time'].to_index()
    temp_c = temp_c.sel(time=~((time_idx.month == 2) & (time_idx.day == 29)))
    
    # 加载到内存
    temp_y = temp_c.load()
    p_baseline = p_baseline.load()
    
    lat = temp_y.latitude
    lon = temp_y.longitude
    
    # 向量化计算超出阈值
    exceed_year = []
    exceed_temp_year = []
    
    for day in range(365):
        exceed = temp_y[day, :, :] > p_baseline[day, :, :]
        exceed_temp = xr.where(exceed, temp_y[day, :, :] - p_baseline[day, :, :], 0.0)
        exceed_year.append(exceed.values)
        exceed_temp_year.append(exceed_temp.values)
    
    exceed_array = np.stack(exceed_year, axis=0)
    exceed_temp_array = np.stack(exceed_temp_year, axis=0)
    
    # 保存exceed标记（使用压缩以减少文件大小）
    da_exceed = xr.DataArray(
        exceed_year,
        name="exceed_flag",
        dims=["time", "latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon}
    )
    da_exceed.attrs["year"] = y
    
    # 将DataArray转换为Dataset以使用encoding
    ds_exceed = da_exceed.to_dataset()
    encoding_exceed = {'exceed_flag': {'zlib': True, 'complevel': 4}}
    ds_exceed.to_netcdf(
        f"{base_dir}/Results/HW_p{percentage}_{min(baseline_years)}_{max(baseline_years)}_{daily_statistic}/exceed_for_{y}_based_on_{min(baseline_years)}_{max(baseline_years)}.nc", 
        format="NETCDF4", encoding=encoding_exceed
    )
    
    da_exceed_temp = xr.DataArray(
        exceed_temp_year,
        name="exceed_temp_flag",
        dims=["time", "latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon}
    )
    da_exceed_temp.attrs["year"] = y
    
    ds_exceed_temp = da_exceed_temp.to_dataset()
    encoding_exceed_temp = {'exceed_temp_flag': {'zlib': True, 'complevel': 4}}
    ds_exceed_temp.to_netcdf(
        f"{base_dir}/Results/HW_p{percentage}_{min(baseline_years)}_{max(baseline_years)}_{daily_statistic}/exceed_temp_for_{y}_based_on_{min(baseline_years)}_{max(baseline_years)}.nc", 
        format="NETCDF4", encoding=encoding_exceed_temp
    )
    
    # 使用优化的向量化函数计算统计指标
    n_lat, n_lon = len(lat), len(lon)
    results = compute_pixel_stats_vectorized(
        exceed_array, exceed_temp_array, threshold_days, n_lat, n_lon
    )
    
    hw_count, hw_days, hw_max_duration, hw_intensity_mean, \
    hw_intensity_max, hw_cumulative_intensity, hw_magnitude_index = results
    
    # 组合各项指标为一个Dataset
    ds = xr.Dataset(
        {
            "hw_count": (["latitude", "longitude"], hw_count),
            "hw_days": (["latitude", "longitude"], hw_days),
            "hw_max_duration": (["latitude", "longitude"], hw_max_duration),
            "hw_intensity_mean": (["latitude", "longitude"], hw_intensity_mean),
            "hw_intensity_max": (["latitude", "longitude"], hw_intensity_max),
            "hw_cumulative_intensity": (["latitude", "longitude"], hw_cumulative_intensity),
            "hw_magnitude_index": (["latitude", "longitude"], hw_magnitude_index),
        },
        coords={"latitude": lat, "longitude": lon},
    )
    
    ds.attrs["year"] = y
    
    # 使用压缩保存
    encoding = {var: {'zlib': True, 'complevel': 4} for var in ds.data_vars}
    ds.to_netcdf(
        f"{base_dir}/Results/HW_p{percentage}_{min(baseline_years)}_{max(baseline_years)}_{daily_statistic}/hw_index_for_{y}_based_on_{min(baseline_years)}_{max(baseline_years)}.nc",
        format="NETCDF4", encoding=encoding
    )
    
    # 清理内存
    ds_temp.close()
    p.close()
    
    return y


def detect_heatwaves_parallel(temp_files, p_file, percentage, threshold_days, baseline_years, base_dir, daily_statistic, var_name, target_years, n_processes=None):
    """
    并行处理多年份的热浪检测
    
    Parameters:
    -----------
    temp_files : list
        温度数据文件列表
    p_file : str
        基线文件路径
    n_processes : int, optional
        并行进程数。None则使用CPU核心数-2
    """
    
    if n_processes is None:
        # 保留2个核心给系统，Ultra 9 285有24核心
        n_processes = max(1, cpu_count() - 2)
    
    years = list(target_years)
    print(f"使用 {n_processes} 个进程并行处理 {len(years)} 个年份")
    
    # 准备参数列表（传递文件路径而不是数据对象）
    args_list = [
        (y, temp_files, p_file, percentage, threshold_days, baseline_years, base_dir, daily_statistic, var_name)
        for y in years
    ]
    
    # 使用进程池并行处理
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_year, args_list),
            total=len(years),
            desc="处理年份"
        ))
    
    print(f"完成！处理了 {len(results)} 个年份")
    return results


# === 参数配置 ===
if __name__ == "__main__":
    disk = 'D'
    base_dir = f"{disk}:/Mediterranean CDHW"

    # === baseline 时间窗（3 种） ===
    baseline_years_list = [
        range(1981, 2011),   # 1981–2010
        range(1981, 2021),   # 1981–2020
        range(1991, 2021)    # 1991–2020
    ]

    # === 百分位阈值（3 种） ===
    percentile_vals = [90, 95, 97, 99]

    # === 日统计量（2 种） ===
    daily_statistics = ["daily_mean", "daily_maximum"]

    # === 目标年份 ===
    target_years = range(1981, 2026) # 更新t2m至2025年

    var_name = "t2m"
    threshold_days = 3

    # 并行进程数（None = 自动）
    n_processes = None

    print("开始批量热浪检测任务...")

    # === 三重循环 ===
    for percentile_val in percentile_vals:
        for daily_statistic in daily_statistics:
            for baseline_years in baseline_years_list:

                b_start = min(baseline_years)
                b_end   = max(baseline_years)

                print(
                    f"\n>>> 正在处理: "
                    f"P{percentile_val}, "
                    f"{daily_statistic}, "
                    f"baseline {b_start}-{b_end}"
                )

                # === 基线文件路径 ===
                p_file = (
                    f"{base_dir}/Results/HW_baselines/"
                    f"p{percentile_val}_{b_start}_{b_end}_{daily_statistic}_baseline.nc"
                )

                # === 温度数据文件列表 ===
                temp_files = [
                    f"{base_dir}/RawData/t2m_{daily_statistic}_era5_land/"
                    f"era5_data_2m_temperature_{year}.nc"
                    for year in target_years
                ]

                print("  正在准备数据...")
                print("  开始热浪检测...")

                # === 调用并行函数 ===
                results = detect_heatwaves_parallel(
                    temp_files=temp_files,
                    p_file=p_file,
                    percentage=percentile_val,
                    threshold_days=threshold_days,
                    baseline_years=baseline_years,
                    base_dir=base_dir,
                    daily_statistic=daily_statistic,
                    var_name=var_name,
                    target_years=target_years,
                    n_processes=n_processes
                )

                print("  ✔ 本组完成")

    print("\n🎉 所有 percentile × baseline × daily_statistic 组合均已完成！")
