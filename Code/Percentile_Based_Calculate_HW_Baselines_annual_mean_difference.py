# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 11:53:38 2025

@author: yujia.yang

智能版本：自动检测和修复空间维度问题
"""

import os
import xarray as xr
import rioxarray
import numpy as np

def prepare_for_raster_export(da, target_crs='EPSG:4326'):
    """
    准备DataArray用于栅格导出
    自动处理维度重命名、CRS设置等问题
    
    Parameters:
    -----------
    da : xarray.DataArray
        输入数据
    target_crs : str
        目标坐标系，默认WGS84
        
    Returns:
    --------
    da_prepared : xarray.DataArray
        准备好的数据
    """
    da_work = da.copy()
    
    # 1. 检测并重命名空间维度
    dim_mapping = {}
    lat_dim = None
    lon_dim = None
    
    for dim in da_work.dims:
        dim_lower = dim.lower()
        # 检查纬度维度
        if dim_lower in ['lat', 'latitude', 'y']:
            lat_dim = dim
            if dim != 'y':
                dim_mapping[dim] = 'y'
        # 检查经度维度
        elif dim_lower in ['lon', 'long', 'longitude', 'x']:
            lon_dim = dim
            if dim != 'x':
                dim_mapping[dim] = 'x'
    
    # 执行重命名
    if dim_mapping:
        da_work = da_work.rename(dim_mapping)
        print(f"    → 维度重命名: {dim_mapping}")
    
    # 2. 如果找不到标准维度名，尝试从坐标推断
    if lat_dim is None or lon_dim is None:
        for coord in da_work.coords:
            coord_lower = coord.lower()
            coord_data = da_work.coords[coord]
            
            # 通过数值范围推断（纬度通常-90到90，经度-180到180）
            if len(coord_data.shape) == 1:
                min_val = float(coord_data.min())
                max_val = float(coord_data.max())
                
                if -90 <= min_val <= 90 and -90 <= max_val <= 90:
                    # 可能是纬度
                    if coord not in da_work.dims:
                        print(f"    ⚠️ {coord} 是坐标但不是维度")
                    elif coord != 'y':
                        da_work = da_work.rename({coord: 'y'})
                        print(f"    → 推断纬度维度: {coord} → y")
                        lat_dim = 'y'
                
                elif -180 <= min_val <= 180 and -180 <= max_val <= 180:
                    # 可能是经度
                    if coord not in da_work.dims:
                        print(f"    ⚠️ {coord} 是坐标但不是维度")
                    elif coord != 'x':
                        da_work = da_work.rename({coord: 'x'})
                        print(f"    → 推断经度维度: {coord} → x")
                        lon_dim = 'x'
    
    # 3. 确保坐标存在且为float64
    if 'x' in da_work.coords:
        da_work['x'] = da_work['x'].astype(np.float64)
    if 'y' in da_work.coords:
        da_work['y'] = da_work['y'].astype(np.float64)
    
    # 4. 设置CRS
    try:
        current_crs = da_work.rio.crs
        if current_crs is None:
            da_work = da_work.rio.write_crs(target_crs)
            print(f"    → 设置CRS: {target_crs}")
        else:
            print(f"    → CRS已存在: {current_crs}")
    except AttributeError:
        # rio属性不存在
        da_work = da_work.rio.write_crs(target_crs)
        print(f"    → 设置CRS: {target_crs}")
    
    # 5. 设置空间维度
    try:
        if 'x' in da_work.dims and 'y' in da_work.dims:
            da_work = da_work.rio.set_spatial_dims(x_dim='x', y_dim='y', inplace=False)
            print(f"    → 设置空间维度: x, y")
    except Exception as e:
        print(f"    ⚠️ 设置空间维度时出错: {e}")
    
    # 6. 检查是否准备好
    if 'x' not in da_work.dims or 'y' not in da_work.dims:
        print(f"    ❌ 警告：空间维度可能仍然不正确")
        print(f"       当前维度: {da_work.dims}")
    
    return da_work


# === 主程序 ===
if __name__ == "__main__":
    disk = 'D'
    base_dir = f"{disk}:/Mediterranean CDHW"

    baseline_years_list = [
        range(1981, 2011),   # 1981–2010
        range(1981, 2021),   # 1981–2020
        range(1991, 2021)    # 1991–2020
    ]

    percentile_vals = [90, 95, 97, 99]
    daily_statistics = ["daily_mean", "daily_maximum"]
    var_name = "t2m"

    for daily_statistic in daily_statistics:
        out_dir = f"{base_dir}/Results/HW_baselines/Annual_mean"
        os.makedirs(out_dir, exist_ok=True)

        for percentile_val in percentile_vals:
            print(f"\n{'='*60}")
            print(f"处理: {daily_statistic} | P{percentile_val}")
            print(f"{'='*60}")

            annual_mean_map = {}

            # === Part 1: 计算并导出每个baseline的annual_mean ===
            for baseline_years in baseline_years_list:
                b_start = min(baseline_years)
                b_end   = max(baseline_years)

                in_nc = (
                    f"{base_dir}/Results/HW_baselines/"
                    f"p{percentile_val}_{b_start}_{b_end}_{daily_statistic}_baseline.nc"
                )

                if not os.path.exists(in_nc):
                    print(f"\n⚠️ 缺少文件: {in_nc}")
                    continue

                print(f"\n📂 处理 baseline {b_start}-{b_end}")
                
                try:
                    ds = xr.open_dataset(in_nc)
                    da = ds[f"{var_name}_p{percentile_val}"]

                    # 对第一维求平均
                    time_dim = da.dims[0]
                    print(f"  → 对 '{time_dim}' 维度求平均")
                    annual_mean = da.mean(dim=time_dim, skipna=True)

                    # 准备导出
                    print(f"  → 准备栅格导出...")
                    annual_mean_prepared = prepare_for_raster_export(annual_mean)

                    # 缓存
                    annual_mean_map[(b_start, b_end)] = annual_mean_prepared

                    # 导出
                    out_file = (
                        f"{out_dir}/p{percentile_val}_{b_start}_{b_end}_"
                        f"{daily_statistic}_baseline.tif"
                    )
                    
                    annual_mean_prepared.rio.to_raster(
                        out_file, 
                        driver='GTiff', 
                        compress='lzw',
                        dtype='float32'
                    )
                    print(f"  ✅ 保存成功: {os.path.basename(out_file)}")

                except Exception as e:
                    print(f"  ❌ 处理失败: {e}")
                    import traceback
                    traceback.print_exc()
                
                finally:
                    if 'ds' in locals():
                        ds.close()

            # === Part 2: 计算并导出差值 ===
            print(f"\n{'─'*60}")
            print(f"计算差值图...")
            print(f"{'─'*60}")
            
            base_ref_key = (1981, 2010)
            ref = annual_mean_map.get(base_ref_key, None)

            if ref is None:
                print("⚠️ 缺少参考 baseline 1981–2010，跳过差值计算")
                continue

            diff_pairs = [
                ((1981, 2020), (1981, 2010)),  # 1981–2020 − 1981–2010
                ((1991, 2020), (1981, 2010)),  # 1991–2020 − 1981–2010
            ]

            for (a_start, a_end), (b_start, b_end) in diff_pairs:
                A = annual_mean_map.get((a_start, a_end), None)
                B = annual_mean_map.get((b_start, b_end), None)

                if (A is None) or (B is None):
                    print(f"\n⚠️ 缺少数据: {a_start}-{a_end} 或 {b_start}-{b_end}")
                    continue

                print(f"\n📊 计算: {a_start}-{a_end} − {b_start}-{b_end}")
                
                try:
                    # 对齐并计算差值
                    A_aligned, B_aligned = xr.align(A, B, join="exact")
                    diff = A_aligned - B_aligned

                    # 确保CRS传递
                    if hasattr(A_aligned, 'rio') and A_aligned.rio.crs:
                        diff = diff.rio.write_crs(A_aligned.rio.crs)

                    # 导出
                    diff_out = (
                        f"{out_dir}/p{percentile_val}_"
                        f"{a_start}_{a_end}_minus_{b_start}_{b_end}_{daily_statistic}_baseline.tif"
                    )
                    
                    diff.rio.to_raster(
                        diff_out, 
                        driver='GTiff', 
                        compress='lzw',
                        dtype='float32'
                    )
                    print(f"  ✅ 差值保存: {os.path.basename(diff_out)}")

                except Exception as e:
                    print(f"  ❌ 差值计算失败: {e}")
                    import traceback
                    traceback.print_exc()

    print("\n" + "="*60)
    print("🎉 全部处理完成！")
