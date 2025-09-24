# -*- coding: utf-8 -*-
"""
科学参数优化模块 (utils/params.py)
Scientific Parameter Optimization Module

使用平均互信息(AMI)和伪最近邻(FNN)算法，为CCM分析确定最佳的嵌入参数。
[v3.6 Final]: 升级tau选择算法，使用更鲁棒的“首次显著下降”准则。
"""

import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mutual_info_score
from scipy.signal import find_peaks
from termcolor import colored
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 导入项目内的模块
from core.systems import generate_time_series, generate_adjacency_matrix

def _calculate_ami(series, max_tau):
    """计算给定序列的平均互信息(AMI)。"""
    amis = []
    n_bins = int(np.sqrt(len(series) / 5)) if len(series) > 25 else 5
    
    for tau in range(1, max_tau + 1):
        x = series[:-tau]
        y = series[tau:]
        hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins)
        mi = mutual_info_score(None, None, contingency=hist_2d)
        amis.append(mi)
        
    return np.array(amis)

def find_optimal_tau(series, max_tau=50, plot=True):
    """
    [改进] 使用平均互信息(AMI)的“首个局部最小值”准则来寻找最佳时间延迟 tau。
    """
    print("1. 正在计算平均互信息(AMI)以寻找最佳 tau...")
    ami = _calculate_ami(series, max_tau)
    
    # 使用 find_peaks 寻找所有局部最小值（波谷）
    valleys, _ = find_peaks(-ami, prominence=0.01)
    
    optimal_tau = -1
    if len(valleys) > 0:
        # [改进] 直接选择第一个局部最小值
        optimal_tau = valleys[0] + 1 # 索引从0开始，tau从1开始
        print_message = f"✅ 找到最佳 tau = {optimal_tau} (基于首个局部最小值准则)"

    else: # 如果没有找到任何波谷
        print(colored("⚠️ 未找到明显的AMI局部最小值，默认返回 tau = 1。", "yellow"))
        optimal_tau = 1
        print_message = f"✅ 默认选择 tau = {optimal_tau}"


    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_tau + 1), ami, marker='o', linestyle='-', zorder=1, label='AMI')
        # [移除] 不再需要 1/e 阈值线
        if len(valleys) > 0:
            plt.scatter(valleys + 1, ami[valleys], c='red', s=100, marker='v', label='Local Minima', zorder=3)
        plt.axvline(optimal_tau, color='r', linestyle='--', label=f'Optimal tau = {optimal_tau}', zorder=4)
        plt.title('Average Mutual Information (AMI) vs. Time Lag')
        plt.xlabel('Time Lag (tau)')
        plt.ylabel('Average Mutual Information')
        plt.grid(True)
        plt.legend()
        plt.show()

    print(colored(print_message, "green"))
    return int(optimal_tau)

def find_optimal_dim(series, tau, max_dim=10, fnn_threshold=1.0, R_tol=15.0, A_tol=2.0, plot=True):
    """
    [v3.5] 使用伪最近邻(FNN)算法和阈值法寻找最佳嵌入维度 Dim。
    """
    print(f"2. 正在使用 FNN 算法 (tau={tau}) 寻找最佳 Dim...")
    A = np.std(series)
    fnn_percentages = []
    
    optimal_dim = -1

    for dim in tqdm(range(1, max_dim + 1), desc="Testing Dimensions"):
        L = len(series)
        shadow_length = L - dim * tau
        if shadow_length <= 1:
            print(colored(f"序列对于 Dim={dim}, tau={tau} 来说太短，停止搜索。", "red"))
            break

        shadow = np.column_stack([series[i * tau : L - (dim - i) * tau] for i in range(dim + 1)])
        
        shadow_d = shadow[:, :-1]
        shadow_d1_vals = shadow[:, -1]

        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(shadow_d)
        distances, indices = nbrs.kneighbors(shadow_d)
        
        nn_distances_d = distances[:, 1]
        nn_indices = indices[:, 1]
        
        num_fnn = 0
        for i in range(len(shadow_d)):
            dist_d1 = np.sqrt(nn_distances_d[i]**2 + (shadow_d1_vals[i] - shadow_d1_vals[nn_indices[i]])**2)
            
            if nn_distances_d[i] > 1e-10 and dist_d1 / nn_distances_d[i] > R_tol:
                num_fnn += 1
                continue

            if dist_d1 / A > A_tol:
                num_fnn += 1
        
        fnn_percentage = 100 * num_fnn / len(shadow_d)
        fnn_percentages.append(fnn_percentage)
        
        if fnn_percentage < fnn_threshold and optimal_dim == -1:
            optimal_dim = dim
    
    if optimal_dim == -1:
        optimal_dim = np.argmin(fnn_percentages) + 1
        print(colored(f"⚠️ FNN比例未能降至 {fnn_threshold}% 以下。选择最小值所在的维度: Dim = {optimal_dim}", "yellow"))
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(fnn_percentages) + 1), fnn_percentages, marker='o', linestyle='-')
        plt.axvline(optimal_dim, color='r', linestyle='--', label=f'Optimal Dim ({optimal_dim})')
        plt.axhline(fnn_threshold, color='g', linestyle=':', label=f'{fnn_threshold}% Threshold')
        plt.title('False Nearest Neighbors (FNN) vs. Embedding Dimension')
        plt.xlabel('Embedding Dimension (Dim)')
        plt.ylabel('Percentage of FNN (%)')
        plt.grid(True)
        plt.legend()
        plt.yscale('log')
        plt.show()

    print(colored(f"✅ 找到最佳 Dim = {optimal_dim} (FNN比例首次低于 {fnn_threshold}%)", "green"))
    return int(optimal_dim)

def optimize_embedding_for_system(system_type, series_length=8000):
    """
    为一个给定的动力学系统运行完整的参数优化流程。
    """
    print(colored(f"--- 开始为 {system_type.capitalize()} 系统进行参数优化 ---", "cyan"))

    # 为了获得可重复的嵌入参数，同时不影响后续 trial 的随机性，
    # 在生成测试序列前暂存随机数生成器的状态。
    np_state = np.random.get_state()
    py_state = random.getstate()

    try:
        # 使用固定种子生成代表性的测试序列
        np.random.seed(42)
        random.seed(42)

        print("正在生成测试时间序列...")
        adjacency_matrix = generate_adjacency_matrix(1, 0)
        test_series = generate_time_series(system_type, 1, adjacency_matrix, series_length, 0.0)[0]

        if np.any(~np.isfinite(test_series)) or np.var(test_series) < 1e-10:
            print(colored("错误：生成的测试序列无效（包含NaN/Inf或无方差）。", "red"))
            return None

    except Exception as e:
        print(colored(f"错误：生成测试序列失败: {e}", "red"))
        return None
    finally:
        # 恢复随机数生成器的状态，避免影响后续 trial 的随机过程
        np.random.set_state(np_state)
        random.setstate(py_state)

    optimal_tau = find_optimal_tau(test_series)
    optimal_dim = find_optimal_dim(test_series, optimal_tau)
    
    print(colored("--- 参数优化完成 ---", "cyan"))
    return {'tau': optimal_tau, 'Dim': optimal_dim}
