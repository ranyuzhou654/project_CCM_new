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


def adaptive_optimize_embedding_params(
    system_type, analysis_type, current_value, base_length=8000, **kwargs
):
    """
    根据分析类型和当前实验变量自适应优化嵌入参数。

    参数:
    - system_type: 系统类型
    - analysis_type: 分析类型 ('length', 'coupling', 'noise', 'degree', 'nodes')
    - current_value: 当前实验变量的值
    - base_length: 基础序列长度
    - **kwargs: 其他参数

    返回:
    - dict: 包含最佳 Dim 和 tau 的字典
    """

    # 敏感系统列表，需要特殊处理
    sensitive_systems = [
        "henon",
        "noisy_henon",
        "henon_dynamic_noise",
        "rossler",
        "noisy_rossler",
        "rossler_dynamic_noise",
    ]

    # 确定是否需要重新优化参数
    should_reoptimize = False
    optimization_params = {}

    if analysis_type == "length":
        # 长度分析：当测试长度与基础长度差异较大时重新优化
        length_ratio = current_value / base_length
        if length_ratio < 0.5 or length_ratio > 2.0 or system_type in sensitive_systems:
            should_reoptimize = True
            optimization_params = {"series_length": current_value}

    elif analysis_type == "noise" and system_type.startswith("noisy_"):
        # 噪声分析：当噪声水平较高时重新优化
        if current_value > 0.1 or system_type in sensitive_systems:
            should_reoptimize = True
            optimization_params = {
                "series_length": base_length,
                "noise_level": current_value,
            }

    elif analysis_type == "coupling":
        # 耦合分析：对敏感系统在强耦合时重新优化
        if (
            current_value > 0.5 and system_type in sensitive_systems
        ) or current_value > 1.0:
            should_reoptimize = True
            optimization_params = {
                "series_length": base_length,
                "coupling": current_value,
            }

    elif system_type in sensitive_systems:
        # 敏感系统：总是进行额外检查
        should_reoptimize = True
        optimization_params = {"series_length": base_length}

    if should_reoptimize:
        print(
            colored(
                f"为 {system_type} 系统在 {analysis_type}={current_value} 条件下重新优化参数...",
                "cyan",
            )
        )

        # 使用特定参数重新优化
        if analysis_type == "length":
            adaptive_params = optimize_embedding_for_system(
                system_type, series_length=current_value
            )
        elif analysis_type == "noise":
            # 为噪声分析创建带噪声的测试序列
            adaptive_params = _optimize_with_noise(
                system_type, base_length, current_value
            )
        else:
            # 其他情况使用默认优化
            adaptive_params = optimize_embedding_for_system(
                system_type, series_length=base_length
            )

        if adaptive_params:
            print(
                colored(
                    f"✅ 自适应参数: Dim={adaptive_params['Dim']},"
                    f" tau={adaptive_params['tau']}",
                    "green",
                )
            )
            return adaptive_params
        else:
            print(colored("❌ 自适应优化失败，回退到默认参数", "yellow"))

    # 回退到默认参数优化
    return optimize_embedding_for_system(system_type, series_length=base_length)


def _optimize_with_noise(system_type, series_length, noise_level):
    """为带噪声的系统优化参数"""
    np_state = np.random.get_state()
    py_state = random.getstate()

    try:
        # 生成带噪声的测试序列
        np.random.seed(42)
        random.seed(42)
        adjacency_matrix = generate_adjacency_matrix(1, 0)

        # 使用噪声系统类型
        if not system_type.startswith("noisy_"):
            noisy_system_type = f"noisy_{system_type}"
        else:
            noisy_system_type = system_type

        test_series = generate_time_series(
            noisy_system_type,
            1,
            adjacency_matrix,
            series_length,
            0.0,
            noise_level=noise_level,
        )[0]

        if np.any(~np.isfinite(test_series)) or np.var(test_series) < 1e-10:
            return None

        # 优化参数
        optimal_tau = find_optimal_tau(
            test_series, max_tau=min(50, series_length // 10), plot=False
        )
        optimal_dim = find_optimal_dim(test_series, optimal_tau, max_dim=10, plot=False)

        return {"Dim": optimal_dim, "tau": optimal_tau}

    except Exception as e:
        print(colored(f"带噪声参数优化失败: {e}", "red"))
        return None
    finally:
        # 恢复随机数生成器状态，避免影响后续流程
        np.random.set_state(np_state)
        random.setstate(py_state)


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
        optimal_tau = valleys[0] + 1  # 索引从0开始，tau从1开始
        print_message = f"✅ 找到最佳 tau = {optimal_tau} (基于首个局部最小值准则)"

    else:  # 如果没有找到任何波谷
        print(colored("⚠️ 未找到明显的AMI局部最小值，默认返回 tau = 1。", "yellow"))
        optimal_tau = 1
        print_message = f"✅ 默认选择 tau = {optimal_tau}"

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, max_tau + 1), ami, marker="o", linestyle="-", zorder=1, label="AMI"
        )
        # [移除] 不再需要 1/e 阈值线
        if len(valleys) > 0:
            plt.scatter(
                valleys + 1,
                ami[valleys],
                c="red",
                s=100,
                marker="v",
                label="Local Minima",
                zorder=3,
            )
        plt.axvline(
            optimal_tau,
            color="r",
            linestyle="--",
            label=f"Optimal tau = {optimal_tau}",
            zorder=4,
        )
        plt.title("Average Mutual Information (AMI) vs. Time Lag")
        plt.xlabel("Time Lag (tau)")
        plt.ylabel("Average Mutual Information")
        plt.grid(True)
        plt.legend()
        plt.show()

    print(colored(print_message, "green"))
    return int(optimal_tau)


def find_optimal_dim(
    series, tau, max_dim=10, fnn_threshold=1.0, R_tol=15.0, A_tol=2.0, plot=True
):
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

        shadow = np.column_stack([
            series[i * tau : L - (dim - i) * tau] for i in range(dim + 1)
        ])

        shadow_d = shadow[:, :-1]
        shadow_d1_vals = shadow[:, -1]

        nbrs = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(shadow_d)
        distances, indices = nbrs.kneighbors(shadow_d)

        nn_distances_d = distances[:, 1]
        nn_indices = indices[:, 1]

        num_fnn = 0
        for i in range(len(shadow_d)):
            dist_d1 = np.sqrt(
                nn_distances_d[i] ** 2
                + (shadow_d1_vals[i] - shadow_d1_vals[nn_indices[i]]) ** 2
            )

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
        print(
            colored(
                f"⚠️ FNN比例未能降至 {fnn_threshold}% 以下。选择最小值所在的维度: Dim ="
                f" {optimal_dim}",
                "yellow",
            )
        )

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, len(fnn_percentages) + 1),
            fnn_percentages,
            marker="o",
            linestyle="-",
        )
        plt.axvline(
            optimal_dim, color="r", linestyle="--", label=f"Optimal Dim ({optimal_dim})"
        )
        plt.axhline(
            fnn_threshold, color="g", linestyle=":", label=f"{fnn_threshold}% Threshold"
        )
        plt.title("False Nearest Neighbors (FNN) vs. Embedding Dimension")
        plt.xlabel("Embedding Dimension (Dim)")
        plt.ylabel("Percentage of FNN (%)")
        plt.grid(True)
        plt.legend()
        plt.yscale("log")
        plt.show()

    print(
        colored(
            f"✅ 找到最佳 Dim = {optimal_dim} (FNN比例首次低于 {fnn_threshold}%)",
            "green",
        )
    )
    return int(optimal_dim)


def optimize_embedding_for_system(system_type, series_length=8000):
    """
    为一个给定的动力学系统运行完整的参数优化流程。
    """
    print(
        colored(f"--- 开始为 {system_type.capitalize()} 系统进行参数优化 ---", "cyan")
    )

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
        test_series = generate_time_series(
            system_type, 1, adjacency_matrix, series_length, 0.0
        )[0]

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
    return {"tau": optimal_tau, "Dim": optimal_dim}
