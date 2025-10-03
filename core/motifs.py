# -*- coding: utf-8 -*-
"""
三节点基序分析模块 (core/motifs.py)
Three-Node Motif Analysis Module

使用CCM和条件传递熵(CTE)两种方法来区分不同的三节点因果结构。
[v3.7 Final]: 集成了高级的部分交叉映射 (Partial CCM) 方法。
"""

import numpy as np
from tqdm import tqdm
from termcolor import colored
from collections import Counter
from datetime import datetime  # 导入 datetime

# --- 自定义传递熵实现 (后备方案) ---
# 这是一个纯Python的实现，用于在pyinform库不可用时作为后备。
# 警告：此实现的性能远低于pyinform的C语言优化版本。


def _get_shannon_entropy(states):
    """根据状态计数的字典计算香农熵。"""
    if not states:
        return 0.0
    total_count = sum(states.values())
    if total_count == 0:
        return 0.0

    entropy = 0.0
    for count in states.values():
        prob = count / total_count
        entropy -= prob * np.log2(prob)
    return entropy


def custom_transfer_entropy(source, target, k=1):
    """
    纯Python实现的传递熵 (TE)。
    TE(X->Y) = H(Y_t+1 | Y_t^k) - H(Y_t+1 | Y_t^k, X_t^k)
    """
    if len(source) != len(target):
        raise ValueError("源序列和目标序列的长度必须相同。")

    n = len(target)

    # 构造历史状态
    y_future = []
    y_history = []
    x_history = []

    for t in range(k, n - 1):
        y_future.append(target[t + 1])
        y_history.append(tuple(target[t - i] for i in range(k)))
        x_history.append(tuple(source[t - i] for i in range(k)))

    # 计算 H(Y_t+1, Y_t^k) 和 H(Y_t^k)
    h_y_future_y_history = _get_shannon_entropy(Counter(zip(y_future, y_history)))
    h_y_history = _get_shannon_entropy(Counter(y_history))

    # 计算 H(Y_t+1 | Y_t^k)
    h_y_future_cond_y_history = h_y_future_y_history - h_y_history

    # 计算 H(Y_t+1, Y_t^k, X_t^k) 和 H(Y_t^k, X_t^k)
    h_y_future_y_history_x_history = _get_shannon_entropy(
        Counter(zip(y_future, y_history, x_history))
    )
    h_y_history_x_history = _get_shannon_entropy(Counter(zip(y_history, x_history)))

    # 计算 H(Y_t+1 | Y_t^k, X_t^k)
    h_y_future_cond_y_x_history = h_y_future_y_history_x_history - h_y_history_x_history

    return h_y_future_cond_y_history - h_y_future_cond_y_x_history


def custom_conditional_transfer_entropy(source, target, condition, k=1):
    """
    纯Python实现的条件传递熵 (CTE)。
    CTE(X->Y|Z) = H(Y_t+1 | Y_t^k, Z_t^k) - H(Y_t+1 | Y_t^k, X_t^k, Z_t^k)
    """
    if not (len(source) == len(target) == len(condition)):
        raise ValueError("所有序列的长度必须相同。")

    n = len(target)

    # 构造历史状态
    y_future = []
    y_history = []
    x_history = []
    z_history = []

    for t in range(k, n - 1):
        y_future.append(target[t + 1])
        y_history.append(tuple(target[t - i] for i in range(k)))
        x_history.append(tuple(source[t - i] for i in range(k)))
        z_history.append(tuple(condition[t - i] for i in range(k)))

    # 计算 H(Y_t+1 | Y_t^k, Z_t^k)
    h_y_future_y_hist_z_hist = _get_shannon_entropy(
        Counter(zip(y_future, y_history, z_history))
    )
    h_y_hist_z_hist = _get_shannon_entropy(Counter(zip(y_history, z_history)))
    h_y_future_cond_y_z_hist = h_y_future_y_hist_z_hist - h_y_hist_z_hist

    # 计算 H(Y_t+1 | Y_t^k, X_t^k, Z_t^k)
    h_y_future_y_x_z_hist = _get_shannon_entropy(
        Counter(zip(y_future, y_history, x_history, z_history))
    )
    h_y_x_z_hist = _get_shannon_entropy(Counter(zip(y_history, x_history, z_history)))
    h_y_future_cond_y_x_z_hist = h_y_future_y_x_z_hist - h_y_x_z_hist

    return h_y_future_cond_y_z_hist - h_y_future_cond_y_x_z_hist


# --- pyinform 库的导入与后备逻辑 ---
try:
    import pyinform
    from pyinform.transferentropy import transfer_entropy, conditional_transfer_entropy

    CTE_AVAILABLE = True
    print(colored("成功加载 'pyinform' 库。将使用C语言优化的传递熵计算。", "green"))
except ImportError:
    CTE_AVAILABLE = False
    transfer_entropy = custom_transfer_entropy
    conditional_transfer_entropy = custom_conditional_transfer_entropy
    print(colored("=" * 80, "yellow"))
    print(
        colored(
            "警告: 'pyinform' 库加载失败。将使用纯Python实现的传递熵函数。", "yellow"
        )
    )
    print(
        colored(
            "分析速度会非常慢。强烈建议通过 'pip install --upgrade --force-reinstall"
            " pyinform' 修复。",
            "cyan",
        )
    )
    print(colored("=" * 80, "yellow"))

# 导入项目模块
from .systems import generate_time_series
from .ccm import parameters, ccm_pearson, generate_surrogates
from utils.params import optimize_embedding_for_system
from .partial_ccm import run_partial_ccm_trial


def _discretize_series(series, n_bins=4):
    """将连续时间序列离散化，为传递熵计算做准备。"""
    try:
        # 使用简单的分位数分箱
        bins = np.quantile(series, np.linspace(0, 1, n_bins + 1))
        bins[0] -= 0.001  # 确保最小值被包含
        bins[-1] += 0.001  # 确保最大值被包含
        return np.digitize(series, bins) - 1
    except Exception as e:
        print(colored(f"离散化序列时出错: {e}", "red"))
        return None


def analyze_three_node_motifs_dual_method(
    system_type, visualizer, time_steps=2000, num_surrogates=200
):
    """
    [v3.7] 使用CCM(所有代理方法), Partial CCM, 和CTE(带p值)三核方法分析基序。
    """
    print(colored("--- [步骤 1/2] 正在自动优化嵌入参数... ---", "cyan"))
    optimal_params = optimize_embedding_for_system(system_type, series_length=8000)
    if not optimal_params:
        print(
            colored(f"错误: 无法为 {system_type} 系统确定最佳参数。分析中止。", "red")
        )
        return
    print(
        colored(
            f"✅ 最佳参数确定: Dim = {optimal_params['Dim']}, tau ="
            f" {optimal_params['tau']}",
            "green",
        )
    )

    te_history_length = optimal_params["Dim"]

    ccm_surrogate_methods = [
        "FFT",
        "AAFT",
        "IAAFT",
        "Time Shift",
        "Random Reorder",
        "Conditional",
    ]

    motifs = {
        "Causal Chain (X→Z→Y)": np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
        "Common Driver (Z→X, Z→Y)": np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]]),
        "Direct Link (X→Y)": np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
    }

    results_for_viz = {}

    print(colored("\n--- [步骤 2/2] 正在分析不同的因果基序... ---", "cyan"))
    for name, adjacency in tqdm(motifs.items(), desc="Analyzing Motifs"):
        epsilon = 0.3 if system_type in ["lorenz", "noisy_lorenz"] else 0.1
        time_series = generate_time_series(
            system_type, 3, adjacency, time_steps, epsilon
        )
        if np.any(~np.isfinite(time_series)):
            continue
        X, Z, Y = time_series[0], time_series[1], time_series[2]

        # --- Partial CCM 分析 ---
        partial_ccm_score = run_partial_ccm_trial(
            X, Y, Z, optimal_params["Dim"], optimal_params["tau"], num_surrogates
        )

        # --- CCM 分析 (使用所有代理方法) ---
        weights_Y, indices_Y = parameters(
            Y, optimal_params["Dim"], optimal_params["tau"]
        )
        if weights_Y is None:
            continue
        shadow_len = weights_Y.shape[0]
        original_ccm_xy = np.abs(ccm_pearson(X, weights_Y, indices_Y, shadow_len))
        ccm_pvals = {}
        ccm_dists = {}
        for method in ccm_surrogate_methods:
            kwargs = {}
            if method == "Conditional":
                kwargs = {
                    "conditional_series": Z,
                    "Dim": optimal_params["Dim"],
                    "tau": optimal_params["tau"],
                }
            surrogates = generate_surrogates(X, method, num_surrogates, **kwargs)
            surrogate_coefs = np.abs(
                ccm_pearson(surrogates, weights_Y, indices_Y, shadow_len)
            )
            p_value = np.sum(surrogate_coefs >= original_ccm_xy) / num_surrogates
            ccm_pvals[method] = p_value
            ccm_dists[method] = surrogate_coefs

        # --- 传递熵 (TE) 分析，并计算其p值 ---
        te_pval, cte_pval = np.nan, np.nan
        # 即使CTE_AVAILABLE为False，我们现在也有了后备函数
        X_d, Y_d, Z_d = (
            _discretize_series(X),
            _discretize_series(Y),
            _discretize_series(Z),
        )
        if X_d is not None and Y_d is not None and Z_d is not None:
            original_te_xy = transfer_entropy(X_d, Y_d, k=te_history_length)
            original_cte_xyz = conditional_transfer_entropy(
                X_d, Y_d, Z_d, k=te_history_length
            )

            te_surrogates_X = generate_surrogates(X, "Time Shift", num_surrogates)
            te_dist, cte_dist = [], []
            for i in range(num_surrogates):
                surr_X_d = _discretize_series(te_surrogates_X[i])
                if surr_X_d is not None:
                    te_dist.append(transfer_entropy(surr_X_d, Y_d, k=te_history_length))
                    cte_dist.append(
                        conditional_transfer_entropy(
                            surr_X_d, Y_d, Z_d, k=te_history_length
                        )
                    )

            te_pval = (
                np.sum(np.array(te_dist) >= original_te_xy) / len(te_dist)
                if te_dist
                else 1.0
            )
            cte_pval = (
                np.sum(np.array(cte_dist) >= original_cte_xyz) / len(cte_dist)
                if cte_dist
                else 1.0
            )
        else:
            print(
                colored(f"警告: 基序 {name} 的序列离散化失败，跳过TE分析。", "yellow")
            )

        results_for_viz[name] = {
            "partial_ccm_score": partial_ccm_score,
            "ccm_pvals": ccm_pvals,
            "ccm_dists": ccm_dists,
            "te_pval": te_pval,
            "cte_pval": cte_pval,
            "original_ccm": original_ccm_xy,
        }

    if results_for_viz:
        # [改进] 添加时间戳以创建唯一文件名，并将其传递给可视化函数
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"motif_analysis_{system_type}_{timestamp}.png"
        visualizer.plot_motif_comparison(results_for_viz, system_type, save_path)
    else:
        print(colored("错误: 所有基序分析均失败，无法生成可视化。", "red"))
