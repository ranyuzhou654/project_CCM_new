# -*- coding: utf-8 -*-
"""
CCM核心算法和代理方法模块 (core/ccm.py)
CCM Core Algorithm and Surrogate Methods Module
[v3.3 Final]: 彻底重构 ccm_pearson 函数，以根除类型错误并提升性能。
"""

import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr, gaussian_kde
from scipy.interpolate import interp1d
from termcolor import colored
import warnings

# ==============================================================================
# 1. CCM核心算法
# ==============================================================================


def parameters(series, Dim, tau):
    """
    为给定的时间序列构建影子流形并计算最近邻的权重。
    """
    if np.any(~np.isfinite(series)):
        print(colored("警告: 输入序列包含 NaN 或 Inf 值。", "yellow"))
        return None, None

    L = len(series)
    shadow_length = L - (Dim - 1) * tau

    if shadow_length <= Dim + 1:
        return None, None

    shadow = np.column_stack([
        series[i * tau : L - (Dim - 1 - i) * tau] for i in range(Dim)
    ])

    nbrs = NearestNeighbors(n_neighbors=Dim + 2, algorithm="ball_tree").fit(shadow)
    distances, indices = nbrs.kneighbors(shadow)

    distances, indices = distances[:, 1:], indices[:, 1:]

    e = 1e-10
    u = np.exp(-distances / (distances[:, [0]] + e))

    sum_u = np.sum(u, axis=1, keepdims=True)
    weights = np.divide(u, sum_u, where=sum_u != 0)

    return weights, indices


def ccm_pearson(target, weights, indices, shadow_length):
    """
    [v3.3 Final] 使用统一的Numpy向量化方法计算皮尔遜相關係數，确保返回类型正确。
    """
    if weights is None or indices is None:
        return np.array([]) if target.ndim == 2 else 0.0

    if np.any(~np.isfinite(target)) or np.any(~np.isfinite(weights)):
        return np.zeros(target.shape[0]) if target.ndim == 2 else 0.0

    # 确保索引不会越界
    if np.max(indices) >= target.shape[-1]:
        return np.zeros(target.shape[0]) if target.ndim == 2 else 0.0

    # 统一处理输入，使其至少为二维
    is_batch = target.ndim == 2
    target_batch = np.atleast_2d(target)

    # 批量进行交叉映射预测
    prediction_batch = np.sum(weights[None, :, :] * target_batch[:, indices], axis=2)

    # 截取原始序列以匹配长度
    original_batch = target_batch[:, :shadow_length]

    # 批量计算皮尔逊相关系数
    mean_orig = original_batch.mean(axis=1, keepdims=True)
    mean_pred = prediction_batch.mean(axis=1, keepdims=True)

    std_orig = original_batch.std(axis=1, ddof=1)
    std_pred = prediction_batch.std(axis=1, ddof=1)

    # 避免除以零
    valid_mask = (std_orig > 1e-10) & (std_pred > 1e-10)

    cov = np.sum(
        (original_batch - mean_orig) * (prediction_batch - mean_pred), axis=1
    ) / (shadow_length - 1)

    std_prod = std_orig * std_pred

    corr = np.zeros_like(std_prod)
    np.divide(cov, std_prod, out=corr, where=valid_mask)

    final_corr = np.nan_to_num(corr)

    # 如果原始输入不是批量，则返回一个Python标量
    return final_corr if is_batch else final_corr.item()


# ==============================================================================
# 2. 代理数据生成方法
# ==============================================================================


def _surrogate_fft(data):
    """傅里叶变换代理：保持功率谱，随机化相位（保证共轭对称）。"""

    data = np.asarray(data)
    n = data.shape[0]

    # 只处理非负频率，确保共轭对称
    freqs = np.fft.rfft(data)
    amplitudes = np.abs(freqs)

    random_phases = np.random.uniform(0.0, 2.0 * np.pi, freqs.shape[0])
    random_phases[0] = np.angle(freqs[0])  # 保持直流分量
    if n % 2 == 0:
        # Nyquist 分量要求返回实数，保留原相位（0 或 π）
        random_phases[-1] = np.angle(freqs[-1])

    surrogate_freqs = amplitudes * np.exp(1j * random_phases)
    surrogate = np.fft.irfft(surrogate_freqs, n=n)

    return surrogate


def _surrogate_iaaft(data, num_iterations=100):
    """迭代幅度调整傅里叶变换代理：更精确地保持分布和功率谱。"""
    surrogate = np.random.permutation(data)
    target_amplitudes = np.abs(np.fft.fft(data))
    sorted_data = np.sort(data)

    for _ in range(num_iterations):
        surrogate_fft = np.fft.fft(surrogate)
        phases = np.angle(surrogate_fft)
        surrogate = np.fft.ifft(target_amplitudes * np.exp(1j * phases)).real
        rank = np.argsort(np.argsort(surrogate))
        surrogate = sorted_data[rank]

    return surrogate


def _surrogate_conditional(source_series, conditional_series, Dim, tau):
    """条件代理：在保持对条件序列依赖性的同时，对源序列进行重排。"""
    L = len(source_series)
    shadow_length = L - (Dim - 1) * tau
    if shadow_length <= Dim + 1:
        return source_series

    conditional_shadow = np.column_stack([
        conditional_series[i * tau : L - (Dim - 1 - i) * tau] for i in range(Dim)
    ])
    nbrs = NearestNeighbors(n_neighbors=Dim + 2, algorithm="ball_tree").fit(
        conditional_shadow
    )
    _, neighbor_indices = nbrs.kneighbors(conditional_shadow)
    neighbor_indices = neighbor_indices[:, 1:]

    surrogate_series = np.copy(source_series)
    head_len = (Dim - 1) * tau

    for t_idx in range(shadow_length):
        chosen_neighbor_idx = random.choice(neighbor_indices[t_idx])
        original_time_idx = chosen_neighbor_idx + head_len
        surrogate_series[t_idx + head_len] = source_series[original_time_idx]

    return surrogate_series


def generate_surrogates(data, method, num_surrogates=1, **kwargs):
    """
    代理方法分发器。
    """
    surrogate_funcs = {
        "FFT": _surrogate_fft,
        "AAFT": lambda d: _surrogate_iaaft(
            d, num_iterations=1
        ),  # AAFT是IAAFT的单次迭代
        "IAAFT": _surrogate_iaaft,
        "Time Shift": lambda d: np.roll(d, np.random.randint(1, len(d))),
        "Random Reorder": np.random.permutation,
        "Conditional": lambda d: _surrogate_conditional(d, **kwargs),
    }

    if method not in surrogate_funcs:
        raise ValueError(f"不支持的代理方法: {method}")

    func = surrogate_funcs[method]

    return np.array([func(data) for _ in range(num_surrogates)])


# ==============================================================================
# 3. 改进的统计检验方法
# ==============================================================================


def improved_ccm_confidence(original_coef, surrogate_coefs, method="kde"):
    """
    计算更平滑的置信度分数，解决离散化问题

    Parameters:
    -----------
    original_coef : float
        原始CCM系数
    surrogate_coefs : array-like
        代理数据的CCM系数数组
    method : str
        计算方法：'kde' (核密度估计) 或 'ecdf' (经验分布函数插值)

    Returns:
    --------
    float : 置信度分数 (0-1之间)
    """
    surrogate_coefs = np.asarray(surrogate_coefs)

    # 处理边界情况
    if len(surrogate_coefs) == 0:
        return 0.5

    if method == "kde":
        try:
            # 使用核密度估计近似经验累积分布函数
            kde = gaussian_kde(surrogate_coefs)
            cdf_value = float(kde.integrate_box_1d(-np.inf, original_coef))
            return np.clip(cdf_value, 0, 1)
        except Exception:
            # KDE失败时回退到ECDF方法
            method = "ecdf"

    if method == "ecdf":
        # 使用经验分布函数插值
        sorted_surrogates = np.sort(surrogate_coefs)
        n = len(sorted_surrogates)

        # 创建经验分布函数
        ecdf_vals = np.arange(1, n + 1) / n

        # 处理重复值
        unique_vals, unique_indices = np.unique(sorted_surrogates, return_index=True)
        unique_ecdf = ecdf_vals[unique_indices]

        if len(unique_vals) < 2:
            # 所有代理值都相同
            return (
                0.5
                if original_coef == unique_vals[0]
                else (1.0 if original_coef > unique_vals[0] else 0.0)
            )

        # 插值函数
        ecdf_func = interp1d(
            unique_vals,
            unique_ecdf,
            bounds_error=False,
            fill_value=(0, 1),
            kind="linear",
        )
        confidence = float(ecdf_func(original_coef))
        return np.clip(confidence, 0, 1)

    # 默认回退到原始方法
    return np.sum(surrogate_coefs < original_coef) / len(surrogate_coefs)


def adaptive_surrogate_testing(
    time_series,
    method,
    weights,
    indices,
    shadow_len,
    min_surrogates=200,
    max_surrogates=2000,
    batch_size=100,
    stability_threshold=0.01,
    max_iterations=20,
):
    """
    自适应地增加代理数量直到置信度分数稳定

    Parameters:
    -----------
    time_series : array-like
        目标时间序列
    method : str
        代理数据生成方法
    weights, indices, shadow_len :
        CCM计算所需参数
    min_surrogates : int
        最小代理数量
    max_surrogates : int
        最大代理数量
    batch_size : int
        每批次代理数量
    stability_threshold : float
        稳定性阈值
    max_iterations : int
        最大迭代次数

    Returns:
    --------
    tuple : (最终置信度分数, 使用的代理数量, 收敛历史)
    """
    # 计算原始CCM系数
    original_coef = np.abs(ccm_pearson(time_series, weights, indices, shadow_len))

    all_surrogate_coefs = []
    confidence_history = []

    for iteration in range(max_iterations):
        # 生成一批代理数据
        current_batch_size = min(batch_size, max_surrogates - len(all_surrogate_coefs))
        if current_batch_size <= 0:
            break

        surrogates = generate_surrogates(time_series, method, current_batch_size)
        batch_coefs = np.abs(ccm_pearson(surrogates, weights, indices, shadow_len))

        # 确保batch_coefs是一维数组
        if batch_coefs.ndim == 0:
            batch_coefs = np.array([batch_coefs])

        all_surrogate_coefs.extend(batch_coefs)

        # 检查是否达到最小数量
        if len(all_surrogate_coefs) < min_surrogates:
            continue

        # 计算当前置信度
        current_confidence = improved_ccm_confidence(
            original_coef, all_surrogate_coefs, method="kde"
        )
        confidence_history.append(current_confidence)

        # 检查稳定性（需要至少3个点）
        if len(confidence_history) >= 3:
            recent_std = np.std(confidence_history[-3:])
            if recent_std < stability_threshold:
                break

    final_confidence = confidence_history[-1] if confidence_history else 0.5

    return final_confidence, len(all_surrogate_coefs), confidence_history


def bootstrap_auroc_confidence(scores, labels, n_bootstrap=1000, confidence_level=0.95):
    """
    计算AUROC的bootstrap置信区间

    Parameters:
    -----------
    scores : array-like
        预测分数
    labels : array-like
        真实标签
    n_bootstrap : int
        bootstrap重采样次数
    confidence_level : float
        置信度水平

    Returns:
    --------
    tuple : (平均AUROC, (下界, 上界))
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.utils import resample

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    if len(scores) != len(labels):
        raise ValueError("分数和标签长度不匹配")

    bootstrap_aucs = []

    for _ in range(n_bootstrap):
        # 重采样
        try:
            indices = resample(
                range(len(scores)), n_samples=len(scores), random_state=None
            )
            boot_scores = scores[indices]
            boot_labels = labels[indices]

            # 检查是否包含两个类别
            if len(np.unique(boot_labels)) < 2:
                continue

            # 计算AUROC
            fpr, tpr, _ = roc_curve(boot_labels, boot_scores)
            auc_val = auc(fpr, tpr)

            if not np.isnan(auc_val):
                bootstrap_aucs.append(auc_val)

        except Exception:
            continue

    if len(bootstrap_aucs) == 0:
        return 0.5, (0.5, 0.5)

    bootstrap_aucs = np.array(bootstrap_aucs)

    # 计算置信区间
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_aucs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_aucs, 100 * (1 - alpha / 2))

    return np.mean(bootstrap_aucs), (ci_lower, ci_upper)
