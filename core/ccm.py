# -*- coding: utf-8 -*-
"""
CCM核心算法和代理方法模块 (core/ccm.py)
CCM Core Algorithm and Surrogate Methods Module
[v3.3 Final]: 彻底重构 ccm_pearson 函数，以根除类型错误并提升性能。
"""

import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
from termcolor import colored

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
        
    shadow = np.column_stack([series[i * tau : L - (Dim - 1 - i) * tau] for i in range(Dim)])
    
    nbrs = NearestNeighbors(n_neighbors=Dim + 2, algorithm='ball_tree').fit(shadow)
    distances, indices = nbrs.kneighbors(shadow)
    
    distances, indices = distances[:, 1:], indices[:, 1:]
    
    e = 1e-10
    u = np.exp(-distances / (distances[:, [0]] + e))
    
    sum_u = np.sum(u, axis=1, keepdims=True)
    weights = np.divide(u, sum_u, where=sum_u!=0)

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
    
    cov = np.sum((original_batch - mean_orig) * (prediction_batch - mean_pred), axis=1) / (shadow_length - 1)
    
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
    """傅里叶变换代理：保持功率谱，随机化相位。"""
    freqs = np.fft.fft(data)
    random_phases = np.exp(1j * np.random.uniform(-np.pi, np.pi, len(data)))
    random_phases[len(data)//2:] = np.conj(random_phases[1:len(data)//2+1][::-1])
    return np.fft.ifft(np.abs(freqs) * random_phases).real

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
        
    conditional_shadow = np.column_stack([conditional_series[i * tau: L - (Dim - 1 - i) * tau] for i in range(Dim)])
    nbrs = NearestNeighbors(n_neighbors=Dim + 2, algorithm='ball_tree').fit(conditional_shadow)
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
        'FFT': _surrogate_fft,
        'AAFT': lambda d: _surrogate_iaaft(d, num_iterations=1), # AAFT是IAAFT的单次迭代
        'IAAFT': _surrogate_iaaft,
        'Time Shift': lambda d: np.roll(d, np.random.randint(1, len(d))),
        'Random Reorder': np.random.permutation,
        'Conditional': lambda d: _surrogate_conditional(d, **kwargs)
    }
    
    if method not in surrogate_funcs:
        raise ValueError(f"不支持的代理方法: {method}")
    
    func = surrogate_funcs[method]
    
    return np.array([func(data) for _ in range(num_surrogates)])
