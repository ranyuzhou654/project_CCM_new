# -*- coding: utf-8 -*-
"""
部分交叉映射 (Partial CCM) 模块 (core/partial_ccm.py)
Partial Cross-Mapping (Partial CCM) Module

[v3.8 Final]: 增强了错误处理，当计算失败时返回0.0而不是NaN，以便在图表中显示。
"""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from termcolor import colored

# 导入项目内的模块
from .ccm import parameters, ccm_pearson, generate_surrogates

def _get_residuals(target_series, driver_series, dim, tau):
    """
    使用K近邻回归模型，计算目标序列中无法被驱动序列解释的部分（残差）。
    """
    L = len(driver_series)
    shadow_length = L - (dim - 1) * tau
    if shadow_length <= dim:
        print(colored("警告: 序列过短，无法计算残差。", "yellow"))
        return np.zeros_like(target_series)

    driver_shadow = np.column_stack([driver_series[i * tau : L - (dim - 1 - i) * tau] for i in range(dim)])
    
    target_aligned = target_series[(dim - 1) * tau:]
    
    knn = KNeighborsRegressor(n_neighbors=dim + 1)
    knn.fit(driver_shadow, target_aligned)
    
    target_predicted = knn.predict(driver_shadow)
    
    residuals = target_aligned - target_predicted
    
    padding = np.zeros(len(target_series) - len(residuals))
    
    return np.concatenate([padding, residuals])


def run_partial_ccm_trial(X, Y, Z, dim, tau, num_surrogates=100):
    """
    [v3.8] 执行一次完整的部分交叉映射分析。
    """
    try:
        residual_X = _get_residuals(X, Z, dim, tau)
        residual_Y = _get_residuals(Y, Z, dim, tau)
        
        # [v3.8] 关键检查：如果残差信号没有动态（方差过小），则直接判定为不显著
        if np.var(residual_X) < 1e-10 or np.var(residual_Y) < 1e-10:
            print(colored("信息: Partial CCM残差信号无方差，判定为不显著。", "blue"))
            return 0.0

        weights_res_Y, indices_res_Y = parameters(residual_Y, dim, tau)
        if weights_res_Y is None:
            return 0.0
        shadow_len = weights_res_Y.shape[0]

        original_corr = np.abs(ccm_pearson(residual_X, weights_res_Y, indices_res_Y, shadow_len))

        surrogates_res_X = generate_surrogates(residual_X, 'Time Shift', num_surrogates)
        surrogate_corrs = np.abs(ccm_pearson(surrogates_res_X, weights_res_Y, indices_res_Y, shadow_len))
        
        p_value = np.sum(surrogate_corrs >= original_corr) / num_surrogates
        
        return 1.0 - p_value

    except Exception as e:
        print(colored(f"部分交叉映射分析失败: {e}", "red"))
        # [v3.8] 当发生任何未预料的错误时，返回0.0，表示不显著
        return 0.0
