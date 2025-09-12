# -*- coding: utf-8 -*-
"""
核心分析流程模块 (core/analysis.py)
Core Analysis Workflow Module

包含运行单次试验和完整多维度性能分析的函数。
[v3.13 Final]: 扩展了节点分析的范围，以探究小系统的表现。
"""

import numpy as np
from tqdm import tqdm
from termcolor import colored
import json
from sklearn.metrics import roc_curve, auc
from datetime import datetime # 导入 datetime

# 导入项目模块
from .systems import generate_time_series, generate_adjacency_matrix
from .ccm import parameters, ccm_pearson, generate_surrogates
from utils.params import optimize_embedding_for_system

def run_single_trial(params):
    """
    运行单次CCM因果分析试验。
    """
    # 解包参数
    system_type = params['system_type']
    time_series_length = params['time_series_length']
    num_systems = params.get('num_systems', 5)
    degree = params.get('degree', 5)
    epsilon = params.get('epsilon', 0.1)
    noise_level = params.get('noise_level', 0.0)
    
    Dim = params['Dim']
    tau = params['tau']
    num_surrogates = params.get('num_surrogates', 100)
    method = params['method']
    
    # 1. 生成数据
    np.random.seed()
    adjacency_matrix = generate_adjacency_matrix(num_systems, degree)
    time_series = generate_time_series(system_type, num_systems, adjacency_matrix, time_series_length, epsilon, noise_level=noise_level)
    
    if np.any(~np.isfinite(time_series)):
        return 0.5, np.full((num_systems, num_systems), 0.5)

    true_causality = adjacency_matrix.T.flatten()
    
    # 2. 计算CCM和代理
    scores = np.zeros((num_systems, num_systems))
    
    for i in range(num_systems):
        weights, indices = parameters(time_series[i], Dim, tau)
        if weights is None:
            continue
            
        shadow_len = weights.shape[0]
        
        for j in range(num_systems):
            if i == j:
                continue
            
            original_coef = np.abs(ccm_pearson(time_series[j], weights, indices, shadow_len))
            
            if method == 'No Surrogate':
                scores[i, j] = original_coef
            else:
                surrogates = generate_surrogates(time_series[j], method, num_surrogates)
                surrogate_coefs = np.abs(ccm_pearson(surrogates, weights, indices, shadow_len))
                p_value_score = np.sum(surrogate_coefs < original_coef) / num_surrogates
                scores[i, j] = p_value_score

    # 3. 计算AUROC
    if scores.flatten().shape != true_causality.shape:
        return 0.5, np.full((num_systems, num_systems), 0.5)

    try:
        fpr, tpr, _ = roc_curve(true_causality, scores.flatten())
        auroc = auc(fpr, tpr)
    except ValueError:
        auroc = 0.5

    return auroc, scores

def run_full_analysis(system_type, analysis_type, visualizer, num_trials=20, num_surrogates=100):
    """
    [v3.13] 运行指定系统和类型的完整多维度分析，并自动处理参数优化。
    """
    methods = ['No Surrogate', 'FFT', 'AAFT', 'IAAFT', 'Time Shift', 'Random Reorder']
    
    print(colored("--- [步骤 1/3] 正在自动优化嵌入参数... ---", "cyan"))
    optimal_params = optimize_embedding_for_system(system_type, series_length=8000)
    if not optimal_params:
        print(colored(f"错误: 无法为 {system_type} 系统确定最佳参数。分析中止。", "red"))
        return
    print(colored(f"✅ 最佳参数确定: Dim = {optimal_params['Dim']}, tau = {optimal_params['tau']}", "green"))
    
    base_params = {**optimal_params}
    base_params.update({'system_type': system_type, 'num_surrogates': num_surrogates})

    variable_param_name, variable_param_values, x_label, title, x_plot_values = (None, [], '', '', [])

    if analysis_type == 'length':
        base_params.update({'num_systems': 5, 'degree': 5, 'epsilon': 0.3})
        variable_param_name = 'time_series_length'
        variable_param_values = [200, 500, 800, 1000, 1200]
        x_label, title = 'Time Series Length', f'AUROC vs. Time Series Length ({system_type.capitalize()})'
        x_plot_values = variable_param_values
    
    elif analysis_type == 'degree':
        num_systems = 5
        base_params.update({'time_series_length': 2000, 'num_systems': num_systems, 'epsilon': 0.3})
        variable_param_name = 'degree'
        degrees = [int(r * num_systems) for r in [1, 2, 4, 6, 8]]
        variable_param_values = degrees
        x_label, title = 'Average Degree', f'AUROC vs. Average Degree ({system_type.capitalize()})'
        x_plot_values = [d / (num_systems - 1) for d in degrees]

    elif analysis_type == 'coupling':
        base_params.update({'time_series_length': 2000, 'num_systems': 5, 'degree': 5})
        variable_param_name = 'epsilon'
        variable_param_values = np.linspace(0.05, 0.8, 5)
        x_label, title = 'Coupling Strength ε', f'AUROC vs. Coupling Strength ({system_type.capitalize()})'
        x_plot_values = variable_param_values

    elif analysis_type == 'nodes':
        # [v3.13] 扩展节点分析范围以包含小系统
        base_params.update({'time_series_length': 4000, 'epsilon': 0.3})
        variable_param_name = 'num_systems'
        variable_param_values = [3, 4, 5, 8, 12, 15] # 新的测试范围
        base_params['avg_degree'] = 2.0 # 设定每个节点平均有2个入边
        x_label, title = 'Number of Nodes N', f'AUROC vs. Number of Nodes ({system_type.capitalize()})'
        x_plot_values = variable_param_values

    elif analysis_type == 'noise':
        base_params.update({'num_systems': 5, 'degree': 5, 'epsilon': 0.3, 'time_series_length': 4000})
        variable_param_name = 'noise_level'
        variable_param_values = np.linspace(0, 0.5, 5)
        x_label, title = 'Noise Level', f'AUROC vs. Noise Level ({system_type.capitalize()})'
        x_plot_values = variable_param_values
        # 保持系统类型不变，让噪声分析直接使用传入的噪声系统类型
        base_params['system_type'] = system_type
    
    else:
        raise ValueError(f"不支持的分析类型: {analysis_type}")

    print(colored("\n--- [步骤 2/3] 正在执行核心性能分析... ---", "cyan"))
    results_raw = {m: [] for m in methods}

    for value in tqdm(variable_param_values, desc=f"Processing {x_label}"):
        for method in methods:
            auroc_trials_for_value = []
            current_params = base_params.copy()
            current_params[variable_param_name] = value
            current_params['method'] = method
            
            if analysis_type == 'nodes':
                num_nodes = value
                # 动态计算度数以保持网络密度大致恒定
                degree = int(base_params['avg_degree'] * num_nodes)
                # 确保度数不超过最大可能值
                max_degree = num_nodes * (num_nodes - 1)
                current_params['degree'] = min(max_degree, degree)

            for _ in range(num_trials):
                auroc, _ = run_single_trial(current_params)
                auroc_trials_for_value.append(auroc)
            
            results_raw[method].append(auroc_trials_for_value)

    results_mean = {m: [np.mean(trials) for trials in results_raw[m]] for m in methods}
    results_std = {m: [np.std(trials) for trials in results_raw[m]] for m in methods}

    print(colored("\n--- [步骤 3/3] 正在生成可视化图表和结果文件... ---", "cyan"))
    
    # [改进] 添加时间戳以创建唯一文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"analysis_{system_type}_{analysis_type}_{timestamp}.png"
    
    # [改进] 提取在本次分析中保持不变的参数
    constant_params = base_params.copy()
    if variable_param_name in constant_params:
        del constant_params[variable_param_name]
    # 移除其他非核心参数信息，避免标题过长
    for key in ['system_type', 'num_surrogates', 'Dim', 'tau', 'method', 'avg_degree']:
        if key in constant_params:
            del constant_params[key]

    visualizer.create_comprehensive_performance_plot(
        x_plot_values, results_mean, results_std, methods, 
        x_label, title, num_trials,
        save_path=save_path,
        raw_results=results_raw,
        num_surrogates=num_surrogates,
        constant_params=constant_params # 传递给可视化函数
    )
    
    results_data = {
        'x_values': x_plot_values if isinstance(x_plot_values, list) else x_plot_values.tolist(),
        'results_mean': results_mean,
        'results_std': results_std,
        'results_raw': results_raw,
        'parameters': {k: v for k, v in base_params.items() if k not in ['Dim', 'tau']},
        'embedding_params': {'Dim': base_params['Dim'], 'tau': base_params['tau']},
        'analysis_info': {
            'system_type': system_type,
            'analysis_type': analysis_type,
            'num_trials': num_trials,
            'num_surrogates': num_surrogates,
            'methods': methods
        }
    }
    
    results_file = f"results_{system_type}_{analysis_type}_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
