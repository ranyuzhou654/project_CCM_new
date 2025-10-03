# -*- coding: utf-8 -*-
"""
核心分析流程模块 (core/analysis.py)
Core Analysis Workflow Module

包含运行单次试验和完整多维度性能分析的函数。
[v3.13 Final]: 扩展了节点分析的范围，以探究小系统的表现。
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime  # 导入 datetime

import numpy as np
from tqdm import tqdm
from termcolor import colored
from sklearn.metrics import roc_curve, auc

# 导入项目模块
from .systems import generate_time_series, generate_adjacency_matrix
from .ccm import (
    parameters,
    ccm_pearson,
    generate_surrogates,
    improved_ccm_confidence,
    adaptive_surrogate_testing,
    bootstrap_auroc_confidence,
)
from utils.params import optimize_embedding_for_system, adaptive_optimize_embedding_params


def _determine_worker_count(requested_workers, num_trials):
    """Return an effective worker count bounded by CPU availability and trials."""

    if num_trials <= 1:
        return 1

    if requested_workers is not None:
        return max(1, min(requested_workers, num_trials))

    cpu_total = os.cpu_count() or 1
    return max(1, min(cpu_total, num_trials))


def _run_single_trial_worker(params):
    """Wrapper to keep ProcessPool submissions picklable."""

    return run_single_trial(params)


def _execute_trials(current_params, num_trials, max_workers=None):
    """
    Run multiple trials either sequentially or in a process pool.
    
    改进: 使用 SeedSequence 为每个试验分配独立的种子，避免多进程环境中的种子冲突。
    """

    worker_count = _determine_worker_count(max_workers, num_trials)

    if worker_count == 1:
        # 为单进程执行生成独立种子
        base_seed = current_params.get("base_seed", None)
        if base_seed is not None:
            seed_sequence = np.random.SeedSequence(base_seed)
            trial_seeds = seed_sequence.spawn(num_trials)
            trial_params_list = []
            for i in range(num_trials):
                params = current_params.copy()
                params["trial_seed"] = trial_seeds[i]
                trial_params_list.append(params)
            return [run_single_trial(params) for params in trial_params_list]
        else:
            return [run_single_trial(current_params) for _ in range(num_trials)]

    # 为多进程执行生成独立种子
    base_seed = current_params.get("base_seed", None)
    if base_seed is not None:
        seed_sequence = np.random.SeedSequence(base_seed)
        trial_seeds = seed_sequence.spawn(num_trials)
    else:
        # 如果没有提供基础种子，生成随机种子序列
        seed_sequence = np.random.SeedSequence()
        trial_seeds = seed_sequence.spawn(num_trials)

    trial_params = []
    for i in range(num_trials):
        params = current_params.copy()
        params["trial_seed"] = trial_seeds[i]
        trial_params.append(params)

    results = []

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_run_single_trial_worker, params) for params in trial_params
        ]
        for future in futures:
            results.append(future.result())

    return results


def run_single_trial(params):
    """
    运行单次CCM因果分析试验 (改进版)

    改进内容:
    - 使用改进的置信度计算方法（KDE或ECDF插值）
    - 可选的自适应代理数量测试
    - Bootstrap置信区间计算
    - 现代化随机种子管理，支持独立试验和可重现性
    """
    # 解包参数
    system_type = params["system_type"]
    time_series_length = params["time_series_length"]
    num_systems = params.get("num_systems", 5)
    degree = params.get("degree", 5)
    epsilon = params.get("epsilon", 0.1)
    noise_level = params.get("noise_level", 0.0)

    Dim = params["Dim"]
    tau = params["tau"]
    num_surrogates = params.get("num_surrogates", 200)  # 默认增加到200
    method = params["method"]

    # 新增参数
    use_adaptive = params.get("use_adaptive", True)  # 是否使用自适应代理测试
    confidence_method = params.get("confidence_method", "kde")  # 置信度计算方法
    compute_bootstrap = params.get(
        "compute_bootstrap", False
    )  # 是否计算bootstrap置信区间
    
    # 现代化随机种子管理
    trial_seed = params.get("trial_seed", None)
    if trial_seed is not None:
        # 处理 SeedSequence 对象
        if hasattr(trial_seed, 'generate_state'):
            # 这是一个 SeedSequence 对象，需要生成整数种子
            rng = np.random.default_rng(trial_seed)
            # 为向后兼容性生成整数种子
            int_seed = int(trial_seed.generate_state(1)[0])
            np.random.seed(int_seed)
        else:
            # 这是一个整数种子
            rng = np.random.default_rng(trial_seed)
            np.random.seed(trial_seed)
    else:
        # 如果没有提供种子，使用默认随机状态
        rng = np.random.default_rng()

    # 1. 生成数据和质量检查
    max_regeneration_attempts = 3  # 最大重新生成次数
    
    for attempt in range(max_regeneration_attempts):
        adjacency_matrix = generate_adjacency_matrix(num_systems, degree)
        time_series = generate_time_series(
            system_type,
            num_systems,
            adjacency_matrix,
            time_series_length,
            epsilon,
            noise_level=noise_level,
        )

        # 基础有效性检查
        if np.any(~np.isfinite(time_series)):
            if attempt == max_regeneration_attempts - 1:
                result = {"auroc": 0.5, "scores": np.full((num_systems, num_systems), 0.5)}
                if compute_bootstrap:
                    result["bootstrap_ci"] = (0.5, 0.5)
                return result
            continue
        
        # 质量检查：检查序列是否"足够混沌"
        series_quality_ok = True
        min_variance_threshold = 1e-8  # 最小方差阈值
        max_variance_ratio = 1000.0    # 最大方差比阈值（避免数值爆炸）
        
        for i in range(num_systems):
            series_var = np.var(time_series[i])
            series_mean = np.abs(np.mean(time_series[i]))
            
            # 检查方差是否过小（可能陷入同步状态）
            if series_var < min_variance_threshold:
                series_quality_ok = False
                break
                
            # 检查方差是否过大（可能数值发散）
            if series_mean > 0 and series_var / (series_mean**2) > max_variance_ratio:
                series_quality_ok = False
                break
                
            # 对于敏感系统（如Henon），进行额外检查
            if system_type in ["henon", "noisy_henon", "henon_dynamic_noise"]:
                # 检查序列是否退化为常数（Henon系统的常见问题）
                if np.std(time_series[i]) < 0.001:
                    series_quality_ok = False
                    break
                    
                # 检查是否存在数值溢出
                if np.max(np.abs(time_series[i])) > 10.0:
                    series_quality_ok = False
                    break
        
        if series_quality_ok:
            break
        elif attempt == max_regeneration_attempts - 1:
            # 如果所有尝试都失败，返回默认结果
            result = {"auroc": 0.5, "scores": np.full((num_systems, num_systems), 0.5)}
            if compute_bootstrap:
                result["bootstrap_ci"] = (0.5, 0.5)
            result["quality_warning"] = "Time series quality check failed after multiple attempts"
            return result

    true_causality = adjacency_matrix.T.flatten()

    # 2. 计算CCM和改进的置信度分数
    scores = np.zeros((num_systems, num_systems))
    adaptive_info = {}  # 存储自适应测试信息

    for i in range(num_systems):
        weights, indices = parameters(time_series[i], Dim, tau)
        if weights is None:
            continue

        shadow_len = weights.shape[0]

        for j in range(num_systems):
            if i == j:
                continue

            original_coef = np.abs(
                ccm_pearson(time_series[j], weights, indices, shadow_len)
            )

            if method == "No Surrogate":
                scores[i, j] = original_coef
            else:
                if use_adaptive:
                    # 使用自适应代理数量测试
                    confidence_score, n_surrogates_used, convergence_history = (
                        adaptive_surrogate_testing(
                            time_series[j],
                            method,
                            weights,
                            indices,
                            shadow_len,
                            min_surrogates=num_surrogates,
                            max_surrogates=min(2000, num_surrogates * 10),
                        )
                    )
                    scores[i, j] = confidence_score
                    adaptive_info[f"{i}->{j}"] = {
                        "n_surrogates_used": n_surrogates_used,
                        "convergence_history": convergence_history,
                    }
                else:
                    # 使用固定数量的代理，但应用改进的置信度计算
                    surrogates = generate_surrogates(
                        time_series[j], method, num_surrogates
                    )
                    surrogate_coefs = np.abs(
                        ccm_pearson(surrogates, weights, indices, shadow_len)
                    )

                    # 确保surrogate_coefs是一维数组
                    if surrogate_coefs.ndim == 0:
                        surrogate_coefs = np.array([surrogate_coefs])

                    confidence_score = improved_ccm_confidence(
                        original_coef, surrogate_coefs, method=confidence_method
                    )
                    scores[i, j] = confidence_score

    # 3. 计算AUROC
    if scores.flatten().shape != true_causality.shape:
        result = {"auroc": 0.5, "scores": np.full((num_systems, num_systems), 0.5)}
        if compute_bootstrap:
            result["bootstrap_ci"] = (0.5, 0.5)
        return result

    try:
        fpr, tpr, _ = roc_curve(true_causality, scores.flatten())
        auroc = auc(fpr, tpr)

        # 计算bootstrap置信区间（如果请求）
        bootstrap_ci = None
        if compute_bootstrap:
            _, bootstrap_ci = bootstrap_auroc_confidence(
                scores.flatten(),
                true_causality,
                n_bootstrap=1000,
                confidence_level=0.95,
            )

    except ValueError:
        auroc = 0.5
        bootstrap_ci = (0.5, 0.5) if compute_bootstrap else None

    # 构建返回结果
    result = {"auroc": auroc, "scores": scores}

    if compute_bootstrap and bootstrap_ci is not None:
        result["bootstrap_ci"] = bootstrap_ci

    if use_adaptive:
        result["adaptive_info"] = adaptive_info

    return result


def run_enhanced_analysis(
    system_type,
    analysis_type,
    visualizer,
    num_trials=20,
    num_surrogates=200,
    use_adaptive=True,
    confidence_method="kde",
    compute_bootstrap=True,
    max_workers=None,
):
    """
    运行增强版完整分析，包含改进的统计方法

    新增功能:
    - 自适应代理数量测试
    - 改进的置信度计算
    - Bootstrap置信区间
    - 详细的统计信息输出
    """
    methods = ["No Surrogate", "FFT", "AAFT", "IAAFT", "Time Shift", "Random Reorder"]

    print(colored("--- [步骤 1/4] 正在自动优化嵌入参数... ---", "cyan"))
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

    base_params = {**optimal_params}
    base_params.update({
        "system_type": system_type,
        "num_surrogates": num_surrogates,
        "use_adaptive": use_adaptive,
        "confidence_method": confidence_method,
        "compute_bootstrap": compute_bootstrap,
    })

    # 确定分析参数
    variable_param_name, variable_param_values, x_label, title, x_plot_values = (
        None,
        [],
        "",
        "",
        [],
    )

    if analysis_type == "length":
        base_params.update({"num_systems": 5, "degree": 5, "epsilon": 0.3})
        variable_param_name = "time_series_length"
        variable_param_values = [50, 70, 100, 150, 200]
        x_label, title = (
            "Time Series Length",
            f"Enhanced AUROC vs. Time Series Length ({system_type.capitalize()})",
        )
        x_plot_values = variable_param_values

    elif analysis_type == "degree":
        num_systems = 5
        base_params.update({
            "time_series_length": 2000, "num_systems": num_systems, "epsilon": 0.3
        })
        variable_param_name = "degree"
        degrees = [int(r * num_systems) for r in [1, 2, 4, 6, 8]]
        variable_param_values = degrees
        x_label, title = (
            "Average Degree",
            f"Enhanced AUROC vs. Average Degree ({system_type.capitalize()})",
        )
        x_plot_values = [d / (num_systems - 1) for d in degrees]

    elif analysis_type == "coupling":
        base_params.update({"time_series_length": 2000, "num_systems": 5, "degree": 5})
        variable_param_name = "epsilon"
        variable_param_values = np.linspace(0.05, 0.8, 5)
        x_label, title = (
            "Coupling Strength ε",
            f"Enhanced AUROC vs. Coupling Strength ({system_type.capitalize()})",
        )
        x_plot_values = variable_param_values

    elif analysis_type == "nodes":
        base_params.update({"time_series_length": 4000, "epsilon": 0.3})
        variable_param_name = "num_systems"
        variable_param_values = [3, 4, 5, 8, 12, 15]
        base_params["avg_degree"] = 2.0
        x_label, title = (
            "Number of Nodes N",
            f"Enhanced AUROC vs. Number of Nodes ({system_type.capitalize()})",
        )
        x_plot_values = variable_param_values

    elif analysis_type == "noise":
        base_params.update({
            "num_systems": 5, "degree": 5, "epsilon": 0.3, "time_series_length": 4000
        })
        variable_param_name = "noise_level"
        variable_param_values = np.linspace(0, 0.5, 5)
        x_label, title = (
            "Noise Level",
            f"Enhanced AUROC vs. Noise Level ({system_type.capitalize()})",
        )
        x_plot_values = variable_param_values
        base_params["system_type"] = system_type

    else:
        raise ValueError(f"不支持的分析类型: {analysis_type}")

    print(colored("\n--- [步骤 2/4] 正在执行增强版性能分析... ---", "cyan"))
    results_raw = {m: [] for m in methods}
    bootstrap_results = {m: [] for m in methods}  # 存储bootstrap置信区间
    adaptive_stats = {m: [] for m in methods}  # 存储自适应统计信息

    for value in tqdm(variable_param_values, desc=f"Processing {x_label}"):
        for method in methods:
            auroc_trials_for_value = []
            bootstrap_cis_for_value = []
            adaptive_info_for_value = []

            current_params = base_params.copy()
            current_params[variable_param_name] = value
            current_params["method"] = method

            if analysis_type == "nodes":
                num_nodes = value
                degree = int(base_params["avg_degree"] * num_nodes)
                max_degree = num_nodes * (num_nodes - 1)
                current_params["degree"] = min(max_degree, degree)

            trial_results = _execute_trials(
                current_params, num_trials, max_workers=max_workers
            )

            for result in trial_results:
                auroc_trials_for_value.append(result["auroc"])

                if compute_bootstrap and "bootstrap_ci" in result:
                    bootstrap_cis_for_value.append(result["bootstrap_ci"])

                if use_adaptive and "adaptive_info" in result:
                    adaptive_info_for_value.append(result["adaptive_info"])

            results_raw[method].append(auroc_trials_for_value)

            if compute_bootstrap:
                bootstrap_results[method].append(bootstrap_cis_for_value)

            if use_adaptive:
                adaptive_stats[method].append(adaptive_info_for_value)

    # 计算统计量
    results_mean = {m: [np.mean(trials) for trials in results_raw[m]] for m in methods}
    results_std = {m: [np.std(trials) for trials in results_raw[m]] for m in methods}

    # 计算bootstrap置信区间的平均值
    bootstrap_mean_cis = {}
    if compute_bootstrap:
        for method in methods:
            method_cis = []
            for value_cis in bootstrap_results[method]:
                if value_cis:  # 确保有数据
                    lower_bounds = [ci[0] for ci in value_cis if ci is not None]
                    upper_bounds = [ci[1] for ci in value_cis if ci is not None]
                    if lower_bounds and upper_bounds:
                        mean_ci = (np.mean(lower_bounds), np.mean(upper_bounds))
                        method_cis.append(mean_ci)
                    else:
                        method_cis.append((0.5, 0.5))
                else:
                    method_cis.append((0.5, 0.5))
            bootstrap_mean_cis[method] = method_cis

    print(colored("\n--- [步骤 3/4] 正在生成增强版可视化图表... ---", "cyan"))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"enhanced_analysis_{system_type}_{analysis_type}_{timestamp}.png"

    # 准备额外的绘图数据
    extra_plot_data = {}
    if compute_bootstrap:
        extra_plot_data["bootstrap_cis"] = bootstrap_mean_cis

    # 创建可视化（需要在可视化模块中添加对bootstrap CI的支持）
    visualizer.create_enhanced_performance_plot(
        x_plot_values,
        results_mean,
        results_std,
        methods,
        x_label,
        title,
        num_trials,
        save_path=save_path,
        extra_data=extra_plot_data,
        use_adaptive=use_adaptive,
        confidence_method=confidence_method,
    )

    print(colored("\n--- [步骤 4/4] 正在保存详细结果数据... ---", "cyan"))

    # 保存增强版结果
    enhanced_results = {
        "metadata": {
            "system_type": system_type,
            "analysis_type": analysis_type,
            "num_trials": num_trials,
            "num_surrogates": num_surrogates,
            "use_adaptive": use_adaptive,
            "confidence_method": confidence_method,
            "compute_bootstrap": compute_bootstrap,
            "optimal_params": optimal_params,
            "timestamp": timestamp,
        },
        "x_values": x_plot_values,
        "results_mean": results_mean,
        "results_std": results_std,
        "results_raw": results_raw,
    }

    if compute_bootstrap:
        enhanced_results["bootstrap_results"] = bootstrap_mean_cis

    if use_adaptive:
        enhanced_results["adaptive_stats"] = adaptive_stats

    results_file = f"enhanced_results_{system_type}_{analysis_type}_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(enhanced_results, f, indent=2, ensure_ascii=False, default=str)

    print(colored(f"✅ 增强版分析完成！", "green"))
    print(colored(f"📊 可视化图表: {save_path}", "blue"))
    print(colored(f"💾 详细结果: {results_file}", "blue"))

    return enhanced_results


def run_full_analysis(
    system_type,
    analysis_type,
    visualizer,
    num_trials=20,
    num_surrogates=100,
    max_workers=None,
):
    """
    [v3.13] 运行指定系统和类型的完整多维度分析，并自动处理参数优化。
    """
    methods = ["No Surrogate", "FFT", "AAFT", "IAAFT", "Time Shift", "Random Reorder"]

    print(colored("--- [步骤 1/3] 正在自动优化嵌入参数... ---", "cyan"))
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

    base_params = {**optimal_params}
    base_params.update({"system_type": system_type, "num_surrogates": num_surrogates})

    variable_param_name, variable_param_values, x_label, title, x_plot_values = (
        None,
        [],
        "",
        "",
        [],
    )

    if analysis_type == "length":
        base_params.update({"num_systems": 5, "degree": 5, "epsilon": 0.3})
        variable_param_name = "time_series_length"
        variable_param_values = [200, 500, 800, 1000, 1200]
        x_label, title = (
            "Time Series Length",
            f"AUROC vs. Time Series Length ({system_type.capitalize()})",
        )
        x_plot_values = variable_param_values

    elif analysis_type == "degree":
        num_systems = 5
        base_params.update({
            "time_series_length": 2000, "num_systems": num_systems, "epsilon": 0.3
        })
        variable_param_name = "degree"
        degrees = [int(r * num_systems) for r in [1, 2, 4, 6, 8]]
        variable_param_values = degrees
        x_label, title = (
            "Average Degree",
            f"AUROC vs. Average Degree ({system_type.capitalize()})",
        )
        x_plot_values = [d / (num_systems - 1) for d in degrees]

    elif analysis_type == "coupling":
        base_params.update({"time_series_length": 2000, "num_systems": 5, "degree": 5})
        variable_param_name = "epsilon"
        variable_param_values = np.linspace(0.05, 0.8, 5)
        x_label, title = (
            "Coupling Strength ε",
            f"AUROC vs. Coupling Strength ({system_type.capitalize()})",
        )
        x_plot_values = variable_param_values

    elif analysis_type == "nodes":
        # [v3.13] 扩展节点分析范围以包含小系统
        base_params.update({"time_series_length": 4000, "epsilon": 0.3})
        variable_param_name = "num_systems"
        variable_param_values = [3, 4, 5, 8, 12, 15]  # 新的测试范围
        base_params["avg_degree"] = 2.0  # 设定每个节点平均有2个入边
        x_label, title = (
            "Number of Nodes N",
            f"AUROC vs. Number of Nodes ({system_type.capitalize()})",
        )
        x_plot_values = variable_param_values

    elif analysis_type == "noise":
        base_params.update({
            "num_systems": 5, "degree": 5, "epsilon": 0.3, "time_series_length": 4000
        })
        variable_param_name = "noise_level"
        variable_param_values = np.linspace(0, 0.5, 5)
        x_label, title = (
            "Noise Level",
            f"AUROC vs. Noise Level ({system_type.capitalize()})",
        )
        x_plot_values = variable_param_values
        # 保持系统类型不变，让噪声分析直接使用传入的噪声系统类型
        base_params["system_type"] = system_type

    else:
        raise ValueError(f"不支持的分析类型: {analysis_type}")

    print(colored("\n--- [步骤 2/3] 正在执行核心性能分析... ---", "cyan"))
    results_raw = {m: [] for m in methods}

    for value in tqdm(variable_param_values, desc=f"Processing {x_label}"):
        
        # 自适应参数优化：根据当前变量值重新评估参数
        adaptive_params = adaptive_optimize_embedding_params(
            system_type, analysis_type, value, base_length=8000
        )
        
        # 如果自适应优化成功，使用新参数
        if adaptive_params and adaptive_params != optimal_params:
            print(colored(f"使用自适应参数 (Dim={adaptive_params['Dim']}, tau={adaptive_params['tau']}) 于 {analysis_type}={value}", "blue"))
            current_base_params = base_params.copy()
            current_base_params.update(adaptive_params)
        else:
            current_base_params = base_params
        
        for method in methods:
            auroc_trials_for_value = []
            current_params = current_base_params.copy()
            current_params[variable_param_name] = value
            current_params["method"] = method

            if analysis_type == "nodes":
                num_nodes = value
                # 动态计算度数以保持网络密度大致恒定
                degree = int(base_params["avg_degree"] * num_nodes)
                # 确保度数不超过最大可能值
                max_degree = num_nodes * (num_nodes - 1)
                current_params["degree"] = min(max_degree, degree)

            trial_results = _execute_trials(
                current_params, num_trials, max_workers=max_workers
            )

            for result in trial_results:
                # 兼容新的返回格式
                if isinstance(result, dict):
                    auroc = result["auroc"]
                else:
                    # 向后兼容旧格式
                    auroc = result[0] if isinstance(result, tuple) else result
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
    for key in ["system_type", "num_surrogates", "Dim", "tau", "method", "avg_degree"]:
        if key in constant_params:
            del constant_params[key]

    visualizer.create_comprehensive_performance_plot(
        x_plot_values,
        results_mean,
        results_std,
        methods,
        x_label,
        title,
        num_trials,
        save_path=save_path,
        raw_results=results_raw,
        num_surrogates=num_surrogates,
        constant_params=constant_params,  # 传递给可视化函数
    )

    results_data = {
        "x_values": (
            x_plot_values if isinstance(x_plot_values, list) else x_plot_values.tolist()
        ),
        "results_mean": results_mean,
        "results_std": results_std,
        "results_raw": results_raw,
        "parameters": {k: v for k, v in base_params.items() if k not in ["Dim", "tau"]},
        "embedding_params": {"Dim": base_params["Dim"], "tau": base_params["tau"]},
        "analysis_info": {
            "system_type": system_type,
            "analysis_type": analysis_type,
            "num_trials": num_trials,
            "num_surrogates": num_surrogates,
            "methods": methods,
        },
    }

    results_file = f"results_{system_type}_{analysis_type}_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
