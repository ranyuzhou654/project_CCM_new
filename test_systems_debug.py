#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统稳定性测试脚本 - 检测和诊断动力学系统的数值问题
"""

import numpy as np
import matplotlib.pyplot as plt
from core.systems import generate_time_series, generate_adjacency_matrix
from termcolor import colored
import warnings

warnings.filterwarnings("ignore")


def test_system_stability(system_name, plot=True):
    """测试单个系统的数值稳定性"""
    print(f"\n{'='*50}")
    print(colored(f"测试系统: {system_name}", "cyan", attrs=["bold"]))
    print(f"{'='*50}")

    # 基本参数
    num_systems = 2
    adjacency_matrix = generate_adjacency_matrix(num_systems, 1)  # 简单连接
    t_steps = 1000
    epsilon = 0.1

    try:
        # 生成时间序列
        series = generate_time_series(
            system_name,
            num_systems,
            adjacency_matrix,
            t_steps,
            epsilon,
            noise_level=0.0,
        )

        # 检查NaN和inf
        has_nan = np.any(np.isnan(series))
        has_inf = np.any(np.isinf(series))

        # 检查数值范围
        min_val = np.min(series)
        max_val = np.max(series)
        mean_val = np.mean(series)
        std_val = np.std(series)

        print(f"时间序列形状: {series.shape}")
        print(f"数值范围: [{min_val:.6f}, {max_val:.6f}]")
        print(f"均值: {mean_val:.6f}, 标准差: {std_val:.6f}")

        if has_nan:
            print(colored("❌ 发现NaN值!", "red", attrs=["bold"]))
            nan_count = np.sum(np.isnan(series))
            print(f"   NaN数量: {nan_count}")

        if has_inf:
            print(colored("❌ 发现无穷大值!", "red", attrs=["bold"]))
            inf_count = np.sum(np.isinf(series))
            print(f"   Inf数量: {inf_count}")

        if abs(max_val) > 1e10 or abs(min_val) > 1e10:
            print(colored("⚠️  数值范围过大，可能存在数值不稳定!", "yellow"))

        if not (has_nan or has_inf) and abs(max_val) < 1e10:
            print(colored("✅ 系统数值稳定", "green", attrs=["bold"]))

        # 绘图
        if plot and not (has_nan or has_inf):
            plt.figure(figsize=(12, 8))

            # 时间序列图
            plt.subplot(2, 2, 1)
            time_axis = np.arange(t_steps)
            plt.plot(time_axis, series[0], "b-", alpha=0.7, linewidth=1, label="系统1")
            plt.plot(time_axis, series[1], "r-", alpha=0.7, linewidth=1, label="系统2")
            plt.title(f"{system_name} - 时间序列")
            plt.xlabel("时间步")
            plt.ylabel("数值")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 相空间图（如果适用）
            if len(series) >= 2:
                plt.subplot(2, 2, 2)
                plt.plot(series[0], series[1], "b-", alpha=0.6, linewidth=0.8)
                plt.title(f"{system_name} - 相空间")
                plt.xlabel("系统1")
                plt.ylabel("系统2")
                plt.grid(True, alpha=0.3)

            # 频率分布
            plt.subplot(2, 2, 3)
            plt.hist(series[0], bins=50, alpha=0.7, color="blue", label="系统1")
            plt.hist(series[1], bins=50, alpha=0.5, color="red", label="系统2")
            plt.title(f"{system_name} - 数值分布")
            plt.xlabel("数值")
            plt.ylabel("频次")
            plt.legend()

            # 功率谱密度
            plt.subplot(2, 2, 4)
            from scipy import signal

            freqs, psd = signal.welch(series[0], nperseg=min(256, len(series[0]) // 4))
            plt.semilogy(freqs, psd)
            plt.title(f"{system_name} - 功率谱密度")
            plt.xlabel("频率")
            plt.ylabel("PSD")
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                f"debug_{system_name}_stability.png", dpi=150, bbox_inches="tight"
            )
            plt.close()
            print(f"图表已保存: debug_{system_name}_stability.png")

        return not (has_nan or has_inf) and abs(max_val) < 1e10

    except Exception as e:
        print(colored(f"❌ 系统生成失败: {str(e)}", "red", attrs=["bold"]))
        import traceback

        print("详细错误信息:")
        traceback.print_exc()
        return False


def test_rossler_parameters():
    """专门测试Rössler系统的参数敏感性"""
    print(f"\n{'='*60}")
    print(colored("Rössler系统参数敏感性测试", "magenta", attrs=["bold"]))
    print(f"{'='*60}")

    # 测试不同的时间步长
    dt_values = [0.1, 0.25, 0.5, 1.0]

    for dt in dt_values:
        print(f"\n测试时间步长 dt = {dt}")

        # 修改systems.py中的dt值进行测试
        import importlib
        from core import systems

        # 临时修改dt
        original_rossler = systems.generate_rossler_series

        def test_rossler_with_dt(
            num_systems,
            adjacency_matrix,
            t_steps,
            epsilon,
            noise_level=0.0,
            dynamic_noise_level=0.0,
        ):
            initial_state = -5 + 10 * np.random.rand(num_systems * 3)
            a, b, c = 0.2, 0.2, 5.7
            test_dt = dt  # 使用测试的dt值
            t = np.arange(0, t_steps * test_dt, test_dt)

            from scipy.integrate import odeint

            transient_t = np.linspace(0, 200, 2000)
            transient_sol = odeint(
                systems._rossler_ode,
                initial_state,
                transient_t,
                args=(a, b, c, np.zeros_like(adjacency_matrix), 0, np.ones(3)),
            ).reshape((-1, 3, num_systems))
            state_std = np.std(transient_sol, axis=(0, 2))

            start_state = transient_sol[-1].flatten()
            args = (a, b, c, adjacency_matrix, epsilon, state_std)

            try:
                solution = odeint(
                    systems._rossler_ode,
                    start_state,
                    t,
                    args=args,
                    rtol=1e-6,
                    atol=1e-8,
                    mxstep=10000,
                ).reshape((-1, 3, num_systems))
                return solution[:, 0, :].T
            except Exception as e:
                print(f"   ❌ 积分失败: {e}")
                return np.full((num_systems, t_steps), np.nan)

        # 测试
        num_systems = 2
        adjacency_matrix = generate_adjacency_matrix(num_systems, 1)
        t_steps = 500
        epsilon = 0.1

        series = test_rossler_with_dt(num_systems, adjacency_matrix, t_steps, epsilon)

        has_nan = np.any(np.isnan(series))
        has_inf = np.any(np.isinf(series))
        max_val = np.max(np.abs(series)) if not (has_nan or has_inf) else float("inf")

        if has_nan or has_inf or max_val > 1e6:
            print(f"   ❌ dt={dt}: 数值不稳定 (max_val={max_val:.2e})")
        else:
            print(f"   ✅ dt={dt}: 稳定 (max_val={max_val:.2e})")


def main():
    """主测试函数"""
    print(colored("动力学系统稳定性诊断", "blue", attrs=["bold"]))

    # 要测试的系统列表
    systems_to_test = [
        "lorenz",
        "rossler",
        "logistic",
        "henon",
        "hindmarsh_rose",
        "kuramoto",
        "mackey_glass",
    ]

    results = {}

    # 测试每个系统
    for system in systems_to_test:
        results[system] = test_system_stability(system, plot=True)

    # 专门测试Rössler
    test_rossler_parameters()

    # 总结报告
    print(f"\n{'='*60}")
    print(colored("测试总结报告", "blue", attrs=["bold"]))
    print(f"{'='*60}")

    stable_systems = []
    unstable_systems = []

    for system, is_stable in results.items():
        if is_stable:
            stable_systems.append(system)
            print(f"✅ {system}: 稳定")
        else:
            unstable_systems.append(system)
            print(f"❌ {system}: 不稳定")

    print(f"\n稳定系统: {len(stable_systems)}/{len(systems_to_test)}")
    print(f"需要修复的系统: {unstable_systems}")

    return unstable_systems


if __name__ == "__main__":
    unstable_systems = main()

    if unstable_systems:
        print(
            colored(
                f"\n发现 {len(unstable_systems)} 个不稳定系统需要修复!",
                "red",
                attrs=["bold"],
            )
        )
    else:
        print(colored("\n所有系统都稳定运行! ✅", "green", attrs=["bold"]))
