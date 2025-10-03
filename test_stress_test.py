#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
压力测试脚本 - 测试更极端条件下的系统稳定性
"""

import numpy as np
import matplotlib.pyplot as plt
from core.systems import generate_time_series, generate_adjacency_matrix
from termcolor import colored
import warnings

warnings.filterwarnings("ignore")


def stress_test_system(system_name, test_conditions):
    """对单个系统进行压力测试"""
    print(f"\n{'='*60}")
    print(colored(f"压力测试: {system_name}", "cyan", attrs=["bold"]))
    print(f"{'='*60}")

    failed_conditions = []

    for condition_name, params in test_conditions.items():
        print(f"\n测试条件: {condition_name}")
        print(f"参数: {params}")

        try:
            series = generate_time_series(
                system_name,
                params["num_systems"],
                params["adjacency_matrix"],
                params["t_steps"],
                params["epsilon"],
                noise_level=params.get("noise_level", 0.0),
            )

            # 检查数值稳定性
            has_nan = np.any(np.isnan(series))
            has_inf = np.any(np.isinf(series))
            max_abs_val = (
                np.max(np.abs(series)) if not (has_nan or has_inf) else float("inf")
            )

            if has_nan:
                print(f"   ❌ NaN值检测到")
                failed_conditions.append(condition_name)
            elif has_inf:
                print(f"   ❌ 无穷大值检测到")
                failed_conditions.append(condition_name)
            elif max_abs_val > 1e8:
                print(f"   ❌ 数值过大: {max_abs_val:.2e}")
                failed_conditions.append(condition_name)
            else:
                print(f"   ✅ 通过 (max_val: {max_abs_val:.2e})")

        except Exception as e:
            print(f"   ❌ 异常: {str(e)}")
            failed_conditions.append(condition_name)

    return failed_conditions


def test_rossler_comprehensive():
    """Rössler系统的全面测试"""
    print(colored("\nRössler系统综合测试", "magenta", attrs=["bold"]))

    # 测试不同的参数组合
    test_cases = {
        "基础测试": {
            "num_systems": 2,
            "adjacency_matrix": generate_adjacency_matrix(2, 1),
            "t_steps": 1000,
            "epsilon": 0.1,
        },
        "长时间序列": {
            "num_systems": 2,
            "adjacency_matrix": generate_adjacency_matrix(2, 1),
            "t_steps": 10000,
            "epsilon": 0.1,
        },
        "强耦合": {
            "num_systems": 2,
            "adjacency_matrix": generate_adjacency_matrix(2, 2),
            "t_steps": 2000,
            "epsilon": 1.0,
        },
        "多系统": {
            "num_systems": 5,
            "adjacency_matrix": generate_adjacency_matrix(5, 10),
            "t_steps": 1000,
            "epsilon": 0.2,
        },
        "极强耦合": {
            "num_systems": 2,
            "adjacency_matrix": generate_adjacency_matrix(2, 2),
            "t_steps": 1000,
            "epsilon": 5.0,
        },
    }

    return stress_test_system("rossler", test_cases)


def test_all_systems_stress():
    """所有系统的压力测试"""
    systems = [
        "lorenz",
        "rossler",
        "logistic",
        "henon",
        "hindmarsh_rose",
        "kuramoto",
        "mackey_glass",
    ]

    # 通用压力测试条件
    stress_conditions = {
        "长序列": {
            "num_systems": 2,
            "adjacency_matrix": generate_adjacency_matrix(2, 1),
            "t_steps": 5000,
            "epsilon": 0.1,
        },
        "强耦合": {
            "num_systems": 3,
            "adjacency_matrix": generate_adjacency_matrix(3, 6),
            "t_steps": 2000,
            "epsilon": 1.0,
        },
        "多系统": {
            "num_systems": 8,
            "adjacency_matrix": generate_adjacency_matrix(8, 20),
            "t_steps": 1000,
            "epsilon": 0.2,
        },
    }

    all_failures = {}

    for system in systems:
        failures = stress_test_system(system, stress_conditions)
        if failures:
            all_failures[system] = failures

    return all_failures


def test_dynamic_noise_systems():
    """测试动态噪声系统"""
    print(f"\n{'='*60}")
    print(colored("动态噪声系统测试", "yellow", attrs=["bold"]))
    print(f"{'='*60}")

    dynamic_systems = [
        "lorenz_dynamic_noise",
        "rossler_dynamic_noise",
        "logistic_dynamic_noise",
        "henon_dynamic_noise",
        "hindmarsh_rose_dynamic_noise",
    ]

    failures = {}

    for system in dynamic_systems:
        print(f"\n测试: {system}")

        try:
            num_systems = 2
            adjacency_matrix = generate_adjacency_matrix(num_systems, 1)
            t_steps = 1000
            epsilon = 0.1

            # 测试不同的噪声水平
            noise_levels = [0.01, 0.05, 0.1]

            system_failures = []

            for noise_level in noise_levels:
                series = generate_time_series(
                    system,
                    num_systems,
                    adjacency_matrix,
                    t_steps,
                    epsilon,
                    noise_level=noise_level,
                )

                has_nan = np.any(np.isnan(series))
                has_inf = np.any(np.isinf(series))
                max_val = (
                    np.max(np.abs(series)) if not (has_nan or has_inf) else float("inf")
                )

                if has_nan or has_inf or max_val > 1e8:
                    system_failures.append(f"noise_level_{noise_level}")
                    print(f"   ❌ 噪声水平 {noise_level}: 失败")
                else:
                    print(f"   ✅ 噪声水平 {noise_level}: 通过")

            if system_failures:
                failures[system] = system_failures

        except Exception as e:
            print(f"   ❌ 系统异常: {str(e)}")
            failures[system] = [f"exception: {str(e)}"]

    return failures


def main():
    """主测试函数"""
    print(colored("系统压力测试开始", "blue", attrs=["bold"]))

    # 1. Rössler系统专项测试
    rossler_failures = test_rossler_comprehensive()

    # 2. 所有系统压力测试
    stress_failures = test_all_systems_stress()

    # 3. 动态噪声系统测试
    dynamic_failures = test_dynamic_noise_systems()

    # 4. 总结报告
    print(f"\n{'='*70}")
    print(colored("压力测试总结报告", "blue", attrs=["bold"]))
    print(f"{'='*70}")

    total_failures = 0

    if rossler_failures:
        print(colored(f"\nRössler系统失败条件: {rossler_failures}", "red"))
        total_failures += len(rossler_failures)
    else:
        print(colored("\nRössler系统: 所有压力测试通过 ✅", "green"))

    if stress_failures:
        print(colored("\n压力测试失败系统:", "red"))
        for system, conditions in stress_failures.items():
            print(f"  {system}: {conditions}")
            total_failures += len(conditions)
    else:
        print(colored("\n压力测试: 所有系统通过 ✅", "green"))

    if dynamic_failures:
        print(colored("\n动态噪声测试失败:", "red"))
        for system, conditions in dynamic_failures.items():
            print(f"  {system}: {conditions}")
            total_failures += len(conditions)
    else:
        print(colored("\n动态噪声测试: 所有系统通过 ✅", "green"))

    print(f"\n总失败数: {total_failures}")

    if total_failures == 0:
        print(colored("🎉 所有系统都通过了压力测试!", "green", attrs=["bold"]))
    else:
        print(
            colored(
                f"⚠️  发现 {total_failures} 个问题需要修复", "yellow", attrs=["bold"]
            )
        )

    return total_failures == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
