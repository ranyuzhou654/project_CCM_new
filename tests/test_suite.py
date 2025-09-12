#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合自动化测试套件 (tests/test_suite.py)
Comprehensive Automated Test Suite

用于验证CCM因果分析工具箱 v3.0的所有核心功能。
[v3.3 Final]: 修正了类型断言，并启用了基序分析的快速测试。
"""

import unittest
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from termcolor import colored

# 将项目根目录添加到Python路径中，以便导入模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.systems import generate_time_series, generate_adjacency_matrix
from core.ccm import parameters, ccm_pearson, generate_surrogates
from core.analysis import run_single_trial
from utils.params import find_optimal_tau, find_optimal_dim
from core.motifs import analyze_three_node_motifs_dual_method
from utils.visualization import VisualizationSuite

class TestCCMToolbox(unittest.TestCase):
    """测试CCM工具箱的核心功能"""

    def test_01_system_generation(self):
        """测试所有动力学系统的生成"""
        print("\n--- 1. 测试系统生成 ---")
        systems = ['lorenz', 'rossler', 'logistic', 'henon', 'noisy_lorenz', 'mackey_glass']
        adj = generate_adjacency_matrix(3, 2)
        for sys_type in systems:
            with self.subTest(system=sys_type):
                print(f"  测试: {sys_type}")
                ts = generate_time_series(sys_type, 3, adj, 1000, 0.1)
                self.assertEqual(ts.shape, (3, 1000))
                self.assertFalse(np.any(np.isnan(ts)), f"{sys_type} 生成了 NaN")
                print(f"  {sys_type} -> OK")

    def test_02_parameter_optimization(self):
        """测试科学的参数优化流程"""
        print("\n--- 2. 测试参数优化 (AMI + FNN) ---")
        ts = generate_time_series('lorenz', 1, np.zeros((1,1)), 5000, 0.0)[0]
        tau = find_optimal_tau(ts, plot=False)
        # [v3.3 Final] 使用更通用的类型检查，接受任何整数类型
        self.assertIsInstance(tau, (int, np.integer), "tau 必须是整数类型")
        self.assertGreater(tau, 0)
        print(f"  AMI -> tau = {tau} -> OK")
        
        dim = find_optimal_dim(ts, tau, plot=False)
        self.assertIsInstance(dim, (int, np.integer), "dim 必须是整数类型")
        self.assertGreater(dim, 0)
        print(f"  FNN -> Dim = {dim} -> OK")

    def test_03_ccm_core_logic(self):
        """测试CCM核心算法"""
        print("\n--- 3. 测试CCM核心逻辑 ---")
        ts = generate_time_series('lorenz', 2, generate_adjacency_matrix(2,1), 1000, 0.2)
        weights, indices = parameters(ts[0], Dim=3, tau=2)
        self.assertIsNotNone(weights)
        self.assertIsNotNone(indices)
        print("  parameters() -> OK")

        corr = ccm_pearson(ts[1], weights, indices, weights.shape[0])
        self.assertIsInstance(corr, float, "单次CCM应返回float")
        self.assertTrue(-1 <= corr <= 1)
        print(f"  ccm_pearson() -> corr = {corr:.3f} -> OK")

    def test_04_surrogate_methods(self):
        """测试所有代理方法的生成"""
        print("\n--- 4. 测试代理方法 ---")
        methods = ['FFT', 'AAFT', 'IAAFT', 'Time Shift', 'Random Reorder']
        data = np.random.randn(1000)
        for method in methods:
            with self.subTest(method=method):
                surr = generate_surrogates(data, method, num_surrogates=1)[0]
                self.assertEqual(len(surr), len(data))
                self.assertFalse(np.any(np.isnan(surr)))
                print(f"  {method} -> OK")

    def test_05_single_trial(self):
        """测试单次试验的运行"""
        print("\n--- 5. 测试单次试验运行 ---")
        params = {
            'system_type': 'logistic', 'time_series_length': 1000,
            'num_systems': 4, 'degree': 3, 'epsilon': 0.1,
            'Dim': 2, 'tau': 1, 'method': 'FFT'
        }
        auroc, _ = run_single_trial(params)
        self.assertTrue(0.0 <= auroc <= 1.0)
        print(f"  run_single_trial() -> AUROC = {auroc:.3f} -> OK")

    def test_06_motif_analysis_fast(self):
        """[v3.3 Final] 对基序分析进行快速功能性测试"""
        print("\n--- 6. 快速测试三节点基序分析 ---")
        try:
            # 使用快速参数以确保测试能迅速完成
            analyze_three_node_motifs_dual_method('logistic', VisualizationSuite(), time_steps=500, num_surrogates=10)
            print("  基序分析函数成功运行 -> OK")
        except Exception as e:
            self.fail(f"基序分析函数运行时出现错误: {e}")

def run_tests():
    """运行所有测试用例"""
    # 关闭matplotlib的交互式窗口，避免在测试过程中弹出图表
    plt.ioff()
    
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCCMToolbox))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print(colored("\n✅ 所有测试通过！工具箱功能正常。", "green", attrs=['bold']))
    else:
        print(colored("\n❌ 部分测试失败！请检查上述错误信息。", "red", attrs=['bold']))

if __name__ == '__main__':
    run_tests()
