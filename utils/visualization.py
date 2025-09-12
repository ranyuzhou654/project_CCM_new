# -*- coding: utf-8 -*-
"""
增强版可视化套件 (utils/visualization.py)
Enhanced Visualization Suite

提供了出版级质量的、统计严谨的图表生成功能。
[v3.5 Final]: 全面升级基序分析可视化，以支持所有代理方法。
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from scipy.stats import mannwhitneyu

class VisualizationSuite:
    """
    一个集成的、功能强大的可视化类，用于展示CCM分析结果。
    """
    def __init__(self):
        self.color_palette = {
            'No Surrogate': '#2E2E2E', 'FFT': '#3498DB', 'AAFT': '#27AE60',
            'IAAFT': '#E74C3C', 'Time Shift': '#8E44AD', 'Random Reorder': '#F39C12',
            'Conditional': '#E67E22', 'Partial CCM': '#d6336c' # 为条件代理添加颜色
        }
        self.marker_styles = {
            'No Surrogate': 'o', 'FFT': 's', 'AAFT': '^',
            'IAAFT': 'd', 'Time Shift': 'p', 'Random Reorder': '*'
        }
        self.line_styles = {
            'No Surrogate': '-', 'FFT': '--', 'AAFT': '-.',
            'IAAFT': ':', 'Time Shift': '--', 'Random Reorder': '-.'
        }
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('seaborn-whitegrid')
        sns.set_palette("husl")
        plt.rcParams.update({
            'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
            'font.size': 12, 'axes.titlesize': 16, 'axes.labelsize': 14,
            'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 12,
            'figure.titlesize': 18, 'axes.grid': True, 'grid.alpha': 0.4
        })

    def create_comprehensive_performance_plot(self, x_values, results_mean, results_std, 
                                            methods, xlabel, title, num_trials,
                                            save_path=None, raw_results=None, num_surrogates=100,
                                            constant_params=None): # [改进] 接收 constant_params
        """创建综合性能分析图表，包含多个子图以提供全面的信息。"""
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[2.5, 1.5, 1.5], width_ratios=[2,1])
        
        ax_main = fig.add_subplot(gs[0, :])
        # [改进] 传递 constant_params 到主图绘制函数
        self._plot_main_performance(ax_main, x_values, results_mean, results_std, 
                                  methods, xlabel, title, num_trials, num_surrogates,
                                  constant_params)
        
        ax_stats = fig.add_subplot(gs[1, 0])
        if raw_results:
            self._plot_statistical_significance(ax_stats, raw_results, methods)
        
        ax_rank = fig.add_subplot(gs[1, 1])
        self._plot_method_ranking(ax_rank, results_mean, methods)

        ax_dist = fig.add_subplot(gs[2, :])
        if raw_results:
            self._plot_performance_distribution(ax_dist, x_values, raw_results, methods, xlabel)

        plt.tight_layout(pad=3.0)
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(colored(f"✅ 图表已保存至: {save_path}", "green"))
            except Exception as e:
                print(colored(f"❌ 保存图表失败: {e}", "red"))
        
        plt.show()
        
    def _plot_main_performance(self, ax, x_values, results_mean, results_std, 
                             methods, xlabel, title_base, num_trials, num_surrogates,
                             constant_params=None): # [改进] 接收 constant_params
        """绘制主要性能曲线、误差带和基准线。"""
        for method in methods:
            if method not in results_mean or not results_mean[method]: continue
            
            mean = np.array(results_mean[method])
            std_err = np.array(results_std[method]) / np.sqrt(num_trials)
            
            ax.plot(x_values, mean, color=self.color_palette.get(method, '#333333'),
                    marker=self.marker_styles.get(method, 'x'), linestyle=self.line_styles.get(method, '-'),
                    linewidth=2.5, markersize=8, label=method, alpha=0.9)
            
            ax.fill_between(x_values, mean - 1.96 * std_err, mean + 1.96 * std_err,
                           color=self.color_palette.get(method, '#333333'), alpha=0.15)

        # [改进] 动态构建包含固定参数信息的副标题
        subtitle_parts = [f"Trials per point: {num_trials}", f"Surrogates: {num_surrogates}"]
        if constant_params:
            param_display_map = {
                'num_systems': 'N', 'degree': 'degree', 'epsilon': 'ε',
                'time_series_length': 'L', 'noise_level': 'noise'
            }
            constant_param_strs = [
                f"{param_display_map.get(k, k)}={v:.2f}" if isinstance(v, float) else f"{param_display_map.get(k, k)}={v}"
                for k, v in constant_params.items()
            ]
            if constant_param_strs:
                subtitle_parts.append(f"Constants: {', '.join(constant_param_strs)}")
        
        subtitle = f"({'; '.join(subtitle_parts)})"
        title_full = f"{title_base}\n{subtitle}"
        
        ax.axhline(0.5, color='red', linestyle=':', lw=2, label='Random Guess')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Area Under ROC Curve (AUROC)')
        ax.set_title(title_full, pad=20)
        ax.set_ylim(0.45, 1.02)
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)

    def _plot_statistical_significance(self, ax, raw_results, methods):
        """绘制基于曼-惠特尼U检验的p值热图。"""
        n_methods = len(methods)
        p_value_matrix = np.ones((n_methods, n_methods))
        
        aggregated_results = {m: [item for sublist in raw_results.get(m, []) for item in sublist] for m in methods}

        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                m1, m2 = methods[i], methods[j]
                if aggregated_results.get(m1) and aggregated_results.get(m2):
                    try:
                        _, p_val = mannwhitneyu(aggregated_results[m1], aggregated_results[m2], alternative='two-sided')
                        p_value_matrix[i, j] = p_value_matrix[j, i] = p_val
                    except ValueError: pass

        sns.heatmap(p_value_matrix, annot=True, fmt=".3f", cmap="viridis_r",
                    xticklabels=methods, yticklabels=methods, ax=ax, vmin=0, vmax=0.1)
        ax.set_title('Statistical Significance (p-value)\n(Mann-Whitney U Test)', pad=15)
        ax.tick_params(axis='x', rotation=45)

    def _plot_method_ranking(self, ax, results_mean, methods):
        """绘制方法的平均性能排名条形图。"""
        avg_perf = {m: np.mean(results_mean[m]) for m in methods if results_mean.get(m)}
        if not avg_perf: return
        
        sorted_perf = sorted(avg_perf.items(), key=lambda item: item[1])
        
        methods_sorted = [item[0] for item in sorted_perf]
        perf_sorted = [item[1] for item in sorted_perf]
        colors = [self.color_palette.get(m, '#333333') for m in methods_sorted]
        
        bars = ax.barh(methods_sorted, perf_sorted, color=colors, alpha=0.8)
        ax.bar_label(bars, fmt='%.3f', padding=3)
        ax.set_xlabel('Average AUROC')
        ax.set_title('Method Performance Ranking', pad=15)
        ax.set_xlim(left=0.4)

    def _plot_performance_distribution(self, ax, x_values, raw_results, methods, xlabel):
        """使用箱形图展示在最后一个参数点上的真实数据分布。"""
        final_value_data = []
        labels = []
        
        for method in methods:
            if method in raw_results and raw_results[method] and raw_results[method][-1]:
                final_value_data.append(raw_results[method][-1])
                labels.append(method)
        
        if not final_value_data: return

        bplot = ax.boxplot(final_value_data, vert=True, patch_artist=True, labels=labels)
        
        for patch, color in zip(bplot['boxes'], [self.color_palette.get(m, '#333333') for m in labels]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylabel('AUROC Distribution')
        ax.set_title(f'Performance Distribution at {xlabel} = {x_values[-1]}', pad=15)
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
        ax.set_ylim(0.4, 1.02)

    def plot_motif_comparison(self, results, system_type, save_path=None): # [改进] 接收 save_path
        """
        [v3.5 Final] 为三节点基序分析创建对比图，支持所有代理方法。
        """
        first_result = next(iter(results.values()))
        cte_was_run = not np.isnan(first_result['te_pval'])

        num_rows = len(results)
        num_cols = 2
        figsize_width = 18
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize_width, 5 * num_rows), 
                                 squeeze=False,
                                 gridspec_kw={'width_ratios': [3, 4]})
        
        title = f'Comprehensive Motif Analysis - {system_type.capitalize()}'
        fig.suptitle(title, fontsize=18, y=1.02)

        for i, (name, data) in enumerate(results.items()):
            # --- 左图：所有方法的显著性得分对比 ---
            ax_bar = axes[i, 0]

            # [v3.7] 构建新的指标列表，将 Partial CCM 放在最前面
            metrics = ['Partial CCM']
            values = [data['partial_ccm_score']]
            colors = [self.color_palette['Partial CCM']]
            
            ccm_pvals = data['ccm_pvals']
            metrics.extend(list(ccm_pvals.keys()))
            values.extend([1 - pval for pval in ccm_pvals.values()])
            colors.extend([self.color_palette.get(m, '#CCCCCC') for m in ccm_pvals.keys()])
            
            if cte_was_run:
                metrics.extend(['TE (p-val)', 'CTE (p-val)'])
                values.extend([1 - data['te_pval'], 1 - data['cte_pval']])
                colors.extend(['#2ECC71', '#9B59B6'])

            bars = ax_bar.bar(metrics, values, color=colors, alpha=0.8)
            ax_bar.bar_label(bars, fmt='%.2f', fontsize=9)
            ax_bar.axhline(0.95, color='r', linestyle='--', label='p=0.05 Threshold')
            ax_bar.set_ylabel('Significance Score (1 - p-value)')
            ax_bar.set_title(f'{name}\nCausality Metrics')
            ax_bar.tick_params(axis='x', rotation=45, labelsize=10)
            ax_bar.set_ylim(0, 1.2)
            ax_bar.legend()

            # --- 右图：所有CCM代理的KDE分布图 ---
            ax_kde = axes[i, 1]
            
            for method, dist in data['ccm_dists'].items():
                dist_data = np.atleast_1d(dist)
                sns.kdeplot(dist_data, ax=ax_kde, label=f'{method}', 
                            color=self.color_palette.get(method, '#CCCCCC'), 
                            fill=True, alpha=0.15, linewidth=2)

            ax_kde.axvline(data['original_ccm'], color='red', lw=3, linestyle='--', 
                          label=f'Original CCM ({data["original_ccm"]:.3f})')
            ax_kde.set_xlabel('|CCM Correlation|')
            ax_kde.set_title('CCM Surrogate Distributions (KDE)')
            ax_kde.legend()

        plt.tight_layout(h_pad=4.0)
        # [改进] 使用传入的 save_path，如果存在
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(colored(f"✅ 基序对比图已保存至: {save_path}", "green"))
            except Exception as e:
                print(colored(f"❌ 保存基序图失败: {e}", "red"))
        plt.show()
