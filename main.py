#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCM因果分析工具箱 v3.0 - 主程序入口 (已修正)
CCM Causal Analysis Toolbox v3.0 - Main Entry Point (Fixed)

此版本修正了噪声分析的逻辑，并增加了对新系统的支持。
"""

import sys
import numpy as np
from termcolor import colored
from rich.console import Console
from rich.table import Table
import argparse

# 导入项目模块
try:
    from core.analysis import run_full_analysis
    from core.motifs import analyze_three_node_motifs_dual_method
    from utils.params import optimize_embedding_for_system
    from utils.visualization import VisualizationSuite
except ImportError:
    print("请确保项目结构正确，且所有模块都可被导入。")
    sys.exit(1)


# 初始化漂亮的终端输出
console = Console()

def print_header():
    """打印项目标题"""
    console.print("=" * 80, style="bold cyan")
    console.print("🎯 CCM因果分析工具箱 v3.0 - 专业版 (修正版)", style="bold cyan", justify="center")
    console.print("=" * 80, style="bold cyan")
    console.print()

def run_analysis_command(args):
    """处理 'run-analysis' 命令"""
    console.print(f"🚀 [bold green]开始运行: [/bold green] [yellow]{args.system.capitalize()}[/yellow] 系统上的 [yellow]{args.analysis_type}[/yellow] 分析...")
    
    visualizer = VisualizationSuite()
    
    run_full_analysis(
        system_type=args.system,
        analysis_type=args.analysis_type,
        visualizer=visualizer,
        num_trials=args.trials,
        num_surrogates=args.surrogates
    )
    console.print("\n✅ [bold green]分析完成！[/bold green] 图表和结果JSON文件已生成。")

def run_motifs_command(args):
    """处理 'run-motifs' 命令"""
    console.print(f"🚀 [bold green]开始运行: [/bold green] [yellow]{args.system.capitalize()}[/yellow] 系统上的三节点基序分析 (CCM + CTE 双核)...")
    
    visualizer = VisualizationSuite()
    
    analyze_three_node_motifs_dual_method(
        system_type=args.system,
        visualizer=visualizer,
        time_steps=args.length,
        num_surrogates=args.surrogates
    )
    console.print("\n✅ [bold green]基序分析完成！[/bold green] 对比图表已生成。")

def optimize_params_command(args):
    """处理 'optimize-params' 命令"""
    console.print(f"🚀 [bold green]开始为 {args.system.capitalize()} 系统优化嵌入参数...[/bold green]")
    console.print(f"将生成长度为 {args.length} 的测试序列进行分析。")
    
    best_params = optimize_embedding_for_system(
        system_type=args.system,
        series_length=args.length
    )
    
    if best_params:
        table = Table(title=f"{args.system.capitalize()} 系统推荐参数")
        table.add_column("参数", justify="right", style="cyan", no_wrap=True)
        table.add_column("推荐值", style="magenta")
        table.add_row("最佳时间延迟 (tau)", str(best_params['tau']))
        table.add_row("最小嵌入维度 (Dim)", str(best_params['Dim']))
        console.print(table)
    else:
        console.print("[bold red]参数优化失败。请检查系统生成过程。[/bold red]")

def main():
    """主函数，负责解析命令行参数并分派任务"""
    print_header()

    parser = argparse.ArgumentParser(
        description="CCM因果分析工具箱 v3.0 - 一个专业、模块化的因果推断框架。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True, help='可用的命令')

    # 更新可用系统列表
    available_systems = [
        'lorenz', 'rossler', 'logistic', 'henon', 'mackey_glass', 'kuramoto',
        'noisy_lorenz', 'noisy_rossler', 'noisy_mackey_glass', 'noisy_kuramoto',
        'lorenz_dynamic_noise', 'rossler_dynamic_noise',
        'logistic_dynamic_noise', 'henon_dynamic_noise',
        'noisy_logistic', 'noisy_henon',
        'hindmarsh_rose', 'noisy_hindmarsh_rose', 'hindmarsh_rose_dynamic_noise'
    ]

    # --- 'run-analysis' 子命令 ---
    parser_analysis = subparsers.add_parser('run-analysis', help='运行多维度性能分析。')
    parser_analysis.add_argument('--system', '-s', type=str, required=True, choices=available_systems, help='要分析的动力学系统。')
    parser_analysis.add_argument('--analysis-type', '-a', type=str, required=True, choices=['length', 'degree', 'coupling', 'nodes', 'noise'], help='要执行的分析类型。')
    parser_analysis.add_argument('--trials', '-t', type=int, default=20, help='每次参数设置的试验次数。')
    parser_analysis.add_argument('--surrogates', '-n', type=int, default=100, help='生成的代理数据数量。')
    parser_analysis.set_defaults(func=run_analysis_command)

    # --- 'run-motifs' 子命令 ---
    parser_motifs = subparsers.add_parser('run-motifs', help='运行三节点基序因果分析 (CCM+CTE)。')
    parser_motifs.add_argument('--system', '-s', type=str, required=True, choices=available_systems, help='要分析的动力学系统。')
    parser_motifs.add_argument('--length', '-l', type=int, default=2000, help='用于分析的时间序列长度。')
    parser_motifs.add_argument('--surrogates', '-n', type=int, default=200, help='生成的代理数据数量。')
    parser_motifs.set_defaults(func=run_motifs_command)

    # --- 'optimize-params' 子命令 ---
    parser_optimize = subparsers.add_parser('optimize-params', help='自动寻找给定系统的最佳嵌入参数 (tau, Dim)。')
    parser_optimize.add_argument('--system', '-s', type=str, required=True, choices=available_systems, help='要优化参数的动力学系统。')
    parser_optimize.add_argument('--length', '-l', type=int, default=8000, help='用于优化的时间序列长度 (推荐较长序列)。')
    parser_optimize.set_defaults(func=optimize_params_command)

    try:
        args = parser.parse_args()
        
        # [修正] 改进噪声分析的逻辑
        if args.command == 'run-analysis' and args.analysis_type == 'noise':
            system_name = args.system
            
            # 仅当用户未明确指定带噪声版本时，才自动选择一个
            if 'noisy_' not in system_name and '_dynamic_noise' not in system_name:
                clean_name = system_name
                # 默认优先选择观测噪声版本
                observational_version = f"noisy_{clean_name}"
                dynamic_version = f"{clean_name}_dynamic_noise"

                if observational_version in available_systems:
                    args.system = observational_version
                    console.print(f"[yellow]提示: 已自动选择观测噪声版本 '{args.system}' 进行噪声分析。[/yellow]")
                elif dynamic_version in available_systems:
                    args.system = dynamic_version
                    console.print(f"[yellow]提示: 已自动选择动态噪声版本 '{args.system}' 进行噪声分析。[/yellow]")
                else:
                    console.print(f"[bold red]错误: 系统 '{clean_name}' 找不到可用的噪声版本。[/bold red]")
                    sys.exit(1)
        
        args.func(args)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]用户中断了程序。[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]程序运行中出现严重错误:[/bold red]")
        console.print_exception(show_locals=True)
        sys.exit(1)

if __name__ == '__main__':
    main()

