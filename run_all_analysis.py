#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动运行所有系统分析脚本 v2.0 - 增强版
Automated script to run analysis for all systems - Enhanced Version

新功能:
- 时间戳文件夹组织结果
- 运行配置和统计记录  
- 改进的错误处理和日志
- Rössler系统稳定性优化
"""

import os
import sys
import time
import json
import shutil
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，防止弹出图片窗口

from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from rich.panel import Panel

# 导入项目模块
try:
    from core.analysis import run_full_analysis
    from utils.visualization import VisualizationSuite
except ImportError:
    print("请确保项目结构正确，且所有模块都可被导入。")
    sys.exit(1)

console = Console()

class ResultsManager:
    """结果文件管理类"""
    def __init__(self, base_dir="results"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = base_dir
        self.run_dir = os.path.join(base_dir, "analysis_runs", f"run_{self.timestamp}")
        self.logs_dir = os.path.join(base_dir, "logs")
        
        # 创建目录结构
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.logs_dir, f"analysis_{self.timestamp}.log")
        
    def save_config(self, systems, analysis_types, num_trials, num_surrogates):
        """保存运行配置"""
        config = {
            "timestamp": self.timestamp,
            "systems": systems,
            "analysis_types": analysis_types,
            "num_trials": num_trials,
            "num_surrogates": num_surrogates,
            "total_tasks": len(systems) * len(analysis_types)
        }
        config_file = os.path.join(self.run_dir, "run_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
    def move_results_to_run_dir(self):
        """将结果文件移动到运行目录（JSON文件）并统计已有的PNG文件"""
        moved_files = []
        existing_files = []
        
        # 移动JSON结果文件
        for file in os.listdir("."):
            if file.endswith(".json") and file.startswith("results_"):
                src = file
                dst = os.path.join(self.run_dir, file)
                if os.path.exists(src):
                    shutil.move(src, dst)
                    moved_files.append(file)
        
        # 统计已经在目标目录中的PNG文件
        if os.path.exists(self.run_dir):
            for file in os.listdir(self.run_dir):
                if file.endswith(".png") and file.startswith("analysis_"):
                    existing_files.append(file)
        
        return moved_files, existing_files

def print_header():
    """打印脚本标题"""
    console.print("=" * 80, style="bold cyan")
    console.print("🚀 CCM 全系统分析自动化脚本 v2.0", style="bold cyan", justify="center")
    console.print("=" * 80, style="bold cyan")
    console.print()

def run_single_analysis_with_recovery(system, analysis_type, visualizer, num_trials, num_surrogates, results_dir):
    """
    运行单个分析任务，包含Rössler系统特殊处理
    """
    # 保存当前工作目录
    original_cwd = os.getcwd()
    
    try:
        # 切换到结果目录，让图片直接保存在正确位置
        os.chdir(results_dir)
        
        # 对于Rössler系统，使用更保守的参数
        if 'rossler' in system.lower():
            console.print(f"    [yellow]检测到Rössler系统，使用稳定性优化参数[/yellow]")
            # Rössler系统使用较少的试验次数以提高稳定性
            rossler_trials = min(num_trials, 15)
            rossler_surrogates = min(num_surrogates, 80)
            
            run_full_analysis(
                system_type=system,
                analysis_type=analysis_type,
                visualizer=visualizer,
                num_trials=rossler_trials,
                num_surrogates=rossler_surrogates
            )
        else:
            run_full_analysis(
                system_type=system,
                analysis_type=analysis_type,
                visualizer=visualizer,
                num_trials=num_trials,
                num_surrogates=num_surrogates
            )
        return True, None
        
    except Exception as e:
        error_msg = str(e)
        # 如果是Rössler相关的数值错误，尝试恢复
        if 'rossler' in system.lower() and ('numerical' in error_msg.lower() or 'nan' in error_msg.lower() or 'inf' in error_msg.lower()):
            console.print(f"    [yellow]Rössler系统数值不稳定，尝试恢复模式[/yellow]")
            try:
                # 使用更保守的参数重试
                run_full_analysis(
                    system_type=system,
                    analysis_type=analysis_type,
                    visualizer=visualizer,
                    num_trials=10,  # 进一步减少试验次数
                    num_surrogates=50  # 进一步减少代理数据
                )
                return True, f"使用恢复模式成功"
            except Exception as retry_e:
                return False, f"恢复失败: {str(retry_e)}"
        else:
            return False, error_msg
    finally:
        # 确保始终切换回原始工作目录
        os.chdir(original_cwd)

def run_all_systems_analysis():
    """
    运行所有系统的分析
    """
    
    # 所有可用系统 - 按稳定性排序，Rössler系统放在后面
    # stable_systems = [
    #     'lorenz', 'logistic', 'henon', 'mackey_glass', 'kuramoto',
    #     'noisy_lorenz', 'noisy_logistic', 'noisy_henon', 'noisy_mackey_glass', 'noisy_kuramoto',
    #     'lorenz_dynamic_noise', 'logistic_dynamic_noise', 'henon_dynamic_noise',
    #     'hindmarsh_rose', 'noisy_hindmarsh_rose', 'hindmarsh_rose_dynamic_noise'
    # ]
    
    stable_systems = [
        'noisy_lorenz', 'noisy_logistic',
        'lorenz_dynamic_noise', 'logistic_dynamic_noise'
    ]
    
    # rossler_systems = [
    #     'rossler', 'noisy_rossler', 'rossler_dynamic_noise'
    # ]

    rossler_systems = [
    
    ]
    
    # 合并系统列表，稳定系统在前
    systems = stable_systems + rossler_systems
    
    # 所有可用分析类型
    analysis_types = ['length', 'coupling', 'noise']
    
    # 默认参数
    num_trials = 3
    num_surrogates = 100
    
    # 初始化结果管理器
    results_manager = ResultsManager()
    
    print_header()
    
    console.print(f"🎯 将运行 {len(systems)} 个系统 × {len(analysis_types)} 种分析类型 = {len(systems) * len(analysis_types)} 个分析任务")
    console.print(f"📊 每个任务: {num_trials} 次试验, {num_surrogates} 个代理数据")
    console.print(f"📁 结果将保存到: {results_manager.run_dir}")
    console.print(f"⚡ Rössler系统将使用稳定性优化参数")
    console.print()
    
    # 保存运行配置
    results_manager.save_config(systems, analysis_types, num_trials, num_surrogates)
    
    # 统计信息
    total_tasks = len(systems) * len(analysis_types)
    completed_tasks = 0
    failed_tasks = []
    rossler_recoveries = 0
    start_time = time.time()
    
    with Progress() as progress:
        main_task = progress.add_task("[green]总体进度...", total=total_tasks)
        
        for system in systems:
            # 特殊标记Rössler系统
            system_emoji = "🔺" if 'rossler' in system.lower() else "🔄"
            console.print(f"\n{system_emoji} [bold blue]处理系统: {system}[/bold blue]")
            
            for analysis_type in analysis_types:
                task_name = f"{system}_{analysis_type}"
                
                console.print(f"  ➤ [yellow]运行 {analysis_type} 分析...[/yellow]")
                
                # 创建可视化套件（使用非交互式后端）
                visualizer = VisualizationSuite()
                
                # 运行分析，包含恢复机制
                success, recovery_info = run_single_analysis_with_recovery(
                    system, analysis_type, visualizer, num_trials, num_surrogates, results_manager.run_dir
                )
                
                if success:
                    completed_tasks += 1
                    if recovery_info and "恢复模式" in recovery_info:
                        rossler_recoveries += 1
                        console.print(f"    ✅ [green]{task_name} 完成[/green] [yellow]({recovery_info})[/yellow]")
                    else:
                        console.print(f"    ✅ [green]{task_name} 完成[/green]")
                else:
                    failed_tasks.append((task_name, recovery_info))
                    console.print(f"    ❌ [red]{task_name} 失败: {recovery_info[:50]}...[/red]")
                
                progress.update(main_task, advance=1)
    
    # 移动结果文件到运行目录
    moved_files, existing_png_files = results_manager.move_results_to_run_dir()
    
    # 输出最终统计
    end_time = time.time()
    total_time = end_time - start_time
    
    # 保存运行统计
    stats = {
        "completed_time": datetime.now().isoformat(),
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "failed_tasks": len(failed_tasks),
        "rossler_recoveries": rossler_recoveries,
        "success_rate": (completed_tasks/total_tasks)*100 if total_tasks > 0 else 0,
        "total_time_hours": total_time/3600,
        "avg_time_per_task": total_time/total_tasks if total_tasks > 0 else 0,
        "failed_task_details": failed_tasks,
        "moved_json_files": moved_files,
        "generated_png_files": existing_png_files
    }
    
    stats_file = os.path.join(results_manager.run_dir, "run_statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    console.print("\n" + "=" * 80, style="bold green")
    console.print("📋 批量分析完成统计", style="bold green", justify="center")
    console.print("=" * 80, style="bold green")
    
    # 创建结果表格
    table = Table(title="分析结果统计")
    table.add_column("指标", justify="right", style="cyan", no_wrap=True)
    table.add_column("数值", style="magenta")
    
    table.add_row("总任务数", str(total_tasks))
    table.add_row("完成任务数", str(completed_tasks))
    table.add_row("失败任务数", str(len(failed_tasks)))
    table.add_row("Rössler恢复成功", str(rossler_recoveries))
    table.add_row("成功率", f"{(completed_tasks/total_tasks)*100:.1f}%")
    table.add_row("总耗时", f"{total_time/3600:.2f} 小时")
    table.add_row("平均每任务", f"{total_time/total_tasks:.1f} 秒")
    table.add_row("JSON文件数", str(len(moved_files)))
    table.add_row("PNG图片数", str(len(existing_png_files)))
    
    console.print(table)
    
    # 如果有失败的任务，显示详情
    if failed_tasks:
        console.print("\n⚠️ [yellow]失败任务详情:[/yellow]")
        for task, error in failed_tasks:
            console.print(f"  • [red]{task}[/red]: {error}")
    
    if rossler_recoveries > 0:
        console.print(f"\n🔺 [yellow]Rössler系统恢复统计: {rossler_recoveries} 个任务使用了恢复模式[/yellow]")
    
    # 显示结果位置信息
    result_panel = Panel(
        f"📁 结果目录: [cyan]{results_manager.run_dir}[/cyan]\n"
        f"📊 统计文件: [cyan]{stats_file}[/cyan]\n"
        f"📋 配置文件: [cyan]{os.path.join(results_manager.run_dir, 'run_config.json')}[/cyan]",
        title="📂 结果文件位置",
        border_style="green"
    )
    console.print(result_panel)
    console.print("🎉 [bold green]批量分析完成！[/bold green]")

if __name__ == "__main__":
    try:
        run_all_systems_analysis()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]用户中断了批量分析。[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]批量分析过程中出现严重错误:[/bold red]")
        console.print(str(e))
        sys.exit(1)