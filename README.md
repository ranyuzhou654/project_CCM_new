# 🌊 CCM因果分析工具箱 v3.0 - 专业级因果推断平台

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)](README.md)

[English README](README_EN.md) | [详细技术文档](docs/技术文档.md) | [算法原理详解](docs/算法原理详解.md)

## 🎯 项目概述

这是一个为**非线性动力学系统**中的因果关系检测而设计的专业级、模块化分析工具箱。基于**收敛交叉映射（CCM）**和**信息论方法**，提供从参数优化、系统模拟、因果分析到深度可视化的完整解决方案。

**版本3.0**经过全面重构，引入科学严谨的算法和统计检验方法，专为科研人员提供可靠、可复现的因果推断工具。

### ✨ 主要特性

- 🔬 **科学参数优化**: 基于AMI和FNN的自动参数选择
- 🌐 **丰富动力学系统**: 7个经典系统 + 噪声变体 + 动态噪声版本
- 🎲 **多种代理检验**: FFT、AAFT、IAAFT、时间移位、随机重排
- 📊 **双核因果分析**: CCM + 条件传递熵(CTE)交叉验证
- 📈 **专业可视化**: 出版级图表，真实统计分布
- 🧪 **全面测试**: 自动化测试套件确保数值稳定性
- 🖥️ **命令行接口**: 简洁易用的CLI工具

---

## 📁 项目架构

```
CCM_Analysis_Toolbox/
├── 📄 main.py                    # CLI入口点，支持argparse接口
├── 📄 run_all_analysis.py        # 🚀 批量分析脚本 (v2.0新增)
├── 📄 run_all_motifs.py          # 🚀 批量基序分析脚本 (v2.0新增)
├── 📄 requirements.txt           # 项目依赖管理
├── 📄 README.md                  # 本文档
├── 📄 CLAUDE.md                  # 开发指南和项目说明
│
├── 📂 core/                      # 🧠 核心算法模块
│   ├── systems.py               # 动力学系统生成器
│   ├── ccm.py                   # CCM算法和代理方法
│   ├── analysis.py              # 分析工作流和单试验执行
│   ├── motifs.py                # 三节点基序分析(CCM+CTE)
│   └── partial_ccm.py           # 偏CCM实现
│
├── 📂 utils/                     # 🛠️ 工具模块
│   ├── params.py                # 科学参数优化(AMI+FNN)
│   └── visualization.py         # 专业可视化套件
│
├── 📂 tests/                     # ✅ 测试模块
│   └── test_suite.py            # 综合自动化测试
│
├── 📂 docs/                      # 📚 文档
│   ├── 技术文档.md                # 详细技术文档
│   └── 算法原理详解.md             # 算法原理和数学基础
│
└── 📂 results/                   # 📊 结果文件夹 (自动创建)
    ├── analysis_runs/           # 批量分析结果
    ├── motifs_runs/             # 批量基序结果  
    └── logs/                    # 运行日志
```

---

## 🚀 快速开始

### 1. 环境要求

- **Python**: >= 3.8
- **建议**: 使用虚拟环境 (venv 或 conda)

### 2. 安装依赖

```bash
# 克隆或下载项目到本地
cd CCM_Analysis_Toolbox

# 安装依赖
pip install -r requirements.txt
```

### 3. 验证安装

```bash
# 运行测试套件
python tests/test_suite.py

# 查看帮助信息
python main.py --help
```

---

## 📊 核心功能详解

### 🌀 动力学系统

| 系统类型 | 基础版本 | 观测噪声版本 | 动态噪声版本 |
|---------|----------|-------------|-------------|
| **连续系统** | | | |
| Lorenz系统 | `lorenz` | `noisy_lorenz` | `lorenz_dynamic_noise` |
| Rössler系统 | `rossler` | `noisy_rossler` | `rossler_dynamic_noise` |
| Hindmarsh-Rose | `hindmarsh_rose` | `noisy_hindmarsh_rose` | `hindmarsh_rose_dynamic_noise` |
| Kuramoto模型 | `kuramoto` | `noisy_kuramoto` | - |
| Mackey-Glass | `mackey_glass` | `noisy_mackey_glass` | - |
| **离散系统** | | | |
| 逻辑映射 | `logistic` | `noisy_logistic` | `logistic_dynamic_noise` |
| Hénon映射 | `henon` | `noisy_henon` | `henon_dynamic_noise` |

#### 🔧 数值稳定性特性

- **自适应积分**: 根据耦合强度和噪声水平自动调整时间步长
- **分块积分**: 长时间序列采用分块策略避免误差累积
- **智能监控**: 实时检测数值异常并采用恢复策略
- **参数限制**: 安全的参数范围确保数值稳定

### 🎯 参数优化算法

#### 平均互信息 (AMI) - 时间延迟优化
```python
python main.py optimize-params --system lorenz --length 8000
```

- 计算时间序列的互信息函数
- 自动选择第一个局部最小值作为最优τ
- 确保嵌入向量的独立性

#### 伪最近邻 (FNN) - 嵌入维度优化
- 检测假近邻的比例随维度的变化
- 找到使吸引子完全展开的最小维度
- 避免过度嵌入造成的噪声放大

### 🔄 代理数据方法

| 方法 | 描述 | 适用场景 |
|------|------|----------|
| **FFT** | 快速傅里叶变换代理 | 线性相关性检验 |
| **AAFT** | 幅度调整傅里叶变换 | 保持幅度分布 |
| **IAAFT** | 迭代AAFT | 精确保持功率谱和分布 |
| **时间移位** | 循环时间移位 | 简单的时间依赖性检验 |
| **随机重排** | 随机打乱顺序 | 基准对照 |

### 📈 分析类型

#### 1. 时间序列长度分析
```bash
python main.py run-analysis --system lorenz --analysis-type length
```
- 测试不同序列长度对CCM性能的影响
- 长度范围: 500-4000步
- 输出AUROC性能曲线

#### 2. 网络度数分析  
```bash
python main.py run-analysis --system rossler --analysis-type degree
```
- 分析网络连接度对因果检测的影响
- 度数范围: 2-12条边
- 评估网络复杂度效应

#### 3. 耦合强度分析
```bash
python main.py run-analysis --system henon --analysis-type coupling
```
- 测试不同耦合强度下的检测能力
- 强度范围: 0.05-1.0
- 识别最优耦合区间

#### 4. 节点数分析
```bash
python main.py run-analysis --system mackey_glass --analysis-type nodes
```
- 评估系统规模对性能的影响
- 节点数: 3-8个系统
- 分析可扩展性

#### 5. 噪声鲁棒性分析
```bash
python main.py run-analysis --system noisy_lorenz --analysis-type noise
```
- 测试算法在噪声环境下的表现
- 噪声水平: 0.01-0.15
- 评估抗噪能力

### 🔗 三节点基序分析

```bash
python main.py run-motifs --system lorenz --length 4000 --surrogates 200
```

#### 双核验证方法

1. **CCM条件代理**: 经典的收敛交叉映射方法
   - 条件代理数据检验
   - 统计显著性测试

2. **条件传递熵(CTE)**: 信息论方法  
   - 直接量化信息流动
   - 排除第三方变量影响

#### 基序类型分析

- **链式基序**: A → B → C
- **叉式基序**: A ← B → C  
- **汇聚基序**: A → C ← B
- **全连接**: A ↔ B ↔ C

---

## 📋 使用示例

### 基础参数优化
```bash
# 优化Lorenz系统参数
python main.py optimize-params --system lorenz

# 自定义序列长度
python main.py optimize-params --system rossler --length 6000
```

### 性能分析
```bash  
# 长度分析
python main.py run-analysis --system lorenz --analysis-type length

# 噪声鲁棒性测试
python main.py run-analysis --system noisy_rossler --analysis-type noise

# 网络规模分析  
python main.py run-analysis --system henon --analysis-type nodes
```

### 基序分析
```bash
# 标准三节点分析
python main.py run-motifs --system lorenz

# 高精度分析（更多代理数据）
python main.py run-motifs --system mackey_glass --length 6000 --surrogates 500
```

### 🚀 批量自动化分析 (新功能)

**v2.0新增**: 提供了两个强大的批量分析脚本，可以自动运行所有系统的完整分析，无需人工干预。

#### 全系统性能分析
```bash
# 运行所有系统的所有分析类型 (95个分析任务)
python run_all_analysis.py
```

**特性**:
- ✅ 自动运行19个系统 × 5种分析类型
- 📁 结果按时间戳自动组织到文件夹
- 🔺 Rössler系统稳定性优化和自动恢复
- 📊 详细的进度跟踪和统计报告
- 💾 运行配置和结果自动备份

#### 全系统基序分析
```bash
# 运行所有系统的三节点基序分析
python run_all_motifs.py
```

**特性**:
- ✅ 自动运行19个系统的基序分析
- 🎯 CCM + CTE双核验证方法
- 🔺 针对Rössler系统的特殊优化
- 📁 结果自动组织和统计

#### 结果文件夹结构

批量分析会自动创建如下目录结构：

```
results/
├── analysis_runs/
│   └── run_YYYYMMDD_HHMMSS/
│       ├── run_config.json          # 运行配置
│       ├── run_statistics.json      # 运行统计
│       ├── results_*.json           # 所有分析结果
│       └── analysis_*.png           # 所有可视化图表
├── motifs_runs/
│   └── run_YYYYMMDD_HHMMSS/
│       ├── run_config.json          # 运行配置  
│       ├── run_statistics.json      # 运行统计
│       └── motif_analysis_*.png     # 所有基序分析图表
└── logs/
    ├── analysis_YYYYMMDD_HHMMSS.log # 分析日志
    └── motifs_YYYYMMDD_HHMMSS.log   # 基序日志
```

#### 🔺 Rössler系统特殊处理

由于Rössler系统的数值敏感性，批量脚本包含了专门的稳定性优化：

**分析脚本优化**:
- 使用较少的试验次数 (≤15) 和代理数据 (≤80)
- 自动检测数值不稳定并启用恢复模式
- 多级参数降级策略确保成功完成

**基序脚本优化**:
- 使用较短的时间序列 (≤1500) 和代理数据 (≤150)  
- 数值异常时自动切换到超保守参数
- 详细的恢复统计和错误报告

---

## 📊 输出文件说明

### 结果文件
- **格式**: JSON
- **命名**: `results_{system}_{analysis}_{timestamp}.json`
- **内容**: 完整的数值结果和参数信息

### 可视化文件  
- **格式**: PNG (300 DPI)
- **命名**: `analysis_{system}_{analysis}_{timestamp}.png`
- **特性**: 出版级质量，包含统计信息

### 基序分析
- **文件**: `motif_analysis_{system}_{timestamp}.png`  
- **内容**: CCM和CTE方法对比，基序分类结果

---

## 🛠️ 故障排除指南

### 🔺 Rössler系统数值稳定性问题

Rössler系统由于其参数敏感性，可能在某些条件下出现数值不稳定。以下是识别和解决方案：

#### 常见问题症状
- ❌ 出现 `NaN` 或 `Inf` 值
- ❌ 系统状态快速发散
- ❌ 积分器报错或超时
- ❌ CCM分析结果异常

#### 自动恢复机制

**批量脚本已内置**以下恢复策略：

1. **参数自动调整**
   ```
   试验次数: 20 → 15 → 10
   代理数据: 100 → 80 → 50
   时间序列: 2000 → 1500 → 1000
   ```

2. **时间步长优化**
   ```
   标准: dt = 0.25
   强耦合: dt = 0.1
   极端情况: dt = 0.05
   ```

3. **积分器参数**
   ```
   rtol: 1e-8, atol: 1e-10
   mxstep: 15000 → 20000
   ```

#### 手动优化建议

如果使用单独的 `main.py` 分析Rössler系统：

```bash
# 保守参数设置
python main.py run-analysis --system rossler --analysis-type length --trials 10 --surrogates 50

# 避免过长的时间序列  
python main.py run-motifs --system noisy_rossler --length 1500 --surrogates 100

# 对于极不稳定的情况，使用其他系统验证
python main.py run-analysis --system lorenz --analysis-type length  # 验证算法本身
```

#### 推荐参数范围

| 系统版本 | 建议试验次数 | 建议代理数据 | 最大时间序列 |
|----------|-------------|-------------|--------------|
| `rossler` | ≤15 | ≤80 | ≤3000 |
| `noisy_rossler` | ≤12 | ≤60 | ≤2500 |
| `rossler_dynamic_noise` | ≤10 | ≤50 | ≤2000 |

### 其他常见问题

#### 内存不足
```bash
# 减少并行试验数量
python main.py run-analysis --system lorenz --trials 10

# 使用较短序列
python main.py run-motifs --system henon --length 1500
```

#### 运行时间过长
```bash
# 减少代理数据数量
python main.py run-analysis --system mackey_glass --surrogates 50

# 选择计算更快的系统进行初步测试
python main.py run-analysis --system logistic --analysis-type length
```

#### 图像显示问题
批量脚本自动使用非交互式后端，如果手动运行遇到显示问题：
```bash
export MPLBACKEND=Agg
python main.py run-analysis --system lorenz --analysis-type length
```

---

## 🔬 算法原理

### CCM (收敛交叉映射)

CCM通过重构吸引子流形检测因果关系:

1. **状态空间重构**: 使用Takens嵌入定理
   ```
   X(t) = [x(t), x(t-τ), ..., x(t-(m-1)τ)]
   ```

2. **交叉映射**: 从Y的流形预测X的值
   ```
   ρ(L) = corr(x(t), x̂(t)|M_Y)
   ```

3. **收敛性检验**: 随序列长度L增加，相关性应收敛

### 条件传递熵 (CTE)

CTE量化在控制变量Z条件下，从X到Y的信息流:

```
CTE_{X→Y|Z} = H(Y_{t+1}|Y_t, Z_t) - H(Y_{t+1}|Y_t, X_t, Z_t)
```

---

## ⚙️ 系统要求

### 最低要求
- **CPU**: 双核2GHz+
- **内存**: 4GB RAM
- **磁盘**: 1GB可用空间

### 推荐配置
- **CPU**: 四核3GHz+  
- **内存**: 8GB+ RAM
- **磁盘**: 2GB+ SSD空间

### 运行时间参考

| 分析类型 | 基础系统 | 噪声系统 | 长序列 |
|----------|----------|----------|--------|
| 参数优化 | 30秒 | 45秒 | 2分钟 |
| 长度分析 | 2分钟 | 5分钟 | 10分钟 |
| 基序分析 | 3分钟 | 8分钟 | 15分钟 |

---

## 🧪 测试验证

### 运行测试套件
```bash
python tests/test_suite.py
```

### 数值稳定性测试
```bash
python test_systems_debug.py    # 基础稳定性
python test_stress_test.py      # 压力测试
```

### 预期测试结果
- ✅ 所有7个基础系统数值稳定
- ✅ 长序列(>5000步)积分正常
- ✅ 强耦合条件下无发散
- ✅ 动态噪声系统收敛

---

## 📚 依赖库

### 核心科学计算
```
numpy>=1.21.0          # 数值计算基础
scipy>=1.7.0           # 科学计算和优化
scikit-learn>=1.0.0    # 机器学习工具
```

### 可视化
```
matplotlib>=3.5.0      # 基础绘图
seaborn>=0.11.0        # 统计可视化
```

### 其他工具
```
tqdm>=4.62.0           # 进度条显示
termcolor>=1.1.0       # 彩色终端输出  
rich>=12.0.0           # 富文本显示
pyinform>=0.3.0        # 信息论计算
```

---

## 🔄 版本历史

### v3.0.0 (当前版本)
- ✨ 完全重构的模块化架构
- 🔧 自适应数值积分和稳定性保障
- 📊 双核基序分析(CCM+CTE)
- 🎨 出版级可视化系统
- 📈 科学参数优化(AMI+FNN)
- 🧪 全面自动化测试

### v2.x (历史版本)  
- 基础CCM实现
- 简单系统支持
- 基础可视化

---

## 🤝 贡献指南

### 代码规范
- 遵循PEP 8代码风格
- 添加详细的函数文档  
- 包含单元测试

### 提交流程
1. Fork项目仓库
2. 创建特性分支
3. 添加测试用例
4. 提交Pull Request

---

## 📞 技术支持

### 问题报告
- 通过GitHub Issues报告bug
- 提供完整的错误日志和环境信息
- 包含重现步骤

### 功能建议  
- 描述期望功能和使用场景
- 提供相关算法参考文献
- 考虑向后兼容性

---

## 📖 引用信息

如果在学术研究中使用本工具，请引用:

```bibtex
@software{ccm_analysis_toolbox_2024,
  title = {CCM因果分析工具箱: 专业级非线性动力学因果推断平台},
  version = {3.0.0},
  year = {2024},
  note = {基于收敛交叉映射和信息论的因果分析工具}
}
```

---

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

感谢以下开源项目和算法的贡献:
- George Sugihara等人的CCM理论基础
- Takens嵌入定理的数学框架  
- scipy生态系统的数值计算支持
- matplotlib/seaborn的可视化能力

---

## 📚 附加文档

- [English README](README_EN.md) - 完整英文版说明文档
- [详细技术文档](docs/技术文档.md) - 包含完整的API参考、架构设计、算法实现细节
- [算法原理详解](docs/算法原理详解.md) - CCM算法的数学理论基础、Takens嵌入定理、动力学系统理论

---

**📧 联系方式**: 如有技术问题或合作意向，欢迎通过GitHub Issues联系

**🌟 如果此项目对您的研究有帮助，请考虑给个Star支持！**