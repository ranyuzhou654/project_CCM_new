# 🌊 CCM Causal Analysis Toolbox v3.0 - Professional Causal Inference Platform

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)](README.md)

[中文版 README](README.md) | [详细技术文档](docs/技术文档.md) | [算法原理详解](docs/算法原理详解.md)

## 🎯 Project Overview

A **professional-grade, modular analysis toolbox** designed for causal relationship detection in **nonlinear dynamical systems**. Built on **Convergent Cross Mapping (CCM)** and **information-theoretic methods**, it provides a complete solution from parameter optimization, system simulation, causal analysis to deep visualization.

**Version 3.0** has been completely refactored with scientifically rigorous algorithms and statistical testing methods, specifically designed to provide reliable and reproducible causal inference tools for researchers.

### ✨ Key Features

- 🔬 **Scientific Parameter Optimization**: Automated parameter selection based on AMI and FNN
- 🌐 **Rich Dynamical Systems**: 7 classic systems + noise variants + dynamic noise versions  
- 🎲 **Multiple Surrogate Tests**: FFT, AAFT, IAAFT, Time Shift, Random Reorder
- 📊 **Dual-Core Causal Analysis**: CCM + Conditional Transfer Entropy (CTE) cross-validation
- 📈 **Professional Visualization**: Publication-quality charts with real statistical distributions
- 🧪 **Comprehensive Testing**: Automated test suite ensuring numerical stability
- 🖥️ **Command Line Interface**: Clean and intuitive CLI tool

---

## 📁 Project Architecture

```
CCM_Analysis_Toolbox/
├── 📄 main.py                    # CLI entry point with argparse interface
├── 📄 run_all_analysis.py        # 🚀 Batch analysis script (v2.0 new)
├── 📄 run_all_motifs.py          # 🚀 Batch motif analysis script (v2.0 new)
├── 📄 requirements.txt           # Project dependencies
├── 📄 README.md                  # Chinese documentation
├── 📄 README_EN.md               # This document (English)
├── 📄 CLAUDE.md                  # Development guide and project instructions
│
├── 📂 core/                      # 🧠 Core algorithm modules
│   ├── systems.py               # Dynamical systems generation
│   ├── ccm.py                   # CCM algorithm and surrogate methods
│   ├── analysis.py              # Analysis workflows and single trial execution
│   ├── motifs.py                # Three-node motif analysis (CCM+CTE)
│   └── partial_ccm.py           # Partial CCM implementations
│
├── 📂 utils/                     # 🛠️ Utility modules
│   ├── params.py                # Scientific parameter optimization (AMI+FNN)
│   └── visualization.py         # Professional visualization suite
│
├── 📂 tests/                     # ✅ Testing modules
│   └── test_suite.py            # Comprehensive automated testing
│
├── 📂 docs/                      # 📚 Documentation
│   ├── 技术文档.md                # Detailed technical documentation (Chinese)
│   └── 算法原理详解.md             # Algorithm principles and mathematical foundations (Chinese)
│
└── 📂 results/                   # 📊 Results folder (auto-created)
    ├── analysis_runs/           # Batch analysis results
    ├── motifs_runs/             # Batch motif results
    └── logs/                    # Execution logs
```

---

## 🚀 Quick Start

### 1. System Requirements

- **Python**: >= 3.8
- **Recommendation**: Use virtual environment (venv or conda)

### 2. Installation

```bash
# Clone or download the project
cd CCM_Analysis_Toolbox

# Install dependencies
pip install -r requirements.txt
```

### 3. Verification

```bash
# Run test suite
python tests/test_suite.py

# View help information
python main.py --help
```

---

## 📊 Core Functionality Detailed

### 🌀 Dynamical Systems

| System Type | Base Version | Observational Noise | Dynamic Noise |
|------------|--------------|-------------------|---------------|
| **Continuous Systems** | | | |
| Lorenz System | `lorenz` | `noisy_lorenz` | `lorenz_dynamic_noise` |
| Rössler System | `rossler` | `noisy_rossler` | `rossler_dynamic_noise` |
| Hindmarsh-Rose | `hindmarsh_rose` | `noisy_hindmarsh_rose` | `hindmarsh_rose_dynamic_noise` |
| Kuramoto Model | `kuramoto` | `noisy_kuramoto` | - |
| Mackey-Glass | `mackey_glass` | `noisy_mackey_glass` | - |
| **Discrete Systems** | | | |
| Logistic Map | `logistic` | `noisy_logistic` | `logistic_dynamic_noise` |
| Hénon Map | `henon` | `noisy_henon` | `henon_dynamic_noise` |

#### 🔧 Numerical Stability Features

- **Adaptive Integration**: Automatically adjusts time step based on coupling strength and noise level
- **Chunked Integration**: Block processing for long time series to prevent error accumulation
- **Smart Monitoring**: Real-time detection of numerical anomalies with recovery strategies
- **Parameter Bounds**: Safe parameter ranges ensuring numerical stability

### 🎯 Parameter Optimization Algorithms

#### Average Mutual Information (AMI) - Time Delay Optimization
```bash
python main.py optimize-params --system lorenz --length 8000
```

- Computes mutual information function of time series
- Automatically selects first local minimum as optimal τ
- Ensures independence of embedding vectors

#### False Nearest Neighbors (FNN) - Embedding Dimension Optimization
- Detects proportion of false neighbors varying with dimension
- Finds minimum dimension for complete attractor unfolding
- Avoids noise amplification from over-embedding

### 🔄 Surrogate Data Methods

| Method | Description | Application |
|--------|-------------|-------------|
| **FFT** | Fast Fourier Transform surrogate | Linear correlation testing |
| **AAFT** | Amplitude Adjusted Fourier Transform | Preserves amplitude distribution |
| **IAAFT** | Iterative AAFT | Precisely preserves power spectrum and distribution |
| **Time Shift** | Cyclic time shifting | Simple temporal dependence testing |
| **Random Shuffle** | Random reordering | Baseline control |

### 📈 Analysis Types

#### 1. Time Series Length Analysis
```bash
python main.py run-analysis --system lorenz --analysis-type length
```
- Tests CCM performance vs different sequence lengths
- Length range: 500-4000 steps
- Outputs AUROC performance curves

#### 2. Network Degree Analysis
```bash
python main.py run-analysis --system rossler --analysis-type degree
```
- Analyzes network connectivity impact on causal detection
- Degree range: 2-12 edges
- Evaluates network complexity effects

#### 3. Coupling Strength Analysis
```bash
python main.py run-analysis --system henon --analysis-type coupling
```
- Tests detection capability under different coupling strengths
- Strength range: 0.05-1.0
- Identifies optimal coupling intervals

#### 4. Node Number Analysis
```bash
python main.py run-analysis --system mackey_glass --analysis-type nodes
```
- Evaluates system scale impact on performance
- Node count: 3-8 systems
- Analyzes scalability

#### 5. Noise Robustness Analysis
```bash
python main.py run-analysis --system noisy_lorenz --analysis-type noise
```
- Tests algorithm performance in noisy environments
- Noise levels: 0.01-0.15
- Evaluates noise resistance

### 🔗 Three-Node Motif Analysis

```bash
python main.py run-motifs --system lorenz --length 4000 --surrogates 200
```

#### Dual Validation Methods

1. **CCM Conditional Surrogate**: Classic convergent cross mapping method
   - Conditional surrogate data testing
   - Statistical significance testing

2. **Conditional Transfer Entropy (CTE)**: Information-theoretic method
   - Direct quantification of information flow
   - Excludes third-party variable influences

#### Motif Types Analysis

- **Chain Motif**: A → B → C
- **Fork Motif**: A ← B → C
- **Convergent Motif**: A → C ← B
- **Fully Connected**: A ↔ B ↔ C

---

## 📋 Usage Examples

### Basic Parameter Optimization
```bash
# Optimize Lorenz system parameters
python main.py optimize-params --system lorenz

# Custom sequence length
python main.py optimize-params --system rossler --length 6000
```

### Performance Analysis
```bash
# Length analysis
python main.py run-analysis --system lorenz --analysis-type length

# Noise robustness testing
python main.py run-analysis --system noisy_rossler --analysis-type noise

# Network scale analysis
python main.py run-analysis --system henon --analysis-type nodes
```

### Motif Analysis
```bash
# Standard three-node analysis
python main.py run-motifs --system lorenz

# High-precision analysis (more surrogate data)
python main.py run-motifs --system mackey_glass --length 6000 --surrogates 500
```

### 🚀 Batch Automated Analysis (New Feature)

**v2.0 New**: Provides two powerful batch analysis scripts that can automatically run complete analysis across all systems without manual intervention.

#### Full System Performance Analysis
```bash
# Run all systems across all analysis types (95 analysis tasks)
python run_all_analysis.py
```

**Features**:
- ✅ Automatically runs 19 systems × 5 analysis types
- 📁 Results automatically organized by timestamp into folders
- 🔺 Rössler system stability optimization with auto-recovery
- 📊 Detailed progress tracking and statistical reports
- 💾 Automatic backup of run configuration and results

#### Full System Motif Analysis
```bash
# Run three-node motif analysis for all systems
python run_all_motifs.py
```

**Features**:
- ✅ Automatically runs motif analysis for 19 systems
- 🎯 CCM + CTE dual validation methods
- 🔺 Special optimization for Rössler system
- 📁 Automatic organization and statistics of results

#### Results Folder Structure

Batch analysis automatically creates the following directory structure:

```
results/
├── analysis_runs/
│   └── run_YYYYMMDD_HHMMSS/
│       ├── run_config.json          # Run configuration
│       ├── run_statistics.json      # Run statistics
│       ├── results_*.json           # All analysis results
│       └── analysis_*.png           # All visualization charts
├── motifs_runs/
│   └── run_YYYYMMDD_HHMMSS/
│       ├── run_config.json          # Run configuration
│       ├── run_statistics.json      # Run statistics
│       └── motif_analysis_*.png     # All motif analysis charts
└── logs/
    ├── analysis_YYYYMMDD_HHMMSS.log # Analysis logs
    └── motifs_YYYYMMDD_HHMMSS.log   # Motif logs
```

#### 🔺 Rössler System Special Handling

Due to the numerical sensitivity of the Rössler system, batch scripts include specialized stability optimization:

**Analysis Script Optimization**:
- Uses fewer trial runs (≤15) and surrogate data (≤80)
- Automatic detection of numerical instability with recovery mode
- Multi-level parameter degradation strategy ensuring successful completion

**Motif Script Optimization**:
- Uses shorter time series (≤1500) and surrogate data (≤150)
- Automatic switch to ultra-conservative parameters on numerical anomalies
- Detailed recovery statistics and error reporting

---

## 📊 Output File Description

### Result Files
- **Format**: JSON
- **Naming**: `results_{system}_{analysis}_{timestamp}.json`
- **Content**: Complete numerical results and parameter information

### Visualization Files
- **Format**: PNG (300 DPI)
- **Naming**: `analysis_{system}_{analysis}_{timestamp}.png`
- **Features**: Publication-quality, includes statistical information

### Motif Analysis
- **File**: `motif_analysis_{system}_{timestamp}.png`
- **Content**: CCM and CTE method comparison, motif classification results

---

## 🛠️ Troubleshooting Guide

### 🔺 Rössler System Numerical Stability Issues

The Rössler system may experience numerical instability under certain conditions due to its parameter sensitivity. Here are identification and solution approaches:

#### Common Problem Symptoms
- ❌ Appearance of `NaN` or `Inf` values
- ❌ Rapid system state divergence
- ❌ Integrator errors or timeouts
- ❌ Anomalous CCM analysis results

#### Auto-Recovery Mechanisms

**Batch scripts have built-in** following recovery strategies:

1. **Automatic Parameter Adjustment**
   ```
   Trial runs: 20 → 15 → 10
   Surrogate data: 100 → 80 → 50
   Time series: 2000 → 1500 → 1000
   ```

2. **Time Step Optimization**
   ```
   Standard: dt = 0.25
   Strong coupling: dt = 0.1
   Extreme cases: dt = 0.05
   ```

3. **Integrator Parameters**
   ```
   rtol: 1e-8, atol: 1e-10
   mxstep: 15000 → 20000
   ```

#### Manual Optimization Recommendations

If using individual `main.py` for Rössler system analysis:

```bash
# Conservative parameter settings
python main.py run-analysis --system rossler --analysis-type length --trials 10 --surrogates 50

# Avoid excessively long time series
python main.py run-motifs --system noisy_rossler --length 1500 --surrogates 100

# For extremely unstable cases, use other systems for verification
python main.py run-analysis --system lorenz --analysis-type length  # Verify algorithm itself
```

#### Recommended Parameter Ranges

| System Version | Recommended Trials | Recommended Surrogates | Max Time Series |
|----------------|-------------------|----------------------|------------------|
| `rossler` | ≤15 | ≤80 | ≤3000 |
| `noisy_rossler` | ≤12 | ≤60 | ≤2500 |
| `rossler_dynamic_noise` | ≤10 | ≤50 | ≤2000 |

### Other Common Issues

#### Memory Insufficient
```bash
# Reduce parallel trial count
python main.py run-analysis --system lorenz --trials 10

# Use shorter sequences
python main.py run-motifs --system henon --length 1500
```

#### Runtime Too Long
```bash
# Reduce surrogate data count
python main.py run-analysis --system mackey_glass --surrogates 50

# Choose computationally faster systems for initial testing
python main.py run-analysis --system logistic --analysis-type length
```

#### Image Display Issues
Batch scripts automatically use non-interactive backend. If encountering display issues when running manually:
```bash
export MPLBACKEND=Agg
python main.py run-analysis --system lorenz --analysis-type length
```

---

## 🔬 Algorithm Principles

For detailed mathematical foundations and algorithm principles, please refer to:
- [详细技术文档 (Detailed Technical Documentation)](docs/技术文档.md)
- [算法原理详解 (Algorithm Principles Detailed)](docs/算法原理详解.md)

### CCM (Convergent Cross Mapping) Overview

CCM detects causal relationships by reconstructing attractor manifolds:

1. **State Space Reconstruction**: Using Takens embedding theorem
   ```
   X(t) = [x(t), x(t-τ), ..., x(t-(m-1)τ)]
   ```

2. **Cross Mapping**: Predict X values from Y's manifold
   ```
   ρ(L) = corr(x(t), x̂(t)|M_Y)
   ```

3. **Convergence Testing**: Correlation should converge as sequence length L increases

### Conditional Transfer Entropy (CTE)

CTE quantifies information flow from X to Y under condition Z:

```
CTE_{X→Y|Z} = H(Y_{t+1}|Y_t, Z_t) - H(Y_{t+1}|Y_t, X_t, Z_t)
```

---

## ⚙️ System Requirements

### Minimum Requirements
- **CPU**: Dual-core 2GHz+
- **Memory**: 4GB RAM
- **Disk**: 1GB available space

### Recommended Configuration
- **CPU**: Quad-core 3GHz+
- **Memory**: 8GB+ RAM
- **Disk**: 2GB+ SSD space

### Runtime Reference

| Analysis Type | Basic Systems | Noisy Systems | Long Sequences |
|---------------|---------------|---------------|----------------|
| Parameter Optimization | 30s | 45s | 2min |
| Length Analysis | 2min | 5min | 10min |
| Motif Analysis | 3min | 8min | 15min |

---

## 🧪 Testing and Validation

### Run Test Suite
```bash
python tests/test_suite.py
```

### Numerical Stability Testing
```bash
python test_systems_debug.py    # Basic stability
python test_stress_test.py      # Stress testing
```

### Expected Test Results
- ✅ All 7 basic systems numerically stable
- ✅ Long sequences (>5000 steps) integrate normally
- ✅ No divergence under strong coupling conditions
- ✅ Dynamic noise systems converge

---

## 📚 Dependencies

### Core Scientific Computing
```
numpy>=1.21.0          # Numerical computing foundation
scipy>=1.7.0           # Scientific computing and optimization
scikit-learn>=1.0.0    # Machine learning tools
```

### Visualization
```
matplotlib>=3.5.0      # Basic plotting
seaborn>=0.11.0        # Statistical visualization
```

### Other Tools
```
tqdm>=4.62.0           # Progress bar display
termcolor>=1.1.0       # Colored terminal output
rich>=12.0.0           # Rich text display
pyinform>=0.3.0        # Information theory calculations
```

---

## 🔄 Version History

### v3.0.0 (Current Version)
- ✨ Completely refactored modular architecture
- 🔧 Adaptive numerical integration and stability assurance
- 📊 Dual-core motif analysis (CCM+CTE)
- 🎨 Publication-quality visualization system
- 📈 Scientific parameter optimization (AMI+FNN)
- 🧪 Comprehensive automated testing

### v2.x (Historical Versions)
- Basic CCM implementation
- Simple system support
- Basic visualization

---

## 🤝 Contributing

### Code Standards
- Follow PEP 8 code style
- Add detailed function documentation
- Include unit tests

### Submission Process
1. Fork the repository
2. Create feature branch
3. Add test cases
4. Submit Pull Request

---

## 📞 Technical Support

### Issue Reporting
- Report bugs through GitHub Issues
- Provide complete error logs and environment information
- Include reproduction steps

### Feature Suggestions
- Describe expected functionality and use cases
- Provide relevant algorithm reference literature
- Consider backward compatibility

---

## 📖 Citation Information

If using this tool in academic research, please cite:

```bibtex
@software{ccm_analysis_toolbox_2024,
  title = {CCM Causal Analysis Toolbox: Professional Nonlinear Dynamics Causal Inference Platform},
  version = {3.0.0},
  year = {2024},
  note = {Causal analysis tool based on Convergent Cross Mapping and information theory}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Thanks to the contributions of the following open source projects and algorithms:
- George Sugihara et al.'s CCM theoretical foundation
- Takens embedding theorem mathematical framework
- scipy ecosystem numerical computing support
- matplotlib/seaborn visualization capabilities

---

## 📚 Additional Documentation

- [中文完整文档 (Chinese Complete Documentation)](README.md)
- [详细技术文档 (Detailed Technical Documentation)](docs/技术文档.md) 
- [算法原理详解 (Algorithm Principles Detailed)](docs/算法原理详解.md)

---

**📧 Contact**: For technical questions or collaboration opportunities, please contact through GitHub Issues

**🌟 If this project helps your research, please consider giving it a Star!**