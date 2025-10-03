# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Plan & Review

### Before Starting Work
- Always in plan mode to make a plan.
- After get the plan, make sure you Write the plan to .claude/tasks/TASK_NAME.md
- The plan should be a detailed implementation plan and the reasoning behind them. As well as tasks broken down.
- If the task requires external knowledge or certain packages, also research to get latest knowledge. (Use Task tool for research)
- Don't over plan it, always think MVP.
- Once the plan is done, ask for approval from the user before starting work.

### While implementing
- You should update the plan as you work.
- After you complete tasks in the plan, you should update and append detailed descriptions of the change you made, so following tasks can be easily hand over to other engineers.


## Project Overview

This is a professional-grade CCM (Convergent Cross Mapping) Causal Analysis Toolbox v3.0 for detecting causal relationships in nonlinear dynamical systems. The toolbox provides a complete solution from parameter optimization, system simulation, causal analysis to deep visualization using CCM and information-theoretic methods.

## Core Architecture

### Module Structure
```
project_CCM/
├── main.py                    # CLI entry point with argparse interface
├── run_all_analysis.py        # Batch analysis script for all systems
├── run_all_motifs.py          # Batch motif analysis script for all systems
├── core/                      # Core algorithm modules
│   ├── systems.py            # Dynamical systems generation (Lorenz, Rössler, etc.)
│   ├── ccm.py                # CCM core algorithm and surrogate methods
│   ├── analysis.py           # Core analysis workflows and single trial execution
│   ├── motifs.py             # Three-node motif analysis (CCM + CTE)
│   └── partial_ccm.py        # Partial CCM implementations
├── utils/                     # Utility modules
│   ├── params.py             # Scientific parameter optimization (AMI + FNN)
│   └── visualization.py      # Professional visualization suite
└── tests/
    └── test_suite.py         # Comprehensive automated test suite
```

### Key Components

**Dynamical Systems (core/systems.py)**: Supports classic systems (Lorenz, Rössler, Logistic Map, Hénon Map) and advanced systems (Mackey-Glass, Kuramoto, Hindmarsh-Rose) with noise variants. Features mathematically sound dynamic noise implementation using step-wise SDE integration for continuous systems.

**CCM Algorithm (core/ccm.py)**: Core convergent cross mapping implementation with multiple surrogate methods (FFT, AAFT, IAAFT, Time Shift, Random Reorder).

**Parameter Optimization (utils/params.py)**: Scientific approach using Average Mutual Information (AMI) for optimal time delay (tau) and False Nearest Neighbors (FNN) for embedding dimension (Dim).

**Analysis Framework (core/analysis.py)**: Comprehensive performance analysis across multiple dimensions (length, degree, coupling, nodes, noise).

**Motif Analysis (core/motifs.py)**: Three-node motif analysis combining CCM with Conditional Transfer Entropy (CTE) for robust causal inference.

## Common Commands

### Installation
```bash
pip install -r requirements.txt
```

### Testing
```bash
# Run comprehensive test suite
python tests/test_suite.py

# Or run from project root
python -m tests.test_suite
```

### Main Operations

**Parameter Optimization**:
```bash
python main.py optimize-params --system lorenz
python main.py optimize-params --system lorenz --length 8000
```

**Full Analysis**:
```bash
# Time series length analysis
python main.py run-analysis --system lorenz --analysis-type length

# Noise robustness analysis  
python main.py run-analysis --system noisy_lorenz --analysis-type noise

# Network degree analysis
python main.py run-analysis --system rossler --analysis-type degree
```

**Motif Analysis**:
```bash
# Three-node motif analysis with dual methods (CCM + CTE)
python main.py run-motifs --system lorenz
python main.py run-motifs --system rossler --length 4000 --surrogates 200
```

**Batch Operations**:
```bash
# Run comprehensive analysis across all systems and analysis types
python run_all_analysis.py

# Run motif analysis across all systems  
python run_all_motifs.py
```

### Available Systems
- Classic: `lorenz`, `rossler`, `logistic`, `henon`, `mackey_glass`, `kuramoto`, `hindmarsh_rose`
- Observational noise: `noisy_lorenz`, `noisy_rossler`, `noisy_mackey_glass`, etc.
- Dynamic noise (SDE): `lorenz_dynamic_noise`, `rossler_dynamic_noise`, `logistic_dynamic_noise`, etc.

### Analysis Types
- `length`: Time series length vs performance
- `degree`: Network connectivity vs performance  
- `coupling`: Coupling strength vs performance
- `nodes`: Number of nodes vs performance
- `noise`: Noise robustness analysis

## Development Notes

### Key Algorithms
- **AMI**: Determines optimal time delay (tau) by finding first local minimum
- **FNN**: Finds minimum embedding dimension (Dim) for complete attractor unfolding
- **CCM**: Core convergent cross mapping with Pearson correlation
- **CTE**: Conditional Transfer Entropy for direct causal quantification
- **Dynamic Noise**: Step-wise SDE integration with proper sqrt(dt) scaling to avoid numerical divergence

### Data Flow
1. System generation with specified parameters and coupling structure
2. Scientific parameter optimization (AMI→tau, FNN→Dim)  
3. CCM analysis with multiple surrogate methods for statistical validation
4. Performance evaluation using AUROC across multiple trials
5. Professional visualization with statistical significance testing

### Output Files
- Results saved as JSON with timestamp: `results_{system}_{analysis}_{timestamp}.json`
- Visualizations saved as PNG: `analysis_{system}_{analysis}_{timestamp}.png`
- Motif analysis: `motif_analysis_{system}_{timestamp}.png`

### Dependencies
Core scientific stack: numpy, scipy, matplotlib, seaborn, scikit-learn, tqdm, termcolor, rich, pyinform

### Code Language and Structure
- The codebase is bilingual (Chinese and English), with comments and documentation in both languages
- The main CLI interface and user-facing messages are primarily in Chinese
- Code structure follows scientific computing conventions with modular, reusable components
- No additional build tools, linting, or formatting configurations are present in this repository

### Debugging and Development
- Debug scripts available: `test_systems_debug.py`, `test_stress_test.py` 
- System stability analysis generates debug plots (e.g., `debug_*_stability.png`)
- Results are automatically timestamped and saved as JSON with corresponding PNG visualizations
- The toolbox uses matplotlib with Agg backend for non-interactive plotting in batch operations