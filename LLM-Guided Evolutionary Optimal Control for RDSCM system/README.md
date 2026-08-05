<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/Optuna-integrated-orange?style=flat-square" alt="Optuna">
</p>

<h1 align="center">🧠 LLM-Guided Evolutionary Framework for Spatiotemporal Control against Malware Propagation in IoUT</h1>

<p align="center">
  <b>An automated closed-loop framework that evolves control-law structures via LLM reasoning and optimizes parameters via CMA-ES.</b><br>
  Applied to malware propagation suppression in IoUT (Internet of Underwater Things) networks.
</p>

---

## 📑 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🏗️ System Architecture](#%EF%B8%8F-system-architecture)
- [📦 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [⚙️ Configuration](#%EF%B8%8F-configuration)
- [🔄 Workflow](#-workflow)
- [📁 File Structure](#-file-structure)
- [🎓 Key Concepts](#-key-concepts)
- [⚠️ Notes](#%EF%B8%8F-notes)

---

## 🎯 Overview

This project implements a **closed-loop evolutionary framework** that couples the structural reasoning power of Large Language Models (LLMs) with the numerical optimization capability of CMA-ES (Covariance Matrix Adaptation Evolution Strategy) to automatically design and refine PDE-constrained bang-bang control laws.

**Target Problem**: In a Reaction-Diffusion malware propagation model describing IoUT networks, design a 6-dimensional bang-bang control policy (immunization, treatment, charging, sleep enforcement, etc.) that maintains infected node densities near desired equilibrium values while minimizing total control expenditure.

**Core Innovations**:
- 🧠 **LLM-Driven Structural Evolution**: The LLM modifies the control-law source code (e.g., adding hysteresis, cross-channel coupling, time-varying gains) based on quantitative diagnostic reports, rather than merely tuning coefficients.
- ⚡ **Single-Stage Adaptive CMA-ES**: A parameter-dimension-aware CMA-ES with auto-scaled population size performs full-fidelity optimization for every trial.
- 📊 **Four-Class Diagnostics (L1–L4)**: Automated extraction of temporal trajectories, switching activity, parameter saturation, and cost decomposition to guide structural improvements.
- 📈 **Diagnostic Visualization**: Evolutionary trajectories and diagnostic indicators are tracked across iterations and can be visualized to inspect convergence and structural changes.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **LLM Closed-Loop Diagnosis** | Automatically generates L1–L4 diagnostic indicators (temporal trajectory, switching frequency, parameter saturation, cost decomposition) to drive LLM-based structural code revision. |
| ⚡ **Single-Stage Adaptive CMA-ES** | Population size auto-scales with parameter dimension (`base + per_dim × d`, capped at 600 and forced even). All trials run full-fidelity PDE simulation. |
| 🔧 **Auto Code Repair** | Built-in AST parser automatically fixes vectorization bugs (e.g., `float(array)` → `array.astype(float)`) in LLM-generated code. |
| 📊 **Four-Class Diagnostic System** | Extracts multi-dimensional metrics: time-series trajectories, switching events, parameter logic consistency, and running/terminal cost decomposition. |
| 📈 **Evolution Visualization** | Tracks and logs diagnostic indicators across evolutionary iterations, enabling post-hoc visualization of structural refinement trajectories. |
| 💾 **Resume from Checkpoint** | Supports resuming evolution from any historical version; automatically loads existing control laws and diagnostic reports. |

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator (Main Loop)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  LLM Agent   │  │  CMA-ES      │  │ PDE Simulator    │  │
│  │  (Structure  │◄─┤  (Parameter  │◄─┤ (RK4 Solver)     │  │
│  │   Evolution) │  │  Optimization)│  │                  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────┘  │
│         │                 │                                  │
│         ▼                 ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Diagnostic Extractor (L1–L4)              │   │
│  │  L1: Time Trajectory   L2: Switching Activity          │   │
│  │  L3: Parameter Sat.    L4: Cost Decomposition         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Requirements
- Python >= 3.9
- NumPy, SciPy
- Optuna
- LangChain + OpenAI-compatible API access

### Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/llm-evolutionary-control.git
cd llm-evolutionary-control

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install numpy scipy optuna langchain-openai
```

---

## 🚀 Quick Start

### 1. Configure LLM and CMA-ES

Edit `config.py` to set your LLM API credentials:

```python
# Example: Moonshot Kimi-K2.5
LLM_MODEL = "moonshotai/Kimi-K2.5"
OPENAI_API_KEY = "your-api-key-here"
BASE_URL = "https://api-inference.modelscope.cn/v1"
LLM_TEMPERATURE = 0.0
LLM_MAX_RETRIES = 30
```

### 2. Launch the Evolution Loop

```bash
python orchestrator.py
```

The program automatically executes:
1. 📝 Generates initial control law `control_v1.py`
2. ⚡ Runs single-stage adaptive CMA-ES (full-fidelity simulation)
3. 🔬 Performs final simulation and extracts L1–L4 diagnostics
4. 🧠 Calls LLM to generate diagnostic report `diagnostic_v1.txt`
5. 🔧 LLM modifies the control law based on diagnosis → `control_v2.py`
6. 🔄 Repeats until convergence or max iterations reached

### 3. Resume from a Specific Version

If the LLM call is interrupted, resume from any existing version:

```python
from orchestrator import resume_from_diagnosis

# Resume from V2 to generate V3
resume_from_diagnosis(2)
```

---

## ⚙️ Configuration

### LLM Settings (`config.py`)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `LLM_MODEL` | Model identifier | `"moonshotai/Kimi-K2.5"` |
| `BASE_URL` | API base URL | `"https://api.openai.com/v1"` |
| `LLM_TEMPERATURE` | Sampling temperature (0.0 recommended for determinism) | `0.0` |
| `LLM_MAX_RETRIES` | Max retry attempts for LLM calls | `30` |

### CMA-ES Settings (`config.py`)

```python
# Single-stage adaptive CMA-ES
CMAES_STAGE1_TRIALS = 8000      # Total evaluation budget
CMAES_STAGE2_TRIALS = 2000      # Added to above for total budget

# Adaptive population sizing
CMAES_POPSIZE_BASE = 4          # Base population size
CMAES_POPSIZE_PER_DIM = 4       # Additional individuals per parameter dimension
# Effective popsize = min(base + per_dim × d, 600), forced to nearest even number

CMAES_SIGMA0_STAGE1 = 0.5       # Initial step size
CMAES_N_STARTUP_TRIALS = 300    # Random startup trials for initial covariance
```

**Adaptive Population Rule**: The population size scales linearly with the number of tunable parameters `d` in the current control law:
```
popsize = min(CMAES_POPSIZE_BASE + CMAES_POPSIZE_PER_DIM × d, 600)
```
The result is rounded to the nearest even number to satisfy CMA-ES requirements.

### Physical & Cost Parameters (`parameter.py`)

- **Desired states**: `Iud=1.2`, `Iad=1.2`, `LSud=0.5`, `LIud=0.2`, `LRud=0.8`
- **Control cost coefficients**: `C1=0.5`, `C2=0.3`, `C3=0.5`, `C4=1.8`, `C5=5.5`, `C6=2.1`

---

## 🔄 Workflow

### One Full Iteration Cycle (Version k)

```text
Control Law π_k 
    │
    ▼
┌─────────────────────┐
│ Single-Stage CMA-ES │  ← Full-fidelity PDE simulation for every trial
│ (Adaptive Popsize)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Final Simulation    │  ← Collect time-series & switching events
│ (L1–L4 Extraction)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ LLM Diagnosis       │  ← Generate structured report with 8 tasks
│ (L1–L4 Analysis)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Structural Modification │  ← LLM edits control_law() source code
│ (Code Generation)   │
└──────────┬──────────┘
           │
           ▼
    Control Law π_{k+1}
```

### Diagnostic Indicator System (L1–L4)

| Indicator | Content | Purpose |
|-----------|---------|---------|
| **L1** | Time-trajectory analysis (early/middle/late phase means, peak detection) | Identify temporal misalignment |
| **L2** | Bang-bang switching analysis (switch counts, min intervals, chattering flags) | Detect control chattering |
| **L3** | Parameter saturation (bounds hitting, sign flips, threshold relationships) | Guide bound adjustment |
| **L4** | Cost decomposition (state-deviation cost vs. control-execution cost per channel) | Identify cost bottlenecks |

### Diagnostic Visualization

Across evolutionary iterations, the framework logs:
- **L1**: RMS state deviation per phase (0–1s, 1–2s, 2–3s)
- **L2**: Aggregate switching activity per control channel
- **L3**: Fraction of parameters at/near bounds
- **L4**: Stacked cost decomposition (state + control + terminal)

These indicators can be plotted to visualize the evolutionary trajectory, inspect whether structural modifications resolve targeted issues, and detect diminishing returns or failure patterns.

---

## 📁 File Structure

```
.
├── orchestrator.py              # Main loop: evolution cycle, single-stage CMA-ES, simulation orchestration
├── model.py                     # PDE solver: MATLAB-strict RK4 with immediate boundary assignment
├── llm_utils.py                 # LLM interaction: initial generation, diagnosis report, control-law modification
├── diagnostic_extractor.py      # Basic diagnostics: metric extraction, anomaly detection, code structure check
├── diagnostic_extractor_v2.py   # Enhanced diagnostics: full L1–L4 analysis
├── evolution_logger.py          # Evolution logger: JSON history, cross-version comparison tables
├── config.py                    # Global configuration: LLM, optimizer, file paths
├── parameter.py                 # Physical parameters: PDE coefficients, transmission rates, cost weights
├── initial_prompt_v1.txt        # LLM initial prompt (control-law generation)
├── control_v1.py                # V1 initial control law (example)
├── control_v2.py                # V2 evolved control law (example, with hysteresis / time-varying gains)
├── diagnostic_v1.txt            # V1 diagnostic report (example)
├── diagnostic_v2.txt            # V2 diagnostic report (example)
└── evolution_log.json           # Auto-generated evolution history log
```

### Key Files

- **`model.py`**  
  Implements the 10-dimensional RDSCM (Reaction-Diffusion malware propagation model) RK4 integrator. Strictly replicates MATLAB's immediate boundary-assignment logic under Neumann boundary conditions.

- **`llm_utils.py`**  
  Three core LLM calls:
  - `generate_initial_control_law()`: Generates V1 from the initial prompt
  - `generate_diagnostic_report()`: Produces structured 8-task diagnostic reports
  - `modify_control_law()`: Revises `control_law()` source code based on diagnosis

- **`orchestrator.py`**  
  Main controller containing:
  - `run_two_stage_cmaes()`: Single-stage adaptive CMA-ES with dimension-aware population sizing
  - `auto_fix_code()`: AST-level automatic repair of vectorization errors
  - `quick_test_control_law()`: Smoke test for shape and bang-bang value validation

---

## 🎓 Key Concepts

### Bang-Bang Control Constraints
- `u1, u2, u3` ∈ `{0, 1}` (on-off controls)
- `u4, u5, u6` ∈ `{0, 0.5}` (half-intensity controls)

All controls must be generated via `np.where(condition, max_value, 0.0)`. Continuous intermediate values are strictly forbidden.

### Control-Law Code Rules
LLM-generated control laws must obey the following red lines:
1. 🚫 **No** `float()`, `int()`, `bool()` applied to arrays
2. 🚫 **No** `and` / `or` inside `np.where`; use `&` / `|` instead
3. 🚫 **No** explicit spatial loops (`for i in range(M+1)`); operations must be vectorized
4. 🚫 **No** reading target values (`Iu_target`, etc.) from `params`; they must be hardcoded constants
5. 🚫 **No** `np.clip()` on `params`; bounds are managed externally by the optimizer
6. ✅ Mandatory `PARAM DECLARATION` block listing every tunable parameter with `(low, high)` ranges

### Single-Stage Adaptive CMA-ES
- **Dimension-Aware Population**: Automatically scales with the number of tunable parameters in the current control law, ensuring sufficient statistical power for high-dimensional structures without manual tuning.
- **Full-Fidelity Evaluation**: Every trial runs the complete PDE simulation (`N=60,000` steps), so the LLM receives diagnostically accurate feedback without multi-fidelity approximation artifacts.
- **Warm Start**: Historical best parameters can be enqueued via `study.enqueue_trial()` to pass knowledge across evolutionary versions.

---

## ⚠️ Notes

1. **API Costs**  
   Each iteration invokes the LLM 2–3 times (diagnosis + modification). Monitor token usage if using commercial endpoints. `LLM_MAX_RETRIES` is capped at 30 to balance robustness and cost.

2. **Computational Load**  
   A single full simulation (`N=60,000` steps, `M=200` spatial grid) takes several seconds to tens of seconds. With thousands of CMA-ES trials per version, total runtime can be significant; multi-core CPU or server environments are recommended.

3. **Version Compatibility**  
   When resuming, ensure both `control_v{k}.py` and `diagnostic_v{k}.txt` exist in the working directory.

4. **Convergence Detection**  
   The framework auto-terminates if two consecutive iterations show no parameter-count increase **and** a relative cost improvement below 1%.

5. **Dynamic Parameter Bounds**  
   The LLM can suggest new parameter ranges via `APPLY_BOUND: param_name (new_low, new_high)` in the diagnostic report. These are parsed and saved to `dynamic_bounds.json` for subsequent iterations.
