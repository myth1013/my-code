# 🚀 LLM-Guided Evolutionary Optimal Control for IoUT

A framework for suppressing malware propagation in the **Internet of Underwater Things (IoUT)** based on the **Reaction‑Diffusion Sleep Control Model (RDSCM)**.

🔄 **Closed‑loop workflow**:
1. 🤖 **LLM** generates adversarial underwater scenarios  
2. ➡️ **PDE** solves malware diffusion dynamics  
3. ➡️ **CMA‑ES** evolves optimal node sleep scheduling  

---

## 📁 Directory Structure

```
./
├── README.md          # This file
├── config.py          # ⚙️ Runtime configuration (LLM, paths, optimiser)
├── parameter.py       # 📊 Fixed physical & epidemiological parameters
└── control_v1.py      # 🎮 First‑version parameterised bang‑bang control law
```

---

## 📄 File Descriptions

### ⚙️ `config.py`

- Loads environment variables from `.env`
- LLM call settings (model, temperature, retries, API endpoint)
- File path templates for prompts, logs, best parameters, diagnostics
- CMA‑ES population size, initial step size, warm‑up trials
- Optuna number of trials and random seed

### 📊 `parameter.py`

Defines the dataclass `RDParams` holding:

- Spatial/temporal discretisation (`X`, `T`, `dx`, `dt`)
- Diffusion coefficients for UWSN (`Du`) and AUV (`Da`)
- Birth, recovery, immunity‑loss, and mortality rates
- Energy depletion and sleep‑wake transition rates
- Running, terminal, and control cost weights (`A`, `B`, `C`)
- Target equilibrium values for all state variables
- Upper bounds for the six control inputs
- Spatially heterogeneous infection rate functions `K1`, `K2`, `K12`, `K21`

All variables are exported globally for use by other modules.

### 🎮 `control_v1.py`

**Core function:** `control_law(t, x, state, params)`

**Inputs:**

| Argument  | Description |
|-----------|-------------|
| `t`       | Current time (float) |
| `x`       | Spatial grid (1D array, length `M+1`) |
| `state`   | 10 state variables at each grid point: `[Su, Iu, Ru, LSu, LIu, LRu, Sa, Ia, Ra, Sus]` |
| `params`  | Dictionary of 9 tunable parameters (see [Tunable Parameters](#-tunable-parameters-decision-variables-for-optimisation)) |

**Output:** Control array of shape `(M+1, 6)` with columns:

| Control | Description                                      | Output     |
|---------|--------------------------------------------------|------------|
| `u1`    | Su → Ru (preventive treatment for UWSN)          | `{0, 1}`   |
| `u2`    | Sa → Ra (preventive treatment for AUV)           | `{0, 1}`   |
| `u3`    | LSu → Su, LIu → Iu, LRu → Ru (wake‑up low‑energy) | `{0, 1}`   |
| `u4`    | Iu → Ru (intensified treatment for UWSN)         | `{0, 0.5}` |
| `u5`    | Ia → Ra (intensified treatment for AUV)          | `{0, 0.5}` |
| `u6`    | Su → Sus (forced sleep)                          | `{0, 0.5}` |

> ⚡ Fully vectorised – **no Python loops** – for performance.

---

## ⚡ Quick Start

### 1️⃣ Set up environment variables

Create a `.env` file in the project root with:

```env
OPENAI_API_KEY=your_key_here
BASE_URL=https://your_llm_endpoint
```

### 2️⃣ Install dependencies

```bash
pip install numpy scipy matplotlib cma optuna python-dotenv
```

### 3️⃣ (Optional) Adjust hyperparameters

Tweak `config.py` and physical constants in `parameter.py` as needed.

---

## 🎯 Tunable Parameters (Decision Variables for Optimisation)

The `params` dictionary passed to `control_law()` contains the following keys with typical search ranges:

| Parameter       | Typical Range | Role                                              |
|-----------------|---------------|---------------------------------------------------|
| `theta_Iu`      | (0.5, 2.0)    | Threshold for triggering `u1`                     |
| `theta_Ia`      | (0.5, 2.0)    | Threshold for triggering `u2`                     |
| `theta_low`     | (0.1, 1.0)    | Threshold for wake‑up (`u3`)                      |
| `w_Iu`          | (0.5, 3.0)    | Weight for `Iu` deviation in `u4`                 |
| `w_Ia`          | (0.5, 3.0)    | Weight for `Ia` deviation in `u5`                 |
| `theta_Su_high` | (1.0, 5.0)    | High `Su` threshold for sleep (`u6`)              |
| `theta_Su_low`  | (0.1, 1.0)    | Low `Su` threshold for sleep (`u6`)               |
| `k_time`        | (0.0, 2.0)    | Time modulation factor                            |
| `alpha_wake`    | (0.0, 1.0)    | `LIu` contribution to wake‑up decision            |
| `beta_sleep`    | (0.0, 1.0)    | Energy balance coefficient for sleep              |

---

## 💡 Notes

- This is the **first version** (`control_v1.py`) – future versions may improve the control logic.
- The framework is designed for **research and simulation**; real‑world deployment would need additional robustness.
- All array operations are **vectorised** – keep this property in any modifications.

---

## 📄 License

[Specify your license here, e.g., MIT, Apache 2.0, or "All rights reserved".]

## 📧 Contact

For questions or contributions, please open an issue or contact the maintainers.
