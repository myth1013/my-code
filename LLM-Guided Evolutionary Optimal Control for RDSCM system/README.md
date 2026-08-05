🚀 LLM-Guided Evolutionary Optimal Control for IoUT
================================================================================

A framework for suppressing malware propagation in the Internet of Underwater
Things (IoUT) based on the Reaction‑Diffusion Sleep Control Model (RDSCM).

🔄 Closed‑loop workflow:
  🤖 LLM generates adversarial underwater scenarios
  ➡️  PDE solves malware diffusion dynamics
  ➡️  CMA‑ES evolves optimal node sleep scheduling

⚠️  This project contains NO reinforcement learning components.

--------------------------------------------------------------------------------
📁 Directory Structure
--------------------------------------------------------------------------------

./
├── README.md                📖 This file
├── config.py                ⚙️ Runtime configuration (LLM, paths, optimiser)
├── parameter.py             📊 Fixed physical & epidemiological parameters
└── control_v1.py            🎮 First‑version parameterised bang‑bang control law

--------------------------------------------------------------------------------
📄 File Descriptions
--------------------------------------------------------------------------------

⚙️ config.py
  • Loads environment variables from .env
  • LLM call settings (model, temperature, retries, API endpoint)
  • File path templates for prompts, logs, best parameters, diagnostics
  • CMA‑ES population size, initial step size, warm‑up trials
  • Optuna number of trials and random seed

📊 parameter.py
  • Defines dataclass RDParams holding:
      - Spatial/temporal discretisation (X, T, dx, dt)
      - Diffusion coefficients for UWSN (Du) and AUV (Da)
      - Birth, recovery, immunity‑loss, and mortality rates
      - Energy depletion and sleep‑wake transition rates
      - Running, terminal, and control cost weights (A, B, C)
      - Target equilibrium values for all state variables
      - Upper bounds for the six control inputs
  • Exports spatially heterogeneous infection rate functions K1, K2, K12, K21
  • All variables are exported globally for use by other modules

🎮 control_v1.py
  • Core function: control_law(t, x, state, params)
  • Inputs:
      t     : current time (float)
      x     : spatial grid (1D array, length M+1)
      state : 10 state variables at each grid point
              [Su, Iu, Ru, LSu, LIu, LRu, Sa, Ia, Ra, Sus]
      params: dictionary of 9 tunable parameters (see Section 🎯 below)
  • Output: control array of shape (M+1, 6) with columns:
      u1 : Su -> Ru (preventive treatment for UWSN)          {0, 1}
      u2 : Sa -> Ra (preventive treatment for AUV)           {0, 1}
      u3 : LSu->Su, LIu->Iu, LRu->Ru (wake‑up low‑energy)   {0, 1}
      u4 : Iu -> Ru (intensified treatment for UWSN)        {0, 0.5}
      u5 : Ia -> Ra (intensified treatment for AUV)         {0, 0.5}
      u6 : Su -> Sus (forced sleep)                         {0, 0.5}
  • Fully vectorised (no Python loops) for performance.

--------------------------------------------------------------------------------
⚡ Quick Start
--------------------------------------------------------------------------------

1️⃣ Set up environment variables
   Create a .env file in the project root with:
       OPENAI_API_KEY=your_key_here
       BASE_URL=https://your_llm_endpoint

2️⃣ Install dependencies
   $ pip install numpy scipy matplotlib cma optuna python-dotenv

3️⃣ (Optional) Adjust hyperparameters in config.py and physical constants in
   parameter.py to match your simulation scenario.

--------------------------------------------------------------------------------
🎯 Tunable Parameters (decision variables for optimisation)
--------------------------------------------------------------------------------

The `params` dictionary passed to control_law() contains the following keys
with typical search ranges:

┌─────────────────┬───────────────┬────────────────────────────────────────────┐
│ Parameter       │ Typical range │ Role                                       │
├─────────────────┼───────────────┼────────────────────────────────────────────┤
│ theta_Iu        │ (0.5, 2.0)    │ Threshold for triggering u1               │
│ theta_Ia        │ (0.5, 2.0)    │ Threshold for triggering u2               │
│ theta_low       │ (0.1, 1.0)    │ Threshold for wake‑up (u3)                │
│ w_Iu            │ (0.5, 3.0)    │ Weight for Iu deviation in u4             │
│ w_Ia            │ (0.5, 3.0)    │ Weight for Ia deviation in u5             │
│ theta_Su_high   │ (1.0, 5.0)    │ High Su threshold for sleep (u6)          │
│ theta_Su_low    │ (0.1, 1.0)    │ Low Su threshold for sleep (u6)           │
│ k_time          │ (0.0, 2.0)    │ Time modulation factor                    │
│ alpha_wake      │ (0.0, 1.0)    │ LIu contribution to wake‑up decision      │
│ beta_sleep      │ (0.0, 1.0)    │ Energy balance coefficient for sleep      │
└─────────────────┴───────────────┴────────────────────────────────────────────┘

--------------------------------------------------------------------------------
💡 Notes
--------------------------------------------------------------------------------
• This is the first version of the control law (control_v1.py). Future versions
  may introduce improved logic.
• The framework is designed for research and simulation; real‑world deployment
  would require additional robustness measures.
• All array operations are vectorised; any modifications should preserve this
  property to maintain efficiency.

================================================================================
                            End of README
================================================================================
