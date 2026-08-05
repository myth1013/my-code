# ========== VECTORIZATION SELF-CHECK ==========
# [x] 1. No `float(`, `int(`, `bool(` applied to any array or comparison expression.
# [x] 2. All `np.where` conditions use `&` or `|`, not `and` or `or`.
# [x] 3. `np.max` / `np.min` used instead of `max` / `min`.
# [x] 4. No `for i in range(M+1)` loops.
# [x] 5. Return array has shape `(M+1, 6)` with columns [u1, u2, u3, u4, u5, u6].
# [x] 6. `u4` and `u5` are correctly implemented (not forced to zero).
# [x] 7. `T = 3.0` is used directly for any time normalization.
# ==============================================

import numpy as np

# ========== PARAM DECLARATION ==========
# @param: theta_Iu (0.5, 2.0)        # Threshold for Iu to trigger preventive treatment u1
# @param: theta_Ia (0.5, 2.0)        # Threshold for Ia to trigger AUV treatment u2
# @param: theta_low (0.1, 1.0)       # Threshold for low-energy nodes to trigger wake-up u3
# @param: w_Iu (0.5, 3.0)            # Weight for Iu deviation in intensified treatment u4
# @param: w_Ia (0.5, 3.0)            # Weight for Ia deviation in AUV intensified treatment u5
# @param: theta_Su_high (1.0, 5.0)   # High threshold for Su to trigger forced sleep u6
# @param: theta_Su_low (0.1, 1.0)    # Low threshold for Su to allow forced sleep u6
# @param: k_time (0.0, 2.0)          # Time modulation factor for control sensitivity
# @param: alpha_wake (0.0, 1.0)      # Coefficient for LIu contribution to wake-up decision
# @param: beta_sleep (0.0, 1.0)       # Coefficient for energy balance in sleep decision
# ========== END PARAM DECLARATION ==========

def control_law(t, x, state, params):
    """
    Parameterized bang-bang control law for underwater sensor network virus propagation.
    
    Parameters:
    -----------
    t : float
        Current time in [0, 3.0]
    x : ndarray, shape (M+1,)
        Spatial grid points
    state : ndarray, shape (M+1, 10)
        State variables: [Su, Iu, Ru, LSu, LIu, LRu, Sa, Ia, Ra, Sus]
    params : dict
        Dictionary containing tunable parameters
        
    Returns:
    --------
    control : ndarray, shape (M+1, 6)
        Control values [u1, u2, u3, u4, u5, u6] for each spatial point
    """
    
    # Extract state variables (all are arrays of shape (M+1,))
    Su = state[:, 0]   # Susceptible UWSN nodes
    Iu = state[:, 1]   # Infected UWSN nodes
    Ru = state[:, 2]   # Recovered UWSN nodes
    LSu = state[:, 3]  # Low-energy susceptible UWSN nodes
    LIu = state[:, 4]  # Low-energy infected UWSN nodes
    LRu = state[:, 5]  # Low-energy recovered UWSN nodes
    Sa = state[:, 6]   # Susceptible AUV nodes
    Ia = state[:, 7]   # Infected AUV nodes
    Ra = state[:, 8]   # Recovered AUV nodes
    Sus = state[:, 9]  # Sleeping UWSN nodes
    
    # Number of spatial points
    M_plus_1 = x.shape[0]
    
    # Target values from cost function
    Iu_target = 1.2
    Ia_target = 1.2
    LSu_target = 0.5
    LIu_target = 0.2
    LRu_target = 0.8
    
    # Extract tunable parameters with defaults
    theta_Iu = params.get('theta_Iu', 1.0)        # Threshold for triggering u1
    theta_Ia = params.get('theta_Ia', 1.0)        # Threshold for triggering u2
    theta_low = params.get('theta_low', 0.5)      # Threshold for low-energy wake-up
    w_Iu = params.get('w_Iu', 1.5)                # Weight for Iu in u4 decision
    w_Ia = params.get('w_Ia', 1.5)                # Weight for Ia in u5 decision
    theta_Su_high = params.get('theta_Su_high', 2.5)  # High Su threshold for sleep
    theta_Su_low = params.get('theta_Su_low', 0.5)    # Low Su threshold for sleep
    k_time = params.get('k_time', 1.0)            # Time modulation factor
    alpha_wake = params.get('alpha_wake', 0.5)    # LIu contribution to wake-up
    beta_sleep = params.get('beta_sleep', 0.3)    # Energy balance coefficient
    
    # Normalized time for temporal modulation (T=3.0 fixed)
    tau = t / 3.0
    
    # Time-dependent sensitivity factor (increases control action as time progresses)
    time_factor = 1.0 + k_time * tau
    
    # ============================================================
    # CONTROL u1: Su -> Ru (preventive treatment for UWSN)
    # Bang-bang: {0, 1}
    # Activate when Iu exceeds threshold, modulated by time and spatial infection pressure
    # ============================================================
    
    # Condition: Iu is above target and significant infection exists
    # Higher Iu deviation from target triggers preventive treatment
    Iu_deviation = Iu - Iu_target
    infection_pressure = Iu + alpha_wake * LIu  # Total infected including low-energy
    
    # u1 activates when infection pressure is high AND Iu exceeds threshold-adjusted target
    u1_condition = (infection_pressure > theta_Iu * time_factor) & (Iu_deviation > -0.3)
    u1 = np.where(u1_condition, 1.0, 0.0)
    
    # ============================================================
    # CONTROL u2: Sa -> Ra (preventive treatment for AUV)
    # Bang-bang: {0, 1}
    # Activate when Ia exceeds threshold, protecting valuable AUV assets
    # ============================================================
    
    # AUV infection deviation
    Ia_deviation = Ia - Ia_target
    
    # u2 activates when AUV infection is significant
    # AUVs are more valuable (lower cost coefficient), so more aggressive protection
    u2_condition = (Ia > theta_Ia * 0.8 * time_factor) | (Ia_deviation > 0.2)
    u2 = np.where(u2_condition, 1.0, 0.0)
    
    # ============================================================
    # CONTROL u3: LSu->Su, LIu->Iu, LRu->Ru (wake-up low-energy nodes)
    # Bang-bang: {0, 1}
    # Activate when low-energy nodes deviate from targets or when workforce is needed
    # ============================================================
    
    # Total low-energy population
    low_energy_total = LSu + LIu + LRu
    low_energy_target_total = LSu_target + LIu_target + LRu_target
    
    # Deviation from low-energy targets
    LSu_dev = np.abs(LSu - LSu_target)
    LIu_dev = np.abs(LIu - LIu_target)
    LRu_dev = np.abs(LRu - LRu_target)
    total_low_dev = LSu_dev + LIu_dev + LRu_dev
    
    # Wake-up needed when: low-energy nodes are too high (need to recover) 
    # OR when infected low-energy nodes need treatment
    # OR when susceptible low-energy nodes are needed for work
    need_workforce = (Su + LSu < 1.0) & (Iu + LIu > 0.8)  # Need more susceptible nodes
    
    u3_condition = ((low_energy_total > low_energy_target_total * (1.0 + theta_low)) |
                    (LIu > LIu_target * 2.0) | 
                    (need_workforce & (LSu > 0.3)))
    u3 = np.where(u3_condition, 1.0, 0.0)
    
    # ============================================================
    # CONTROL u4: Iu -> Ru (intensified treatment for UWSN)
    # Bang-bang: {0, 0.5}
    # Stronger intervention when Iu deviates significantly from target
    # ============================================================
    
    # Weighted deviation from target
    weighted_Iu_dev = w_Iu * np.abs(Iu - Iu_target)
    
    # Intensified treatment when Iu is far from target (either too high or too low)
    # But primarily activated when Iu is too high (infection control)
    u4_condition = (Iu > Iu_target * (1.0 + 0.3 * time_factor)) & (weighted_Iu_dev > 0.4)
    u4 = np.where(u4_condition, 0.5, 0.0)
    
    # ============================================================
    # CONTROL u5: Ia -> Ra (intensified treatment for AUV)
    # Bang-bang: {0, 0.5}
    # Expensive control, used only when AUV infection is critical
    # ============================================================
    
    # Weighted deviation for AUV
    weighted_Ia_dev = w_Ia * np.abs(Ia - Ia_target)
    
    # u5 is expensive (cost 5.5), so use sparingly - only when critical
    # Activate when Ia significantly exceeds target
    u5_condition = (Ia > Ia_target * (1.0 + 0.5 * time_factor)) & (weighted_Ia_dev > 0.6)
    u5 = np.where(u5_condition, 0.5, 0.0)
    
    # ============================================================
    # CONTROL u6: Su -> Sus (forced sleep for energy saving)
    # Bang-bang: {0, 0.5}
    # Activate when too many susceptible nodes exist and energy conservation is needed
    # ============================================================
    
    # Energy balance: check if we have excess susceptible nodes
    total_active = Su + Iu + Ru
    total_low = LSu + LIu + LRu
    
    # Sleep condition: high Su relative to needs, AND sufficient low-energy reserve
    # OR when energy conservation is prioritized (high beta_sleep)
    energy_ratio = total_low / (total_active + 1e-10)  # Avoid division by zero
    
    # Forced sleep when: Su is high AND (we have low-energy backup OR energy ratio is good)
    u6_condition = ((Su > theta_Su_high) & ((LSu > LSu_target * 0.5) | (energy_ratio > beta_sleep))) | \
                   ((Su > theta_Su_low * 3.0) & (Iu + LIu < 0.5) & (tau > 0.5))
    u6 = np.where(u6_condition, 0.5, 0.0)
    
    # ============================================================
    # ASSEMBLE RETURN ARRAY
    # Shape must be (M+1, 6) with columns [u1, u2, u3, u4, u5, u6]
    # ============================================================
    
    control = np.zeros((M_plus_1, 6))
    control[:, 0] = u1
    control[:, 1] = u2
    control[:, 2] = u3
    control[:, 3] = u4
    control[:, 4] = u5
    control[:, 5] = u6
    
    return control
