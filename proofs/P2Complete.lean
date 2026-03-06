import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic

set_option linter.unusedVariables false
-- Individual heartbeat overrides applied per-theorem if needed

/-!
# Formal Verification of Core Mathematical Claims

All 4 core mathematical claims from the paper, verified in Lean 4.
Paper: "Compounding Vulnerability: Hub Removal Triggers Cascade Phase
        Transitions While Degrading Percolation Robustness in Scale-Free
        Networks" (Cachero, 2026)
arXiv: 2603.04838
-/

/-!
## Claim 1: Cascade Stable-Node Condition (§2.5, Appendix A.1)

A node with degree k > 1/φ requires ≥ 2 active neighbors to activate.
Formally: k > 1/φ ⟹ k·φ > 1.
This operationalizes the Watts (2002) stable-node mechanism.
-/
theorem cascade_stable_node (k : ℝ) (φ : ℝ)
    (hφ_pos : 0 < φ) (hk_pos : 0 < k)
    (h_fw : k > 1 / φ) :
    k * φ > 1 := by
  have h1 : k * φ > (1 / φ) * φ :=
    mul_lt_mul_of_pos_right h_fw hφ_pos
  rw [div_mul_cancel₀ 1 (ne_of_gt hφ_pos)] at h1
  linarith

/-!
## Claim 2: Molloy-Reed Monotonicity (§3.1, §5.1)

p_c = 1/(κ - 1) is strictly decreasing in κ for κ > 1.
Hub removal reduces κ ⟹ p_c increases ⟹ percolation robustness decreases.
-/
theorem molloy_reed_monotone (κ₁ κ₂ : ℝ) (h₁ : κ₁ > 1) (h₂ : κ₂ > 1)
    (h_decrease : κ₁ > κ₂) :
    1 / (κ₁ - 1) < 1 / (κ₂ - 1) := by
  apply div_lt_div_of_pos_left one_pos (by linarith) (by linarith)

/-!
## Claim 3: Necessary Condition φ* ≥ √α/m (§5.2, Appendix A.2)

For BA networks: minimum hub degree k_min ≈ m/√α.
If k_min > 1/φ (hubs are stable nodes), then φ > √α/m.
This gives a network-size-independent lower bound on the transition.
(Derived as a scaling argument; see Proposition 2 in the paper.)
-/
theorem necessary_condition (m α φ : ℝ)
    (hm_pos : 0 < m) (hα_pos : 0 < α) (hφ_pos : 0 < φ)
    (h_sqrt_pos : 0 < Real.sqrt α)
    (h_firewall : m / Real.sqrt α > 1 / φ) :
    φ > Real.sqrt α / m := by
  rw [gt_iff_lt] at h_firewall ⊢
  field_simp at h_firewall ⊢
  nlinarith

/-!
## Claim 4a: Hub Dual Role (§6.1)

A hub with k ≥ 2 and 0 < φ ≤ 1 always satisfies k² ≥ k·φ.
The hub's contribution to ⟨k²⟩ always exceeds its cascade activation cost.
-/
theorem hub_dual_role (k : ℕ) (φ : ℝ) (hk : k ≥ 2) (hφ_pos : 0 < φ) (hφ_le : φ ≤ 1) :
    (k : ℝ) ^ 2 ≥ (k : ℝ) * φ := by
  have hk_real : (k : ℝ) ≥ 2 := by exact_mod_cast hk
  nlinarith

/-!
## Claim 4b: Compounding Vulnerability (§6, core thesis)

When k > 2⟨k⟩ AND k > 1/φ:
  (i)  k²/⟨k⟩ > 4⟨k⟩  (disproportionate percolation contribution)
  (ii) k·φ > 1           (cascade stable-node property)

Removing such a node destroys BOTH protective functions simultaneously.
-/
theorem compounding_vulnerability (k : ℕ) (φ mean_k : ℝ)
    (hk : k ≥ 2) (hφ_pos : 0 < φ) (hφ_le : φ ≤ 1)
    (h_mean_pos : 0 < mean_k)
    (h_firewall : (k : ℝ) > 1 / φ)
    (h_hub : (k : ℝ) > 2 * mean_k) :
    (k : ℝ) ^ 2 / mean_k > 4 * mean_k
    ∧ (k : ℝ) * φ > 1 := by
  have hk_real : (k : ℝ) ≥ 2 := by exact_mod_cast hk
  constructor
  · -- k²/⟨k⟩ > 4⟨k⟩ when k > 2⟨k⟩
    rw [gt_iff_lt]
    field_simp
    nlinarith
  · -- k*φ > 1 (stable-node condition)
    have h1 : (k : ℝ) * φ > (1 / φ) * φ :=
      mul_lt_mul_of_pos_right h_firewall hφ_pos
    rw [div_mul_cancel₀ 1 (ne_of_gt hφ_pos)] at h1
    linarith

/-!
## Verification Summary

All 4 claims are formally verified without `sorry`.
No axioms beyond standard Mathlib foundations.

| Claim | Paper Section | Statement | Status |
|-------|--------------|-----------|--------|
| 1 | §2.5, A.1 | k > 1/φ ⟹ k·φ > 1 (stable node) | ✅ |
| 2 | §3.1, §5.1 | κ₁ > κ₂ > 1 ⟹ 1/(κ₁-1) < 1/(κ₂-1) | ✅ |
| 3 | §5.2, A.2 | m/√α > 1/φ ⟹ φ > √α/m | ✅ |
| 4a | §6.1 | k ≥ 2, φ ∈ (0,1] ⟹ k² ≥ k·φ | ✅ |
| 4b | §6 | k > 2⟨k⟩, k > 1/φ ⟹ k²/⟨k⟩ > 4⟨k⟩ ∧ k·φ > 1 | ✅ |
-/
