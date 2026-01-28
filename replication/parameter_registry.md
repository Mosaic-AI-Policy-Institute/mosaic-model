# MOSAIC Parameter Registry

This document catalogs every parameter used in the MOSAIC paper with its source, type, and verification location.

## Parameter Types

| Type | Description | Example |
|------|-------------|---------|
| **Design** | Policy choices made by authors | Taper rate (50%) |
| **Source** | Direct external data citation | GDP (BOI 2024) |
| **Extrapolated** | Derived from data via regression | VAT pass-through (event study) |
| **Calculated** | Computed from other parameters | Post-AI GDP (CES model) |
| **Simulated** | Monte Carlo or microsimulation | NIT floor (Brent solver) |

---

## A. Design Parameters (Policy Choices)

These parameters are design choices by the paper authors, not empirical estimates.

| Parameter | Symbol | Value | Rationale | CALC Key | Notebook |
|-----------|--------|-------|-----------|----------|----------|
| Taper rate | τ | 0.50 | Below Saez optimal (0.44) for political feasibility | `taper_implemented` | Layer 1 |
| Disregard | D | 1,000 NIS/mo | Work incentive preservation | `disregard` | Layer 1 |
| Single-parent top-up | ψ | 900 NIS/mo | Additional needs recognition | `single_parent_topup` | Layer 1 |
| Child equiv weight | - | 0.30 | OECD standard | `child_equiv_weight` | Layer 1 |
| Ring-fence rate | ρ | 0.75 | Balance revenue capture vs. investment (applies to all 3 routes) | `ringfence_rate` | Layer 1 |
| AI attribution | κ | 1.0 | Full attribution (conservative sensitivity: 0.6-1.0) | `ai_attribution_coef` | Layer 1 |
| Take-up rate | - | 85% | Based on EITC experience | `nit_take_up_rate` | Layer 1 |
| GDP buffer | - | 3% | Fiscal safety margin | `nit_gdp_buffer_pct` | Layer 1 |
| Estate tax rate | τ_E | 30% | Policy choice for Channel 3 | `tau_estate` | Layer 4 |
| Estate exemption | - | 5M NIS | Threshold for estate tax | `estate_exemption_threshold` | Layer 4 |

### Scenario Assumptions

| Scenario | TFP (α) | Labor Share (s_L') | Unemployment (u) |
|----------|---------|-------------------|------------------|
| Low Displacement | 1.4 | 0.50 | 6% |
| Strong | 1.6 | 0.42 | 10% |
| AGI | 2.2 | 0.35 | 40% |
| ASI | 3.0 | 0.25 | 60% |

**CALC Keys:** `alpha_{scenario}`, `s_L_prime_{scenario}`, `u_{scenario}`

---

## B. Source Parameters (External Data - Direct)

These are directly cited values from external data sources.

### B.1 Israeli Economy Data

| Parameter | Symbol | Value | Source | CALC Key | Notebook |
|-----------|--------|-------|--------|----------|----------|
| Initial GDP | Y₀ | 1,694B NIS | BOI 2024 | `Y0` | Layer 1 |
| Capital stock | K₀ | 4,708B NIS | PWT 2023 | `K_0` | Layer 1 |
| Labor share | s_L | 0.56 | Adva 2024 | `labor_share` | Layer 1 |
| Labor force | L | 4.5M | BOI 2024 | `labor_force` | Layer 1 |
| Natural unemployment | u₀ | 3.0% | BOI 2024 | `natural_unemployment` | Layer 1 |
| Median wage | - | 10,586 NIS/mo | NII 2025 | `median_wage` | Layer 1 |
| Total households | - | 2.98M | CBS 2024 | `total_households` | Layer 1 |

### B.2 Poverty & Inequality Data

| Parameter | Symbol | Value | Source | CALC Key | Notebook |
|-----------|--------|-------|--------|----------|----------|
| Poverty line | z | 3,324 NIS/mo | BTL 2023 | `poverty_line` | Layer 1 |
| Poverty rate | - | 20.7% | BTL 2023 | `poverty_rate_initial` | Layer 1 |
| Poverty gap | - | 39.5% | BTL 2023 | `poverty_gap_initial` | Layer 1 |
| Gini coefficient | - | 0.363 | BTL 2023 | `gini_initial` | Layer 1 |

### B.3 Elasticity Parameters

| Parameter | Symbol | Value | Source | CALC Key | Notebook |
|-----------|--------|-------|--------|----------|----------|
| Intensive elasticity | ε_int | 0.51 | Taub 2024 | `epsilon_intensive` | Layer 1 |
| Extensive elasticity | ε_ext | 0.28 | Taub 2024 | `epsilon_extensive` | Layer 1 |
| Elasticity of substitution | σ | 1.5 | Chirinko 2008 | `sigma` | Layer 1 |

### B.4 Fiscal & Tax Data

| Parameter | Symbol | Value | Source | CALC Key | Notebook |
|-----------|--------|-------|--------|----------|----------|
| VAT rate | v₀ | 18% | ITA 2024 | `vat_baseline` | Layer 1 |
| Corporate tax rate | - | 23% | MOF 2024 | `corporate_tax_rate` | Layer 1 |
| Capital gains rate | - | 25% | MOF 2024 | `capital_gains_rate` | Layer 1 |
| Public sector wages | W | 203B NIS | MOF 2024 | `public_sector_wages` | Layer 1 |

### B.5 Social Insurance Data

| Parameter | Value | Source | CALC Key | Notebook |
|-----------|-------|--------|----------|----------|
| Income support | 1.6B NIS | BTL 2023 | `btl_income_support` | Layer 1 |
| Unemployment benefits | 5.6B NIS | BTL 2023 | `btl_unemployment` | Layer 1 |
| Disability benefits | 15.7B NIS | BTL 2023 | `btl_disability` | Layer 1 |

### B.6 Deflation Anchor

| Parameter | Symbol | Value | Source | CALC Key | Notebook |
|-----------|--------|-------|--------|----------|----------|
| BLS software deflation | δ_BLS | 5%/yr | BLS CPI (1997-2025) | `bls_deflation_rate` | Layer 1 |
| AI consumption exposure | θ | 40% | OECD 2025 (conservative) | `ai_consumption_exposure` | Layer 1 |
| Consumption share | c | 53.9% | CEIC Data | `consumption_share` | Layer 1 |

---

## C. Extrapolated Parameters (Regression-Based)

These parameters are derived from data via regression or extrapolation. Each section below shows the full derivation.

### C.1 VAT Pass-through (β = 92%)

**Notebook:** verify_layer2.ipynb §2.4a
**CALC Key:** `vat_passthrough`

**Method:** Event study of VAT reforms

**Raw Data:** 16 VAT rate changes 1982-2025 (from `israel_vat_history.csv`) + Monthly CPI (from `price_indices.csv`)

**Formula:**
```
Pass-through = ΔP_actual / ΔP_expected
where ΔP_expected = Δτ / (1 + τ)
```

**Results:**
| Period | Events | Mean PT | Notes |
|--------|--------|---------|-------|
| All (1982-2025) | 16 | Variable | Includes stabilization era |
| Post-1995 increases | 4 | 92% | Used for projection |

**Stored Value:** `vat_passthrough = 0.92`

---

### C.2 Corporate Tax Trend (T₀ = 50.3B)

**Notebook:** verify_layer2.ipynb §2.5c
**CALC Keys:** `profit_trend_linear`, `profit_trend_gdp_linked`, `profit_trend_baseline`

**Method:** Dual extrapolation to 2024

**Raw Data:** Corporate tax revenue 2015-2019 (from `corporate_tax_revenue.csv`)

| Year | Revenue (B NIS) | GDP (B NIS) | Tax/GDP |
|------|-----------------|-------------|---------|
| 2015 | 32.8 | 1,179 | 2.78% |
| 2016 | 35.8 | 1,236 | 2.90% |
| 2017 | 39.8 | 1,290 | 3.09% |
| 2018 | 40.8 | 1,354 | 3.01% |
| 2019 | 39.9 | 1,428 | 2.79% |

**Method 1: Linear Extrapolation**
```
Revenue = α + β × Year
OLS regression → Extrapolate to 2024
Result: 51.2B NIS
```

**Method 2: GDP-Linked**
```
Avg(Tax/GDP) × Y₀ = 2.91% × 1,694B = 49.3B NIS
```

**Final Value:** Midpoint = (51.2 + 49.3) / 2 = **50.3B NIS**

**Stored Values:**
- `profit_trend_linear = 51.2`
- `profit_trend_gdp_linked = 49.3`
- `profit_trend_baseline = 50.3`

---

### C.3 Capital Gains Tax Trend (8.6B)

**Notebook:** verify_layer2.ipynb §2.5e
**CALC Keys:** `cg_trend_linear`, `cg_trend_gdp_linked`, `cg_trend_baseline`

**Method:** Dual extrapolation to 2024 (excluding 2017 outlier)

**Raw Data:** CG tax revenue 2012-2019 (from `capital_gains_tax_history.csv`)

| Year | Revenue (B NIS) | Notes |
|------|-----------------|-------|
| 2012 | 10.2 | |
| 2013 | 7.5 | |
| 2014 | 7.3 | |
| 2015 | 8.4 | |
| 2016 | 7.1 | |
| 2017 | 19.6 | **OUTLIER** (dividend spike 15.5B) |
| 2018 | 7.2 | |
| 2019 | 7.7 | |

**Outlier Note:** 2017 had exceptional dividend distributions (15.5B vs normal ~4B) due to corporate tax reform anticipation. Excluded from trend.

**Method 1: Linear Extrapolation (excl 2017)**
```
Revenue = α + β × Year
OLS regression → Extrapolate to 2024
Result: 5.8B NIS
```

**Method 2: GDP-Linked (excl 2017)**
```
Avg(CGTax/GDP) × Y₀ = 0.67% × 1,694B = 11.3B NIS
```

**Final Value:** Midpoint = (5.8 + 11.3) / 2 = **8.6B NIS**

**Stored Values:**
- `cg_trend_linear = 5.8`
- `cg_trend_gdp_linked = 11.3`
- `cg_trend_baseline = 8.6`

---

## D. Calculated Parameters (Layer 2)

These are computed from source/design parameters via explicit formulas.

### D.1 CES Production Model

| Parameter | Formula | CALC Key | Notebook |
|-----------|---------|----------|----------|
| CES β | Calibrated to match s_K = 0.44 | `ces_beta` | Layer 2 §2.1 |
| Post-AI GDP (Y₁) | CES with α scaling | `Y1_{scenario}` | Layer 2 §2.1 |
| Growth rate (g_Y) | (Y₁ - Y₀) / Y₀ | `g_Y_{scenario}` | Layer 2 §2.1 |

**CES Results by Scenario:**

| Scenario | Y₁ (B NIS) | g_Y |
|----------|------------|-----|
| Low Displacement | 2,330 | 38% |
| Strong | 2,598 | 53% |
| AGI | 2,877 | 70% |

### D.2 Surplus Decomposition

| Parameter | Formula | CALC Key | Notebook |
|-----------|---------|----------|----------|
| Capital windfall | ΔΠ = s_K' × Y₁ - s_K × Y₀ | `capital_windfall_{scenario}` | Layer 2 §2.2 |
| Labor change | ΔS = s_L' × Y₁ - s_L × Y₀ | `labor_change_{scenario}` | Layer 2 §2.2 |

### D.3 Break-even & Taper

| Parameter | Formula | CALC Key | Notebook |
|-----------|---------|----------|----------|
| Break-even income | y* = M / τ | `breakeven_income` | Layer 2 §2.3 |
| Saez optimal taper | τ* = 1 / (1 + ε) | `taper_optimal` | Layer 2 §2.3 |

### D.4 Deflation Formula

| Parameter | Formula | CALC Key | Notebook |
|-----------|---------|----------|----------|
| Deflation (δ) | δ_BLS + θ × (α - α_BLS) | `deflation_{scenario}` | Layer 2 §2.4 |
| VAT adjusted | v₁ = (1 + v₀) / (1 - δ) - 1 | `vat_adjusted_{scenario}` | Layer 2 §2.4 |

**Deflation Results:**

| Scenario | α | δ (%) | v₁ (%) |
|----------|---|-------|--------|
| Low Displacement | 1.4 | 5.0 | 24.2 |
| Strong | 1.6 | 5.8 | 25.3 |
| AGI | 2.2 | 8.2 | 28.6 |

### D.5 Revenue Components

| Component | Formula | CALC Keys | Notebook |
|-----------|---------|-----------|----------|
| VAT revenue | Y₁ × c × Δv × β | `vat_revenue_{scenario}` | Layer 2 §2.5 |
| Ring-fenced corporate | (Π - T₀) × κ × ρ | `ringfenced_{scenario}` | Layer 2 §2.5c |
| Ring-fenced capital gains | (CG - CG₀) × κ × ρ | `ringfenced_cg_{scenario}` | Layer 2 §2.5e |
| Ring-fenced government | W × u × ρ | `ringfenced_gov_{scenario}` | Layer 2 §2.6 |
| Consolidation | BTL programs | `consolidation_revenue` | Layer 2 §2.7 |
| **Total Revenue** | Sum of above | `revenue_{scenario}` | Layer 2 §2.8 |

**Revenue Summary (B NIS):**

| Component | Low Displacement | Strong | AGI |
|-----------|------------------|--------|-----|
| VAT | 64 | 84 | 126 |
| Ring-fenced corporate | 6 | 13 | 22 |
| Ring-fenced capital gains | 3 | 6 | 13 |
| Ring-fenced government | 9 | 15 | 61 |
| Consolidation | 7 | 7 | 7 |
| **Total** | 89 | 125 | 229 |

---

## E. Simulated Parameters (Layer 3)

These are computed via microsimulation or numerical optimization.

### E.1 Revenue-Constrained Floors

**Method:** Brent's algorithm to find M where Cost(M) = Revenue

| Scenario | Revenue (B NIS) | Floor M (NIS/mo) | Break-even (NIS/mo) |
|----------|-----------------|------------------|---------------------|
| Low Displacement | 89 | ~5,900 | ~11,800 |
| Strong | 125 | ~7,500 | ~15,000 |
| AGI | 266* | ~9,250 | ~18,500 |

*AGI uses 85% take-up + 3% buffer

**CALC Keys:** `floor_{scenario}`, `breakeven_{scenario}`
**Notebook:** Layer 3 §3.0

### E.2 Poverty & Inequality Outcomes

**Method:** Microsimulation using CBS 2022 decile data with Beta/Gamma distributions

| Outcome | Pre-NIT | Post-NIT (AGI) | CALC Key |
|---------|---------|----------------|----------|
| Poverty rate | 20.7% | ~1.4% | `poverty_{scenario}` |
| Gini coefficient | 0.363 | ~0.316 | `gini_{scenario}` |

**Notebook:** Layer 3 §3.1

### E.3 Decile-Level Analysis (Table 11)

**Method:** Microsimulation with 100,000 synthetic households per decile

| Decile | Mean Gross | NIT Benefit | Cost at 85% |
|--------|------------|-------------|-------------|
| D1 | 6,193 | 5,111 | 61.5B |
| D2 | 9,790 | 4,177 | 50.2B |
| ... | ... | ... | ... |
| D8 | 55,141 | 792 | 9.5B |
| **Total** | - | - | **266B** |

**CALC Keys:** `agi_decile_{d}_cost_85pct`, `agi_decile_{d}_benefit_per_hh`
**Notebook:** Layer 3 §3.1a

---

## F. Channel 3 Parameters (Layer 4)

### F.1 Estate Tax

| Parameter | Value | Source/Method | CALC Key |
|-----------|-------|---------------|----------|
| Total wealth | 6,500B NIS | Berl 2020 | `wealth_total_2020` |
| Top decile share | 50% | Berl 2020 | `wealth_top_decile_share` |
| Annual flow rate | 3% | Piketty 2010 | `mu_inheritance` |
| Estate tax revenue | ~40B NIS | Calculated | `estate_tax_revenue` |

**Notebook:** Layer 4 §4.2

### F.2 Other Channel 3 Sources

| Source | Revenue (B NIS) | Method | CALC Key |
|--------|-----------------|--------|----------|
| Data dividend | 3.6 | HH × GB × rate | `data_dividend` |
| Pillar Two expansion | 6-10 | MNE tax recovery | `pillar_two_revenue` |
| Yozma 2.0 dividends | 3-5 | VC fund returns | `vc_dividend` |

---

## Data Files Reference

| File | Content | Used In |
|------|---------|---------|
| `cbs_household_income_2022.csv` | CBS decile income data | Layer 3 microsim |
| `exchange_rates.csv` | USD/ILS rates 2000-2026 | Currency conversion |
| `price_indices.csv` | CPI 1980-2025 | VAT pass-through |
| `israel_vat_history.csv` | VAT rate changes | VAT pass-through |
| `corporate_tax_revenue.csv` | Corp tax 2015-2019 | Profit trend |
| `capital_gains_tax_history.csv` | CG tax 2012-2019 | CG trend |
| `israel_gdp_history_dollar.csv` | GDP history | Trend extrapolation |

---

## Verification Cross-Reference

To verify any paper claim:

1. Find the parameter in this registry
2. Check the CALC key in `calc_outputs.json`
3. Run the specified notebook section to reproduce

**Example:** Paper claims "VAT pass-through is 92%"
- Registry: Section C.1, CALC key `vat_passthrough`
- Notebook: verify_layer2.ipynb §2.4a
- Verification: Run cell, check stored value matches

---

*Last updated: January 2026*
