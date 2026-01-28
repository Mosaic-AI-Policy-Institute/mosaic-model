# Replication Files

This directory contains all code and data needed to replicate the numerical results in "When AI Takes Our Jobs, It Should Also Pay Our Wages."

## Quick Start

```bash
# Run all notebooks in sequence
python run_all.py

# Or run individual notebooks
jupyter notebook layer1_inputs.ipynb
```

**Important:** Notebooks must be run in order: Layer 1 → Layer 2 → Layer 3 → Layer 4

## File Structure

```
replication/
├── config.py               # Design parameters and source data
├── functions.py            # All calculation functions
├── layer1_inputs.ipynb     # Layer 1: Exogenous inputs
├── layer2_calculations.ipynb     # Layer 2: Calculated values (CES, revenue)
├── layer3_simulation.ipynb     # Layer 3: Simulated values (floors, poverty)
├── layer4_channel3.ipynb     # Layer 4: Channel 3 analysis (UHI funding)
├── run_all.py           # Runner script for all layers
├── calc_outputs.json       # Computed results (auto-generated)
├── parameter_registry.md   # Complete parameter documentation
├── README.md               # This file
└── data/                   # Source data files (CSV only)
    ├── cbs_household_income_2022.csv
    ├── corporate_tax_revenue.csv
    ├── capital_gains_tax_history.csv
    ├── capital_gains_tax_profits.csv
    ├── price_indices.csv
    ├── israel_vat_history.csv
    ├── israel_gdp_history_dollar.csv
    └── exchange_rates.csv
```

## Architecture

```
DATA (config.py)
    ↓
layer1_inputs.ipynb → CALC["layer1"]
    ↓
layer2_calculations.ipynb → CALC["layer2"]
    ↓
layer3_simulation.ipynb → CALC["layer3"]
    ↓
layer4_channel3.ipynb → CALC["layer4"]
    ↓
calc_outputs.json
```

**Key Functions:**
- `store(key, value, layer)` - Save computed value to CALC
- `get(key)` - Retrieve value (checks CALC first, then DATA)

## Layer Descriptions

### Layer 1: Exogenous Inputs
Stores external data from official sources (BOI, CBS, BTL, MOF). No calculations.
- Israeli economy data (GDP, labor share, wages)
- Poverty data (poverty line, Gini)
- Scenario parameters (α, s_L', u)
- Design parameters (τ, D, ρ)

### Layer 2: Calculated Values
Derives values from Layer 1 using explicit formulas.

**CES Production:**
- `calibrate_ces_period_0()` - Calibrate β from initial labor share
- `ces_output()` - Y = A[βK^ρ + (1-β)L^ρ]^(1/ρ)
- `calc_growth_ces()` - Calculate g_Y for each scenario

**Surplus Decomposition:**
- `calc_automation_surplus()` - ΔS = s_L'·Y₁ - s_L·Y₀
- `calc_capital_windfall()` - ΔΠ = (1-s_L')·Y₁ - (1-s_L)·Y₀

**Revenue Components:**
- `calc_vat_revenue()` - Dynamic VAT with deflation adjustment
- `calc_ringfenced_profit()` - Ring-fenced corporate tax
- `calc_ringfenced_cg()` - Ring-fenced capital gains
- `calc_gov_ringfence()` - Ring-fenced government wage savings
- `calc_consolidation_revenue()` - Program consolidation savings

**Extrapolations:**
- `calc_profit_trend_gdp_linked()` - GDP-linked profit trend
- `calc_cg_trend_gdp_linked()` - GDP-linked CG trend

### Layer 3: Simulated Values
Uses microsimulation and numerical solvers.

**NIT Cost Functions:**
- `calc_nit_cost_microsim()` - Microsim-based cost estimation
- `calc_floor_balanced_microsim()` - Solve for floor given revenue
- `calc_floor_with_takeup_and_cap()` - Floor with take-up and caps

**Poverty & Inequality:**
- `calc_gini_from_deciles()` - Gini coefficient calculation
- `calc_poverty_gap_fgt1()` - FGT(1) poverty gap index
- `calc_emtr_by_decile()` - Effective marginal tax rates

**Distribution Generation:**
- `generate_decile_households()` - Synthetic household generation
- `generate_beta_incomes()` - Beta-distributed incomes within bounds
- `get_decile_bounds_gross()` - Gross income bounds from net

### Layer 4: Channel 3 Analysis
UHI funding mechanisms beyond basic NIT.
- Estate tax revenue calculations
- Capital windfall capture rates
- UHI target cost at 85% take-up

## Data Sources

| File | Content | Source |
|------|---------|--------|
| `cbs_household_income_2022.csv` | Decile income data | CBS 2022 |
| `corporate_tax_revenue.csv` | Corp tax 2015-2019 | Knesset 2021 |
| `capital_gains_tax_history.csv` | CG tax 2012-2019 | Knesset 2024 |
| `capital_gains_tax_profits.csv` | CG profits data | Knesset 2024 |
| `price_indices.csv` | CPI 1980-2025 | CBS |
| `israel_vat_history.csv` | VAT rate changes | ITA |
| `israel_gdp_history_dollar.csv` | GDP history USD | World Bank |
| `exchange_rates.csv` | USD/ILS rates | BOI |

## Parameter Categories

See `parameter_registry.md` for complete documentation. Summary:

| Category | Description | Example |
|----------|-------------|---------|
| **Design** | Policy choices | τ = 0.50, ρ = 0.75 |
| **Source** | External data | Y₀ = 1,694B (BOI 2024) |
| **Extrapolated** | Regression-derived | β = 92% (VAT pass-through) |
| **Calculated** | Formula-based | Y₁ = CES(α, K, L) |
| **Simulated** | Numerical methods | M* = solve C(M) = R |

## Verification

To verify any paper claim:

1. Find the parameter in `parameter_registry.md`
2. Note the CALC key and notebook section
3. Run the notebook and check `calc_outputs.json`

Example: Verify "UHI cost at 85% take-up is 885B NIS"
- Registry: Section D, key `uhi_target_cost`
- Notebook: layer4_channel3.ipynb
- Check: `calc_outputs.json["layer4"]["uhi_target_cost"] ≈ 884.8`

## Requirements

- Python 3.10+
- pandas, numpy, scipy
- jupyter, nbformat, nbclient

```bash
pip install pandas numpy scipy jupyter nbformat nbclient
```

## Troubleshooting

**"KeyError: 'xyz'"**
- Run notebooks in order (Layer 1 first)
- Check `calc_outputs.json` for the expected key

**"Must run layer{n} first"**
- Notebooks enforce dependency order
- Run layer1_inputs.ipynb before layer2, etc.

**Stale results**
- Delete `calc_outputs.json` and rerun all notebooks
- `init_calc()` in Layer 1 clears previous values

---

*Last updated: January 2026*
