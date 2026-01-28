"""MOSAIC verification data configuration.

This module contains the DATA dictionary used for numerical verification
of the MOSAIC paper claims.

Architecture:
    DATA → verify_layer1 → CALC (calc_outputs.json) → verify_layer2/3/4

DATA: External data sources and design parameters
    - Design choices (NIT parameters, policy assumptions)
    - Source data (verified from BTL, CBS, BOI, etc.)
    - Scenario definitions (TFP multipliers, labor shares)

CALC: Computed outputs (stored in calc_outputs.json)
    - Layer 1: Exogenous inputs stored
    - Layer 2: Calculated values (CES, revenue components)
    - Layer 3: Simulated values (floors, poverty outcomes)
    - Layer 4: Channel 3 analysis

Note: PAPER dict was removed - all verification now uses CALC values
      compared against paper claims in notebooks.
"""

from typing import Optional


DATA: dict[str, Optional[float]] = {
    # =========================================================================
    # SECTION 1: DESIGN PARAMETERS (Policy Choices)
    # These are author design decisions, not empirical estimates.
    # =========================================================================

    # NIT Mechanism Parameters
    "floor_reference": 6350,            # NIS/mo (60% of median) - initial design goal
    "disregard": 1000,                  # NIS/mo (D in paper)
    "single_parent_topup": 900,         # NIS/mo (ψ in paper)
    "taper_implemented": 0.50,          # τ = 0.50 (below Saez optimal for feasibility)
    "child_equiv_weight": 0.30,         # OECD standard equivalence scale

    # Fiscal Design
    "nit_take_up_rate": 85,             # % of eligible households that claim
    "nit_gdp_buffer_pct": 3,            # Buffer % of GDP for 100% take-up

    # Funding Mechanism Parameters
    "ai_attribution_coef": 100,         # κ=1 (100% AI attribution)
    "ringfence_rate": 75,               # ρ = 75% earmarked for NIT

    # =========================================================================
    # SECTION 2: SCENARIO DEFINITIONS
    # TFP multipliers (α), post-AI labor shares (s_L'), unemployment rates (u)
    # =========================================================================

    # TFP Multipliers (α)
    "alpha_low_displacement": 1.4,
    "alpha_strong": 1.6,
    "alpha_agi": 2.2,
    "alpha_asi": 3.0,

    # Post-Automation Labor Shares (s_L')
    "s_L_prime_low_displacement": 0.50,
    "s_L_prime_strong": 0.42,
    "s_L_prime_agi": 0.35,
    "s_L_prime_asi": 0.25,

    # Unemployment Rates (%)
    "u_low_displacement": 6,
    "u_strong": 10,
    "u_agi": 40,
    "u_asi": 60,

    # =========================================================================
    # SECTION 3: SOURCE DATA - Israeli Economy
    # Verified external data with citations
    # =========================================================================

    # GDP and Capital (BOI 2024, PWT 2023)
    "Y0": 1694,                         # BOI 2024: GDP in B NIS
    "K_0": 4708.1,                      # PWT 2023: capital stock in B NIS
    "labor_share": 0.56,                # Adva Center 2024
    "labor_force": 4.5043,              # BOI 2024: 4,504.3K workers
    "natural_unemployment": 3.0,        # BOI 2024: baseline unemployment

    # Household Data (CBS 2024)
    "total_households": 2.9794,         # CBS 2024: 2,979,400 households

    # Wage Data (NII 2025)
    "median_wage": 10586,               # NII 2025: H1 2025 median wage
    "P80_income": 21118,                # BTL2025: 80th percentile income

    # Poverty Data (BTL 2023)
    "poverty_line": 3324,               # BTL2023: poverty line per std person
    "poverty_rate_initial": 20.7,       # BTL 2023 poverty report
    "poverty_gap_initial": 39.5,        # BTL2023: poverty gap index (%)
    "gini_initial": 0.363,              # BTL2023: pre-NIT Gini coefficient

    # =========================================================================
    # SECTION 4: SOURCE DATA - Elasticities
    # =========================================================================

    # Labor Supply Elasticities (Taub 2024)
    "epsilon_intensive": 0.51,          # Men 0.45, Women 0.56 -> midpoint
    "epsilon_extensive": 0.28,          # Men 0.25, Women 0.31 -> midpoint
    "income_tax_wedge": 10,             # ~10% at low earnings

    # CES Parameters
    "sigma": 1.5,                       # Chirinko 2008: elasticity of substitution

    # =========================================================================
    # SECTION 5: SOURCE DATA - Fiscal & Tax
    # =========================================================================

    # VAT (ITA 2024)
    "vat_baseline": 18,                 # Current VAT rate %

    # Tax Rates (MOF 2024)
    "corporate_tax_rate": 23,           # Corporate tax rate %
    "capital_gains_rate": 25,           # Capital gains tax rate %

    # Public Sector (MOF 2024)
    "public_sector_wages": 203,         # B NIS (MOF2024Financial)
    "public_sector_substitutable": 40.3,  # % (for reference only)

    # Deflation Anchor (BLS CPI)
    "bls_deflation_rate": 5,            # % (BLS historical software deflation)
    "bls_deflation_alpha": 1.4,         # Alpha anchor for BLS rate
    "ai_consumption_exposure": 40,      # % sensitivity to excess TFP (OECD 2025)
    "consumption_share": 53.9,          # % of GDP (CEIC Data: Israel avg)
    "private_consumption": 1.0,         # Trillion NIS (baseline)

    # =========================================================================
    # SECTION 6: SOURCE DATA - Corporate Tax Revenue (Knesset 2021)
    # Used for profit trend extrapolation
    # =========================================================================

    "corp_tax_2015": 32.766,            # B NIS
    "corp_tax_2016": 35.777,
    "corp_tax_2017": 39.839,
    "corp_tax_2018": 40.782,
    "corp_tax_2019": 39.855,

    # Historical GDP 2015-2019 (for GDP-linked calculation)
    "gdp_2015": 1179,                   # B NIS
    "gdp_2016": 1236,
    "gdp_2017": 1290,
    "gdp_2018": 1354,
    "gdp_2019": 1428,

    # =========================================================================
    # SECTION 7: SOURCE DATA - Capital Gains Tax Revenue (Knesset 2024)
    # Used for CG trend extrapolation (2017 is outlier - dividend spike)
    # =========================================================================

    "cg_tax_2012": 10.155,              # B NIS
    "cg_tax_2013": 7.532,
    "cg_tax_2014": 7.270,
    "cg_tax_2015": 8.404,
    "cg_tax_2016": 7.126,
    "cg_tax_2017": 19.630,              # OUTLIER (dividend spike 15.5B)
    "cg_tax_2018": 7.150,
    "cg_tax_2019": 7.669,

    # =========================================================================
    # SECTION 8: SOURCE DATA - Program Consolidation (BTL 2023)
    # =========================================================================

    "btl_income_support": 1.643,        # B NIS - הבטחת הכנסה
    "btl_unemployment": 5.563,          # B NIS - דמי אבטלה
    "btl_disability": 15.7,             # B NIS - נכות כללית
    "btl_disability_recipients": 311011,
    "btl_disability_avg_monthly": 4214,
    "consolidation_admin_rate": 3,      # % admin savings

    # =========================================================================
    # SECTION 9: SOURCE DATA - Wealth & Estate Tax (Berl 2020, Piketty 2010)
    # Used for Channel 3 analysis
    # =========================================================================

    "wealth_total_2020": 6500,          # B NIS (6.5T total household wealth)
    "wealth_top_decile_share": 50,      # % of wealth held by top decile
    "wealth_top_percentile_share": 25,  # % of wealth held by top percentile
    "wealth_top_decile_avg": 14.1,      # M NIS average per HH in top decile
    "wealth_top_percentile_min": 55,    # M NIS minimum per HH in top percentile
    "mu_inheritance": 3,                # % annual inheritance flow rate
    "tau_estate": 30,                   # % estate tax rate (policy choice)
    "estate_exemption_threshold": 5,    # M NIS exemption threshold

    # =========================================================================
    # SECTION 10: SOURCE DATA - Data Dividend & Pillar Two
    # Channel 3 revenue sources
    # =========================================================================

    # Natural Gas Benchmark
    "gas_production_2023": 25,          # BCM
    "wealth_fund_transfer_2024": 1,     # B NIS

    # Data Dividend
    "data_per_hh_2030": 1000,           # GB/month (1 TB/month forecast)
    "data_rate_nis_per_gb": 0.10,       # NIS/GB
    "data_dividend": 3.6,               # B NIS/year (calculated)

    # Pillar Two Expansion
    "pillar_two_tax_loss_pct_gdp": 0.3,  # % of GDP lost to MNE tax incentives
    "pillar_two_current_loss": 6,        # B NIS/year (current, at Y0)
    "pillar_two_nvidia_example": 10,     # B NIS/year (single large MNE)

    # =========================================================================
    # SECTION 11: SOURCE DATA - Government VC Fund (OECD)
    # Yozma 2.0 analysis
    # =========================================================================

    "vc_gross_irr": 17.5,               # % gross IRR
    "vc_net_irr": 13,                   # % net IRR (after fees)
    "vc_initial_investment": 50,        # B NIS initial government investment
    "vc_investment_horizon": 5,         # years
    "vc_dividend_rate": 5,              # % dividend on accumulated profits

    # =========================================================================
    # SECTION 12: CHANNEL 3 TARGETS
    # =========================================================================

    "uhi_floor_target": 15000,          # NIS/mo (~142% of median)

    "kohelet_consolidation": 43,        # B NIS (Kohelet proposal)
    "kohelet_tax_reform": 78,           # B NIS (Kohelet proposal)
}


# =============================================================================
# BTL2025 Wage Deciles
# Source: BTL2025 - NII H1 2025 wage data
# Format: (decile, workers, mean, median, max_wage)
# =============================================================================

BTL2025_DECILES = [
    (1, 366516, 1287, 1283, 2442),
    (2, 366516, 3565, 3558, 4702),
    (3, 366636, 5664, 5681, 6552),
    (4, 366402, 7525, 7516, 8513),
    (5, 366530, 9538, 9531, 10586),
    (6, 366605, 11758, 11741, 13017),
    (7, 366422, 14539, 14507, 16200),
    (8, 366518, 18460, 18365, 21118),
    (9, 366519, 25408, 25063, 31121),
    (10, 366517, 53236, 42273, None),  # No max for top decile
]
