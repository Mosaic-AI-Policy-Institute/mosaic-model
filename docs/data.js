/**
 * MOSAIC Calculator - Data Configuration
 *
 * All exogenous data from config.py and CBS household income data.
 */

// =============================================================================
// EXOGENOUS DATA (Layer 1 - External Sources)
// =============================================================================

const DATA = {
    // -------------------------------------------------------------------------
    // Israeli Economy (Layer 1.1)
    // -------------------------------------------------------------------------
    Y0: 1694,                    // GDP in billion NIS (BOI 2024)
    labor_share: 0.56,           // Adva Center 2024
    labor_force: 4.5043,         // BOI 2024: 4,504.3K workers
    natural_unemployment: 3.0,   // BOI 2024
    total_households: 2.9794,    // CBS 2024: millions
    poverty_line: 3324,          // BTL2023Poverty: per std person
    gini_initial: 0.363,         // BTL2023Poverty
    poverty_rate_initial: 20.7,  // BTL2023Poverty: pre-NIT poverty rate (%)
    poverty_gap_initial: 39.5,   // BTL2023Poverty: poverty gap index (%)
    median_wage: 10586,          // BTL2025: median wage per employee (NIS/mo)

    // -------------------------------------------------------------------------
    // CES Production Parameters (Layer 1.2)
    // -------------------------------------------------------------------------
    K_0: 4708.1,                 // PWT2023: capital stock (B NIS)
    sigma: 1.5,                  // Elasticity of substitution

    // -------------------------------------------------------------------------
    // Labor Supply Elasticities (Saez 2001 calibration)
    // -------------------------------------------------------------------------
    epsilon_intensive: 0.51,     // Intensive margin elasticity (Chetty et al. 2011)
    epsilon_extensive: 0.28,     // Extensive margin elasticity (Chetty et al. 2011)
    income_tax_wedge: 10,        // % average tax wedge

    // -------------------------------------------------------------------------
    // Funding Mechanism (Layer 1.5)
    // -------------------------------------------------------------------------
    vat_baseline: 18,            // ITA2024VAT (%)
    vat_passthrough: 0.92,       // Event study estimate (§2.4a)
    consumption_share: 53.9,     // % of GDP (CEIC 1995-2023 avg)
    private_consumption: 1.0,    // Trillion NIS (baseline, scales with Y)

    // Deflation (derived from alpha via BLS anchor)
    bls_deflation_rate: 5,       // % (BLS historical software deflation 1997-2025)
    bls_deflation_alpha: 1.4,    // Alpha anchor for BLS rate (baseline scenario)
    ai_consumption_exposure: 40, // % sensitivity of deflation to excess TFP

    // -------------------------------------------------------------------------
    // Corporate Tax & Profit Trend (Layer 1.5)
    // -------------------------------------------------------------------------
    corporate_tax_rate: 23,      // MOF 2024 (%)
    profit_trend_baseline: 50.3, // B NIS (midpoint of linear & GDP-linked)
    ai_attribution_coef: 100,    // % of over-trend profit attributed to AI

    // Historical corporate tax revenue (2015-2019, for trend calculation)
    corp_tax_2015: 32.766,       // B NIS
    corp_tax_2016: 35.777,
    corp_tax_2017: 39.839,
    corp_tax_2018: 40.782,
    corp_tax_2019: 39.855,

    // -------------------------------------------------------------------------
    // Capital Gains Tax (Layer 1.5)
    // -------------------------------------------------------------------------
    capital_gains_rate: 25,      // MOF 2024 (%)

    // Historical capital gains tax revenue (2012-2019, for trend calculation)
    // Note: 2017 is outlier year (19.6B vs ~8B average)
    cg_tax_2012: 10.155,         // B NIS
    cg_tax_2013: 7.532,
    cg_tax_2014: 7.270,
    cg_tax_2015: 8.404,
    cg_tax_2016: 7.126,
    cg_tax_2017: 19.630,         // Outlier year
    cg_tax_2018: 7.150,
    cg_tax_2019: 7.669,

    // Historical GDP (for GDP-linked trends)
    gdp_2012: 1029,              // B NIS
    gdp_2013: 1078,
    gdp_2014: 1120,
    gdp_2015: 1179,
    gdp_2016: 1236,
    gdp_2017: 1290,
    gdp_2018: 1354,
    gdp_2019: 1428,

    // -------------------------------------------------------------------------
    // Public Sector (Ring-fenced Government)
    // -------------------------------------------------------------------------
    public_sector_wages: 203,    // B NIS (MOF2024Financial)
    public_sector_substitutable: 40.3, // % (MOSAIC2024AI projection)

    // -------------------------------------------------------------------------
    // Program Consolidation (BTL2023)
    // -------------------------------------------------------------------------
    btl_income_support: 1.643,   // B NIS
    btl_unemployment: 5.563,     // B NIS
    consolidation_admin_rate: 3, // % savings

    // -------------------------------------------------------------------------
    // NIT Design Reference Parameters
    // -------------------------------------------------------------------------
    floor_reference: 6350,       // Reference floor (NIS/mo) - CBS median
    single_parent_topup: 900,    // Psi: single-parent top-up (NIS/mo)
    child_equiv_weight: 0.30,    // OECD modified scale child weight
    nit_take_up_rate: 85,        // % take-up rate
    nit_gdp_buffer_pct: 3,       // % GDP buffer for revenue cap

    // -------------------------------------------------------------------------
    // UHI (Universal High Income) Funding - Third Channel
    // -------------------------------------------------------------------------
    uhi_floor_target: 15000,     // NIS/mo (~142% of median wage)
    kohelet_consolidation: 43,   // B NIS - program cancellation (6 programs)
    kohelet_tax_reform: 78,      // B NIS - income tax reform (25% floor)
    estate_tax_rate: 30,         // % estate tax rate
    estate_tax_revenue: 29,      // B NIS - estate tax (30%, high exemption)
    data_dividend: 4,            // B NIS - resource royalty
    pillar_two_revenue: 19,      // B NIS - global min tax (requires coordination)
    yozma_revenue: 3,            // B NIS - govt VC equity returns
};

// =============================================================================
// DEFAULT DESIGN PARAMETERS (Editable by user)
// =============================================================================

const DEFAULTS = {
    // NIT Design Parameters
    disregard: 1000,             // D: Income disregard (NIS/mo)
    taper: 0.50,                 // tau: Taper rate
    ringfence_rate: 75,          // rho: Ring-fence rate (%)

    // Scenario Parameters
    // Note: deflation is derived from alpha, not user-editable
    scenarios: {
        baseline: {
            alpha: 1.4,          // TFP multiplier
            s_L_prime: 0.50,     // Post-automation labor share
            u: 6,                // Unemployment rate (%)
            ringfence_rate: 75,  // % ring-fenced
        },
        strong: {
            alpha: 1.6,
            s_L_prime: 0.42,
            u: 10,
            ringfence_rate: 75,
        },
        agi: {
            alpha: 2.2,
            s_L_prime: 0.35,
            u: 40,
            ringfence_rate: 75,
        },
        asi: {
            alpha: 3.0,
            s_L_prime: 0.25,
            u: 60,
            ringfence_rate: 75,
        }
    },
};

// =============================================================================
// CBS 2022 HOUSEHOLD INCOME DATA
// =============================================================================
// Source: CBS average_monthly_income_by_percentiles_2022.xlsx
// Format: { decile, households, std_persons, equiv_income }
// equiv_income = GROSS income / std_persons (used for NIT benefit calculation)

const CBS_DECILES = [
    { decile: 1, households: 291600, std_persons: 3.30, equiv_income: 1877 },
    { decile: 2, households: 292300, std_persons: 3.00, equiv_income: 3263 },
    { decile: 3, households: 291100, std_persons: 2.60, equiv_income: 4358 },
    { decile: 4, households: 293000, std_persons: 2.70, equiv_income: 5291 },
    { decile: 5, households: 292000, std_persons: 2.60, equiv_income: 6622 },
    { decile: 6, households: 292200, std_persons: 2.60, equiv_income: 7812 },
    { decile: 7, households: 291800, std_persons: 2.60, equiv_income: 9278 },
    { decile: 8, households: 292100, std_persons: 2.50, equiv_income: 11244 },
    { decile: 9, households: 291900, std_persons: 2.30, equiv_income: 14586 },
    { decile: 10, households: 292700, std_persons: 2.30, equiv_income: 23974 },
];

// =============================================================================
// MICROSIMULATION PARAMETERS (Pre-computed from verify_layer3)
// =============================================================================
// These match the values computed in verify_layer3.ipynb Section 3.0

const MICROSIM_PARAMS = {
    // Distribution parameters (computed from CBS 2022 cross-decile variance)
    // Matches verify_layer3.ipynb Section 3.0
    beta_concentration: 2.16,    // For Beta-distributed incomes within deciles
    gamma_shape: 85.12,          // For Gamma-distributed household sizes

    // Decile bounds (GROSS income, converted from CBS NET thresholds)
    // Source: CBS 2022 Row 5 upper limits, converted via gross/net ratio per decile
    decile_bounds: [
        { lower: 0,     upper: 3229 },    // D1
        { lower: 3229,  upper: 4506 },    // D2
        { lower: 4506,  upper: 5647 },    // D3
        { lower: 5647,  upper: 6957 },    // D4
        { lower: 6957,  upper: 8327 },    // D5
        { lower: 8327,  upper: 10015 },   // D6
        { lower: 10015, upper: 11945 },   // D7
        { lower: 11945, upper: 14555 },   // D8
        { lower: 14555, upper: 19004 },   // D9
        { lower: 19004, upper: 33363 },   // D10 (extrapolated upper)
    ],

    n_samples: 1000,  // Samples per decile (matches verify_layer3)
};

// =============================================================================
// DATA SOURCE TOOLTIPS (for UI)
// =============================================================================

const DATA_SOURCES = {
    // Economy
    Y0: "GDP: Bank of Israel 2024",
    labor_share: "Labor share: Adva Center 2024",
    K_0: "Capital stock: Penn World Table 2023",
    sigma: "CES elasticity: Acemoglu & Restrepo (2018)",
    poverty_line: "Poverty line: BTL 2023 Poverty Report",
    gini_initial: "Gini coefficient: BTL 2023 Poverty Report",
    median_wage: "Median wage: BTL 2025 (H1)",

    // Elasticities
    epsilon_intensive: "Intensive elasticity: Chetty et al. (2011)",
    epsilon_extensive: "Extensive elasticity: Chetty et al. (2011)",

    // Funding mechanism
    consumption_share: "Private consumption: CEIC Data (1995-2023 avg)",
    vat_passthrough: "VAT pass-through: Event study (16 VAT reforms 1982-2025)",
    bls_deflation_rate: "Software deflation: BLS (1997-2025 avg ~5%/yr)",

    // Profits & CG
    profit_trend_baseline: "Profit trend: Knesset 2021 (2015-2019 linear + GDP-linked avg)",
    corp_tax_2015: "Corporate tax: Knesset 2021 State Budget data",
    capital_gains_rate: "Capital gains rate: MOF 2024",
    cg_tax_2012: "CG tax revenue: Israel Tax Authority (2012-2019)",

    // Public sector
    public_sector_wages: "Public wages: MOF 2024 Financial Report",
    public_sector_substitutable: "Substitutable share: MOSAIC 2024 AI labor research",

    // BTL
    btl_income_support: "Income support: BTL 2023 Annual Report",
    btl_unemployment: "Unemployment benefits: BTL 2023 Annual Report",

    // NIT design
    floor_reference: "Reference floor: CBS median-based calculation",
    nit_take_up_rate: "Take-up rate: Literature estimate (BTL programs)",
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { DATA, DEFAULTS, CBS_DECILES, DATA_SOURCES };
}
