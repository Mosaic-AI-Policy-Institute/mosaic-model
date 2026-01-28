/**
 * MOSAIC Calculator - Formulas
 *
 * All calculation functions ported from functions.py
 */

// =============================================================================
// CES PRODUCTION FUNCTION
// =============================================================================

/**
 * CES production function output.
 * Y = A * [beta * K^rho + (1-beta) * L^rho]^(1/rho)
 */
function cesOutput(A, beta, K, L, sigma) {
    if (sigma === 1) {
        return A * Math.pow(K, beta) * Math.pow(L, 1 - beta);
    }
    const rho = 1 - 1 / sigma;
    const inner = beta * Math.pow(K, rho) + (1 - beta) * Math.pow(L, rho);
    return A * Math.pow(inner, 1 / rho);
}

/**
 * Labor share under CES production.
 * sL = (1-beta) * L^rho / [beta * K^rho + (1-beta) * L^rho]
 */
function cesLaborShare(beta, K, L, sigma) {
    if (sigma === 1) {
        return 1 - beta;
    }
    const rho = 1 - 1 / sigma;
    const K_term = beta * Math.pow(K, rho);
    const L_term = (1 - beta) * Math.pow(L, rho);
    return L_term / (K_term + L_term);
}

/**
 * Calibrate CES parameters (A, beta) to match initial conditions.
 */
function calibrateCES(Y_0, L_0, s_L, sigma, K_0) {
    const s_K = 1 - s_L;
    const rho = 1 - 1 / sigma;

    const L_rho = Math.pow(L_0, rho);
    const K_rho = Math.pow(K_0, rho);

    const beta = (s_K * L_rho) / (s_L * K_rho + s_K * L_rho);
    const inner = beta * K_rho + (1 - beta) * L_rho;
    const A = Y_0 / Math.pow(inner, 1 / rho);

    return { A, beta, rho };
}

/**
 * Calculate Y1, g_Y, and Delta S using CES production function.
 */
function calcGrowthCES(params) {
    const { alpha, s_L_prime, u, Y_0, s_L, L_force, u_0, sigma, K_0 } = params;

    // Period 0 calibration
    const L_0 = L_force * (1 - u_0);
    const calib = calibrateCES(Y_0, L_0, s_L, sigma, K_0);

    // Period 1
    const A_1 = alpha * calib.A;
    const L_1 = L_force * (1 - u);
    const K_1 = K_0;

    const Y_1 = cesOutput(A_1, calib.beta, K_1, L_1, sigma);
    const g_Y = (Y_1 - Y_0) / Y_0;

    // Automation surplus
    const delta_S = s_L_prime * Y_1 - s_L * Y_0;

    return {
        Y_1,
        g_Y,
        g_Y_pct: g_Y * 100,
        delta_S,
        A_0: calib.A,
        beta: calib.beta,
        L_0,
        L_1,
    };
}

// =============================================================================
// VAT AND REVENUE CALCULATIONS
// =============================================================================

/**
 * Price-neutral VAT rate after AI deflation.
 * v1 = (1+v0)/(1-delta) - 1
 */
function calcVatAdjustment(v0, delta) {
    return (1 + v0) / (1 - delta) - 1;
}

/**
 * Derive deflation rate from TFP multiplier (alpha).
 * BLS software deflation (5%) is the baseline floor. AI-driven deflation
 * adds on top, scaled by ai_exposure (sensitivity to excess TFP).
 *
 * Formula:
 *   - Baseline (α <= α_BLS): δ = δ_BLS
 *   - Above baseline: δ = δ_BLS × [1 + ai_exposure × (α - α_BLS) / (α_BLS - 1)]
 *
 * With ai_exposure = 40%: Low Displacement=5%, Strong=6%, AGI=9%
 *
 * @param {number} alpha - TFP multiplier (1.4 = low displacement, 1.6 = strong, 2.2 = AGI)
 * @param {number} bls_deflation - BLS historical software deflation rate (default 0.05)
 * @param {number} alpha_bls - Alpha value for BLS rate (default 1.4)
 * @param {number} ai_exposure - Sensitivity of deflation to excess TFP (default 0.40)
 * @returns {number} Deflation rate as decimal (e.g., 0.05 for 5%)
 */
function calcDeflationFromAlpha(alpha, bls_deflation = 0.05, alpha_bls = 1.4, ai_exposure = 0.40) {
    if (alpha <= alpha_bls) {
        return bls_deflation;
    }
    const excess = (alpha - alpha_bls) / (alpha_bls - 1);
    return bls_deflation * (1 + ai_exposure * excess);
}

/**
 * Additional VAT revenue from dynamic VAT.
 * Revenue = C × Δv × β  where C = consumption_share × Y
 *
 * @param {number} Y - GDP (billion NIS) - use Y_1 for post-AI scenarios
 * @param {number} consumption_share - Private consumption as share of GDP (decimal, e.g., 0.539)
 * @param {number} v0 - Initial VAT rate (decimal, e.g., 0.18)
 * @param {number} delta - Deflation rate (decimal)
 * @param {number} passthrough - VAT pass-through rate (β)
 * @returns {number} Additional revenue (billion NIS)
 */
function calcVatRevenue(Y, consumption_share, v0, delta, passthrough) {
    const v1 = calcVatAdjustment(v0, delta);
    const C = consumption_share * Y;  // Consumption in billion NIS
    return C * (v1 - v0) * passthrough;
}

/**
 * Ring-fenced profit for NIT fund.
 * Ringfenced = (actual - baseline) * attribution * earmark
 */
function calcRingfencedProfit(baseline, actual, attribution, earmark) {
    const over_trend = actual - baseline;
    const ai_attributed = over_trend * attribution;
    return ai_attributed * earmark;
}

/**
 * Linear extrapolation of capital gains tax trend (excluding 2017 outlier).
 * Uses 2012-2016, 2018-2019 data for linear regression.
 *
 * @param {object} data - DATA object with cg_tax_YYYY fields
 * @param {number} target_year - Year to extrapolate to (default 2024)
 * @returns {number} Extrapolated CG tax revenue (B NIS)
 */
function calcCGTrendLinear(data, target_year = 2024) {
    // Years and values (excluding 2017 outlier)
    const years = [2012, 2013, 2014, 2015, 2016, 2018, 2019];
    const values = [
        data.cg_tax_2012, data.cg_tax_2013, data.cg_tax_2014,
        data.cg_tax_2015, data.cg_tax_2016, data.cg_tax_2018, data.cg_tax_2019
    ];

    // Simple linear regression: y = mx + b
    const n = years.length;
    let sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    for (let i = 0; i < n; i++) {
        sum_x += years[i];
        sum_y += values[i];
        sum_xy += years[i] * values[i];
        sum_xx += years[i] * years[i];
    }
    const m = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    const b = (sum_y - m * sum_x) / n;

    return m * target_year + b;
}

/**
 * GDP-linked capital gains baseline.
 * CG_trend = avg(CG/GDP) * Y1
 *
 * @param {object} data - DATA object with cg_tax_YYYY and gdp_YYYY fields
 * @param {number} Y1 - Post-AI GDP (B NIS)
 * @returns {number} GDP-linked CG baseline (B NIS)
 */
function calcCGTrendGDPLinked(data, Y1) {
    // Years (excluding 2017 outlier)
    const years = [2012, 2013, 2014, 2015, 2016, 2018, 2019];
    const cg_values = [
        data.cg_tax_2012, data.cg_tax_2013, data.cg_tax_2014,
        data.cg_tax_2015, data.cg_tax_2016, data.cg_tax_2018, data.cg_tax_2019
    ];
    const gdp_values = [
        data.gdp_2012, data.gdp_2013, data.gdp_2014,
        data.gdp_2015, data.gdp_2016, data.gdp_2018, data.gdp_2019
    ];

    // Calculate average CG/GDP ratio
    let sum_ratio = 0;
    for (let i = 0; i < years.length; i++) {
        sum_ratio += cg_values[i] / gdp_values[i];
    }
    const avg_ratio = sum_ratio / years.length;

    return avg_ratio * Y1;
}

/**
 * Ring-fenced capital gains tax for NIT fund.
 * Uses average of linear and GDP-linked trends as baseline.
 *
 * @param {object} data - DATA object
 * @param {number} Y1 - Post-AI GDP (B NIS)
 * @param {number} g_Y - GDP growth rate (decimal)
 * @param {number} attribution - AI attribution coefficient (default 1.0)
 * @param {number} earmark - Ring-fence rate (decimal, e.g., 0.75)
 * @returns {number} Ring-fenced CG revenue (B NIS)
 */
function calcRingfencedCG(data, Y1, g_Y, attribution, earmark) {
    // Calculate baseline as average of linear and GDP-linked
    const linear = calcCGTrendLinear(data);
    const gdp_linked = calcCGTrendGDPLinked(data, data.Y0);  // Baseline GDP
    const baseline = (linear + gdp_linked) / 2;

    // Actual CG grows with GDP (simplified assumption)
    const actual = baseline * (1 + g_Y);

    // Ring-fenced amount
    const over_trend = actual - baseline;
    return over_trend * attribution * earmark;
}

/**
 * Ring-fenced Government Wage Savings.
 * R_gov = Personnel * u * ringfence_rate
 *
 * Assumes public sector automation mirrors economy-wide displacement:
 * if u% of the economy's labor is displaced by AI, then u% of public
 * sector labor costs are saved. Ring-fence rate determines share for NIT.
 */
function calcGovRingfence(personnel, unemployment_rate, ringfence_rate) {
    const gross_savings = personnel * unemployment_rate;
    return gross_savings * ringfence_rate;
}

/**
 * Program consolidation revenue.
 * Consolidation = IS + UE + AdminSavings
 */
function calcConsolidationRevenue(income_support, unemployment, admin_rate) {
    const redirected = income_support + unemployment;
    const admin_savings = redirected * admin_rate;
    return redirected + admin_savings;
}

/**
 * Calculate UHI (Universal High Income) funding breakdown.
 * Returns detailed breakdown of funding sources and gap analysis.
 *
 * @param {number} existingRevenue - Basic NIT revenue (VAT + Ring-fencing + Consolidation)
 * @param {number} deltaPi - Capital windfall (s_K' * Y_1 - s_K * Y_0)
 * @param {number} uhiTargetCost - Target UHI cost (B NIS)
 * @param {object} data - Data object with UHI parameters
 */
function calcUHIFunding(existingRevenue, deltaPi, uhiTargetCost, data) {
    // Third-channel funding sources (fixed estimates from research)
    const koheletConsolidation = data.kohelet_consolidation || 43;
    const koheletTaxReform = data.kohelet_tax_reform || 78;
    const estateTax = data.estate_tax_revenue || 29;
    const dataDividend = data.data_dividend || 4;
    const pillarTwo = data.pillar_two_revenue || 19;
    const yozma = data.yozma_revenue || 3;

    // Sum of all non-windfall sources
    const extendedConsolidation = koheletConsolidation + koheletTaxReform;
    const newInstruments = estateTax + dataDividend + pillarTwo + yozma;
    const otherSources = existingRevenue + extendedConsolidation + newInstruments;

    // Windfall levy = remainder needed to reach UHI target
    const windfallLevy = Math.max(0, uhiTargetCost - otherSources);
    const windfallLevyPct = deltaPi > 0 ? (windfallLevy / deltaPi) * 100 : 0;

    // Total funding
    const totalFunding = otherSources + windfallLevy;

    return {
        // Existing mechanisms
        existingRevenue,

        // Extended consolidation (Kohelet)
        koheletConsolidation,
        koheletTaxReform,
        extendedConsolidation,

        // New instruments
        estateTax,
        dataDividend,
        pillarTwo,
        yozma,
        newInstruments,

        // Direct windfall capture
        windfallLevy,
        windfallLevyPct,

        // Totals
        totalFunding,
        uhiTargetCost,
        deltaPi,

        // Gap analysis
        gap: uhiTargetCost - totalFunding,
        feasible: totalFunding >= uhiTargetCost
    };
}

// =============================================================================
// NIT COST AND FLOOR CALCULATIONS
// =============================================================================

/**
 * Calculate NIT benefit for a household.
 * B = max(0, M - tau * max(0, y - D))
 */
function calcNITBenefit(income, floor, taper, disregard) {
    const taxable = Math.max(0, income - disregard);
    const benefit = floor - taper * taxable;
    return Math.max(0, benefit);
}

/**
 * Calculate annual NIT cost given floor M and household decile data.
 * Cost = sum(benefit_per_hh * households * 12) / 1e9
 */
function calcNITCostFromDeciles(M, deciles, taper, disregard) {
    const breakeven = disregard + M / taper;
    let total_cost = 0;

    for (const d of deciles) {
        let benefit_per_std;
        if (d.equiv_income >= breakeven) {
            benefit_per_std = 0;
        } else {
            const taxable = Math.max(0, d.equiv_income - disregard);
            benefit_per_std = Math.max(0, M - taper * taxable);
        }
        const benefit_per_hh = benefit_per_std * d.std_persons;
        total_cost += benefit_per_hh * d.households * 12;
    }

    return total_cost / 1e9;
}

/**
 * CBS 2022 effective tax rates by decile.
 * Calculated as (gross - net) / gross from CBS income data.
 */
const CBS_TAX_RATES = [0.055, 0.069, 0.097, 0.109, 0.115, 0.132, 0.150, 0.162, 0.200, 0.266];

/**
 * Calculate per-decile NIT cost breakdown.
 * Returns array with benefit and cost details per decile.
 * Costs are shown at 85% take-up to match Table 11 in paper.
 * (Floor M is calibrated so that total cost x 0.85 = revenue)
 */
function calcDecileBreakdown(M, deciles, taper, disregard, takeup = 0.85) {
    const breakeven = disregard + M / taper;
    const breakdown = [];

    for (const d of deciles) {
        let benefit_per_std;
        let emtr = null;

        if (d.equiv_income >= breakeven) {
            benefit_per_std = 0;
        } else {
            const taxable = Math.max(0, d.equiv_income - disregard);
            benefit_per_std = Math.max(0, M - taper * taxable);
            // Calculate EMTR: t + tau (additive, since taper on gross income)
            const t = CBS_TAX_RATES[d.decile - 1] || 0;
            emtr = t + taper;
        }
        const benefit_per_hh = benefit_per_std * d.std_persons;
        // Cost at take-up rate (85% by default, matches Table 11)
        const annual_cost = benefit_per_hh * d.households * 12 * takeup / 1e9; // B NIS

        breakdown.push({
            decile: d.decile,
            households: d.households,
            equiv_income: d.equiv_income,
            std_persons: d.std_persons,
            benefit_per_std: benefit_per_std,
            benefit_per_hh: benefit_per_hh,
            annual_cost: annual_cost,
            emtr: emtr,
        });
    }

    return breakdown;
}

/**
 * Calculate revenue-constrained floor using bisection.
 * Find M where cost(M) = revenue
 */
// =============================================================================
// MICROSIMULATION: Random Number Generation
// =============================================================================

/**
 * Simple seeded random number generator (Mulberry32).
 */
function createRNG(seed) {
    let state = seed;
    return function() {
        state |= 0; state = state + 0x6D2B79F5 | 0;
        let t = Math.imul(state ^ state >>> 15, 1 | state);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

/**
 * Generate Beta-distributed random numbers using rejection sampling.
 * For Beta(α, β): mean = α/(α+β), variance = αβ/((α+β)²(α+β+1))
 */
function betaRandom(alpha, beta, rng) {
    // Use Gamma method: if X ~ Gamma(α,1) and Y ~ Gamma(β,1), then X/(X+Y) ~ Beta(α,β)
    const x = gammaRandom(alpha, 1, rng);
    const y = gammaRandom(beta, 1, rng);
    return x / (x + y);
}

/**
 * Generate Gamma-distributed random numbers using Marsaglia and Tsang's method.
 */
function gammaRandom(shape, scale, rng) {
    if (shape < 1) {
        // For shape < 1, use: Gamma(α) = Gamma(α+1) * U^(1/α)
        return gammaRandom(shape + 1, scale, rng) * Math.pow(rng(), 1 / shape);
    }

    const d = shape - 1/3;
    const c = 1 / Math.sqrt(9 * d);

    while (true) {
        let x, v;
        do {
            x = normalRandom(rng);
            v = 1 + c * x;
        } while (v <= 0);

        v = v * v * v;
        const u = rng();
        const x2 = x * x;

        if (u < 1 - 0.0331 * x2 * x2) {
            return d * v * scale;
        }
        if (Math.log(u) < 0.5 * x2 + d * (1 - v + Math.log(v))) {
            return d * v * scale;
        }
    }
}

/**
 * Generate standard normal random numbers using Box-Muller transform.
 */
function normalRandom(rng) {
    const u1 = rng();
    const u2 = rng();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// =============================================================================
// MICROSIMULATION: Distribution Parameter Estimation
// =============================================================================

/**
 * Estimate Beta concentration parameter from cross-decile income variance.
 * From verify_layer3: c = (U-L)² × m(1-m) / Var - 1
 */
function estimateBetaConcentration(grossMeans) {
    const variance = calcVariance(grossMeans);
    const range = Math.max(...grossMeans) - Math.min(...grossMeans);
    const m = 0.5; // Typical normalized mean
    const c = (range * range) * m * (1 - m) / variance - 1;
    return Math.max(2.0, c);
}

/**
 * Estimate Gamma shape parameter from cross-decile std_persons variance.
 * From verify_layer3: α = mean² / variance
 */
function estimateGammaShape(stdPersonsMeans) {
    const mean = stdPersonsMeans.reduce((a, b) => a + b, 0) / stdPersonsMeans.length;
    const variance = calcVariance(stdPersonsMeans);
    if (variance <= 0) return 10.0;
    return Math.max(2.0, mean * mean / variance);
}

/**
 * Calculate variance of an array.
 */
function calcVariance(arr) {
    const n = arr.length;
    const mean = arr.reduce((a, b) => a + b, 0) / n;
    return arr.reduce((sum, x) => sum + (x - mean) * (x - mean), 0) / n;
}

// =============================================================================
// MICROSIMULATION: Decile Bounds and Household Generation
// =============================================================================

/**
 * Get decile bounds (GROSS income) from CBS data.
 * Uses gross/net ratio to convert CBS NET thresholds to GROSS.
 */
function getDecileBoundsGross(decile, grossMeans, totalRange = null) {
    const i = decile - 1;

    // Approximate bounds based on position in distribution
    // Lower deciles have narrower ranges, upper have wider
    const minIncome = 0;
    const maxIncome = grossMeans[9] * 1.5; // Estimate D10 upper bound

    // Interpolate bounds
    let lower, upper;
    if (decile === 1) {
        lower = minIncome;
        upper = (grossMeans[0] + grossMeans[1]) / 2;
    } else if (decile === 10) {
        lower = (grossMeans[8] + grossMeans[9]) / 2;
        upper = maxIncome;
    } else {
        lower = (grossMeans[i - 1] + grossMeans[i]) / 2;
        upper = (grossMeans[i] + grossMeans[i + 1]) / 2;
    }

    return { lower, upper };
}

/**
 * Generate synthetic households for a decile using Beta (income) and Gamma (std_persons).
 */
function generateDecileHouseholds(n, meanIncome, lowerIncome, upperIncome,
                                   meanStdPersons, betaConcentration, gammaShape, rng) {
    const households = [];

    for (let i = 0; i < n; i++) {
        // Generate income from Beta distribution on [lower, upper]
        let income;
        if (upperIncome <= lowerIncome) {
            income = meanIncome;
        } else {
            const m = Math.max(0.01, Math.min(0.99, (meanIncome - lowerIncome) / (upperIncome - lowerIncome)));
            const alpha = m * betaConcentration;
            const beta = (1 - m) * betaConcentration;
            const x = betaRandom(alpha, beta, rng);
            income = lowerIncome + (upperIncome - lowerIncome) * x;
        }

        // Generate std_persons from Gamma distribution
        const scale = meanStdPersons / gammaShape;
        const stdPersons = gammaRandom(gammaShape, scale, rng);

        households.push({
            equiv_gross: income,
            std_persons: Math.max(1, stdPersons) // At least 1 person
        });
    }

    return households;
}

// =============================================================================
// MICROSIMULATION: Cost Calculation
// =============================================================================

/**
 * Calculate NIT cost using microsimulation (exactly as in verify_layer3).
 */
function calcNITCostMicrosim(M, cbsDeciles, decileBounds, betaConcentration, gammaShape,
                              taper, disregard, nSamples = 500, seed = 42) {
    const rng = createRNG(seed);
    const breakeven = disregard + M / taper;
    let totalCost = 0;

    for (let i = 0; i < cbsDeciles.length; i++) {
        const d = cbsDeciles[i];
        const bounds = decileBounds[i];

        // Generate synthetic households
        const synthHH = generateDecileHouseholds(
            nSamples,
            d.equiv_income,
            bounds.lower,
            bounds.upper,
            d.std_persons,
            betaConcentration,
            gammaShape,
            rng
        );

        // Calculate total benefit for synthetic households
        let totalBenefit = 0;
        for (const hh of synthHH) {
            let benefitPerStd = 0;
            if (hh.equiv_gross < breakeven) {
                const taxable = Math.max(0, hh.equiv_gross - disregard);
                benefitPerStd = Math.max(0, M - taper * taxable);
            }
            totalBenefit += benefitPerStd * hh.std_persons;
        }

        // Average benefit per household
        const avgBenefitPerHH = totalBenefit / nSamples;

        // Annual cost for this decile (billion NIS)
        const decileCost = avgBenefitPerHH * d.households * 12 / 1e9;
        totalCost += decileCost;
    }

    return totalCost;
}

/**
 * Calculate per-decile NIT cost breakdown using microsimulation.
 * Uses within-decile Beta distribution (concentration c=2.16) to capture
 * income variation, matching Table 11 methodology.
 * Costs shown at 85% take-up.
 */
function calcDecileBreakdownMicrosim(M, cbsDeciles, decileBounds, betaConcentration, gammaShape,
                                      taper, disregard, takeup = 0.85, nSamples = 500, seed = 42) {
    const rng = createRNG(seed);
    const breakeven = disregard + M / taper;
    const breakdown = [];

    for (let i = 0; i < cbsDeciles.length; i++) {
        const d = cbsDeciles[i];
        const bounds = decileBounds[i];

        // Generate synthetic households with Beta-distributed incomes
        const synthHH = generateDecileHouseholds(
            nSamples,
            d.equiv_income,
            bounds.lower,
            bounds.upper,
            d.std_persons,
            betaConcentration,
            gammaShape,
            rng
        );

        // Calculate total benefit for synthetic households
        let totalBenefit = 0;
        for (const hh of synthHH) {
            let benefitPerStd = 0;
            if (hh.equiv_gross < breakeven) {
                const taxable = Math.max(0, hh.equiv_gross - disregard);
                benefitPerStd = Math.max(0, M - taper * taxable);
            }
            totalBenefit += benefitPerStd * hh.std_persons;
        }

        // Average benefit per household (from microsim)
        const avgBenefitPerHH = totalBenefit / nSamples;

        // Annual cost at take-up rate (85% by default)
        const annualCost = avgBenefitPerHH * d.households * 12 * takeup / 1e9;

        // EMTR (using decile average for display, as in paper)
        const t = CBS_TAX_RATES[d.decile - 1] || 0;
        const emtr = avgBenefitPerHH > 0 ? t + taper : null;

        breakdown.push({
            decile: d.decile,
            households: d.households,
            equiv_income: d.equiv_income,
            std_persons: d.std_persons,
            benefit_per_hh: avgBenefitPerHH,
            annual_cost: annualCost,
            emtr: emtr,
        });
    }

    return breakdown;
}

// =============================================================================
// FLOOR CALCULATION WITH TAKE-UP AND GDP CAP (matches verify_layer3)
// =============================================================================

/**
 * Calculate revenue-constrained NIT floor with take-up rate and GDP cap.
 * This matches exactly the logic in verify_layer3.ipynb:
 *
 * 1. Find M where Cost(M) × take_up = Revenue
 * 2. Check if Cost(M, 100%) > Revenue + gdp_buffer × Y1
 * 3. If cap binds, reduce M to ensure fiscal safety
 */
function calcFloorBalanced(revenue, deciles, taper, disregard, Y1 = null, takeUp = 0.85, gdpBuffer = 0.03) {
    // Use pre-computed microsim parameters from data.js (matches verify_layer3)
    const M = typeof MICROSIM_PARAMS !== 'undefined' ? MICROSIM_PARAMS : null;

    let betaConcentration, gammaShape, decileBounds, nSamples;

    if (M) {
        // Use pre-computed values (exact match with verify_layer3)
        betaConcentration = M.beta_concentration;
        gammaShape = M.gamma_shape;
        decileBounds = M.decile_bounds;
        nSamples = M.n_samples;
    } else {
        // Fallback: estimate from decile data
        const grossMeans = deciles.map(d => d.equiv_income);
        const stdPersonsMeans = deciles.map(d => d.std_persons);
        betaConcentration = estimateBetaConcentration(grossMeans);
        gammaShape = estimateGammaShape(stdPersonsMeans);
        decileBounds = deciles.map((d, i) => getDecileBoundsGross(i + 1, grossMeans));
        nSamples = 500;
    }

    // Helper: calculate microsim cost for a given floor
    const calcCost = (M) => calcNITCostMicrosim(
        M, deciles, decileBounds, betaConcentration, gammaShape,
        taper, disregard, nSamples, 42
    );

    // Bisection search to find M where cost = targetCost
    const findFloorForCost = (targetCost) => {
        let low = 100;
        let high = 50000;
        let M = (low + high) / 2;

        for (let i = 0; i < 50; i++) {
            const cost = calcCost(M);
            if (Math.abs(cost - targetCost) < 0.1) break;

            if (cost < targetCost) {
                low = M;
            } else {
                high = M;
            }
            M = (low + high) / 2;
        }
        return M;
    };

    // Step 1: Find M where Cost(M) × takeUp = Revenue
    const targetCost100 = revenue / takeUp;
    const M_uncapped = findFloorForCost(targetCost100);
    const cost100_uncapped = calcCost(M_uncapped);

    // Step 2: Check GDP cap
    let M_final = M_uncapped;
    let capBinds = false;
    let maxCost100 = Infinity;

    if (Y1 !== null && Y1 > 0) {
        maxCost100 = revenue + gdpBuffer * Y1;

        if (cost100_uncapped > maxCost100) {
            // Cap binds - find M where Cost(M) = maxCost100
            M_final = findFloorForCost(maxCost100);
            capBinds = true;
        }
    }

    // Calculate final costs
    const cost100_final = calcCost(M_final);
    const breakeven = disregard + M_final / taper;

    return {
        floor: M_final,
        floor_uncapped: M_uncapped,
        cost_at_takeup: cost100_final * takeUp,
        cost_at_100pct: cost100_final,
        max_cost_100pct: maxCost100,
        cap_binds: capBinds,
        breakeven: breakeven,
        // Legacy compatibility
        cost: cost100_final * takeUp,
        balance: revenue - cost100_final * takeUp,
    };
}

/**
 * Calculate break-even income.
 * y* = D + M/tau
 */
function calcBreakeven(D, M, tau) {
    return D + M / tau;
}

/**
 * Calculate household-specific NIT floor using modified OECD equivalence scale.
 *
 * Formula: F_h = M * scale + psi (if single parent)
 * Scale: 1.0 (first adult) + 0.5 (each additional adult) + 0.3 (each child)
 *
 * @param {number} M - Reference single-adult floor
 * @param {number} adults - Number of adults in household
 * @param {number} children - Number of children
 * @param {boolean} singleParent - Whether single-parent household
 * @param {number} psi - Single-parent top-up (default 900)
 * @returns {number} Household-specific floor F_h (NIS/mo)
 */
function calcHouseholdFloor(M, adults, children, singleParent, psi = 900) {
    const scale = 1.0 + 0.5 * (adults - 1) + 0.3 * children;
    const topup = singleParent ? psi : 0;
    return M * scale + topup;
}

// =============================================================================
// GINI AND POVERTY CALCULATIONS
// =============================================================================

/**
 * Calculate Gini coefficient from decile data using trapezoid rule.
 * G = 1 - 2 * (Area under Lorenz curve)
 */
function calcGiniFromDeciles(deciles, incomeKey = 'equiv_income', weightKey = 'households') {
    // Sort by income
    const sorted = [...deciles].sort((a, b) => a[incomeKey] - b[incomeKey]);

    // Calculate totals
    let total_income = 0;
    let total_pop = 0;
    for (const d of sorted) {
        total_income += d[incomeKey] * d[weightKey];
        total_pop += d[weightKey];
    }

    // Build Lorenz curve
    let cum_pop = 0;
    let cum_income = 0;
    const lorenz = [[0, 0]];

    for (const d of sorted) {
        cum_pop += d[weightKey];
        cum_income += d[incomeKey] * d[weightKey];
        lorenz.push([cum_pop / total_pop, cum_income / total_income]);
    }

    // Calculate area using trapezoid rule
    let area = 0;
    for (let i = 1; i < lorenz.length; i++) {
        const [x0, y0] = lorenz[i - 1];
        const [x1, y1] = lorenz[i];
        area += (x1 - x0) * (y0 + y1) / 2;
    }

    return 1 - 2 * area;
}

/**
 * Apply NIT to deciles and return post-NIT incomes.
 */
function applyNITToDeciles(deciles, floor, taper, disregard) {
    return deciles.map(d => {
        const benefit = calcNITBenefit(d.equiv_income, floor, taper, disregard);
        return {
            ...d,
            equiv_income: d.equiv_income + benefit,
            benefit: benefit,
        };
    });
}

/**
 * Calculate median from decile data (interpolate between D5 and D6).
 */
function calcMedianFromDeciles(deciles) {
    const sorted = [...deciles].sort((a, b) => a.equiv_income - b.equiv_income);
    // For 10 deciles, median is between D5 and D6
    const d5 = sorted[4].equiv_income;
    const d6 = sorted[5].equiv_income;
    return (d5 + d6) / 2;
}

/**
 * Calculate poverty rate from deciles (fraction with income < poverty_line).
 * Note: Using decile means, this is approximate.
 */
function calcPovertyRate(deciles, poverty_line) {
    const sorted = [...deciles].sort((a, b) => a.equiv_income - b.equiv_income);
    let poor_hh = 0;
    let total_hh = 0;

    for (const d of sorted) {
        total_hh += d.households;
        if (d.equiv_income < poverty_line) {
            poor_hh += d.households;
        }
    }

    return (poor_hh / total_hh) * 100;
}

/**
 * Calculate poverty threshold: pre-NIT income below which a person stays poor.
 *
 * Formula:
 *   If M >= z: no one remains poor (threshold = 0)
 *   If z - M <= D: threshold = z - M
 *   If z - M > D: threshold = (z - M - tau*D) / (1 - tau)
 */
function calcPovertyThreshold(z, M, taper, disregard) {
    if (M >= z) {
        return 0;  // Floor covers poverty line, no one remains poor
    }
    const gap = z - M;
    if (gap <= disregard) {
        return gap;  // Pre-NIT income < gap remains poor
    } else {
        return (z - M - taper * disregard) / (1 - taper);
    }
}

// =============================================================================
// CAPITAL WINDFALL CALCULATION
// =============================================================================

/**
 * Calculate capital windfall from AI-driven growth.
 * Formula: ΔW = s_K' × (Y₁ - Y₀)
 *
 * This is capital's share of OUTPUT GROWTH only, not total capital income change.
 * Represents capital income that wouldn't exist without AI-driven growth.
 *
 * @param {number} Y0 - Pre-automation GDP (billion NIS)
 * @param {number} Y1 - Post-automation GDP (billion NIS)
 * @param {number} s_L_prime - Post-automation labor share
 * @returns {number} Capital windfall (billion NIS)
 */
/**
 * Capital windfall: capital's share of GDP growth.
 * ΔW = s_K' × (Y₁ - Y₀)
 */
function calcCapitalWindfall(Y0, Y1, s_L_prime) {
    const s_K_prime = 1 - s_L_prime;
    return s_K_prime * (Y1 - Y0);
}

/**
 * Capital profit change: total change in capital income.
 * ΔΠ = s_K' × Y₁ - s_K × Y₀
 * Symmetric with automation surplus: ΔS = s_L' × Y₁ - s_L × Y₀
 */
function calcCapitalProfitChange(Y0, Y1, s_L, s_L_prime) {
    const s_K = 1 - s_L;
    const s_K_prime = 1 - s_L_prime;
    return s_K_prime * Y1 - s_K * Y0;
}


// =============================================================================
// FLOOR PROPOSAL: Microsimulation Cost Wrapper
// =============================================================================

/**
 * Calculate NIT cost for floor proposal mode using microsimulation.
 * Follows exact same pattern as calcFloorBalanced for consistency.
 */
function calcNITCostForProposal(M, deciles, taper, disregard) {
    // Use pre-computed microsim parameters from data.js (matches verify_layer3)
    const params = typeof MICROSIM_PARAMS !== 'undefined' ? MICROSIM_PARAMS : null;

    if (params) {
        // Use pre-computed values (exact match with verify_layer3/layer4)
        return calcNITCostMicrosim(
            M, deciles, params.decile_bounds,
            params.beta_concentration, params.gamma_shape,
            taper, disregard, params.n_samples || 500, 42
        );
    } else {
        // Fallback: use decile-average method
        return calcNITCostFromDeciles(M, deciles, taper, disregard);
    }
}

// =============================================================================
// FLOOR PROPOSAL SOLVERS
// =============================================================================

/**
 * Calculate total revenue given Y1 and scenario parameters.
 * Helper function for inverse solvers.
 */
function calcTotalRevenueForY1(Y1, g_Y, scenarioParams, data) {
    const D = data;
    const ringfence_rate = scenarioParams.ringfence_rate / 100;

    // Derive deflation from alpha (BLS-anchored)
    const bls_delta = D.bls_deflation_rate / 100;
    const alpha_bls = D.bls_deflation_alpha;
    const ai_exposure = D.ai_consumption_exposure / 100;
    const delta = calcDeflationFromAlpha(scenarioParams.alpha, bls_delta, alpha_bls, ai_exposure);

    // VAT revenue
    const v0 = D.vat_baseline / 100;
    const consumption_share = D.consumption_share / 100;
    const vat_revenue = calcVatRevenue(Y1, consumption_share, v0, delta, D.vat_passthrough);

    // Ring-fenced profits
    const profit_baseline = D.profit_trend_baseline;
    const profit_actual = profit_baseline * (1 + g_Y);
    const ringfenced = calcRingfencedProfit(
        profit_baseline, profit_actual, 1.0, ringfence_rate
    );

    // Ring-fenced capital gains
    const cg_ringfenced = calcRingfencedCG(D, Y1, g_Y, 1.0, ringfence_rate);

    // Ring-fenced government wages (uses unemployment rate and same ringfence_rate)
    const gov_ringfence = calcGovRingfence(
        D.public_sector_wages,
        scenarioParams.u / 100,
        ringfence_rate
    );

    // Consolidation (fixed)
    const consolidation = calcConsolidationRevenue(
        D.btl_income_support, D.btl_unemployment, D.consolidation_admin_rate / 100
    );

    return { vat_revenue, ringfenced, cg_ringfenced, gov_ringfence, consolidation, total: vat_revenue + ringfenced + cg_ringfenced + gov_ringfence + consolidation };
}

/**
 * Solve for Y1 given a proposed floor.
 * Find Y1 such that revenue(Y1) = cost(proposedFloor)
 *
 * @param {number} proposedFloor - Proposed NIT floor (NIS/mo)
 * @param {object} scenarioParams - Scenario parameters (alpha, s_L_prime, u, ringfence_rate)
 * @param {array} deciles - CBS household decile data
 * @param {number} taper - NIT taper rate
 * @param {number} disregard - NIT disregard amount
 * @param {object} data - Exogenous data (DATA object)
 * @returns {object} { Y1, g_Y, revenue, cost, converged, message }
 */
function solveY1ForFloor(proposedFloor, scenarioParams, deciles, taper, disregard, data, takeUp = 0.85) {
    const cost100 = calcNITCostForProposal(proposedFloor, deciles, taper, disregard);
    const cost = cost100 * takeUp;  // Revenue needed at given take-up rate
    const Y0 = data.Y0;

    // Bisection bounds
    let low = Y0;                // Minimum: no growth
    let high = Y0 * 10;          // Maximum: 900% growth (very generous)

    // Check if solution exists at upper bound
    const highRevenue = calcTotalRevenueForY1(high, (high - Y0) / Y0, scenarioParams, data);
    if (highRevenue.total < cost) {
        return {
            Y1: null,
            g_Y: null,
            revenue: highRevenue,
            cost: cost,
            converged: false,
            message: `Infeasible: Floor ${proposedFloor} requires more than ${highRevenue.total.toFixed(0)}B (max at 900% growth)`
        };
    }

    let Y1 = (low + high) / 2;
    let iterations = 0;
    const maxIter = 100;

    while (iterations < maxIter) {
        const g_Y = (Y1 - Y0) / Y0;
        const revenue = calcTotalRevenueForY1(Y1, g_Y, scenarioParams, data);

        if (Math.abs(revenue.total - cost) < 0.01) {
            return {
                Y1: Y1,
                g_Y: g_Y * 100,
                revenue: revenue,
                cost: cost,
                converged: true,
                iterations: iterations,
                message: `Converged in ${iterations} iterations`
            };
        }

        if (revenue.total < cost) {
            low = Y1;
        } else {
            high = Y1;
        }
        Y1 = (low + high) / 2;
        iterations++;
    }

    const finalRevenue = calcTotalRevenueForY1(Y1, (Y1 - Y0) / Y0, scenarioParams, data);
    return {
        Y1: Y1,
        g_Y: ((Y1 - Y0) / Y0) * 100,
        revenue: finalRevenue,
        cost: cost,
        converged: false,
        iterations: iterations,
        message: `Did not converge after ${maxIter} iterations`
    };
}

/**
 * Solve for alpha (TFP multiplier) given a proposed floor.
 * Hold s_L' and u constant, find alpha such that revenue = cost(proposedFloor)
 */
function solveAlphaForFloor(proposedFloor, s_L_prime, u, scenarioParams, deciles, taper, disregard, data, takeUp = 0.85) {
    const cost100 = calcNITCostForProposal(proposedFloor, deciles, taper, disregard);
    const cost = cost100 * takeUp;  // Revenue needed at given take-up rate
    const Y0 = data.Y0;
    const s_L = data.labor_share;
    const L_force = data.labor_force;
    const u_0 = data.natural_unemployment / 100;
    const sigma = data.sigma;
    const K_0 = data.K_0;

    // Bisection bounds for alpha
    let low = 1.0;   // Minimum: no TFP gain
    let high = 20.0; // Maximum: extreme TFP gain

    let alpha = (low + high) / 2;
    let iterations = 0;
    const maxIter = 100;

    while (iterations < maxIter) {
        // Calculate Y1 from CES with this alpha
        const growth = calcGrowthCES({
            alpha: alpha,
            s_L_prime: s_L_prime,
            u: u / 100,
            Y_0: Y0,
            s_L: s_L,
            L_force: L_force,
            u_0: u_0,
            sigma: sigma,
            K_0: K_0,
        });

        const Y1 = growth.Y_1;
        const g_Y = growth.g_Y;
        const revenue = calcTotalRevenueForY1(Y1, g_Y, scenarioParams, data);

        if (Math.abs(revenue.total - cost) < 0.01) {
            return {
                alpha: alpha,
                Y1: Y1,
                g_Y: g_Y * 100,
                revenue: revenue,
                cost: cost,
                converged: true,
                iterations: iterations,
                message: `Found α = ${alpha.toFixed(2)}`
            };
        }

        if (revenue.total < cost) {
            low = alpha;
        } else {
            high = alpha;
        }
        alpha = (low + high) / 2;
        iterations++;
    }

    return {
        alpha: alpha,
        converged: false,
        message: `Did not converge after ${maxIter} iterations`
    };
}

/**
 * Solve for unemployment rate given a proposed floor.
 * Hold alpha and s_L' constant, find u such that revenue = cost(proposedFloor)
 */
function solveUForFloor(proposedFloor, alpha, s_L_prime, scenarioParams, deciles, taper, disregard, data, takeUp = 0.85) {
    const cost100 = calcNITCostForProposal(proposedFloor, deciles, taper, disregard);
    const cost = cost100 * takeUp;  // Revenue needed at given take-up rate
    const Y0 = data.Y0;
    const s_L = data.labor_share;
    const L_force = data.labor_force;
    const u_0 = data.natural_unemployment / 100;
    const sigma = data.sigma;
    const K_0 = data.K_0;

    // Bisection bounds for u (unemployment as decimal)
    // Note: Lower u = higher L1 = higher Y1 = higher revenue
    let low = 0.0;    // 0% unemployment
    let high = 0.99;  // 99% unemployment

    // Check if solution exists
    const lowGrowth = calcGrowthCES({ alpha, s_L_prime, u: low, Y_0: Y0, s_L, L_force, u_0, sigma, K_0 });
    const lowRevenue = calcTotalRevenueForY1(lowGrowth.Y_1, lowGrowth.g_Y, scenarioParams, data);

    if (lowRevenue.total < cost) {
        return {
            u: null,
            converged: false,
            message: `Infeasible: Even 0% unemployment gives only ${lowRevenue.total.toFixed(0)}B revenue`
        };
    }

    let u = (low + high) / 2;
    let iterations = 0;
    const maxIter = 100;

    while (iterations < maxIter) {
        const growth = calcGrowthCES({
            alpha: alpha,
            s_L_prime: s_L_prime,
            u: u,
            Y_0: Y0,
            s_L: s_L,
            L_force: L_force,
            u_0: u_0,
            sigma: sigma,
            K_0: K_0,
        });

        const Y1 = growth.Y_1;
        const g_Y = growth.g_Y;
        const revenue = calcTotalRevenueForY1(Y1, g_Y, scenarioParams, data);

        if (Math.abs(revenue.total - cost) < 0.01) {
            return {
                u: u * 100,  // Return as percentage
                Y1: Y1,
                g_Y: g_Y * 100,
                revenue: revenue,
                cost: cost,
                converged: true,
                iterations: iterations,
                message: `Found u = ${(u * 100).toFixed(1)}%`
            };
        }

        // Lower u = higher revenue, so if revenue < cost, we need lower u
        if (revenue.total < cost) {
            high = u;
        } else {
            low = u;
        }
        u = (low + high) / 2;
        iterations++;
    }

    return {
        u: u * 100,
        converged: false,
        message: `Did not converge after ${maxIter} iterations`
    };
}

/**
 * Calculate windfall analysis for all scenarios.
 * Returns data for the windfall capture table.
 *
 * When proposedFloor is null:
 * - Shows each scenario's revenue-constrained floor (gap ≈ 0)
 *
 * When proposedFloor is set:
 * - Shows the SAME proposed floor for ALL scenarios
 * - Cost is same for all (depends only on floor, taper, disregard)
 * - Gap = Cost - Revenue (different per scenario)
 * - Capture % = Gap / ΔW (what fraction of windfall needed)
 */
function calcWindfallAnalysis(results, proposedFloor, deciles, taper, disregard, data, takeUp = 0.85) {
    const Y0 = data.Y0;
    const s_L = data.labor_share;
    const rows = [];

    // Calculate cost of proposed floor (same for all scenarios)
    const proposedCost100 = proposedFloor
        ? calcNITCostForProposal(proposedFloor, deciles, taper, disregard)
        : null;
    const proposedCost = proposedCost100 ? proposedCost100 * takeUp : null;

    for (const [name, r] of Object.entries(results)) {
        let floor, cost;

        if (proposedFloor) {
            // Feature 1: Use proposed floor for ALL scenarios
            floor = proposedFloor;
            cost = proposedCost;
        } else {
            // Default: Use each scenario's revenue-constrained floor
            floor = r.floor;
            cost = r.cost;
        }

        const revenue = r.total_revenue;
        const gap = cost - revenue;

        // ΔW = s_K' × (Y₁ - Y₀) - capital's share of growth
        const deltaW = calcCapitalWindfall(Y0, r.Y_1, r.s_L_prime);

        // ΔΠ = s_K' × Y₁ - s_K × Y₀ - total change in capital income
        const deltaPi = calcCapitalProfitChange(Y0, r.Y_1, s_L, r.s_L_prime);

        // Capture percentages for both measures
        const captureW = deltaW > 0 ? (gap / deltaW) * 100 : 0;
        const capturePi = deltaPi > 0 ? (gap / deltaPi) * 100 : 0;

        rows.push({
            scenario: name,
            floor: floor,
            cost: cost,
            revenue: revenue,
            gap: gap,
            deltaW: deltaW,
            deltaPi: deltaPi,
            captureW: captureW,
            capturePi: capturePi,
            isProposed: !!proposedFloor
        });
    }

    return rows;
}

// =============================================================================
// MAIN CALCULATION FUNCTION
// =============================================================================

/**
 * Run all calculations for all scenarios.
 * @param {object} params - Design parameters
 * @param {array} deciles - CBS household income deciles
 * @param {object} data - Optional: exogenous data (defaults to global DATA)
 */
function calculateAll(params, deciles, data = null) {
    const {
        disregard,
        taper,
        scenarios,
        // Design parameters (with fallbacks to DATA)
        sigma: paramSigma,
        single_parent_topup,
        child_equiv_weight,
        nit_take_up_rate,
        nit_gdp_buffer_pct,
        ai_consumption_exposure: paramAIExposure,
        ai_attribution_coef: paramAttribution,
        public_sector_substitutable: paramSubstitutable,
        consolidation_admin_rate: paramAdminRate,
    } = params;

    // Use provided data or fall back to global DATA
    const D = data || (typeof DATA !== 'undefined' ? DATA : null);
    if (!D) throw new Error('DATA not available');

    const results = {};

    // Base economic parameters (from data sources)
    const Y_0 = D.Y0;
    const s_L = D.labor_share;
    const L_force = D.labor_force;
    const u_0 = D.natural_unemployment / 100;
    const poverty_line = D.poverty_line;
    const K_0 = D.K_0;

    // Design parameters (from params, with fallbacks)
    const sigma = paramSigma !== undefined ? paramSigma : D.sigma;
    const ai_exposure = (paramAIExposure !== undefined ? paramAIExposure : D.ai_consumption_exposure) / 100;
    const attribution = (paramAttribution !== undefined ? paramAttribution : D.ai_attribution_coef) / 100;
    const substitutable = (paramSubstitutable !== undefined ? paramSubstitutable : D.public_sector_substitutable) / 100;
    const admin_rate = (paramAdminRate !== undefined ? paramAdminRate : D.consolidation_admin_rate) / 100;
    const take_up = (nit_take_up_rate !== undefined ? nit_take_up_rate : D.nit_take_up_rate) / 100;
    const gdp_buffer = (nit_gdp_buffer_pct !== undefined ? nit_gdp_buffer_pct : D.nit_gdp_buffer_pct) / 100;

    for (const [scenarioName, scenario] of Object.entries(scenarios)) {
        // CES growth calculation (sigma from design params)
        const growth = calcGrowthCES({
            alpha: scenario.alpha,
            s_L_prime: scenario.s_L_prime,
            u: scenario.u / 100,
            Y_0,
            s_L,
            L_force,
            u_0,
            sigma,
            K_0,
        });

        // Derive deflation from alpha (BLS-anchored, ai_exposure from design params)
        const bls_delta = D.bls_deflation_rate / 100;
        const alpha_bls = D.bls_deflation_alpha;
        const delta = calcDeflationFromAlpha(scenario.alpha, bls_delta, alpha_bls, ai_exposure);

        // VAT revenue - now using Y_1 and consumption_share
        const v0 = D.vat_baseline / 100;
        const consumption_share = D.consumption_share / 100;  // 53.9% -> 0.539
        const vat_revenue = calcVatRevenue(
            growth.Y_1,
            consumption_share,
            v0,
            delta,
            D.vat_passthrough
        );

        // Ring-fenced profits (attribution from design params)
        const profit_baseline = D.profit_trend_baseline;
        const profit_actual = profit_baseline * (1 + growth.g_Y);
        const ringfenced = calcRingfencedProfit(
            profit_baseline,
            profit_actual,
            attribution,
            scenario.ringfence_rate / 100
        );

        // Ring-fenced capital gains (attribution from design params)
        const cg_ringfenced = calcRingfencedCG(
            D,
            growth.Y_1,
            growth.g_Y,
            attribution,
            scenario.ringfence_rate / 100
        );

        // Ring-fenced government wages (uses unemployment rate and same ringfence_rate)
        const gov_ringfence = calcGovRingfence(
            D.public_sector_wages,
            scenario.u / 100,
            scenario.ringfence_rate / 100
        );

        // Consolidation (admin_rate from design params)
        const consolidation = calcConsolidationRevenue(
            D.btl_income_support,
            D.btl_unemployment,
            admin_rate
        );

        // Total revenue (now includes CG)
        const total_revenue = vat_revenue + ringfenced + cg_ringfenced + gov_ringfence + consolidation;

        // Revenue-constrained floor (with take-up rate and GDP cap, matching verify_layer3)
        const floor_result = calcFloorBalanced(
            total_revenue, deciles, taper, disregard,
            growth.Y_1, take_up, gdp_buffer
        );

        // Apply NIT and calculate post-NIT metrics
        const post_deciles = applyNITToDeciles(deciles, floor_result.floor, taper, disregard);
        const gini_post = calcGiniFromDeciles(post_deciles);

        // Static poverty (using pre-NIT z = 3324)
        const static_poverty = calcPovertyRate(post_deciles, poverty_line);
        const static_threshold = calcPovertyThreshold(poverty_line, floor_result.floor, taper, disregard);

        // Dynamic poverty (using z' = 50% of post-NIT median)
        const post_median = calcMedianFromDeciles(post_deciles);
        const dynamic_z = post_median * 0.5;
        const dynamic_poverty = calcPovertyRate(post_deciles, dynamic_z);
        const dynamic_threshold = calcPovertyThreshold(dynamic_z, floor_result.floor, taper, disregard);

        results[scenarioName] = {
            // Growth
            g_Y: growth.g_Y_pct,
            Y_1: growth.Y_1,
            delta_S: growth.delta_S,
            delta_S_pct: (growth.delta_S / Y_0) * 100,

            // Scenario params (for windfall calculation)
            s_L_prime: scenario.s_L_prime,
            deflation: delta * 100,  // Derived from alpha, stored as %

            // Revenue components
            vat_revenue,
            ringfenced,
            cg_ringfenced,
            gov_ringfence,
            consolidation,
            total_revenue,

            // Floor (microsim with take-up and GDP cap)
            floor: floor_result.floor,
            breakeven: floor_result.breakeven,
            cost_at_takeup: floor_result.cost_at_takeup,
            cost_at_100pct: floor_result.cost_at_100pct,
            max_cost_100pct: floor_result.max_cost_100pct,
            cap_binds: floor_result.cap_binds,
            cost: floor_result.cost,
            balance: floor_result.balance,
            floor_over_z: floor_result.floor / poverty_line,

            // Poverty & inequality
            gini_post,
            static_poverty,
            static_threshold,
            dynamic_poverty,
            dynamic_z,
            dynamic_threshold,
        };
    }

    return results;
}

// Export for use in app.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        cesOutput,
        cesLaborShare,
        calibrateCES,
        calcGrowthCES,
        calcVatAdjustment,
        calcDeflationFromAlpha,
        calcVatRevenue,
        calcRingfencedProfit,
        calcCGTrendLinear,
        calcCGTrendGDPLinked,
        calcRingfencedCG,
        calcGovRingfence,
        calcConsolidationRevenue,
        calcUHIFunding,
        calcNITBenefit,
        calcNITCostFromDeciles,
        calcDecileBreakdown,
        calcDecileBreakdownMicrosim,
        calcFloorBalanced,
        calcBreakeven,
        calcGiniFromDeciles,
        applyNITToDeciles,
        calcMedianFromDeciles,
        calcPovertyRate,
        calcPovertyThreshold,
        calcCapitalWindfall,
        calcCapitalProfitChange,
        calcTotalRevenueForY1,
        solveY1ForFloor,
        solveAlphaForFloor,
        solveUForFloor,
        calcWindfallAnalysis,
        calculateAll,
    };
}
