"""MOSAIC verification functions.

This module contains all calculation and verification functions used
in the MOSAIC numerical verification system.
"""

import json
from pathlib import Path

from config import DATA


# =============================================================================
# CALC INFRASTRUCTURE
# =============================================================================

CALC_FILE = Path(__file__).parent / "calc_outputs.json"
CALC: dict = {"layer1": {}, "layer2": {}, "layer3": {}}


def init_calc() -> dict:
    """Initialize CALC fresh, clearing any existing values.

    Creates a fresh empty CALC structure and persists to disk.
    Call at start of verify_layer1.ipynb to ensure reproducible runs.

    Output: The fresh empty CALC dictionary
    """
    global CALC
    CALC = {"layer1": {}, "layer2": {}, "layer3": {}, "layer4": {}}
    save_calc()
    return CALC


def load_calc() -> dict:
    """Load CALC from JSON file.

    Loads CALC from persistent storage. Call at start of layer2-4 notebooks
    to continue from previous layer. If file doesn't exist, initializes empty.

    Output: The loaded CALC dictionary
    """
    global CALC
    if CALC_FILE.exists():
        with open(CALC_FILE, encoding="utf-8") as f:
            CALC = json.load(f)
    else:
        CALC = {"layer1": {}, "layer2": {}, "layer3": {}, "layer4": {}}
    return CALC


def save_calc() -> None:
    """Persist CALC to JSON file.

    Saves current CALC state to disk. Called automatically by store().
    """
    with open(CALC_FILE, "w", encoding="utf-8") as f:
        json.dump(CALC, f, indent=2)


def store(key: str, value: float, layer: str = "layer2") -> float:
    """Store a calculated value and persist to disk.

    Inputs:
        key: Name of the value (e.g., 'breakeven_income')
        value: Calculated value
        layer: Which layer (layer1, layer2, layer3)

    Output: The stored value (for chaining)
    """
    if layer not in CALC:
        CALC[layer] = {}
    CALC[layer][key] = value
    save_calc()
    return value


def get(key: str, layer: str = None) -> float:
    """Get a value from CALC, then DATA.

    Search order:
    1. If layer specified, search only that layer
    2. Otherwise, search all CALC layers (layer1, layer2, layer3)
    3. Finally, search DATA dict

    Inputs:
        key: Name of the value
        layer: Optional layer to search (if None, searches all)

    Output: The requested value

    Raises:
        KeyError: If key not found in CALC or DATA
    """
    if layer:
        if key in CALC.get(layer, {}):
            return CALC[layer][key]
    else:
        for l in ["layer1", "layer2", "layer3"]:
            if key in CALC.get(l, {}):
                return CALC[l][key]
    # Fallback to DATA
    if key in DATA:
        return DATA[key]
    raise KeyError(f"Key '{key}' not found in CALC or DATA")


# =============================================================================
# VERIFICATION FUNCTION
# =============================================================================

def verify(name: str, calculated: float, paper: float,
           tolerance: float = 0.01, layer: str = "layer2") -> bool:
    """Verify calculated value against paper and store in CALC.

    This function:
    1. Stores the calculated value to CALC (persisted to disk)
    2. Compares against the PAPER reference value
    3. Reports [OK] or [MISMATCH]

    Inputs:
        name: Identifier for the value (used as key in CALC)
        calculated: Our calculated value
        paper: Reference value from PAPER dict
        tolerance: Relative tolerance (default 1%)
        layer: Which layer to store in (layer1, layer2, layer3)

    Output: True if values match within tolerance, False otherwise
    """
    # Store the calculated value
    store(name, calculated, layer)

    # Compare to paper
    if paper == 0:
        match = abs(calculated) < tolerance
    else:
        match = abs(calculated - paper) / abs(paper) < tolerance

    if match:
        status = f"[OK] {calculated:.4g}"
    else:
        status = f"[MISMATCH] calc={calculated:.4g}, paper={paper:.4g}"

    print(f"  {name}: {status}")
    return match


# =============================================================================
# LAYER 2 CALCULATION FUNCTIONS
# =============================================================================

def calc_breakeven(D: float, M: float, tau: float) -> float:
    """Break-even income where NIT phases out.

    Formula: y* = D + M/tau

    Inputs:
        D: Disregard (earnings exempt from taper)
        M: Floor (guaranteed minimum)
        tau: Taper rate

    Output: Break-even income (NIS/mo)
    """
    return D + M / tau


def calc_income_distribution_at_threshold(
    deciles: list[tuple], y_threshold: float
) -> dict:
    """Calculate CDF and mean income below a threshold from wage decile data.

    Method:
    1. F(y*): Linear interpolation within decile containing threshold
    2. Mean below y*: Weighted sum of full deciles + partial decile

    Inputs:
        deciles: List of tuples (decile, workers, mean, median, max_wage)
                 max_wage is None for top decile
        y_threshold: Income threshold (e.g., break-even income)

    Output: Dict with F_y (CDF), mean_below, target_decile, details
    """
    # Find which decile contains the threshold
    lower_bound = 0
    target_decile = None
    decile_lower = 0
    decile_upper = None

    for i, (d, workers, mean, median, max_wage) in enumerate(deciles):
        upper = max_wage if max_wage is not None else float('inf')
        if lower_bound <= y_threshold <= upper:
            target_decile = d
            decile_lower = lower_bound
            decile_upper = max_wage
            break
        lower_bound = max_wage if max_wage is not None else lower_bound

    if target_decile is None:
        raise ValueError(f"Threshold {y_threshold} outside distribution range")

    # Calculate F(y*) via linear interpolation
    base_cdf = (target_decile - 1) / 10
    if decile_upper is not None and decile_upper > decile_lower:
        fraction_in_decile = (y_threshold - decile_lower) / (decile_upper - decile_lower)
    else:
        fraction_in_decile = 0.5  # If in top decile, assume middle

    F_y = base_cdf + 0.10 * fraction_in_decile

    # Calculate mean income below threshold
    total_income = 0
    total_workers = 0

    for i, (d, workers, mean, median, max_wage) in enumerate(deciles):
        upper = max_wage if max_wage is not None else float('inf')
        d_lower = 0 if i == 0 else deciles[i-1][4]

        if upper <= y_threshold:
            # Entire decile is below threshold
            total_income += workers * mean
            total_workers += workers
        elif d_lower < y_threshold < upper:
            # Partial decile - threshold falls within
            fraction = (y_threshold - d_lower) / (upper - d_lower)
            partial_workers = fraction * workers
            # Approximate mean for truncated portion (uniform assumption)
            partial_mean = (d_lower + y_threshold) / 2
            total_income += partial_workers * partial_mean
            total_workers += partial_workers

    mean_below = total_income / total_workers if total_workers > 0 else 0

    return {
        'F_y': F_y,
        'mean_below': mean_below,
        'target_decile': target_decile,
        'decile_lower': decile_lower,
        'decile_upper': decile_upper,
        'fraction_in_decile': fraction_in_decile,
        'total_workers_below': total_workers,
    }


def calc_optimal_taper_saez(epsilon: float, y_bar: float, y_star: float,
                            F_y_star: float) -> float:
    """Saez (2001) optimal taper formula.

    Formula: tau/(1-tau) = (1/eps) * (ybar/y*) * ((1-F(y*))/F(y*))

    Inputs:
        epsilon: Labor supply elasticity
        y_bar: Mean income below break-even
        y_star: Break-even income
        F_y_star: CDF at break-even (fraction below y*)

    Output: Optimal taper rate tau*
    """
    rhs = (1/epsilon) * (y_bar/y_star) * ((1-F_y_star)/F_y_star)
    return rhs / (1 + rhs)


def calc_household_floor(M: float, adults: int, children: int,
                         single_parent: bool, psi: float) -> float:
    """Household floor using modified OECD equivalence scale.

    Formula: F_h = M * scale + psi (if single parent)
    Scale: 1.0 (first adult) + 0.5 (each additional adult) + 0.3 (each child)

    Inputs:
        M: Reference single-adult floor
        adults: Number of adults in household
        children: Number of children
        single_parent: Whether single-parent household
        psi: Single-parent top-up

    Output: Household-specific floor F_h (NIS/mo)
    """
    scale = 1.0 + 0.5 * (adults - 1) + 0.3 * children
    topup = psi if single_parent else 0
    return M * scale + topup


def calc_adjustment_triggered(u: float, s_L: float,
                              u_0: float = 3.0, s_L_0: float = 0.56,
                              alpha_u: float = 0.40, alpha_L: float = 0.60) -> float:
    """Floor adjustment formula (triggered growth rate).

    Formula: dF/F = alpha_u * (u - u_0)/u_0 + alpha_L * (s_L_0 - s_L)/s_L_0

    Inputs:
        u: Current unemployment rate (percent)
        s_L: Current labor share
        u_0: Natural unemployment rate (percent)
        s_L_0: Baseline labor share
        alpha_u: Unemployment trigger coefficient
        alpha_L: Labor share trigger coefficient

    Output: Triggered floor growth rate (before caps)
    """
    unemployment_adj = alpha_u * (u - u_0) / u_0
    labor_share_adj = alpha_L * (s_L_0 - s_L) / s_L_0
    return unemployment_adj + labor_share_adj


def calc_scenario_floor(M_base: float, adjustment: float) -> float:
    """Scenario-specific floor from base floor and adjustment (UNCAPPED).

    Formula: M_scenario = M_base * (1 + adjustment)

    Inputs:
        M_base: Base reference floor
        adjustment: Total adjustment rate

    Output: Scenario-specific floor (NIS/mo)
    """
    return M_base * (1 + adjustment)


def calc_nit_cost_from_deciles(M: float, deciles: list, taper: float = 0.5,
                                disregard: float = 1000) -> float:
    """Calculate annual NIT cost given floor M and household decile data.

    Formula: Cost = Σ(benefit_per_hh × households × 12) / 1e9

    Inputs:
        M: Floor per standard person (NIS/mo)
        deciles: List of dicts with 'equiv_income', 'std_persons', 'households'
        taper: Taper rate (default 0.5)
        disregard: Disregard amount (NIS/mo, default 1000)

    Output: Annual cost (billion NIS)
    """
    breakeven = disregard + M / taper
    total_cost = 0

    for d in deciles:
        if d['equiv_income'] >= breakeven:
            benefit_per_std = 0
        else:
            taxable = max(0, d['equiv_income'] - disregard)
            benefit_per_std = max(0, M - taper * taxable)

        benefit_per_hh = benefit_per_std * d['std_persons']
        total_cost += benefit_per_hh * d['households'] * 12

    return total_cost / 1e9


def calc_floor_balanced(revenue: float, deciles: list, taper: float = 0.5,
                        disregard: float = 1000) -> dict:
    """Calculate revenue-constrained floor: find M where cost(M) = revenue.

    This is the core of the revenue-constrained NIT mechanism. The floor is
    set to the maximum level that can be sustained by available revenue.

    Formula: Solve for M such that C(M) = R

    Where:
        C(M) = Σ[B_d(M) × n_d × H_d × 12] / 1e9
        B_d(M) = max(0, M - τ × max(0, y_d - D))

    Inputs:
        revenue: Available revenue (billion NIS)
        deciles: List of dicts with 'equiv_income', 'std_persons', 'households'
        taper: Taper rate (default 0.5)
        disregard: Disregard amount (NIS/mo, default 1000)

    Output: Dict with:
        - floor: Revenue-constrained floor M* (NIS/mo)
        - cost: Annual cost at M* (billion NIS, ≈ revenue)
        - breakeven: Break-even income y* = D + M*/τ
        - balance: Revenue - Cost (≈ 0 by construction)
    """
    from scipy.optimize import brentq

    def cost_minus_revenue(M):
        return calc_nit_cost_from_deciles(M, deciles, taper, disregard) - revenue

    # Search for M where cost = revenue
    # Lower bound: 100 NIS/mo, upper bound: 50,000 NIS/mo
    try:
        M_balanced = brentq(cost_minus_revenue, 100, 50000)
    except ValueError:
        # If no solution in range, return edge case
        if cost_minus_revenue(100) > 0:
            M_balanced = 100  # Even minimum floor exceeds revenue
        else:
            M_balanced = 50000  # Revenue exceeds maximum floor cost

    cost = calc_nit_cost_from_deciles(M_balanced, deciles, taper, disregard)
    breakeven = disregard + M_balanced / taper

    return {
        'floor': M_balanced,
        'cost': cost,
        'breakeven': breakeven,
        'balance': revenue - cost
    }


def calc_floor_revenue_capped(M_base: float, adjustment_triggered: float,
                               revenue: float, deciles: list,
                               taper: float = 0.5, disregard: float = 1000,
                               balance_threshold: float = -0.03) -> dict:
    """Calculate revenue-capped floor using the full adjustment formula.

    Formula:
        ΔF/F = 𝟙[B ≥ -3%] × min{
            α_u(u-u_0)/u_0 + α_L(s_L_0-s_L)/s_L_0,  (triggered adjustment)
            max(0, (R - R_0)/C)                      (revenue cap)
        }

    For initial implementation, R_0 = 0 (no baseline AI revenue), so cap = R/C.

    Inputs:
        M_base: Base reference floor (NIS/mo)
        adjustment_triggered: Economic trigger adjustment (from calc_adjustment_triggered)
        revenue: Available revenue (billion NIS)
        deciles: List of dicts with 'equiv_income', 'std_persons', 'households'
        taper: Taper rate
        disregard: Disregard amount (NIS/mo)
        balance_threshold: Minimum balance ratio to allow adjustment (default -3%)

    Output: Dict with floor, adjustment_used, cost, balance, capped (bool)
    """
    from scipy.optimize import brentq

    # Calculate uncapped floor
    M_uncapped = M_base * (1 + adjustment_triggered)

    # Calculate cost at uncapped floor
    cost_uncapped = calc_nit_cost_from_deciles(M_uncapped, deciles, taper, disregard)

    # Check balance at uncapped floor
    balance_uncapped = revenue - cost_uncapped
    balance_ratio = balance_uncapped / revenue if revenue > 0 else -1

    # If balance is acceptable, use uncapped floor
    if balance_ratio >= balance_threshold:
        return {
            'floor': M_uncapped,
            'adjustment_used': adjustment_triggered,
            'cost': cost_uncapped,
            'balance': balance_uncapped,
            'capped': False,
            'balance_ratio': balance_ratio
        }

    # Otherwise, find the floor where cost = revenue (balance = 0)
    # This is the maximum affordable floor
    def cost_minus_revenue(M):
        return calc_nit_cost_from_deciles(M, deciles, taper, disregard) - revenue

    # Search for M where cost = revenue
    # Lower bound: some minimum floor, upper bound: uncapped floor
    try:
        M_capped = brentq(cost_minus_revenue, 100, M_uncapped)
    except ValueError:
        # If even M=100 costs more than revenue, set to minimum
        M_capped = 100

    cost_capped = calc_nit_cost_from_deciles(M_capped, deciles, taper, disregard)
    adjustment_capped = (M_capped / M_base) - 1

    return {
        'floor': M_capped,
        'adjustment_used': adjustment_capped,
        'cost': cost_capped,
        'balance': revenue - cost_capped,
        'capped': True,
        'balance_ratio': 0.0  # By construction, balance ≈ 0
    }


def calc_deflation_from_alpha(
    alpha: float,
    bls_deflation: float = 0.05,
    alpha_bls: float = 1.4,
    ai_exposure: float = 0.40
) -> float:
    """Derive deflation rate from TFP multiplier.

    BLS software deflation (5%) is the baseline floor. AI-driven deflation
    adds on top, scaled by ai_exposure (sensitivity to excess TFP).

    Formula:
        - Baseline (α ≤ α_BLS): δ = δ_BLS
        - Above baseline: δ = δ_BLS × [1 + ai_exposure × (α - α_BLS) / (α_BLS - 1)]

    With ai_exposure = 40%:
        - Baseline (α=1.4): 5%
        - Strong (α=1.6): 6%
        - AGI (α=2.2): 9%

    Args:
        alpha: TFP multiplier (1.4 = baseline, 1.6 = strong, 2.2 = AGI)
        bls_deflation: BLS historical software deflation rate (default 5%)
        alpha_bls: Alpha value corresponding to BLS rate (default 1.4 = baseline)
        ai_exposure: Sensitivity of deflation to excess TFP (default 40%)

    Returns:
        Deflation rate as decimal (e.g., 0.05 for 5%)
    """
    if alpha <= alpha_bls:
        return bls_deflation
    excess = (alpha - alpha_bls) / (alpha_bls - 1)
    return bls_deflation * (1 + ai_exposure * excess)


def calc_vat_adjustment(v0: float, delta: float) -> float:
    """Price-neutral VAT rate after AI deflation.

    Formula: v1 = (1+v0)/(1-delta) - 1

    Inputs:
        v0: Initial VAT rate (decimal, e.g., 0.18)
        delta: Deflation rate (decimal, e.g., 0.03)

    Output: New VAT rate (decimal)
    """
    return (1 + v0) / (1 - delta) - 1


def calc_vat_revenue(Y: float, consumption_share: float, v0: float, delta: float,
                     passthrough: float) -> float:
    """Additional VAT revenue from dynamic VAT.

    Formula: Revenue = C × Δv × β
             where C = consumption_share × Y

    Inputs:
        Y: GDP (billion NIS) - use Y₁ for post-AI scenarios
        consumption_share: Private consumption as share of GDP (decimal, e.g., 0.539)
        v0: Initial VAT rate (decimal, e.g., 0.18)
        delta: Deflation rate (decimal)
        passthrough: VAT pass-through rate (β) - from 2.4a event study calculation

    Output: Additional revenue (billion NIS)
    """
    v1 = calc_vat_adjustment(v0, delta)
    C = consumption_share * Y  # Consumption in billion NIS
    return C * (v1 - v0) * passthrough


def calc_ringfenced_profit(baseline: float, actual: float,
                           attribution: float, earmark: float) -> float:
    """Ring-fenced profit for NIT fund.

    Formula: Ringfenced = (actual - baseline) * attribution * earmark

    Inputs:
        baseline: Pre-AI trend revenue (billion NIS)
        actual: Post-AI actual revenue (billion NIS)
        attribution: AI attribution coefficient (decimal)
        earmark: Earmark percentage (decimal)

    Output: Ring-fenced amount (billion NIS)
    """
    over_trend = actual - baseline
    ai_attributed = over_trend * attribution
    return ai_attributed * earmark


def calc_gov_ringfence(personnel_cost: float, unemployment_rate: float,
                       ringfence_rate: float = 0.75) -> float:
    """Ring-fenced government wage savings.

    Formula: R_gov = Personnel × u × ρ

    Assumes public sector automation mirrors economy-wide displacement:
    if u% of the economy's labor is displaced by AI, then u% of public
    sector labor costs are saved. The ring-fence rate ρ determines
    what share is earmarked for the NIT fund.

    Inputs:
        personnel_cost: Total public sector wages (billion NIS)
        unemployment_rate: Economy-wide unemployment rate (decimal)
        ringfence_rate: Share earmarked for NIT fund (decimal, default 0.75)

    Output: Ring-fenced government savings contribution to NIT fund (billion NIS)
    """
    gross_savings = personnel_cost * unemployment_rate
    return gross_savings * ringfence_rate


def calc_automation_surplus(Y0: float, Y1: float, s_L: float, s_L_prime: float) -> float:
    """Automation surplus.

    Formula: dS = s_L' * Y1 - s_L * Y0

    Inputs:
        Y0: Pre-automation GDP (billion NIS)
        Y1: Post-automation GDP (billion NIS)
        s_L: Pre-automation labor share
        s_L_prime: Post-automation labor share

    Output: Automation surplus (billion NIS, positive = net gain)
    """
    return s_L_prime * Y1 - s_L * Y0


def calc_gap_closure_pct(delta_S: float, delta_Pi: float) -> float:
    """Percentage of capital windfall needed to close labor income gap.

    Formula: gap_closure_pct = |ΔS| / ΔΠ × 100 when ΔS < 0
             0 when ΔS >= 0 (no gap to close)

    Inputs:
        delta_S: Labor income change (billion NIS, negative = loss)
        delta_Pi: Capital windfall (billion NIS)

    Output: Percentage of capital windfall needed (0-100)
    """
    if delta_S >= 0:
        return 0.0
    return abs(delta_S) / delta_Pi * 100


def calc_capital_windfall(Y0: float, Y1: float, s_L_prime: float) -> float:
    """Capital windfall from AI-driven growth.

    Formula: ΔΠ = s_K' × (Y₁ - Y₀)

    This is capital's share of OUTPUT GROWTH only, not total capital income change.
    It represents the capital income that wouldn't exist without AI-driven growth.

    Note: An alternative formula (ΔΠ = s_K'Y₁ - s_KY₀) measures total capital income
    change, but that includes redistribution from labor, not just new value creation.

    Inputs:
        Y0: Pre-automation GDP (billion NIS)
        Y1: Post-automation GDP (billion NIS)
        s_L_prime: Post-automation labor share

    Output: Capital windfall (billion NIS)
    """
    s_K_prime = 1 - s_L_prime
    delta_Y = Y1 - Y0
    return s_K_prime * delta_Y


def calc_poverty_reduction(initial: float, final: float) -> float:
    """Calculate percentage reduction in poverty measure.

    Formula: reduction = (initial - final) / initial * 100

    Inputs:
        initial: Initial poverty rate (%)
        final: Final poverty rate (%)

    Output: Percentage reduction (%)
    """
    return (initial - final) / initial * 100


def calc_poverty_gap(headcount_ratio: float, mean_poor: float,
                     poverty_line: float) -> float:
    """Calculate poverty gap index (FGT-1).

    Formula: PG = H * I
    Where:
        H = headcount ratio (fraction below poverty line)
        I = income gap ratio = (z - mu_poor) / z
        z = poverty line
        mu_poor = mean income of the poor

    Equivalent to: PG = (1/N) * sum_i max(0, (z - y_i)/z)

    Inputs:
        headcount_ratio: H, fraction of population below poverty line
        mean_poor: mu_poor, mean income of those below poverty line
        poverty_line: z, the poverty threshold

    Output: Poverty gap index as percentage (0-100)
    """
    if poverty_line <= 0:
        raise ValueError("Poverty line must be positive")
    income_gap_ratio = (poverty_line - mean_poor) / poverty_line
    return headcount_ratio * income_gap_ratio * 100


def calc_poverty_gap_post_nit(pre_headcount: float, pre_mean_poor: float,
                               poverty_line: float, floor: float,
                               post_headcount: float) -> float:
    """Estimate post-NIT poverty gap.

    Under NIT with floor M:
    - Those remaining poor have income close to (but below) poverty line
    - Residual poverty gap reflects behavioral non-take-up

    Simplified model: Assume remaining poor have mean income = M
    (floor lifts most incomes, residual poverty from edge cases)

    Inputs:
        pre_headcount: Initial headcount ratio (decimal)
        pre_mean_poor: Initial mean income of poor
        poverty_line: Poverty line
        floor: NIT floor (guaranteed minimum)
        post_headcount: Post-NIT headcount ratio (decimal)

    Output: Estimated post-NIT poverty gap (%)
    """
    # Post-NIT, remaining poor are edge cases near the floor
    # Approximate their mean income as slightly below poverty line
    # (since floor ~= 60% median, and poverty line = 50% median)
    if post_headcount <= 0:
        return 0.0
    # Residual poor have income gap only from behavioral factors
    # Estimate mean of residual poor as 90% of poverty line
    mean_residual_poor = 0.90 * poverty_line
    income_gap = (poverty_line - mean_residual_poor) / poverty_line
    return post_headcount * income_gap * 100


def calc_gini_from_deciles(deciles: list[dict], income_key: str = 'equiv_income',
                           weight_key: str = 'households') -> float:
    """Calculate Gini coefficient from decile data using trapezoid rule.

    Formula: G = 1 - 2 * (Area under Lorenz curve)
             G = 1 - sum (x_i - x_{i-1}) * (y_i + y_{i-1})

    Where:
        x_i = cumulative population share
        y_i = cumulative income share

    Inputs:
        deciles: List of dicts with income and weight keys, sorted by income
        income_key: Key for income values
        weight_key: Key for population weights

    Output: Gini coefficient (0-1)
    """
    # Sort by income
    sorted_deciles = sorted(deciles, key=lambda d: d[income_key])

    # Calculate total income and population
    total_income = sum(d[income_key] * d[weight_key] for d in sorted_deciles)
    total_pop = sum(d[weight_key] for d in sorted_deciles)

    # Build cumulative shares
    cum_pop = 0
    cum_income = 0
    lorenz_points = [(0, 0)]  # Start at origin

    for d in sorted_deciles:
        cum_pop += d[weight_key]
        cum_income += d[income_key] * d[weight_key]
        x = cum_pop / total_pop
        y = cum_income / total_income
        lorenz_points.append((x, y))

    # Calculate area under Lorenz curve using trapezoid rule
    area = 0
    for i in range(1, len(lorenz_points)):
        x0, y0 = lorenz_points[i-1]
        x1, y1 = lorenz_points[i]
        area += (x1 - x0) * (y0 + y1) / 2

    # Gini = 1 - 2 * area
    gini = 1 - 2 * area
    return gini


def calc_poverty_gap_from_deciles(deciles: list, poverty_line: float,
                                   income_key: str = 'equiv_income',
                                   weight_key: str = 'households') -> float:
    """Calculate poverty gap index from decile data.

    Formula: PG = (1/N) * sum_i max(0, (z - y_i)/z) * w_i

    This measures the average depth of poverty as a fraction of the poverty line.
    NOTE: Using decile averages underestimates true poverty gap since it loses
    within-decile variation. Use for relative comparisons only.

    Inputs:
        deciles: List of dicts with income and household count
        poverty_line: z, the poverty threshold
        income_key: Key for income in decile dict (default 'equiv_income')
        weight_key: Key for household count in decile dict (default 'households')

    Output: Poverty gap index (0 to 1, NOT percentage)
    """
    total_gap = 0.0
    total_pop = 0.0

    for d in deciles:
        y = d[income_key]
        w = d[weight_key]
        total_pop += w

        if y < poverty_line:
            shortfall = (poverty_line - y) / poverty_line
            total_gap += shortfall * w

    return total_gap / total_pop if total_pop > 0 else 0.0


# =============================================================================
# MICROSIMULATION FUNCTIONS
# =============================================================================

def calc_poverty_gap_fgt1(incomes: list, poverty_line: float) -> float:
    """Calculate poverty gap index using FGT-1 formula.

    Formula: PG = (1/N) * sum_i max(0, (z - y_i)/z)

    Parameters:
        N = len(incomes) - total population
        q = number with income < poverty_line
        z = poverty_line
        y_i = income of household i

    Inputs:
        incomes: List of all household incomes
        poverty_line: z, the poverty threshold

    Output: Poverty gap as percentage (0-100)
    """
    N = len(incomes)
    if N == 0:
        return 0.0

    total_gap = 0.0
    for y in incomes:
        if y < poverty_line:
            total_gap += (poverty_line - y) / poverty_line

    return (total_gap / N) * 100


def calc_nit_benefit(income: float, floor: float, taper: float,
                     disregard: float) -> float:
    """Calculate NIT benefit for a household.

    Formula: B = max(0, M - tau * max(0, y - D))

    Parameters:
        y = income
        M = floor (guaranteed minimum)
        tau = taper rate
        D = disregard

    Inputs:
        income: Household income (y)
        floor: Guaranteed minimum (M)
        taper: Withdrawal rate (tau)
        disregard: Earnings exemption (D)

    Output: NIT benefit amount (NIS/mo)
    """
    taxable_income = max(0, income - disregard)
    benefit = floor - taper * taxable_income
    return max(0, benefit)


def calc_median_from_deciles(deciles: list[dict], income_key: str = 'equiv_income',
                             weight_key: str = 'households') -> float:
    """Calculate median income from decile data via interpolation.

    Inputs:
        deciles: List of dicts with income and weight keys
        income_key: Key for income values
        weight_key: Key for population weights

    Output: Interpolated median income
    """
    sorted_d = sorted(deciles, key=lambda d: d[income_key])
    total = sum(d[weight_key] for d in sorted_d)
    cum = 0
    for i, d in enumerate(sorted_d):
        prev = cum
        cum += d[weight_key]
        if cum >= total / 2:
            if i > 0:
                prev_inc = sorted_d[i-1][income_key]
            else:
                prev_inc = 0
            frac = (total/2 - prev) / d[weight_key]
            return prev_inc + frac * (d[income_key] - prev_inc)
    return sorted_d[-1][income_key]


def generate_lognormal_incomes(mean: float, lower: float, upper: float,
                                n_samples: int, rng=None) -> list:
    """Generate log-normal distributed incomes within bounds.

    Fits log-normal to match mean and approximate bounds.

    Inputs:
        mean: Target mean income
        lower: Lower bound (decile min)
        upper: Upper bound (decile max)
        n_samples: Number of samples to generate
        rng: numpy random generator (optional)

    Output: List of income samples
    """
    import numpy as np

    if rng is None:
        rng = np.random.default_rng(42)  # Reproducible

    # Estimate log-normal parameters
    if upper > lower and lower > 0:
        sigma = np.log(upper / lower) / 4
    else:
        sigma = 0.3

    mu = np.log(max(mean, 1)) - sigma**2 / 2

    samples = rng.lognormal(mu, sigma, n_samples)
    samples = np.clip(samples, lower, upper)
    return samples.tolist()


# -----------------------------------------------------------------------------
# Within-Decile Distribution Functions
# -----------------------------------------------------------------------------

def estimate_beta_concentration(gross_means: list) -> float:
    """Estimate Beta concentration parameter from cross-decile income variance.

    For Beta on [L,U]: Var(Y) = (U-L)² × m(1-m) / (c+1)
    Solving: c = (U-L)² × m(1-m) / Var(Y) - 1

    Uses cross-decile variance as proxy for within-decile variance.

    Inputs:
        gross_means: List of equivalized gross income means per decile (D1-D10)

    Output: Estimated concentration parameter c = α + β
    """
    import numpy as np

    var_gross = np.var(gross_means)
    range_gross = max(gross_means) - min(gross_means)

    # Typical normalized mean across deciles (assume ~0.5 for estimation)
    m = 0.5

    # Estimate c from: Var = (U-L)² × m(1-m) / (c+1)
    # => c = (U-L)² × m(1-m) / Var - 1
    c = (range_gross ** 2) * m * (1 - m) / var_gross - 1

    return max(2.0, c)  # Ensure c >= 2 for reasonable distribution


def estimate_gamma_shape(std_persons_means: list) -> float:
    """Estimate Gamma shape parameter from cross-decile std_persons variance.

    For Gamma: mean = α × β, variance = α × β²
    Solving: α = mean² / variance

    Uses cross-decile variance as proxy for within-decile variance.

    Inputs:
        std_persons_means: List of mean std_persons per decile (D1-D10)

    Output: Estimated shape parameter α
    """
    import numpy as np

    mean_overall = np.mean(std_persons_means)
    var_overall = np.var(std_persons_means)

    if var_overall <= 0:
        return 10.0  # Default if no variance

    alpha = mean_overall ** 2 / var_overall
    return max(2.0, alpha)  # Ensure α >= 2


def generate_beta_incomes(n: int, lower: float, upper: float, mean: float,
                          concentration: float, rng=None) -> "np.ndarray":
    """Generate Beta-distributed incomes on [lower, upper] matching mean.

    Inputs:
        n: Number of samples
        lower: Decile lower bound (gross income)
        upper: Decile upper bound (gross income)
        mean: Decile mean (gross income)
        concentration: α + β parameter (from estimate_beta_concentration)
        rng: numpy random generator

    Output: Array of n income values in [lower, upper]
    """
    import numpy as np

    if rng is None:
        rng = np.random.default_rng()

    # Handle edge case where bounds are invalid
    if upper <= lower:
        return np.full(n, mean)

    # Normalized mean
    m = (mean - lower) / (upper - lower)
    m = np.clip(m, 0.01, 0.99)  # Avoid edge cases

    # Beta parameters
    alpha = m * concentration
    beta = (1 - m) * concentration

    # Sample and transform to [lower, upper]
    x = rng.beta(alpha, beta, n)
    return lower + (upper - lower) * x


def generate_gamma_std_persons(n: int, mean: float, shape: float,
                                rng=None) -> "np.ndarray":
    """Generate Gamma-distributed household sizes matching mean.

    Inputs:
        n: Number of samples
        mean: Decile mean std_persons
        shape: Gamma shape parameter α (from estimate_gamma_shape)
        rng: numpy random generator

    Output: Array of n std_persons values
    """
    import numpy as np

    if rng is None:
        rng = np.random.default_rng()

    # Scale parameter to match decile mean: β = mean / α
    scale = mean / shape
    return rng.gamma(shape, scale, n)


def get_decile_bounds_gross(decile: int, upper_net: list,
                            gross_means: list, net_means: list) -> tuple:
    """Convert NET income bounds to GROSS income bounds using gross/net ratio.

    CBS provides upper limits for NET income per standard person (used to sort
    households into deciles). This function converts to GROSS income bounds
    for NIT benefit calculation.

    Inputs:
        decile: Decile number (1-10)
        upper_net: CBS upper limits for NET income [D1, D2, ..., D10] (D10 can be None)
        gross_means: Equivalized gross income means per decile
        net_means: Equivalized net income means per decile

    Output: (lower_gross, upper_gross) tuple
    """
    i = decile - 1

    # Gross/net ratio for this decile
    ratio = gross_means[i] / net_means[i]

    # Lower bound = upper bound of previous decile (converted to gross)
    if decile == 1:
        lower = 0
    else:
        prev_ratio = gross_means[i - 1] / net_means[i - 1]
        lower = upper_net[i - 1] * prev_ratio

    # Upper bound = CBS upper limit converted to gross
    if decile == 10 or upper_net[i] is None:
        # Extrapolate for D10 (unbounded in CBS)
        upper = gross_means[9] + (gross_means[9] - gross_means[8])
    else:
        upper = upper_net[i] * ratio

    return lower, upper


def generate_decile_households(n: int, mean_income: float, lower_income: float,
                                upper_income: float, mean_std_persons: float,
                                beta_concentration: float, gamma_shape: float,
                                rng=None) -> list:
    """Generate synthetic households within a decile.

    Samples income from Beta distribution and std_persons from Gamma distribution,
    independently.

    Inputs:
        n: Number of households to generate
        mean_income: Decile average gross income (equivalized)
        lower_income: Decile lower bound (gross)
        upper_income: Decile upper bound (gross)
        mean_std_persons: Decile average household size (OECD scale)
        beta_concentration: Concentration parameter for Beta distribution
        gamma_shape: Shape parameter for Gamma distribution
        rng: numpy random generator

    Output: List of dicts with 'equiv_gross' and 'std_persons' for each household
    """
    import numpy as np

    if rng is None:
        rng = np.random.default_rng(42)  # Reproducible

    incomes = generate_beta_incomes(n, lower_income, upper_income, mean_income,
                                     beta_concentration, rng)
    std_persons = generate_gamma_std_persons(n, mean_std_persons, gamma_shape, rng)

    return [{'equiv_gross': float(inc), 'std_persons': float(sp)}
            for inc, sp in zip(incomes, std_persons)]


# =============================================================================
# CES PRODUCTION FUNCTION (Option 2)
# =============================================================================

def ces_output(A: float, beta: float, K: float, L: float, sigma: float) -> float:
    """CES production function output.

    Formula: Y = A × [β × K^ρ + (1-β) × L^ρ]^(1/ρ)
    where ρ = 1 - 1/σ

    Inputs:
        A: Total factor productivity
        beta: Capital distribution parameter
        K: Capital stock
        L: Labor input
        sigma: Elasticity of substitution (σ > 0)

    Output: GDP (Y)
    """
    if sigma == 1:
        # Cobb-Douglas limit
        return A * (K ** beta) * (L ** (1 - beta))
    rho = 1 - 1 / sigma
    inner = beta * (K ** rho) + (1 - beta) * (L ** rho)
    return A * (inner ** (1 / rho))


def ces_labor_share(beta: float, K: float, L: float, sigma: float) -> float:
    """Labor share under CES production.

    Formula: sL = (1-β) × L^ρ / [β × K^ρ + (1-β) × L^ρ]

    Inputs:
        beta: Capital distribution parameter
        K: Capital stock
        L: Labor input
        sigma: Elasticity of substitution

    Output: Labor share (0-1)
    """
    if sigma == 1:
        return 1 - beta
    rho = 1 - 1 / sigma
    K_term = beta * (K ** rho)
    L_term = (1 - beta) * (L ** rho)
    return L_term / (K_term + L_term)


def ces_marginal_product_K(A: float, beta: float, K: float, L: float,
                            sigma: float) -> float:
    """Marginal product of capital under CES.

    Formula: MPK = β × A^ρ × (Y/K)^(1-ρ)

    Inputs:
        A: Total factor productivity
        beta: Capital distribution parameter
        K: Capital stock
        L: Labor input
        sigma: Elasticity of substitution

    Output: Marginal product of capital
    """
    Y = ces_output(A, beta, K, L, sigma)
    if sigma == 1:
        return beta * Y / K
    rho = 1 - 1 / sigma
    return beta * (A ** rho) * ((Y / K) ** (1 - rho))


def ces_marginal_product_L(A: float, beta: float, K: float, L: float,
                            sigma: float) -> float:
    """Marginal product of labor under CES.

    Formula: MPL = (1-β) × A^ρ × (Y/L)^(1-ρ)

    Inputs:
        A: Total factor productivity
        beta: Capital distribution parameter
        K: Capital stock
        L: Labor input
        sigma: Elasticity of substitution

    Output: Marginal product of labor
    """
    Y = ces_output(A, beta, K, L, sigma)
    if sigma == 1:
        return (1 - beta) * Y / L
    rho = 1 - 1 / sigma
    return (1 - beta) * (A ** rho) * ((Y / L) ** (1 - rho))


def calibrate_ces_period_0(Y_0: float, L_0: float, s_L_target: float,
                            sigma: float, K_0: float = None) -> dict:
    """Calibrate CES parameters (A, β) to match initial conditions.

    Given: Y₀, L₀, sL, σ, and optionally K₀
    Find: A, β (and K₀ if not provided)

    Method:
    1. If K₀ not provided, derive from capital share: K₀ = (sK × Y₀) / r
       where r is the interest rate (assumed 5%)
    2. Solve for β from labor share condition:
       sL = (1-β) × L^ρ / [β × K^ρ + (1-β) × L^ρ]
    3. Solve for A from output condition:
       Y₀ = A × [β × K^ρ + (1-β) × L^ρ]^(1/ρ)

    Inputs:
        Y_0: Initial GDP
        L_0: Initial labor (employment)
        s_L_target: Target labor share
        sigma: Elasticity of substitution
        K_0: Initial capital (optional, will be estimated if None)

    Output: Dict with A, beta, K_0, verification values
    """
    from scipy.optimize import fsolve
    import numpy as np

    s_K_target = 1 - s_L_target
    rho = 1 - 1 / sigma

    # Step 1: Estimate K_0 if not provided
    if K_0 is None:
        # Use capital-output ratio method
        # Typical K/Y ratio for developed economy: ~3
        # Or derive from: r × K = sK × Y → K = sK × Y / r
        r = 0.05  # Assumed real interest rate
        K_0 = s_K_target * Y_0 / r  # This gives K in same units as Y

    # Step 2: Solve for β from labor share condition
    # sL = (1-β) × L^ρ / [β × K^ρ + (1-β) × L^ρ]
    # Rearranging: sL × [β × K^ρ + (1-β) × L^ρ] = (1-β) × L^ρ
    # sL × β × K^ρ + sL × (1-β) × L^ρ = (1-β) × L^ρ
    # sL × β × K^ρ = (1-β) × L^ρ × (1 - sL)
    # sL × β × K^ρ = (1-β) × L^ρ × sK
    # β / (1-β) = sK × L^ρ / (sL × K^ρ)
    # β = sK × L^ρ / (sL × K^ρ + sK × L^ρ)

    L_rho = L_0 ** rho
    K_rho = K_0 ** rho

    beta = s_K_target * L_rho / (s_L_target * K_rho + s_K_target * L_rho)

    # Step 3: Solve for A from output condition
    inner = beta * K_rho + (1 - beta) * L_rho
    A = Y_0 / (inner ** (1 / rho))

    # Verify
    Y_check = ces_output(A, beta, K_0, L_0, sigma)
    sL_check = ces_labor_share(beta, K_0, L_0, sigma)

    return {
        'A': A,
        'beta': beta,
        'K_0': K_0,
        'sigma': sigma,
        'rho': rho,
        'Y_0_target': Y_0,
        'Y_0_check': Y_check,
        'sL_target': s_L_target,
        'sL_check': sL_check,
        'MPK': ces_marginal_product_K(A, beta, K_0, L_0, sigma),
        'MPL': ces_marginal_product_L(A, beta, K_0, L_0, sigma),
    }


def calc_growth_ces(alpha: float, s_L_prime: float, u: float,
                    Y_0: float, s_L: float, L_force: float, u_0: float,
                    sigma: float, K_0: float) -> dict:
    """Calculate Y₁ and g_Y using CES production function.

    Method:
    1. Calibrate CES to period 0 (find A, β)
    2. Apply TFP shock: A₁ = α × A₀
    3. Compute L₁ from unemployment
    4. K₁ = K₀ (capital constant)
    5. Compute Y₁ = CES(A₁, β, K₁, L₁)

    Inputs:
        alpha: TFP multiplier (A₁/A₀)
        s_L_prime: Post-automation labor share (for surplus calc)
        u: Post-automation unemployment rate (decimal)
        Y_0: Initial GDP (B NIS)
        s_L: Initial labor share
        L_force: Labor force (millions)
        u_0: Natural unemployment rate (decimal)
        sigma: Elasticity of substitution
        K_0: Initial capital stock (B NIS, from Penn World Table)

    Output: Dict with Y_1, g_Y, calibration results
    """
    # Period 0 calibration
    L_0 = L_force * (1 - u_0)
    calib = calibrate_ces_period_0(Y_0, L_0, s_L, sigma, K_0)

    A_0 = calib['A']
    beta = calib['beta']

    # Period 1: TFP increases, K constant, L falls with unemployment
    A_1 = alpha * A_0
    L_1 = L_force * (1 - u)
    K_1 = K_0  # Capital constant

    Y_1 = ces_output(A_1, beta, K_1, L_1, sigma)
    g_Y = (Y_1 - Y_0) / Y_0
    sL_1 = ces_labor_share(beta, K_1, L_1, sigma)

    # Automation surplus
    delta_S = s_L_prime * Y_1 - s_L * Y_0

    return {
        'Y_1': Y_1,
        'g_Y': g_Y,
        'g_Y_pct': g_Y * 100,
        'delta_S': delta_S,
        'A_0': A_0,
        'A_1': A_1,
        'beta': beta,
        'K_0': K_0,
        'K_1': K_1,
        'L_0': L_0,
        'L_1': L_1,
        'sL_0': calib['sL_check'],
        'sL_1': sL_1,
        'calibration': calib,
    }


# =============================================================================
# PROFIT TREND CALCULATION FUNCTIONS
# =============================================================================

def calc_profit_trend_linear(years: list[int], revenues: list[float],
                              target_year: int) -> dict:
    """Linear extrapolation of pre-AI profit trend.

    Formula: Revenue_t = alpha + beta * t
    Extrapolate: Baseline = alpha + beta * target_year

    Inputs:
        years: List of years for fitting (e.g., [2015, 2016, 2017, 2018, 2019])
        revenues: Corporate tax revenue in B NIS for each year
        target_year: Year to extrapolate to (e.g., 2024)

    Output: Dict with slope, intercept, r_squared, baseline
    """
    from scipy import stats
    import numpy as np

    slope, intercept, r_value, p_value, std_err = stats.linregress(years, revenues)
    r_squared = r_value ** 2
    baseline = intercept + slope * target_year

    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'p_value': p_value,
        'std_err': std_err,
        'baseline': baseline,
        'target_year': target_year,
    }


def calc_profit_trend_gdp_linked(revenues: list[float], gdps: list[float],
                                  current_gdp: float) -> dict:
    """GDP-linked profit trend baseline.

    Formula: profit_share = mean(Revenue_t / GDP_t)
             Baseline = profit_share * current_GDP

    Inputs:
        revenues: Corporate tax revenue in B NIS for each year
        gdps: GDP in B NIS for each year (same length as revenues)
        current_gdp: Current GDP in B NIS (e.g., 1694)

    Output: Dict with shares, avg_share, baseline
    """
    import numpy as np

    shares = [r / g for r, g in zip(revenues, gdps)]
    avg_share = np.mean(shares)
    baseline = avg_share * current_gdp

    return {
        'shares': shares,
        'avg_share': avg_share,
        'avg_share_pct': avg_share * 100,
        'baseline': baseline,
        'current_gdp': current_gdp,
    }


def calc_cg_trend_linear(years: list[int], revenues: list[float],
                         target_year: int, exclude_years: list[int] = None) -> dict:
    """Linear extrapolation of capital gains tax trend.

    Similar to calc_profit_trend_linear but allows excluding outlier years.

    Formula: Revenue_t = alpha + beta * t
    Extrapolate: Baseline = alpha + beta * target_year

    Inputs:
        years: List of years for fitting (e.g., [2012, 2013, ..., 2019])
        revenues: Capital gains tax revenue in B NIS for each year
        target_year: Year to extrapolate to (e.g., 2024)
        exclude_years: Years to exclude from regression (e.g., [2017] for outliers)

    Output: Dict with slope, intercept, r_squared, baseline
    """
    from scipy import stats
    import numpy as np

    # Filter out excluded years
    if exclude_years:
        filtered = [(y, r) for y, r in zip(years, revenues) if y not in exclude_years]
        years_fit = [y for y, r in filtered]
        revenues_fit = [r for y, r in filtered]
    else:
        years_fit = years
        revenues_fit = revenues

    slope, intercept, r_value, p_value, std_err = stats.linregress(years_fit, revenues_fit)
    r_squared = r_value ** 2
    baseline = intercept + slope * target_year

    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'p_value': p_value,
        'std_err': std_err,
        'baseline': baseline,
        'target_year': target_year,
        'years_used': years_fit,
        'excluded_years': exclude_years or [],
    }


def calc_cg_trend_gdp_linked(revenues: list[float], gdps: list[float],
                             current_gdp: float, exclude_indices: list[int] = None) -> dict:
    """GDP-linked capital gains tax trend baseline.

    Formula: cg_share = mean(Revenue_t / GDP_t)
             Baseline = cg_share * current_GDP

    Inputs:
        revenues: Capital gains tax revenue in B NIS for each year
        gdps: GDP in B NIS for each year (same length as revenues)
        current_gdp: Current GDP in B NIS (e.g., 1694)
        exclude_indices: Indices to exclude from average (e.g., [5] for 2017)

    Output: Dict with shares, avg_share, baseline
    """
    import numpy as np

    shares = [r / g for r, g in zip(revenues, gdps)]

    # Filter out excluded indices
    if exclude_indices:
        shares_fit = [s for i, s in enumerate(shares) if i not in exclude_indices]
    else:
        shares_fit = shares

    avg_share = np.mean(shares_fit)
    baseline = avg_share * current_gdp

    return {
        'shares': shares,
        'shares_used': shares_fit,
        'avg_share': avg_share,
        'avg_share_pct': avg_share * 100,
        'baseline': baseline,
        'current_gdp': current_gdp,
        'excluded_indices': exclude_indices or [],
    }


def calc_ringfenced_cg(baseline: float, g_Y: float,
                       attribution: float, earmark: float) -> dict:
    """Ring-fenced capital gains tax for NIT fund.

    Formula: Ringfenced = (CG_post - CG_baseline) * attribution * earmark
             CG_post = CG_baseline * (1 + g_Y)

    Capital gains tax revenue scales with GDP growth (conservative assumption).

    Inputs:
        baseline: Pre-AI trend CG tax revenue (billion NIS)
        g_Y: GDP growth rate (decimal, e.g., 0.38 for 38%)
        attribution: AI attribution coefficient (decimal, e.g., 1.0)
        earmark: Earmark percentage (decimal, e.g., 0.75)

    Output: Dict with post_ai, over_trend, ringfenced
    """
    post_ai = baseline * (1 + g_Y)
    over_trend = post_ai - baseline
    ai_attributed = over_trend * attribution
    ringfenced = ai_attributed * earmark

    return {
        'baseline': baseline,
        'g_Y': g_Y,
        'post_ai': post_ai,
        'over_trend': over_trend,
        'attribution': attribution,
        'earmark': earmark,
        'ringfenced': ringfenced,
    }


# =============================================================================
# PROGRAM CONSOLIDATION FUNCTIONS
# =============================================================================

def calc_consolidation_revenue(income_support: float, unemployment: float,
                                admin_rate: float = 0.03) -> dict:
    """Program consolidation revenue from merging legacy transfers into NIT.

    Formula: Consolidation = IncomeSupport + Unemployment + AdminSavings
             AdminSavings = (IncomeSupport + Unemployment) × AdminRate

    Components:
    1. Income Support (הבטחת הכנסה): Fully replaced by NIT
    2. Unemployment (דמי אבטלה): Fully replaced by NIT
    3. Admin Savings: Reduced overhead from unified eligibility system

    Inputs:
        income_support: Annual income support spending (billion NIS)
        unemployment: Annual unemployment benefits spending (billion NIS)
        admin_rate: Administrative savings rate (default 3%)

    Output: Dict with components and total consolidation revenue

    Source: BTL2023 Annual Report (Bituach Leumi)
    """
    # Programs fully redirected to NIT
    redirected = income_support + unemployment

    # Administrative savings from unified system
    admin_savings = redirected * admin_rate

    # Total consolidation revenue
    total = redirected + admin_savings

    return {
        'income_support': income_support,
        'unemployment': unemployment,
        'redirected_total': redirected,
        'admin_rate': admin_rate,
        'admin_savings': admin_savings,
        'total': total,
    }


# =============================================================================
# EMTR CALCULATION FUNCTIONS
# =============================================================================

def calc_emtr_by_decile(gross_income: list, net_income: list,
                        floor: float, taper: float, disregard: float) -> list:
    """Calculate EMTR by decile.

    EMTR = t + tau for those in phase-out region.

    The effective marginal tax rate combines the NIT taper with
    existing income taxation. Since the NIT taper applies to gross
    income (pre-tax), the rates are additive, not multiplicative.

    Inputs:
        gross_income: Gross income by decile (list of 10 values)
        net_income: Net income by decile (list of 10 values)
        floor: NIT floor M (NIS/mo)
        taper: Taper rate tau (e.g., 0.50)
        disregard: Disregard D (NIS/mo)

    Output: List of EMTR values (None for those above breakeven)
    """
    breakeven = disregard + floor / taper
    emtr_list = []

    for g, n in zip(gross_income, net_income):
        if g >= breakeven:
            emtr_list.append(None)  # Above breakeven, no NIT
        else:
            t = (g - n) / g if g > 0 else 0  # Effective tax rate
            emtr = t + taper
            emtr_list.append(emtr)

    return emtr_list


def calc_effective_tax_rate(gross: float, net: float) -> float:
    """Calculate effective income tax rate.

    Formula: t = (gross - net) / gross

    Inputs:
        gross: Gross income (before tax)
        net: Net income (after tax)

    Output: Effective tax rate (decimal, e.g., 0.10 for 10%)
    """
    if gross <= 0:
        return 0.0
    return (gross - net) / gross


# =============================================================================
# MICROSIMULATION-BASED FLOOR CALCULATION
# =============================================================================

def calc_nit_cost_microsim(M: float, cbs_deciles: list, decile_bounds: list,
                           beta_concentration: float, gamma_shape: float,
                           taper: float = 0.5, disregard: float = 1000,
                           n_samples: int = 1000, seed: int = 42) -> float:
    """Calculate NIT cost using microsimulation with within-decile distributions.

    Instead of using decile averages, this generates synthetic households
    with Beta-distributed incomes and Gamma-distributed household sizes,
    then calculates benefits at the household level.

    Inputs:
        M: Floor per standard person (NIS/mo)
        cbs_deciles: List of dicts with 'equiv_gross', 'std_persons', 'households'
        decile_bounds: List of dicts with 'lower', 'upper' for each decile
        beta_concentration: Concentration parameter for Beta distribution
        gamma_shape: Shape parameter for Gamma distribution
        taper: Taper rate (default 0.5)
        disregard: Disregard amount (NIS/mo, default 1000)
        n_samples: Number of synthetic households per decile (default 1000)
        seed: Random seed for reproducibility (default 42)

    Output: Annual cost (billion NIS)
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    breakeven = disregard + M / taper
    total_cost = 0

    for i, d in enumerate(cbs_deciles):
        bounds = decile_bounds[i]

        # Generate synthetic households for this decile
        synth_hh = generate_decile_households(
            n=n_samples,
            mean_income=d['equiv_gross'],
            lower_income=bounds['lower'],
            upper_income=bounds['upper'],
            mean_std_persons=d['std_persons'],
            beta_concentration=beta_concentration,
            gamma_shape=gamma_shape,
            rng=rng
        )

        # Calculate total benefit for synthetic households
        total_benefit_microsim = 0
        for hh in synth_hh:
            if hh['equiv_gross'] >= breakeven:
                benefit_per_std = 0
            else:
                taxable = max(0, hh['equiv_gross'] - disregard)
                benefit_per_std = max(0, M - taper * taxable)
            total_benefit_microsim += benefit_per_std * hh['std_persons']

        # Average benefit per household (microsim)
        avg_benefit_per_hh = total_benefit_microsim / n_samples

        # Annual cost for this decile
        decile_cost = avg_benefit_per_hh * d['households'] * 12 / 1e9
        total_cost += decile_cost

    return total_cost


def calc_floor_balanced_microsim(revenue: float, cbs_deciles: list,
                                  decile_bounds: list,
                                  beta_concentration: float, gamma_shape: float,
                                  taper: float = 0.5, disregard: float = 1000,
                                  n_samples: int = 1000, seed: int = 42) -> dict:
    """Calculate revenue-constrained floor using microsimulation.

    This is the microsim equivalent of calc_floor_balanced(). Instead of
    using decile averages, it generates synthetic households and finds
    the floor M where microsim_cost(M) = revenue.

    Because microsimulation captures within-decile variation, the resulting
    floor will typically be HIGHER than the decile-average method for the
    same revenue (microsim costs are lower for a given M).

    Inputs:
        revenue: Available revenue (billion NIS)
        cbs_deciles: List of dicts with 'equiv_gross', 'std_persons', 'households'
        decile_bounds: List of dicts with 'lower', 'upper' for each decile
        beta_concentration: Concentration parameter for Beta distribution
        gamma_shape: Shape parameter for Gamma distribution
        taper: Taper rate (default 0.5)
        disregard: Disregard amount (NIS/mo, default 1000)
        n_samples: Number of synthetic households per decile (default 1000)
        seed: Random seed for reproducibility (default 42)

    Output: Dict with:
        - floor: Revenue-constrained floor M* (NIS/mo)
        - cost: Annual cost at M* (billion NIS, ≈ revenue)
        - breakeven: Break-even income y* = D + M*/τ
        - balance: Revenue - Cost (≈ 0 by construction)
    """
    from scipy.optimize import brentq

    def cost_minus_revenue(M):
        cost = calc_nit_cost_microsim(
            M, cbs_deciles, decile_bounds,
            beta_concentration, gamma_shape,
            taper, disregard, n_samples, seed
        )
        return cost - revenue

    # Search for M where cost = revenue
    # Lower bound: 100 NIS/mo, upper bound: 50,000 NIS/mo
    try:
        M_balanced = brentq(cost_minus_revenue, 100, 50000, xtol=1.0)
    except ValueError:
        # If no solution in range, return edge case
        if cost_minus_revenue(100) > 0:
            M_balanced = 100  # Even minimum floor exceeds revenue
        else:
            M_balanced = 50000  # Revenue exceeds maximum floor cost

    cost = calc_nit_cost_microsim(
        M_balanced, cbs_deciles, decile_bounds,
        beta_concentration, gamma_shape,
        taper, disregard, n_samples, seed
    )
    breakeven = disregard + M_balanced / taper

    return {
        'floor': M_balanced,
        'cost': cost,
        'breakeven': breakeven,
        'balance': revenue - cost
    }


def calc_floor_with_takeup_and_cap(
    revenue: float,
    Y_1: float,
    cbs_deciles: list,
    decile_bounds: list,
    beta_concentration: float,
    gamma_shape: float,
    take_up: float = 0.85,
    gdp_buffer_pct: float = 0.03,
    taper: float = 0.5,
    disregard: float = 1000,
    n_samples: int = 1000,
    seed: int = 42
) -> dict:
    """Calculate floor with take-up rate and GDP buffer cap.

    This extends calc_floor_balanced_microsim() with:
    1. Take-up rate: Only take_up% of eligible households claim
       → Effective cost = Cost(M) × take_up
       → We can afford higher M for same revenue

    2. GDP buffer cap: If 100% take-up would exceed Revenue + 3% GDP,
       cap M to ensure fiscal safety margin

    Logic:
        Step 1: Find M where Cost(M) × take_up = Revenue
        Step 2: Check if Cost(M, 100%) > Revenue + gdp_buffer_pct × Y_1
        Step 3: If cap binds, reduce M; otherwise use revenue-constrained M

    Inputs:
        revenue: Available revenue (billion NIS)
        Y_1: Post-AI GDP (billion NIS) for cap calculation
        cbs_deciles: List of dicts with 'equiv_gross', 'std_persons', 'households'
        decile_bounds: List of dicts with 'lower', 'upper' for each decile
        beta_concentration: Concentration parameter for Beta distribution
        gamma_shape: Shape parameter for Gamma distribution
        take_up: Take-up rate (default 0.85 = 85%)
        gdp_buffer_pct: GDP buffer as fraction (default 0.03 = 3%)
        taper: Taper rate (default 0.5)
        disregard: Disregard amount (NIS/mo, default 1000)
        n_samples: Number of synthetic households per decile
        seed: Random seed for reproducibility

    Output: Dict with:
        - floor: Final floor M (NIS/mo), capped if necessary
        - floor_uncapped: Revenue-constrained floor (before cap check)
        - cost_at_takeup: Actual cost at take_up% (billion NIS, ≈ revenue)
        - cost_at_100pct: Cost if 100% take-up (billion NIS)
        - max_cost_100pct: Revenue + GDP buffer (billion NIS)
        - cap_binds: True if cap reduced the floor
        - breakeven: Break-even income y* = D + M/τ
    """
    from scipy.optimize import brentq

    # Helper to calculate microsim cost for a given floor
    def calc_cost(M):
        return calc_nit_cost_microsim(
            M, cbs_deciles, decile_bounds,
            beta_concentration, gamma_shape,
            taper, disregard, n_samples, seed
        )

    # Step 1: Find M where Cost(M) × take_up = Revenue
    # Rearranged: Cost(M) = Revenue / take_up
    target_cost_100pct = revenue / take_up

    def cost_minus_target(M):
        return calc_cost(M) - target_cost_100pct

    try:
        M_uncapped = brentq(cost_minus_target, 100, 50000, xtol=1.0)
    except ValueError:
        if cost_minus_target(100) > 0:
            M_uncapped = 100
        else:
            M_uncapped = 50000

    # Calculate cost at M_uncapped (100% take-up)
    cost_100pct_uncapped = calc_cost(M_uncapped)

    # Step 2: Calculate max allowable 100% cost
    max_cost_100pct = revenue + gdp_buffer_pct * Y_1

    # Step 3: Check if cap binds
    if cost_100pct_uncapped > max_cost_100pct:
        # Cap binds - find M where Cost(M) = max_cost_100pct
        def cost_minus_cap(M):
            return calc_cost(M) - max_cost_100pct

        try:
            M_final = brentq(cost_minus_cap, 100, M_uncapped, xtol=1.0)
        except ValueError:
            M_final = 100  # Fallback
        cap_binds = True
    else:
        M_final = M_uncapped
        cap_binds = False

    # Calculate final costs
    cost_100pct_final = calc_cost(M_final)
    cost_at_takeup = cost_100pct_final * take_up
    breakeven = disregard + M_final / taper

    return {
        'floor': M_final,
        'floor_uncapped': M_uncapped,
        'cost_at_takeup': cost_at_takeup,
        'cost_at_100pct': cost_100pct_final,
        'max_cost_100pct': max_cost_100pct,
        'cap_binds': cap_binds,
        'breakeven': breakeven,
        'take_up_rate': take_up,
        'gdp_buffer_pct': gdp_buffer_pct,
    }