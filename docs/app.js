/**
 * MOSAIC Calculator - Application Logic
 *
 * UI state management, event handlers, and rendering.
 */

// =============================================================================
// STATE MANAGEMENT
// =============================================================================

let state = {
    // NIT Design (user-editable)
    disregard: DEFAULTS.disregard,
    taper: DEFAULTS.taper,
    ringfence_rate: DEFAULTS.ringfence_rate || 75,
    scenarios: JSON.parse(JSON.stringify(DEFAULTS.scenarios)),

    // Design Parameters (editable assumptions)
    sigma: DATA.sigma,
    single_parent_topup: DATA.single_parent_topup,
    nit_take_up_rate: DATA.nit_take_up_rate,
    nit_gdp_buffer_pct: DATA.nit_gdp_buffer_pct,
    ai_consumption_exposure: DATA.ai_consumption_exposure,
    ai_attribution_coef: DATA.ai_attribution_coef,
    consolidation_admin_rate: DATA.consolidation_admin_rate,

    // Feature 1: Gap Analysis (show proposed floor cost for all scenarios)
    gapAnalysisEnabled: false,
    proposedFloor: 13750,

    // Feature 2: Solve for Feasibility (find params for gap = 0)
    feasibilitySolverEnabled: false,
    solveFor: 'Y1',           // 'Y1', 'alpha', or 'u'
    baseScenario: 'agi',      // Scenario to use for held params
    targetScenario: 'agi',    // Scenario to apply results to
};

let results = null;
let prevResults = null;
let proposalResult = null;   // Result from floor proposal solver
let prevDeflation = {};      // Track previous deflation values for highlighting
let prevDecileTotal = null;  // Track previous decile total for highlighting
let prevDecileBreakdown = [];  // Track previous decile breakdown for highlighting
let prevWindfall = {};         // Track previous windfall values for highlighting

// =============================================================================
// INITIALIZATION
// =============================================================================

function init() {
    // Bind event listeners
    bindInputs();
    bindFloorProposalInputs();

    // Initialize floor proposal toggle
    const floorProposalToggle = document.getElementById('floor-proposal-toggle');
    const floorProposalContent = document.getElementById('floor-proposal-content');
    floorProposalToggle.addEventListener('click', () => {
        floorProposalContent.classList.toggle('hidden');
        floorProposalToggle.textContent = floorProposalContent.classList.contains('hidden')
            ? '+ Floor Proposal Mode'
            : '- Floor Proposal Mode';
    });

    // Initialize advanced section toggle
    const advancedToggle = document.getElementById('advanced-toggle');
    const advancedContent = document.getElementById('advanced-content');
    advancedToggle.addEventListener('click', () => {
        advancedContent.classList.toggle('hidden');
        advancedToggle.textContent = advancedContent.classList.contains('hidden')
            ? '+ Advanced Parameters'
            : '- Advanced Parameters';
    });

    // Initialize data sources toggle
    const sourcesToggle = document.getElementById('sources-toggle');
    const sourcesContent = document.getElementById('sources-content');
    sourcesToggle.addEventListener('click', () => {
        sourcesContent.classList.toggle('hidden');
        sourcesToggle.textContent = sourcesContent.classList.contains('hidden')
            ? '+ Data Sources'
            : '- Data Sources';
    });

    // Reset button
    document.getElementById('reset-btn').addEventListener('click', resetDefaults);

    // Export button
    document.getElementById('export-btn').addEventListener('click', exportResults);

    // Initial calculation
    recalculate();
}

function bindInputs() {
    // NIT parameters
    bindInput('disregard', 'disregard');
    bindInput('taper', 'taper');

    // Scenario parameters
    for (const scenario of ['baseline', 'strong', 'agi', 'asi']) {
        bindScenarioInput(scenario, 'alpha');
        bindScenarioInput(scenario, 's_L_prime');
        bindScenarioInput(scenario, 'u');
        // Note: deflation is derived from alpha, not user-editable
    }
    
    // Single ring-fence rate (applies to all scenarios)
    bindInput('ringfence-rate', 'ringfence_rate');

    // Design parameters (editable assumptions)
    bindInput('sigma', 'sigma');
    bindInput('single-parent-topup', 'single_parent_topup');
    bindInput('child-equiv-weight', 'child_equiv_weight');
    bindInput('nit-take-up-rate', 'nit_take_up_rate');
    bindInput('nit-gdp-buffer-pct', 'nit_gdp_buffer_pct');
    bindInput('ai-consumption-exposure', 'ai_consumption_exposure');
    bindInput('ai-attribution-coef', 'ai_attribution_coef');
    bindInput('consolidation-admin-rate', 'consolidation_admin_rate');
}

function bindInput(elementId, stateKey) {
    const el = document.getElementById(elementId);
    if (!el) return;

    el.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        if (!isNaN(value)) {
            state[stateKey] = value;
            recalculate();
        }
    });
}

function bindScenarioInput(scenario, param) {
    const elId = `${scenario}-${param.replace(/_/g, '-')}`;
    const el = document.getElementById(elId);
    if (!el) return;

    el.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        if (!isNaN(value)) {
            state.scenarios[scenario][param] = value;
            recalculate();
        }
    });
}

function bindFloorProposalInputs() {
    // Feature 1: Gap Analysis checkbox
    const gapAnalysisCheckbox = document.getElementById('enable-gap-analysis');
    const gapAnalysisInputs = document.getElementById('gap-analysis-inputs');
    const feature1Box = document.getElementById('feature1-box');

    // Feature 2: Feasibility Solver checkbox
    const feasibilityCheckbox = document.getElementById('enable-feasibility-solver');
    const feasibilityInputs = document.getElementById('feasibility-inputs');
    const feature2Box = document.getElementById('feature2-box');
    const feature2Prereq = document.getElementById('feature2-prereq');

    // Helper to update Feature 2's enabled state based on Feature 1
    function updateFeature2State() {
        if (state.gapAnalysisEnabled) {
            feasibilityCheckbox.disabled = false;
            feature2Box.classList.remove('disabled');
            feature2Prereq.classList.add('hidden');
        } else {
            feasibilityCheckbox.disabled = true;
            feasibilityCheckbox.checked = false;
            state.feasibilitySolverEnabled = false;
            feasibilityInputs.classList.add('hidden');
            feature2Box.classList.remove('active');
            feature2Box.classList.add('disabled');
            feature2Prereq.classList.remove('hidden');
        }
    }

    if (gapAnalysisCheckbox) {
        gapAnalysisCheckbox.addEventListener('change', (e) => {
            state.gapAnalysisEnabled = e.target.checked;
            gapAnalysisInputs.classList.toggle('hidden', !state.gapAnalysisEnabled);
            feature1Box.classList.toggle('active', state.gapAnalysisEnabled);
            updateFeature2State();
            recalculate();
        });
    }

    // Proposed floor input
    const proposedFloorInput = document.getElementById('proposed-floor');
    if (proposedFloorInput) {
        proposedFloorInput.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            if (!isNaN(value) && value > 0) {
                state.proposedFloor = value;
                recalculate();
            }
        });
    }

    if (feasibilityCheckbox) {
        feasibilityCheckbox.addEventListener('change', (e) => {
            state.feasibilitySolverEnabled = e.target.checked;
            feasibilityInputs.classList.toggle('hidden', !state.feasibilitySolverEnabled);
            feature2Box.classList.toggle('active', state.feasibilitySolverEnabled);
            recalculate();
        });
    }

    // Initialize Feature 2 state
    updateFeature2State();

    // Solve-for dropdown
    const solveForSelect = document.getElementById('solve-for');
    const heldParamsSection = document.getElementById('held-params-section');

    if (solveForSelect) {
        solveForSelect.addEventListener('change', (e) => {
            state.solveFor = e.target.value;
            // Show held params section only when solving for alpha or u
            const showHeldParams = (state.solveFor === 'alpha' || state.solveFor === 'u');
            heldParamsSection.classList.toggle('hidden', !showHeldParams);
            recalculate();
        });
    }

    // Base scenario dropdown (for held params)
    const baseScenarioSelect = document.getElementById('base-scenario');
    if (baseScenarioSelect) {
        baseScenarioSelect.addEventListener('change', (e) => {
            state.baseScenario = e.target.value;
            recalculate();
        });
    }

    // Target scenario dropdown
    const targetScenarioSelect = document.getElementById('target-scenario');
    if (targetScenarioSelect) {
        targetScenarioSelect.addEventListener('change', (e) => {
            state.targetScenario = e.target.value;
            recalculate();
        });
    }
}

// =============================================================================
// CALCULATION
// =============================================================================

function recalculate() {
    prevResults = results;
    // Apply single ringfence_rate to all scenarios
    const stateWithRingfence = {
        ...state,
        scenarios: Object.fromEntries(
            Object.entries(state.scenarios).map(([name, s]) => [
                name,
                { ...s, ringfence_rate: state.ringfence_rate }
            ])
        )
    };
    results = calculateAll(stateWithRingfence, CBS_DECILES);

    // Handle Feature 2: Feasibility Solver
    proposalResult = null;
    if (state.feasibilitySolverEnabled && state.gapAnalysisEnabled) {
        const baseScenario = state.scenarios[state.baseScenario];
        const targetScenario = state.scenarios[state.targetScenario];

        if (state.solveFor === 'Y1') {
            // Solve for Y1 given proposed floor
            proposalResult = solveY1ForFloor(
                state.proposedFloor,
                targetScenario,
                CBS_DECILES,
                state.taper,
                state.disregard,
                DATA,
                state.nit_take_up_rate / 100
            );
            proposalResult.s_L_prime = targetScenario.s_L_prime;
        } else if (state.solveFor === 'alpha') {
            // Solve for alpha given proposed floor, holding s_L' and u from base scenario
            proposalResult = solveAlphaForFloor(
                state.proposedFloor,
                baseScenario.s_L_prime,
                baseScenario.u,
                targetScenario,
                CBS_DECILES,
                state.taper,
                state.disregard,
                DATA,
                state.nit_take_up_rate / 100
            );
            proposalResult.s_L_prime = baseScenario.s_L_prime;
        } else if (state.solveFor === 'u') {
            // Solve for u given proposed floor, holding alpha and s_L' from base scenario
            proposalResult = solveUForFloor(
                state.proposedFloor,
                baseScenario.alpha,
                baseScenario.s_L_prime,
                targetScenario,
                CBS_DECILES,
                state.taper,
                state.disregard,
                DATA,
                state.nit_take_up_rate / 100
            );
            proposalResult.s_L_prime = baseScenario.s_L_prime;
        }

        // Update solver status display
        updateSolverStatus(proposalResult);
    } else {
        // Clear solver status if not enabled
        updateSolverStatus(null);
    }

    render();
}

function updateSolverStatus(result) {
    const statusEl = document.getElementById('solver-status');
    if (!statusEl) return;

    if (!result) {
        statusEl.innerHTML = '';
        statusEl.classList.remove('status-ok', 'status-error');
        return;
    }

    if (result.converged) {
        let statusHtml = `<div class="status-ok">`;
        statusHtml += `<strong>✓ Converged</strong><br>`;

        if (state.solveFor === 'Y1' && result.Y1) {
            statusHtml += `Required Y₁ = ${formatB(result.Y1)} (g<sub>Y</sub> = ${result.g_Y.toFixed(1)}%)`;
        } else if (state.solveFor === 'alpha' && result.alpha) {
            statusHtml += `Required α = ${result.alpha.toFixed(2)}`;
            if (result.Y1) statusHtml += ` → Y₁ = ${formatB(result.Y1)}`;
        } else if (state.solveFor === 'u' && result.u !== null) {
            statusHtml += `Required u = ${result.u.toFixed(1)}%`;
            if (result.Y1) statusHtml += ` → Y₁ = ${formatB(result.Y1)}`;
        }

        statusHtml += `</div>`;
        statusEl.innerHTML = statusHtml;
        statusEl.classList.add('status-ok');
        statusEl.classList.remove('status-error');
    } else {
        statusEl.innerHTML = `<div class="status-error"><strong>✗ ${result.message || 'No solution found'}</strong></div>`;
        statusEl.classList.add('status-error');
        statusEl.classList.remove('status-ok');
    }
}

/**
 * Check if a value changed and return the highlight class if so.
 */
function getChangeClass(scenario, key, newValue, tolerance = 0.01) {
    if (!prevResults || !prevResults[scenario]) return '';
    const oldValue = prevResults[scenario][key];
    if (oldValue === undefined) return '';

    // Check if values differ (with tolerance for floating point)
    const diff = Math.abs(newValue - oldValue);
    const threshold = Math.abs(oldValue) * tolerance;
    if (diff > Math.max(threshold, 0.001)) {
        return 'value-changed';
    }
    return '';
}

// =============================================================================
// RENDERING
// =============================================================================

function render() {
    renderDeflationDisplay();
    renderGrowthTable();
    renderRevenueTable();
    renderFloorTable();
    renderHouseholdTable();
    renderDecileTable();
    renderPovertyTable();
    renderWindfallTable();
    renderUHITable();

    // Update personal calculator with new floor
    if (typeof updateCalculator === 'function') {
        updateCalculator();
    }
}

function renderDeflationDisplay() {
    // Update derived deflation display values for each scenario
    // Uses state.ai_consumption_exposure so it updates when AI% changes
    const bls_delta = DATA.bls_deflation_rate / 100;
    const alpha_bls = DATA.bls_deflation_alpha;
    const ai_exposure = (state.ai_consumption_exposure !== undefined
        ? state.ai_consumption_exposure : DATA.ai_consumption_exposure) / 100;

    for (const scenario of ['baseline', 'strong', 'agi', 'asi']) {
        const alpha = state.scenarios[scenario].alpha;
        const deflation = calcDeflationFromAlpha(alpha, bls_delta, alpha_bls, ai_exposure);
        const deflationPct = (deflation * 100).toFixed(0);

        const el = document.getElementById(`${scenario}-deflation-display`);
        if (el) {
            el.textContent = `${deflationPct}%`;

            // Add change highlighting
            const prevVal = prevDeflation[scenario];
            if (prevVal !== undefined && Math.abs(deflation - prevVal) > 0.001) {
                el.classList.add('value-changed');
                // Remove the class after animation completes
                setTimeout(() => el.classList.remove('value-changed'), 1500);
            }
            prevDeflation[scenario] = deflation;
        }
    }
}

function renderGrowthTable() {
    const tbody = document.getElementById('growth-tbody');
    tbody.innerHTML = '';

    for (const [name, r] of Object.entries(results)) {
        const row = document.createElement('tr');
        const isAGI = name === 'agi';
        if (isAGI) row.classList.add('agi-row');

        // Color code ΔS: green if positive, red if negative
        const deltaSClass = r.delta_S >= 0 ? 'surplus-positive' : 'surplus-negative';

        row.innerHTML = `
            <td>${formatScenario(name)}</td>
            <td class="num ${getChangeClass(name, 'g_Y', r.g_Y)}">${r.g_Y.toFixed(1)}%</td>
            <td class="num ${getChangeClass(name, 'Y_1', r.Y_1)}">${formatB(r.Y_1)}</td>
            <td class="num ${deltaSClass} ${getChangeClass(name, 'delta_S', r.delta_S)}">${formatB(r.delta_S)}</td>
        `;
        tbody.appendChild(row);
    }
}

function renderRevenueTable() {
    const tbody = document.getElementById('revenue-tbody');
    tbody.innerHTML = '';

    for (const [name, r] of Object.entries(results)) {
        const row = document.createElement('tr');
        const isAGI = name === 'agi';
        if (isAGI) row.classList.add('agi-row');

        // Get CG value (may be undefined in older results)
        const cg = r.cg_ringfenced || 0;

        row.innerHTML = `
            <td>${formatScenario(name)}</td>
            <td class="num ${getChangeClass(name, 'vat_revenue', r.vat_revenue)}">${r.vat_revenue.toFixed(1)}</td>
            <td class="num ${getChangeClass(name, 'ringfenced', r.ringfenced)}">${r.ringfenced.toFixed(1)}</td>
            <td class="num ${getChangeClass(name, 'cg_ringfenced', cg)}">${cg.toFixed(1)}</td>
            <td class="num ${getChangeClass(name, 'gov_ringfence', r.gov_ringfence)}">${r.gov_ringfence.toFixed(1)}</td>
            <td class="num ${getChangeClass(name, 'consolidation', r.consolidation)}">${r.consolidation.toFixed(1)}</td>
            <td class="num total ${getChangeClass(name, 'total_revenue', r.total_revenue)}">${r.total_revenue.toFixed(1)}</td>
        `;
        tbody.appendChild(row);
    }

    // Feature 2: Show solved revenue when feasibility solver is enabled
    if (state.feasibilitySolverEnabled && proposalResult && proposalResult.converged && proposalResult.revenue) {
        const solvedRow = document.createElement('tr');
        solvedRow.classList.add('solved-row');

        const rev = proposalResult.revenue;
        const solveLabel = state.solveFor === 'Y1' ? `Y₁=${formatB(proposalResult.Y1)}` :
                           state.solveFor === 'alpha' ? `α=${proposalResult.alpha.toFixed(2)}` :
                           `u=${proposalResult.u.toFixed(1)}%`;
        const revCG = rev.cg_ringfenced || 0;

        solvedRow.innerHTML = `
            <td><strong>Solved</strong><br><small>(${solveLabel})</small></td>
            <td class="num">${rev.vat_revenue.toFixed(1)}</td>
            <td class="num">${rev.ringfenced.toFixed(1)}</td>
            <td class="num">${revCG.toFixed(1)}</td>
            <td class="num">${rev.gov_ringfence.toFixed(1)}</td>
            <td class="num">${rev.consolidation.toFixed(1)}</td>
            <td class="num total">${rev.total.toFixed(1)}</td>
        `;
        tbody.appendChild(solvedRow);
    }
}

function renderFloorTable() {
    const tbody = document.getElementById('floor-tbody');
    tbody.innerHTML = '';

    const z = DATA.poverty_line;
    const useProposedFloor = state.gapAnalysisEnabled;

    for (const [name, r] of Object.entries(results)) {
        const row = document.createElement('tr');
        const isAGI = name === 'agi';
        if (isAGI) row.classList.add('agi-row');

        // Use proposed floor when Feature 1 is enabled
        const floor = useProposedFloor ? state.proposedFloor : r.floor;
        const breakeven = calcBreakeven(state.disregard, floor, state.taper);
        const floorOverZ = floor / z;

        const meetsZ = floor >= z;
        const statusClass = meetsZ ? 'status-ok' : 'status-warn';
        const statusText = meetsZ ? 'M ≥ z' : 'M < z';

        // Add "+ proposed" suffix when using proposed floor
        let scenarioName = formatScenario(name);
        if (useProposedFloor) {
            scenarioName += ' + proposed';
            row.classList.add('proposal-mode');
        }

        row.innerHTML = `
            <td>${scenarioName}</td>
            <td class="num ${getChangeClass(name, 'floor', floor)}">${formatNum(floor)}</td>
            <td class="num ${getChangeClass(name, 'breakeven', breakeven)}">${formatNum(breakeven)}</td>
            <td class="num ${getChangeClass(name, 'floor_over_z', floorOverZ)}">${floorOverZ.toFixed(2)}x</td>
            <td class="num ${statusClass}">${statusText}</td>
        `;
        tbody.appendChild(row);
    }
}

function renderHouseholdTable() {
    const tbody = document.getElementById('household-tbody');
    tbody.innerHTML = '';

    const psi = 900; // Single parent top-up
    const useProposedFloor = state.gapAnalysisEnabled;

    // Household types with their scale factors
    const householdTypes = [
        { name: 'Single', key: 'single', adults: 1, children: 0, singleParent: false },
        { name: 'Couple', key: 'couple', adults: 2, children: 0, singleParent: false },
        { name: 'Couple + 2 kids', key: 'couple2', adults: 2, children: 2, singleParent: false },
        { name: 'Single parent + 2', key: 'single_parent2', adults: 1, children: 2, singleParent: true },
    ];

    for (const hh of householdTypes) {
        const row = document.createElement('tr');

        // Calculate floor for each scenario
        const floors = {};
        const prevFloors = {};
        for (const [scenarioName, r] of Object.entries(results)) {
            // Use proposed floor when Feature 1 is enabled
            const baseFloor = useProposedFloor ? state.proposedFloor : r.floor;
            floors[scenarioName] = calcHouseholdFloor(
                baseFloor, hh.adults, hh.children, hh.singleParent, psi
            );
            // Calculate previous floors for comparison
            if (prevResults && prevResults[scenarioName]) {
                prevFloors[scenarioName] = calcHouseholdFloor(
                    prevResults[scenarioName].floor, hh.adults, hh.children, hh.singleParent, psi
                );
            }
        }

        // Check for changes in household floors
        const getHHChangeClass = (scenario) => {
            if (!prevFloors[scenario]) return '';
            const diff = Math.abs(floors[scenario] - prevFloors[scenario]);
            return diff > 1 ? 'value-changed' : '';
        };

        // Add proposal mode class when using proposed floor
        if (useProposedFloor) {
            row.classList.add('proposal-mode');
        }

        row.innerHTML = `
            <td>${hh.name}</td>
            <td class="num ${getHHChangeClass('baseline')}">${formatNum(floors.baseline)}</td>
            <td class="num ${getHHChangeClass('strong')}">${formatNum(floors.strong)}</td>
            <td class="num agi-highlight ${getHHChangeClass('agi')}">${formatNum(floors.agi)}</td>
            <td class="num asi-highlight ${getHHChangeClass('asi')}">${formatNum(floors.asi)}</td>
        `;
        tbody.appendChild(row);
    }
}

function renderDecileTable() {
    const tbody = document.getElementById('decile-tbody');
    const tfoot = document.getElementById('decile-tfoot');
    tbody.innerHTML = '';
    tfoot.innerHTML = '';

    const useProposedFloor = state.gapAnalysisEnabled;

    // Use AGI scenario for the decile breakdown
    const agiResult = results.agi;
    // Use proposed floor when Feature 1 is enabled
    const floor = useProposedFloor ? state.proposedFloor : agiResult.floor;

    // Use microsimulation with within-decile Beta distribution (matches Table 11)
    const breakdown = calcDecileBreakdownMicrosim(
        floor,
        CBS_DECILES,
        MICROSIM_PARAMS.decile_bounds,
        MICROSIM_PARAMS.beta_concentration,
        MICROSIM_PARAMS.gamma_shape,
        state.taper,
        state.disregard,
        0.85,  // 85% take-up
        MICROSIM_PARAMS.n_samples,
        42     // seed
    );

    let totalCost = 0;
    let totalHouseholds = 0;

    for (let i = 0; i < breakdown.length; i++) {
        const d = breakdown[i];
        const row = document.createElement('tr');
        totalCost += d.annual_cost;
        totalHouseholds += d.households;

        // Highlight rows that receive benefits
        if (d.benefit_per_hh > 0) {
            row.classList.add('receives-benefit');
        }

        // Check for changes in benefit and cost values
        const prev = prevDecileBreakdown[i];
        const benefitChanged = prev && Math.abs(d.benefit_per_hh - prev.benefit_per_hh) > 1;
        const costChanged = prev && Math.abs(d.annual_cost - prev.annual_cost) > 0.1;
        const benefitClass = benefitChanged ? 'value-changed' : '';
        const costClass = costChanged ? 'value-changed' : '';

        row.innerHTML = `
            <td>D${d.decile}</td>
            <td class="num">${formatNum(d.equiv_income)}</td>
            <td class="num">${d.std_persons.toFixed(1)}</td>
            <td class="num">${formatNum(d.households)}</td>
            <td class="num ${benefitClass}">${d.benefit_per_hh > 0 ? formatNum(Math.round(d.benefit_per_hh)) : '-'}</td>
            <td class="num">${d.emtr !== null ? (d.emtr * 100).toFixed(1) + '%' : '-'}</td>
            <td class="num ${costClass}">${d.annual_cost > 0.001 ? d.annual_cost.toFixed(1) : '-'}</td>
        `;
        tbody.appendChild(row);
    }

    // Store current breakdown for next comparison
    prevDecileBreakdown = breakdown.map(d => ({ benefit_per_hh: d.benefit_per_hh, annual_cost: d.annual_cost }));

    // Add total row with change highlighting
    const totalRow = document.createElement('tr');
    totalRow.classList.add('total-row');

    // Check if total cost changed
    const totalChangeClass = (prevDecileTotal !== null && Math.abs(totalCost - prevDecileTotal) > 0.1)
        ? 'value-changed' : '';
    prevDecileTotal = totalCost;

    totalRow.innerHTML = `
        <td><strong>Total</strong></td>
        <td></td>
        <td></td>
        <td class="num"><strong>${(totalHouseholds / 1e6).toFixed(2)}M</strong></td>
        <td></td>
        <td></td>
        <td class="num ${totalChangeClass}"><strong>${totalCost.toFixed(1)}</strong></td>
    `;
    tfoot.appendChild(totalRow);
}

function renderPovertyTable() {
    const tbody = document.getElementById('poverty-tbody');
    tbody.innerHTML = '';

    const useProposedFloor = state.gapAnalysisEnabled;
    const poverty_line = DATA.poverty_line;

    // Pre-NIT row
    const preRow = document.createElement('tr');
    preRow.classList.add('pre-nit-row');
    preRow.innerHTML = `
        <td>Pre-NIT</td>
        <td class="num">${DATA.gini_initial.toFixed(3)}</td>
        <td class="num">${DATA.poverty_rate_initial}%</td>
        <td class="num">-</td>
    `;
    tbody.appendChild(preRow);

    for (const [name, r] of Object.entries(results)) {
        const row = document.createElement('tr');
        const isAGI = name === 'agi';
        if (isAGI) row.classList.add('agi-row');

        // Use proposed floor when Feature 1 is enabled, recalculate poverty metrics
        let gini_post, static_poverty, static_threshold;

        if (useProposedFloor) {
            const floor = state.proposedFloor;
            // Recalculate poverty metrics for proposed floor
            const post_deciles = applyNITToDeciles(CBS_DECILES, floor, state.taper, state.disregard);
            gini_post = calcGiniFromDeciles(post_deciles);
            static_poverty = calcPovertyRate(post_deciles, poverty_line);
            static_threshold = calcPovertyThreshold(poverty_line, floor, state.taper, state.disregard);
            row.classList.add('proposal-mode');
        } else {
            gini_post = r.gini_post;
            static_poverty = r.static_poverty;
            static_threshold = r.static_threshold;
        }

        // Format threshold
        let threshText;
        if (static_threshold === 0) {
            threshText = 'None';
        } else {
            threshText = `< ${formatNum(Math.round(static_threshold))}`;
        }

        // Add "+ proposed" suffix when using proposed floor
        let scenarioName = formatScenario(name);
        if (useProposedFloor) {
            scenarioName += ' + proposed';
        }

        row.innerHTML = `
            <td>${scenarioName}</td>
            <td class="num ${getChangeClass(name, 'gini_post', gini_post)}">${gini_post.toFixed(3)}</td>
            <td class="num ${getChangeClass(name, 'static_poverty', static_poverty)}">${static_poverty.toFixed(1)}%</td>
            <td class="num threshold ${getChangeClass(name, 'static_threshold', static_threshold)}">${threshText}</td>
        `;
        tbody.appendChild(row);
    }
}

function renderWindfallTable() {
    const tbody = document.getElementById('windfall-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';

    // Calculate windfall analysis
    // Feature 1 (Gap Analysis) shows proposed floor cost for all scenarios
    const windfallData = calcWindfallAnalysis(
        results,
        state.gapAnalysisEnabled ? state.proposedFloor : null,
        CBS_DECILES,
        state.taper,
        state.disregard,
        DATA,
        state.nit_take_up_rate / 100
    );

    for (const row of windfallData) {
        const tr = document.createElement('tr');

        // Styling for proposed mode
        if (row.isProposed) {
            tr.classList.add('proposal-mode');
        }
        if (row.scenario === 'agi') {
            tr.classList.add('agi-row');
        }

        // Format capture percentage with warning colors
        const capturePiClass = row.capturePi > 100 ? 'capture-warning' : (row.capturePi > 50 ? 'capture-high' : '');

        // Format scenario name - add "+ proposed" suffix when in proposal mode
        let scenarioName = formatScenario(row.scenario);
        if (row.isProposed) {
            scenarioName += ' + proposed';
        }

        // Check for changes in windfall values
        const prevKey = row.scenario + (row.isProposed ? '_proposed' : '');
        const prev = prevWindfall[prevKey];
        const getWfChangeClass = (val, prevVal, tol = 0.1) =>
            prev && Math.abs(val - prevVal) > tol ? 'value-changed' : '';

        tr.innerHTML = `
            <td>${scenarioName}</td>
            <td class="num ${getWfChangeClass(row.floor, prev?.floor, 1)}">${formatNum(Math.round(row.floor))}</td>
            <td class="num ${getWfChangeClass(row.cost, prev?.cost)}">${row.cost.toFixed(1)}</td>
            <td class="num ${getWfChangeClass(row.revenue, prev?.revenue)}">${row.revenue.toFixed(1)}</td>
            <td class="num ${row.gap > 0.5 ? 'gap-positive' : ''} ${getWfChangeClass(row.gap, prev?.gap)}">${row.gap.toFixed(1)}</td>
            <td class="num ${getWfChangeClass(row.deltaPi, prev?.deltaPi, 1)}">${row.deltaPi.toFixed(0)}</td>
            <td class="num ${capturePiClass} ${getWfChangeClass(row.capturePi, prev?.capturePi)}">${row.capturePi.toFixed(1)}%</td>
        `;
        tbody.appendChild(tr);

        // Store current values for next comparison
        prevWindfall[prevKey] = {
            floor: row.floor, cost: row.cost, revenue: row.revenue, gap: row.gap,
            deltaPi: row.deltaPi, capturePi: row.capturePi
        };
    }
}

let prevUHIFunding = null;  // Track previous UHI values for highlighting

function renderUHITable() {
    const tbody = document.getElementById('uhi-tbody');
    const tfoot = document.getElementById('uhi-tfoot');
    if (!tbody || !tfoot) return;
    tbody.innerHTML = '';
    tfoot.innerHTML = '';

    // Use AGI scenario for UHI calculations
    const agiResult = results.agi;
    if (!agiResult) return;

    // Use proposed floor if gap analysis is enabled, otherwise use default UHI target
    const useProposedFloor = state.gapAnalysisEnabled;
    const uhiFloor = useProposedFloor ? state.proposedFloor : DATA.uhi_floor_target;
    const uhiCost100 = calcNITCostForProposal(uhiFloor, CBS_DECILES, state.taper, state.disregard);
    const uhiTargetCost = uhiCost100 * (state.nit_take_up_rate / 100);

    // Update description text
    const descEl = document.getElementById('uhi-floor-description');
    if (descEl) {
        const medianPct = (uhiFloor / DATA.median_wage * 100).toFixed(0);
        if (useProposedFloor) {
            descEl.textContent = `Funding for proposed floor of ${formatNum(uhiFloor)} NIS/mo (~${medianPct}% of median wage)`;
        } else {
            descEl.textContent = `Path to Universal High Income floor of ${formatNum(uhiFloor)} NIS/mo (~${medianPct}% of median wage)`;
        }
    }

    // Get capital profit change (ΔΠ) for AGI
    const deltaPi = calcCapitalProfitChange(
        DATA.Y0,
        agiResult.Y_1,
        DATA.labor_share,
        agiResult.s_L_prime
    );

    // Calculate UHI funding breakdown
    const uhi = calcUHIFunding(agiResult.total_revenue, deltaPi, uhiTargetCost, DATA);

    // Helper for change highlighting
    const getUHIChangeClass = (key, value) => {
        if (!prevUHIFunding) return '';
        const prev = prevUHIFunding[key];
        if (prev === undefined) return '';
        return Math.abs(value - prev) > 0.1 ? 'value-changed' : '';
    };

    // Build table rows
    const rows = [
        // Existing mechanisms
        {
            section: 'Existing mechanisms (Section 4)',
            isSection: true
        },
        {
            name: 'VAT + Ring-fencing + Consolidation',
            value: uhi.existingRevenue,
            key: 'existingRevenue',
            mechanism: 'Automatic',
            indent: true
        },

        // Extended consolidation
        {
            section: 'Extended consolidation (Kohelet)',
            isSection: true
        },
        {
            name: 'Program cancellation (6 programs)',
            value: uhi.koheletConsolidation,
            key: 'koheletConsolidation',
            mechanism: 'Replacement',
            indent: true
        },
        {
            name: 'Income tax reform (25% floor)',
            value: uhi.koheletTaxReform,
            key: 'koheletTaxReform',
            mechanism: 'Base broadening',
            indent: true
        },

        // New instruments
        {
            section: 'New instruments',
            isSection: true
        },
        {
            name: 'Estate tax (30%, high exemption)',
            value: uhi.estateTax,
            key: 'estateTax',
            mechanism: 'Wealth transfer',
            indent: true
        },
        {
            name: 'Data dividend',
            value: uhi.dataDividend,
            key: 'dataDividend',
            mechanism: 'Resource royalty',
            indent: true
        },
        {
            name: 'Pillar Two expansion*',
            value: uhi.pillarTwo,
            key: 'pillarTwo',
            mechanism: 'Global min. tax',
            indent: true
        },
        {
            name: 'Yozma 2.0 (govt. VC)',
            value: uhi.yozma,
            key: 'yozma',
            mechanism: 'Equity returns',
            indent: true
        },

        // Direct windfall capture
        {
            section: 'Direct windfall capture',
            isSection: true
        },
        {
            name: `AI windfall levy (${uhi.windfallLevyPct.toFixed(0)}% of ΔΠ)`,
            value: uhi.windfallLevy,
            key: 'windfallLevy',
            mechanism: 'Other methods',
            indent: true,
            highlight: true
        },
    ];

    for (const row of rows) {
        const tr = document.createElement('tr');

        if (row.isSection) {
            // Section header row
            tr.classList.add('section-header');
            tr.innerHTML = `
                <td colspan="4"><em>${row.section}</em></td>
            `;
        } else {
            // Data row
            const pct = (row.value / uhi.uhiTargetCost * 100).toFixed(1);
            const changeClass = getUHIChangeClass(row.key, row.value);

            if (row.highlight) {
                tr.classList.add('windfall-row');
            }

            tr.innerHTML = `
                <td>${row.indent ? '&nbsp;&nbsp;&nbsp;&nbsp;' : ''}${row.name}</td>
                <td class="num ${changeClass}">${row.value.toFixed(1)}</td>
                <td class="num">${pct}%</td>
                <td>${row.mechanism}</td>
            `;
        }

        tbody.appendChild(tr);
    }

    // Total row in footer
    const totalChangeClass = getUHIChangeClass('totalFunding', uhi.totalFunding);
    const totalRow = document.createElement('tr');
    totalRow.classList.add('total-row');
    totalRow.innerHTML = `
        <td><strong>Total</strong></td>
        <td class="num ${totalChangeClass}"><strong>${uhi.totalFunding.toFixed(0)}</strong></td>
        <td class="num"><strong>100%</strong></td>
        <td></td>
    `;
    tfoot.appendChild(totalRow);

    // Store current values for next comparison
    prevUHIFunding = {
        existingRevenue: uhi.existingRevenue,
        koheletConsolidation: uhi.koheletConsolidation,
        koheletTaxReform: uhi.koheletTaxReform,
        estateTax: uhi.estateTax,
        dataDividend: uhi.dataDividend,
        pillarTwo: uhi.pillarTwo,
        yozma: uhi.yozma,
        windfallLevy: uhi.windfallLevy,
        totalFunding: uhi.totalFunding
    };
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function formatNum(n) {
    return n.toLocaleString('en-US', { maximumFractionDigits: 0 });
}

function formatB(n) {
    return n.toFixed(1) + 'B';
}

function formatScenario(s) {
    // Special case for AGI and ASI (all caps)
    if (s === 'agi') return '**AGI';
    if (s === 'asi') return 'ASI';
    // Capitalize first letter
    return s.charAt(0).toUpperCase() + s.slice(1);
}

function resetDefaults() {
    state = {
        disregard: DEFAULTS.disregard,
        taper: DEFAULTS.taper,
        ringfence_rate: DEFAULTS.ringfence_rate || 75,
        scenarios: JSON.parse(JSON.stringify(DEFAULTS.scenarios)),
        // Design parameters
        sigma: DATA.sigma,
        single_parent_topup: DATA.single_parent_topup,
            nit_take_up_rate: DATA.nit_take_up_rate,
        nit_gdp_buffer_pct: DATA.nit_gdp_buffer_pct,
        ai_consumption_exposure: DATA.ai_consumption_exposure,
        ai_attribution_coef: DATA.ai_attribution_coef,
        consolidation_admin_rate: DATA.consolidation_admin_rate,
        // Reset floor proposal state
        gapAnalysisEnabled: false,
        proposedFloor: 13750,
        feasibilitySolverEnabled: false,
        solveFor: 'Y1',
        baseScenario: 'agi',
        targetScenario: 'agi',
    };

    // Update all input fields
    document.getElementById('disregard').value = state.disregard;
    document.getElementById('taper').value = state.taper;

    // Update design parameter inputs
    const designInputs = {
        'ringfence-rate': state.ringfence_rate,
        'sigma': state.sigma,
        'single-parent-topup': state.single_parent_topup,
        'nit-take-up-rate': state.nit_take_up_rate,
        'nit-gdp-buffer-pct': state.nit_gdp_buffer_pct,
        'ai-consumption-exposure': state.ai_consumption_exposure,
        'ai-attribution-coef': state.ai_attribution_coef,
        'consolidation-admin-rate': state.consolidation_admin_rate,
    };
    for (const [id, value] of Object.entries(designInputs)) {
        const el = document.getElementById(id);
        if (el) el.value = value;
    }

    for (const [scenario, params] of Object.entries(state.scenarios)) {
        for (const [param, value] of Object.entries(params)) {
            const el = document.getElementById(`${scenario}-${param.replace(/_/g, '-')}`);
            if (el) el.value = value;
        }
    }

    // Reset floor proposal UI
    const gapAnalysisCheckbox = document.getElementById('enable-gap-analysis');
    if (gapAnalysisCheckbox) gapAnalysisCheckbox.checked = false;
    const gapAnalysisInputs = document.getElementById('gap-analysis-inputs');
    if (gapAnalysisInputs) gapAnalysisInputs.classList.add('hidden');
    const feature1Box = document.getElementById('feature1-box');
    if (feature1Box) feature1Box.classList.remove('active');

    const feasibilityCheckbox = document.getElementById('enable-feasibility-solver');
    if (feasibilityCheckbox) {
        feasibilityCheckbox.checked = false;
        feasibilityCheckbox.disabled = true;
    }
    const feasibilityInputs = document.getElementById('feasibility-inputs');
    if (feasibilityInputs) feasibilityInputs.classList.add('hidden');
    const feature2Box = document.getElementById('feature2-box');
    if (feature2Box) {
        feature2Box.classList.remove('active');
        feature2Box.classList.add('disabled');
    }
    const feature2Prereq = document.getElementById('feature2-prereq');
    if (feature2Prereq) feature2Prereq.classList.remove('hidden');

    const proposedFloorInput = document.getElementById('proposed-floor');
    if (proposedFloorInput) proposedFloorInput.value = 13750;
    const solveForSelect = document.getElementById('solve-for');
    if (solveForSelect) solveForSelect.value = 'Y1';
    const heldParamsSection = document.getElementById('held-params-section');
    if (heldParamsSection) heldParamsSection.classList.add('hidden');

    // Clear previous value trackers to prevent highlighting after reset
    prevResults = null;
    prevDeflation = {};
    prevDecileTotal = null;
    prevDecileBreakdown = [];
    prevWindfall = {};
    prevUHIFunding = null;

    recalculate();
}

function exportResults() {
    const exportData = {
        parameters: state,
        results: results,
        timestamp: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'mosaic-results.json';
    a.click();
    URL.revokeObjectURL(url);
}


// =============================================================================
// PERSONAL BENEFIT CALCULATOR
// =============================================================================

function initCalculator() {
    // Bind calculator inputs
    const inputs = ['calc-gross-income', 'calc-adults', 'calc-children'];
    inputs.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('input', updateCalculator);
        }
    });
    
    // Single parent checkbox
    const singleParentEl = document.getElementById('calc-single-parent');
    if (singleParentEl) {
        singleParentEl.addEventListener('change', updateCalculator);
    }
    
    // Initial calculation
    updateCalculator();
}

function updateCalculator() {
    // Get input values
    const grossIncome = parseFloat(document.getElementById('calc-gross-income').value) || 0;
    const adults = parseInt(document.getElementById('calc-adults').value) || 1;
    const children = parseInt(document.getElementById('calc-children').value) || 0;
    const isSingleParent = document.getElementById('calc-single-parent').checked;
    
    // Get floor: use proposed floor if gap analysis enabled, otherwise AGI floor
    let floor = 7251;  // Default fallback
    if (state.gapAnalysisEnabled) {
        floor = state.proposedFloor;
    } else if (results && results.agi && results.agi.floor) {
        floor = results.agi.floor;
    }
    
    // Get parameters from state
    const taper = state.taper;
    const disregard = state.disregard;
    const singleParentTopup = state.single_parent_topup || 900;
    
    // Calculate equivalence scale (OECD modified)
    // 1.0 for first adult + 0.5 for each additional adult + 0.3 for each child
    const stdPersons = 1.0 + 0.5 * (adults - 1) + 0.3 * children;
    
    // Calculate equivalized income
    const equivIncome = grossIncome / stdPersons;
    
    // Calculate break-even (per std person)
    const breakeven = disregard + floor / taper;
    
    // Calculate benefit per standard person
    let benefitPerStd = 0;
    if (equivIncome < breakeven) {
        const taxable = Math.max(0, equivIncome - disregard);
        benefitPerStd = Math.max(0, floor - taper * taxable);
    }
    
    // Total household benefit
    let totalBenefit = benefitPerStd * stdPersons;
    
    // Add single parent top-up
    if (isSingleParent && totalBenefit > 0) {
        totalBenefit += singleParentTopup;
    }
    
    // Calculate EMTR (if receiving benefit)
    let emtr = null;
    if (totalBenefit > 0) {
        // Approximate tax rate based on income (simplified)
        const approxTaxRate = equivIncome < 5000 ? 0.10 : (equivIncome < 10000 ? 0.14 : 0.20);
        emtr = approxTaxRate + taper;
    }
    
    // Update intro text based on whether proposed floor is used
    const introEl = document.getElementById('calc-intro');
    if (introEl) {
        if (state.gapAnalysisEnabled) {
            introEl.textContent = `Enter your household details to see your NIT benefit under the proposed floor (${formatNum(floor)} NIS/mo).`;
        } else {
            introEl.textContent = 'Enter your household details to see your NIT benefit under the AGI scenario.';
        }
    }

    // Update display
    document.getElementById('calc-floor').textContent = formatNum(Math.round(floor)) + ' NIS';
    document.getElementById('calc-std-persons').textContent = stdPersons.toFixed(2);
    document.getElementById('calc-equiv-income').textContent = formatNum(Math.round(equivIncome)) + ' NIS';
    document.getElementById('calc-breakeven').textContent = formatNum(Math.round(breakeven)) + ' NIS';
    
    const benefitEl = document.getElementById('calc-benefit');
    if (totalBenefit > 0) {
        benefitEl.textContent = formatNum(Math.round(totalBenefit)) + ' NIS/mo';
        benefitEl.classList.remove('no-benefit');
    } else {
        benefitEl.textContent = 'None (above break-even)';
        benefitEl.classList.add('no-benefit');
    }
    
    const emtrEl = document.getElementById('calc-emtr');
    if (emtr !== null) {
        emtrEl.textContent = (emtr * 100).toFixed(1) + '%';
    } else {
        emtrEl.textContent = '-';
    }
}

// =============================================================================
// INITIALIZATION
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    init();
    initCalculator();
});
