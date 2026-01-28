# When AI Takes Our Jobs, It Should Also Pay Our Wages

A Negative Income Tax framework funded by AI-driven capital windfalls, calibrated to the Israeli economy.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## The Key Insight

AI-driven unemployment differs fundamentally from classical unemployment:

| | Classical Unemployment | AI-driven Unemployment |
|---|------------------------|------------------------|
| Employment | ΔL < 0 | ΔL < 0 |
| Capital | ΔK = 0 | ΔK_AI > 0 |
| Output | ΔY < 0 | ΔY > 0 |
| Tax base | Shrinks | **Expands** |

When employment falls but output rises, GDP growth accrues disproportionately to capital. We call this increment the **capital windfall** (ΔΠ). The more disruptive the automation, the larger the windfall available for redistribution.

## Two-Channel Funding Architecture

The framework captures AI-generated value through two automatic channels:

1. **Dynamic VAT**: As AI raises productivity and lowers prices, VAT rates adjust to harvest deflation gains without raising consumer prices (72--280B NIS across scenarios)

2. **Ring-fenced Capital Income**: Above-trend corporate and capital gains tax receipts are earmarked for redistribution (16--40B NIS across scenarios)

These channels, supplemented by Government Automation Dividends and program consolidation, fund a revenue-constrained Negative Income Tax that eliminates poverty across all scenarios.

## Key Results

| Scenario | α | s_L' | u | Revenue | Floor | Gini |
|----------|---|------|---|---------|-------|------|
| Low Displacement | 1.4 | 50% | 6% | 105B | 4,073 | 0.195 |
| Strong | 1.6 | 42% | 10% | 144B | 4,798 | 0.167 |
| **AGI*** | 2.2 | 35% | 40% | 266B | 6,751 | 0.112 |
| ASI | 3.0 | 25% | 60% | 418B | 8,868 | 0.079 |

*AGI is the "decision-relevant" scenario

**Key findings:**
- All floors exceed the poverty line (3,324 NIS) -- poverty eliminated in all scenarios
- Gini coefficient falls from 0.363 to 0.079--0.195 (46--78% reduction)
- Fiscal balance by construction: floors adjust automatically to available revenue

## Repository Structure

```
mosaic-model/
├── docs/                    # Web interface (GitHub Pages)
│   ├── index.html
│   ├── app.js, data.js, formulas.js
│   └── style.css
│
├── replication/             # Replication code
│   ├── config.py            # Parameters and data sources
│   ├── functions.py         # Calculation functions
│   ├── run_all.py           # Execute all notebooks
│   ├── layer1_inputs.ipynb  # Exogenous inputs
│   ├── layer2_calculations.ipynb  # CES, revenue
│   ├── layer3_simulation.ipynb    # Microsimulation
│   ├── layer4_channel3.ipynb      # UHI funding
│   └── data/                # Source data (CSV)
│
└── paper/                   # LaTeX paper
    ├── mosaic.tex
    ├── BibFile.bib
    └── figures/
```

## Quick Start

### Run Replication

```bash
cd replication
pip install pandas numpy scipy jupyter nbformat nbclient
python run_all.py
```

### Build Paper

```bash
cd paper
xelatex mosaic && bibtex mosaic && xelatex mosaic && xelatex mosaic
```

## Citation

```bibtex
@techreport{mosaic2026,
  title={When AI Takes Our Jobs, It Should Also Pay Our Wages},
  author={Schreiber, Daniel and Shapira, Niv},
  year={2026},
  month={January},
  institution={Mosaic AI Policy Institute},
  url={https://github.com/Mosaic-AI-Policy-Institute/mosaic-model}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
