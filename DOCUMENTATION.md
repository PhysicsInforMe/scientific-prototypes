# Documentation Tasks

## Task 1: Complete Code Documentation

Review ALL Python files in timeseries_toolkit/ and ensure each has:

### Module Level
- Module docstring explaining purpose, main classes/functions, and usage example

### Class Level
- Class docstring with description, attributes list, and example usage
- All __init__ parameters documented

### Function/Method Level
- Google-style docstring with:
  - One-line description
  - Args: each parameter with type and description
  - Returns: type and description
  - Raises: exceptions that may be raised
  - Example: working code example

### Inline Comments
- Comment non-obvious logic
- Explain mathematical formulas with references
- Note any assumptions or edge cases

## Task 2: Create Technical Documentation

Create docs/TECHNICAL.md with these sections:

### 1. Architecture Overview
- Package structure diagram
- Module dependencies
- Data flow between components

### 2. Module Documentation

For each module, document:

#### preprocessing/fractional_diff.py
- Mathematical foundation (FFD method)
- Weight computation formula: w_k = -w_{k-1} * (d-k+1) / k
- Stationarity vs memory trade-off
- Reference: López de Prado (2018)

#### preprocessing/filtering.py
- Two-stage approach: STL + SARIMA
- Seasonal period auto-detection logic
- Signal Dominance Index formula
- When to use vs skip filtering

#### preprocessing/imputation.py
- MICE algorithm explanation
- Mixed-frequency alignment strategy
- Monthly to quarterly pivot logic

#### models/kalman.py
- State-space representation
- UnobservedComponents specification
- Comparison with ARIMA approach

#### models/regime.py
- Hidden Markov Model theory
- GMMHMM vs standard HMM
- BIC-based model selection
- Viterbi decoding explanation

#### models/forecaster.py
- GlobalBoostForecaster design
- Feature engineering approach
- Why LightGBM for time series

#### validation/causality.py
- CCM theory (Sugihara et al. 2012)
- Shadow manifold reconstruction
- Granger causality assumptions
- When CCM vs Granger

#### validation/diagnostics.py
- Seven diagnostic tests explained
- Scoring system logic
- Interpretation guidelines

### 3. Usage Examples
- End-to-end workflow example
- Common use cases
- Integration patterns

### 4. API Reference
- Complete function signatures
- Parameter constraints
- Return value specifications

## Task 3: Update README.md

Ensure README.md has:
- Project description
- Installation instructions
- Quick start example
- Link to technical docs
- Test coverage summary
- License

## Task 4: Create requirements.txt

List all dependencies with versions.

Run these tasks in order.
