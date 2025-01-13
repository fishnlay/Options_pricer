# Options_pricer

This python module aims to provide various option pricing models.

As of now (1/13/2024), it contains the following pricing models:
- Vanilla Black-Scholes for European Options
- Vanilla Monte-Carlo for European Options
- Least-Square Monte-Carlo for Vanilla American Options
- Cox-Ross-Rubstein binomial tree for Vanilla European and American Options

The paths dependant models (Monte-Carlo and LSM) contains 3 distribution types:
- LogNormal (or Geometric Brownian Motion)
- Merton Jump diffusion
- Variance Gamma

User guide:

The necessary modules must be imported:
```python
from option_pricer import VanillaOption
from pricing_models import BlackScholes, MonteCarlo, LeastSquareMonteCarlo, CoxRossRubstein
from distribution_models import LogNormalDistribution, MertonJumpDiffusion, VarianceGamma
```

An object containing the option is the created (VanillaOption is the only one for now):
```python
call_100 = VanillaOption(S, K, T, r, sigma, q, "call")
```

The model choosen can be inputed into the price and get_greeks method as follow:
```python
call_100.price(BlackScholes)
call_100.get_greeks(BlackScholes)
```

For path-dependant pricing models, additional arguments can be entered:
```python
call_100.price(MonteCarlo, distribution=LogNormalDistribution, n=100_000)

#Or

call_100.price(CoxRossRubstein, american=False)
```
