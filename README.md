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
