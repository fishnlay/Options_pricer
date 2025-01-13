import input_validation as iv

class Options:
    S = iv.validated_number(validator=iv.validate_positive)
    K = iv.validated_number(validator=iv.validate_positive)
    T = iv.validated_number(validator=iv.validate_positive_t)
    r = iv.validated_number(validator=iv.validate_r)
    sigma = iv.validated_number(validator=iv.validate_sigma)
    q = iv.validated_number(validator=iv.validate_q)

    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, q: float, option_type: str):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.option_type = option_type  # Validate option_type

    @property
    def option_type(self):
        return self._option_type

    @option_type.setter
    def option_type(self, value):
        if value not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")
        self._option_type = value

class VanillaOption(Options):
    def __init__(self, S, K, T, r, sigma, q, option_type):
        """
        S: Current price of the underlying asset (stock price)
        K: Strike price of the option
        T: Time to expiration (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility of the underlying asset (annualized)
        q: Annlualized Dividend Yield
        option_type: "call" or "put"
        """
        super().__init__(S, K, T, r, sigma, q, option_type)
    
    def price(self, model, **kwargs):
        return model.price(self, **kwargs)
    
    def get_greeks(self, model, **kwargs):
        """
        Get the greeks for the option

        Parameters:
        - model: the pricing model used (class)
        """
        greeks = {
            "Delta": model.delta(self, **kwargs),
            "Gamma": model.gamma(self, **kwargs),
            "Theta": model.theta(self, **kwargs),
        }
         # Add "Rho" if the model has the method
        if hasattr(model, 'vega'):
            greeks["Vega"] = model.vega(self, **kwargs)
        if hasattr(model, 'rho'):
            greeks["Rho"] = model.rho(self, **kwargs)
        return greeks




