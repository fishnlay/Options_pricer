import math, numpy as np
from scipy.stats import norm

class BlackScholes:
    """
    Black-Scholes option pricing model for European Vanilla Options.

    """
    @staticmethod
    def _d1(option):
        return (math.log(option.S / option.K) + (option.r - option.q + \
                0.5 * option.sigma**2) * option.T) / (option.sigma * math.sqrt(option.T))

    @staticmethod
    def price(option):
        """
        Price an option using the Black-Scholes formula.
        """
        d1 = BlackScholes._d1(option)
        d2 = d1 - option.sigma * math.sqrt(option.T)

        if option.option_type.lower() == "call":
            price = option.S * math.exp(-option.q * option.T) * norm.cdf(d1) - \
                option.K * math.exp(-option.r * option.T) * norm.cdf(d2)
        elif option.option_type.lower() == "put":
            price = option.K * math.exp(-option.r * option.T) * norm.cdf(-d2) - \
                option.S * math.exp(-option.q * option.T) * norm.cdf(-d1)
        return price   

    @staticmethod
    def delta(option):
        if option.option_type.lower() == "call":
            return math.exp(-option.q * option.T) * norm.cdf(BlackScholes._d1(option))
        elif option.option_type.lower() == "put":
            return math.exp(-option.q * option.T) * (norm.cdf(BlackScholes._d1(option)) - 1)

    @staticmethod
    def gamma(option):
        return math.exp(-option.q * option.T) * norm.pdf(BlackScholes._d1(option)) / (option.S * \
                        option.sigma * math.sqrt(option.T))

    @staticmethod
    def vega(option):
        return option.S * math.exp(-option.q * option.T) * norm.pdf(BlackScholes._d1(option)) * math.sqrt(option.T)
    
    @staticmethod
    def theta(option):
        d1 = BlackScholes._d1(option)
        d2 = d1 - option.sigma * math.sqrt(option.T)
        term1 = -(option.S * math.exp(-option.q * option.T) * norm.pdf(d1) * option.sigma) / (2 * math.sqrt(option.T))
        if option.option_type.lower() == "call":
            term2 = option.q * option.S * math.exp(-option.q * option.T) * norm.cdf(d1)
            term3 = option.r * option.K * math.exp(-option.r * option.T) * norm.cdf(d2)
            return term1 - term2 - term3
        elif option.option_type.lower() == "put":
            term2 = option.q * option.S * math.exp(-option.q * option.T) * norm.cdf(-d1)
            term3 = option.r * option.K * math.exp(-option.r * option.T) * norm.cdf(-d2)
            return term1 + term2 + term3

    @staticmethod
    def rho(option):
        d2 = BlackScholes._d1(option) - option.sigma * math.sqrt(option.T)
        if option.option_type.lower() == "call":
            return option.K * option.T * math.exp(-option.r * option.T) * norm.cdf(d2)
        elif option.option_type.lower() == "put":
            return -option.K * option.T * math.exp(-option.r * option.T) * norm.cdf(-d2)

class MonteCarlo:
    """
    Monte-Carlo pricing model for European Vanilla options

    Inputs:
    - distribution (required): distribution model for paths generation (LogNormal, JumpMerton, VarianceGamma)
    - m: number of simulations
    - n: number of steps
    - random_seed: seed for random number generation
    - epsilon: increment for greeks computation

    """
    @staticmethod
    def price(option, distribution, m=1_000_000, n=100, random_seed=42):
        """
        Returns the price of a European option using Monte-Carlo simulation

        m: number of simulations 
        n: number of steps
        random_seed: random seed in the simulation
        """
        S, K, T, r, sigma, q, option_type= option.S, option.K, option.T, option.r, option.sigma, option.q, option.option_type
        discount_factor = math.exp(-r * T)

        paths = distribution.generate_paths(S, T, r, sigma, q, n, m, random_seed)

        if option_type.lower() == 'call':
            payoffs = np.maximum(paths[:, -1] - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - paths[:, -1], 0)
        else:
            raise ValueError("Invalid option_type. Use 'call' or 'put'.")

        price = discount_factor * np.mean(payoffs)
        return price
    
    @staticmethod
    def delta(option, distribution, m=10_000, n=100, epsilon=1e-4, random_seed=42):
        """
        Calculate Delta using finite difference methods.

        Parameters:
        - m: Number of simulations
        - n: Number of time steps
        - epsilon: Small perturbation for finite difference calculations
        """
        option.S += epsilon
        price_up = option.price(model=MonteCarlo(), distribution=distribution, m=m, n=n, random_seed=random_seed)
        option.S -= 2 * epsilon
        price_down = option.price(model=MonteCarlo(), distribution=distribution, m=m, n=n, random_seed=random_seed)

        #Reset S to its initial value
        option.S += epsilon
        return (price_up - price_down) / (2 * epsilon)

    @staticmethod
    def gamma(option, distribution, m=10_000, n=100, epsilon=1e-4, random_seed=42):
        """
        Calculate Gamma using finite difference methods.

        Parameters:
        - m: Number of simulations
        - n: Number of time steps
        - epsilon: Small perturbation for finite difference calculations
        """
        base_price = option.price(model=MonteCarlo(), distribution=distribution, m=m, n=n, random_seed=random_seed)
        option.S += epsilon
        price_up = option.price(model=MonteCarlo(), distribution=distribution, m=m, n=n, random_seed=random_seed)
        option.S -= 2 * epsilon
        price_down = option.price(model=MonteCarlo(), distribution=distribution, m=m, n=n, random_seed=random_seed)

        #Reset S to its initial value
        option.S += epsilon
        return (price_up - 2 * base_price + price_down) / (epsilon**2)

    @staticmethod
    def vega(option, distribution, m=10_000, n=100, epsilon=1e-4, random_seed=42):
        """
        Calculate Vega using finite difference methods.

        Parameters:
        - m: Number of simulations
        - n: Number of time steps
        - epsilon: Small perturbation for finite difference calculations
        """
        base_price = option.price(model=MonteCarlo(), distribution=distribution, m=m, n=n, random_seed=random_seed)
        option.sigma += epsilon
        vega_price = option.price(model=MonteCarlo(), distribution=distribution, m=m, n=n, random_seed=random_seed)

        #Reset sigma to its initial value
        option.sigma -= epsilon
        return (vega_price - base_price) / epsilon

    @staticmethod
    def theta(option, distribution, m=10_000, n=100, epsilon=1e-4, random_seed=42):
        """
        Calculate Theta using finite difference methods.

        Parameters:
        - m: Number of simulations
        - n: Number of time steps
        - epsilon: Small perturbation for finite difference calculations
        """
        base_price = option.price(model=MonteCarlo(), distribution=distribution, m=m, n=n, random_seed=random_seed)
        option.T -= epsilon
        theta_price = option.price(model=MonteCarlo(), distribution=distribution, m=m, n=n, random_seed=random_seed)

        #Reset theta to its initial value
        option.T += epsilon
        return (theta_price - base_price) / epsilon

class LeastSquareMonteCarlo:
    """
    Least-Square Monte-Carlo pricing model for American options

    Inputs:
    - distribution (required): distribution model for paths generation (LogNormal, JumpMerton, VarianceGamma)
    - m: number of simulations
    - n: number of steps
    - random_seed: seed for random number generation
    - epsilon: increment for greeks computation

    """
    @staticmethod
    def price(option, distribution, m=1_000_000, n=100, random_seed=42):
        """
        Returns the price of a American option using Least-Square Monte-Carlo simulation

        m: number of simulations 
        n: number of steps
        """
        S, K, T, r, sigma, q, option_type= option.S, option.K, option.T, option.r, option.sigma, option.q, option.option_type
        discount_factor = math.exp(-r * T)

        paths = distribution.generate_paths(S, T, r, sigma, q, n, m, random_seed)

        if option_type == "call":
            payoffs = np.maximum(paths[:, -1] - K, 0)
        elif option_type == "put":
            payoffs = np.maximum(K - paths[:, -1], 0)
        
        # Backward induction
        for t in range(n - 2, -1, -1):
            in_the_money = (paths[:, t] < K) if option_type == "put" else (paths[:, t] > K)
            itm_indices = np.where(in_the_money)[0]

            if len(itm_indices) > 0:
                # Regression to estimate continuation value
                X = paths[itm_indices, t]
                Y = payoffs[itm_indices] * discount_factor
                regression_coeffs = np.polyfit(X, Y, 2)  # get the [a,b,c] coefficient for the degree 2 quadratic (to fit X and Y)
                continuation_values = np.polyval(regression_coeffs, X) #get the X0 and X1 values out of {aX0**2 + bX1 + c}

                # Immediate exercise value
                exercise_values = (K - X) if option_type == "put" else (X - K)

                # Determine whether to exercise or continue
                exercise = exercise_values > continuation_values
                payoffs[itm_indices[exercise]] = exercise_values[exercise]

            payoffs *= discount_factor  # Discount the payoffs

        # Calculate the option price as the average of discounted payoffs
        option_price = np.mean(payoffs) * discount_factor
        return option_price

    @staticmethod
    def delta(option, distribution, m=1_000_000, n=100, random_seed=42, epsilon=1e-5):
        """
        Calculate the Delta of the option: Sensitivity to stock price.
        """
        option.S += epsilon
        price_up = LeastSquareMonteCarlo.price(option, distribution=distribution, m=m, n=n, random_seed=random_seed)
        option.S -= 2*epsilon
        price_down = LeastSquareMonteCarlo.price(option, distribution=distribution, m=m, n=n, random_seed=random_seed)

        #Reset S to its initial value
        option.S += epsilon
        delta = (price_up - price_down) / (2 * epsilon)
        return delta

    @staticmethod
    def gamma(option, distribution, m=1_000_000, n=100, random_seed=42, epsilon=1e-5):
        """
        Calculate the Gamma of the option: Sensitivity to Delta.
        """        
        option.S += epsilon
        delta_up = LeastSquareMonteCarlo.delta(option, distribution=distribution, m=m, n=n, random_seed=random_seed)
        option.S -= 2* epsilon
        delta_down = LeastSquareMonteCarlo.delta(option, distribution=distribution, m=m, n=n, random_seed=random_seed)

        #Reset S to its initial value
        option.S += epsilon
        gamma = (delta_up - delta_down) / epsilon
        return gamma

    @staticmethod
    def vega(option, distribution, m=1_000_000, n=100, random_seed=42, epsilon=1e-5):
        """
        Calculate the Vega of the option: Sensitivity to volatility.
        """        
        option.sigma += epsilon
        price_up = LeastSquareMonteCarlo.price(option, distribution=distribution, m=m, n=n, random_seed=random_seed)
        
        option.sigma -= 2*epsilon
        price_down = LeastSquareMonteCarlo.price(option, distribution=distribution, m=m, n=n, random_seed=random_seed)
        
        #Reset sigma to its initial value
        option.sigma += epsilon
        vega = (price_up - price_down) / (2 * epsilon)
        return vega

    @staticmethod
    def theta(option, distribution, m=1_000_000, n=100, random_seed=42, epsilon=1e-5):
        """
        Calculate the Theta of the option: Sensitivity to time to maturity.
        """        
        option.T += epsilon
        price_up = LeastSquareMonteCarlo.price(option, distribution=distribution, m=m, n=n, random_seed=random_seed)
        
        option.T -= 2*epsilon
        price_down = LeastSquareMonteCarlo.price(option, distribution=distribution, m=m, n=n, random_seed=random_seed)
        
        #Reset T to its initial value
        option.T += epsilon
        theta = (price_up - price_down) / (2 * epsilon)
        return theta   

class CoxRossRubstein:
    """
    Cox-Ross-Rubstein binomial tree pricing model

    Parameters:
    - american(bool): True for yes
    - n: number of steps
    """
    @staticmethod
    def price(option, american:bool, n=100):
        S, K, T, r, sigma, q, option_type = option.S, option.K, option.T, option.r, option.sigma, \
                                                option.q, option.option_type
        
        dt = T/n
        u = math.exp(sigma * math.sqrt(dt))
        d = 1/u
        p = (math.exp((r - q) * dt) - d) / (u - d)
        stock_prices = [S * (u**j) * (d**(n-j)) for j in range(n+1)]

        if option_type == "call":
            option_values = [max(0, price - K) for price in stock_prices]
        elif option_type == "put":
            option_values = [max(0, K - price) for price in stock_prices]
        
        # Work backwards through the tree
        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                continuation_value = math.exp(-r * dt) * (p * option_values[j + 1] + (1 - p) * option_values[j])
                if american:
                    if option_type == "call":
                        intrinsic_value = max(0, stock_prices[j] - K)
                    elif option_type == "put":
                        intrinsic_value = max(0, K - stock_prices[j])
                    
                    option_values[j] = max(continuation_value, intrinsic_value)
                else:
                    option_values[j] = continuation_value
                
                stock_prices[j] = stock_prices[j] * d

        return option_values[0]
    
    @staticmethod
    def delta(option, american:bool, n=100):
        """
        Calculate the Delta of an option using the CRR model.
        """
        S,T,sigma = option.S, option.T, option.sigma

        dt = T / n
        u = math.exp(sigma * math.sqrt(dt))
        d = 1 / u
        og_S = option.S
        n -= 1
        option.S = S*u
        option_up = CoxRossRubstein.price(option, american, n)

        option.S = og_S
        option.S = S*d
        option_down = CoxRossRubstein.price(option, american, n)

        #Reset n and S to their initial value
        n += 1
        option.S = og_S
        return (option_up - option_down) / (S * (u - d))

    @staticmethod
    def gamma(option, american:bool, n=100):
        """
        Calculate the Gamma of an option using the CRR model.
        """
        S, T, sigma = option.S, option.T, option.sigma
        
        dt = T / n
        u = math.exp(sigma * math.sqrt(dt))
        d = 1 / u

        og_S = option.S
        option.S = S * u
        n -= 1
        option_up = CoxRossRubstein.price(option, american, n)

        option.S = og_S
        option.S = S * d
        option_down = CoxRossRubstein.price(option, american, n)

        option.S = og_S
        option_base = CoxRossRubstein.price(option, american, n)

        #Reset n to its initial value
        n += 1
        return (option_up - 2 * option_base + option_down) / ((S * (u - d)) ** 2)

    @staticmethod
    def theta(option, american:bool, n=100):
        """
        Calculate the Theta of an option using the CoxRossRubstein model.
        """
        T = option.T
        
        option_now = CoxRossRubstein.price(option, american, n)

        n -= 1
        option_future = CoxRossRubstein.price(option, american, n)

        #Reset the value of n
        n += 1
        dt = T/n

        return (option_future - option_now) / dt