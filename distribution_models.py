import numpy as np

class LogNormalDistribution:
    @staticmethod
    def generate_paths(S, T, r, sigma, q, n, m, random_seed=42):
        """
        n:number of time steps
        m: number of simulated paths
        """
        np.random.seed(random_seed)
        dt = T / n
        paths = np.zeros((m, n + 1))
        paths[:, 0] = S
        for t in range(1, n + 1):
            z = np.random.standard_normal(m)
            paths[:, t] = paths[:, t - 1] * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) *  z)
            # S_t = S_t-1 * exp(r - 0.5 * sigma**2) * dt + sigma * sqrt(dt) * Z
        return paths

class MertonJumpDiffusion:
    def __init__(self, jump_intensity, jump_mean, jump_std):
        """
        jump_intensity: Lambda, the average number of jumps per year
        jump_mean: Mean of the jump size
        jump_std: Standard deviation of the jump size
        """
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std

    def generate_paths(self, S, T, r, sigma, q, n, m, random_seed=42):
        """
        Generate Monte Carlo paths using the Merton Jump Diffusion model.

        Parameters are the same as LogNormalDistribution, with additional jump parameters.
        """
        np.random.seed(random_seed)
        dt = T / n
        drift = (r - q - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)

        # Simulate Poisson jumps
        poisson_events = np.random.poisson(self.jump_intensity * dt, (m, n))
        jump_sizes = np.random.normal(self.jump_mean, self.jump_std, (m, n))
        jumps = poisson_events * (np.exp(jump_sizes) - 1)

        # Simulate paths
        increments = np.random.normal(drift, diffusion, (m, n)) + jumps
        paths = S * np.exp(np.cumsum(increments, axis=1))
        paths = np.insert(paths, 0, S, axis=1)  # Include initial price
        return paths

class VarianceGamma:
    def __init__(self, theta, nu, sigma):
        """
        theta: Drift of the process
        nu: Variance of the subordinator
        sigma: Volatility of the Brownian motion
        """
        self.theta = theta
        self.nu = nu
        self.sigma = sigma

    def generate_paths(self, S, T, r, q, n, m, random_seed=42):
        """
        Generate Monte Carlo paths using the Variance Gamma model.

        Parameters are similar to LogNormalDistribution, with additional VG-specific parameters.
        """
        np.random.seed(random_seed)
        dt = T / n

        # Simulate Gamma process
        gamma_process = np.random.gamma(shape=dt / self.nu, scale=self.nu, size=(m, n))

        # Simulate Variance Gamma increments
        drift = (r - q) * dt + self.theta * gamma_process
        diffusion = self.sigma * np.sqrt(gamma_process)
        increments = drift + np.random.normal(0, 1, (m, n)) * diffusion

        # Simulate paths
        paths = S * np.exp(np.cumsum(increments, axis=1))
        paths = np.insert(paths, 0, S, axis=1)  # Include initial price
        return paths
