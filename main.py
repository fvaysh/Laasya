import numpy as np
import pandas as pd
import scipy.stats as si
import matplotlib.pyplot as plt

# Black-Scholes Option Pricing
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)

    return price

# Delta Hedging
def delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return si.norm.cdf(d1)

def delta_neutral_hedging(S0, K, T, r, sigma, steps=100, dt=1/252):
    S = [S0]
    np.random.seed(42)
    
    for _ in range(steps):
        S_t = S[-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.normal())
        S.append(S_t)

    S = np.array(S)
    deltas = delta(S, K, T, r, sigma)
    
    hedge_pnl = np.zeros(steps)
    hedge_pnl[0] = 0

    for i in range(1, steps):
        hedge_pnl[i] = hedge_pnl[i - 1] + (deltas[i] - deltas[i - 1]) * S[i]

    return S, hedge_pnl

# Monte Carlo Simulation
def monte_carlo_risk(S0, K, T, r, sigma, simulations=10000, steps=100, dt=1/252):
    final_pnls = []

    for _ in range(simulations):
        S, hedge_pnl = delta_neutral_hedging(S0, K, T, r, sigma, steps, dt)
        final_pnls.append(hedge_pnl[-1])

    return final_pnls

# Portfolio Optimization using Markowitz Model
def portfolio_optimization(returns, risk_free_rate=0.02):
    mean_returns = np.mean(returns, axis=0)
    cov_matrix = np.cov(returns.T)

    num_assets = len(mean_returns)
    num_portfolios = 10000

    results = np.zeros((3, num_portfolios))

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        portfolio_return = np.sum(weights * mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = sharpe_ratio

    return results

# Generate Synthetic Data for Portfolio Optimization
def generate_synthetic_stock_data(n_assets=5, n_days=252):
    np.random.seed(42)
    returns = np.random.normal(loc=0.0005, scale=0.02, size=(n_days, n_assets))
    return returns

# Parameters
S0 = 100
K = 100
T = 1
r = 0.05
sigma = 0.2
steps = 100

# Monte Carlo Simulation
final_pnls = monte_carlo_risk(S0, K, T, r, sigma)

# Portfolio Optimization
synthetic_returns = generate_synthetic_stock_data()
optimization_results = portfolio_optimization(synthetic_returns)

# Plot Monte Carlo Risk Analysis
plt.figure(figsize=(12, 6))
plt.hist(final_pnls, bins=50, alpha=0.7, color="blue", edgecolor="black")
plt.title("Monte Carlo Simulated P&L Distribution")
plt.xlabel("Profit/Loss")
plt.ylabel("Frequency")
plt.grid()
plt.show()

# Plot Portfolio Optimization Results
plt.figure(figsize=(12, 6))
plt.scatter(optimization_results[1, :], optimization_results[0, :], c=optimization_results[2, :], cmap="viridis", marker="o")
plt.xlabel("Portfolio Volatility")
plt.ylabel("Portfolio Return")
plt.colorbar(label="Sharpe Ratio")
plt.title("Efficient Frontier with Monte Carlo Simulations")
plt.grid()
plt.show()
