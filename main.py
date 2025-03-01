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

# Delta-Neutral Hedging Strategy
def delta_neutral_hedging(S0, K, T, r, sigma, steps=100, dt=1/252):
    S = [S0]
    np.random.seed(42)
    
    for _ in range(steps):
        S_t = S[-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.normal())
        S.append(S_t)

    S = np.array(S)
    deltas = si.norm.cdf((np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)))
    
    hedge_pnl = np.zeros(steps)
    hedge_pnl[0] = 0  # Initial hedge cost

    for i in range(1, steps):
        hedge_pnl[i] = hedge_pnl[i - 1] + (deltas[i] - deltas[i - 1]) * S[i]

    return S, hedge_pnl

# Monte Carlo Simulation for Risk Analysis
def monte_carlo_risk(S0, K, T, r, sigma, simulations=10000, steps=100, dt=1/252):
    final_pnls = []

    for _ in range(simulations):
        S, hedge_pnl = delta_neutral_hedging(S0, K, T, r, sigma, steps, dt)
        final_pnls.append(hedge_pnl[-1])

    return final_pnls

# Parameters
S0 = 100  # Initial stock price
K = 100   # Strike price
T = 1     # Time to expiration (1 year)
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility
steps = 100  # Steps for hedging

# Running Monte Carlo Risk Analysis
final_pnls = monte_carlo_risk(S0, K, T, r, sigma)

# Plot results
plt.hist(final_pnls, bins=50, alpha=0.7, color="blue", edgecolor="black")
plt.title("Monte Carlo Simulated P&L Distribution")
plt.xlabel("Profit/Loss")
plt.ylabel("Frequency")
plt.grid()
plt.show()
