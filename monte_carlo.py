import math
import numpy as np
import pandas as pd
import datetime
import scipy.stats as stats
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr

# Import stock data
def get_stock_data(stocks, start, end):
    stock_data = pdr.get_data_yahoo(stocks, start, end)
    stock_data = stock_data['Close']
    returns = stock_data.pct_change()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix

# List of stocks
stock_list = ['CBA', 'BHP', 'TLS', 'NAB', 'WBC', 'STO']
stocks = [stock + '.AX' for stock in stock_list]
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=300)

mean_returns, cov_matrix = get_stock_data(stocks, start_date, end_date)

# Generate random weights for portfolio and normalize
weights = np.random.random(len(mean_returns))
weights /= np.sum(weights)

# Monte Carlo simulation parameters
mc_sims = 400 # number of simulations
time_horizon = 100 # timeframe in days

# Create mean matrix
mean_matrix = np.full(shape=(time_horizon, len(weights)), fill_value=mean_returns).T

# Initialize matrix to store simulation results
portfolio_simulations = np.full(shape=(time_horizon, mc_sims), fill_value=0.0)

initial_portfolio_value = 10000

# Run Monte Carlo simulation
for sim in range(mc_sims):
    random_normals = np.random.normal(size=(time_horizon, len(weights))) # uncorrelated random variables
    lower_triangular_matrix = np.linalg.cholesky(cov_matrix) # Cholesky decomposition to lower triangular matrix
    daily_returns = mean_matrix + np.inner(lower_triangular_matrix, random_normals) # correlated daily returns
    portfolio_simulations[:, sim] = np.cumprod(np.inner(weights, daily_returns.T) + 1) * initial_portfolio_value

# Plot Monte Carlo simulation results
plt.plot(portfolio_simulations)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('MC simulation of a stock portfolio')
plt.show()

# Function to calculate Value at Risk (VaR)
def calculate_var(returns, alpha=5):
    """ Input: pandas series of returns
        Output: percentile on return distribution to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    else:
        raise TypeError("Expected a pandas data series.")

# Function to calculate Conditional Value at Risk (CVaR)
def calculate_cvar(returns, alpha=5):
    """ Input: pandas series of returns
        Output: CVaR or Expected Shortfall to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        below_var = returns <= calculate_var(returns, alpha=alpha)
        return returns[below_var].mean()
    else:
        raise TypeError("Expected a pandas data series.")

# Calculate VaR and CVaR
portfolio_results = pd.Series(portfolio_simulations[-1, :])
var_5 = initial_portfolio_value - calculate_var(portfolio_results, alpha=5)
cvar_5 = initial_portfolio_value - calculate_cvar(portfolio_results, alpha=5)

print('VaR_5 ${}'.format(round(var_5, 2)))
print('CVaR_5 ${}'.format(round(cvar_5, 2)))

# Initial derivative parameters for option pricing
stock_price = 101.15          # stock price
strike_price = 98.01          # strike price
volatility = 0.0991           # volatility (%)
risk_free_rate = 0.01         # risk-free rate (%)
num_time_steps = 10           # number of time steps
num_simulations = 1000        # number of simulations

market_option_value = 3.86    # market price of option
time_to_maturity = ((datetime.date(2022, 3, 17) - datetime.date(2022, 1, 17)).days + 1) / 365  # time in years
print(time_to_maturity)

# Precompute constants for option pricing
delta_t = time_to_maturity / num_time_steps
drift = (risk_free_rate - 0.5 * volatility**2) * delta_t
volatility_sqrt_t = volatility * np.sqrt(delta_t)
log_stock_price = np.log(stock_price)

# Standard error placeholders
sum_CT = 0
sum_CT_squared = 0

# Monte Carlo simulation for option pricing
for i in range(num_simulations):
    log_stock_price_t = log_stock_price
    for j in range(num_time_steps):
        log_stock_price_t = log_stock_price_t + drift + volatility_sqrt_t * np.random.normal()

    simulated_stock_price = np.exp(log_stock_price_t)
    payoff = max(0, simulated_stock_price - strike_price)
    sum_CT += payoff
    sum_CT_squared += payoff * payoff

# Compute expectation and standard error
option_value = np.exp(-risk_free_rate * time_to_maturity) * sum_CT / num_simulations
option_variance = np.sqrt((sum_CT_squared - sum_CT * sum_CT / num_simulations) * np.exp(-2 * risk_free_rate * time_to_maturity) / (num_simulations - 1))
standard_error = option_variance / np.sqrt(num_simulations)

print("Call value is ${0} with SE +/- {1}".format(np.round(option_value, 2), np.round(standard_error, 2)))

# Precompute constants for vectorized Monte Carlo simulation
delta_t = time_to_maturity / num_time_steps
drift = (risk_free_rate - 0.5 * volatility**2) * delta_t
volatility_sqrt_t = volatility * np.sqrt(delta_t)
log_stock_price = np.log(stock_price)

# Vectorized Monte Carlo simulation for option pricing
random_normals = np.random.normal(size=(num_time_steps, num_simulations))
delta_log_stock_price_t = drift + volatility_sqrt_t * random_normals
log_stock_price_t = log_stock_price + np.cumsum(delta_log_stock_price_t, axis=0)
log_stock_price_t = np.concatenate((np.full(shape=(1, num_simulations), fill_value=log_stock_price), log_stock_price_t))

# Compute expectation and standard error
simulated_stock_prices = np.exp(log_stock_price_t)
option_payoffs = np.maximum(0, simulated_stock_prices - strike_price)
option_value = np.exp(-risk_free_rate * time_to_maturity) * np.sum(option_payoffs[-1]) / num_simulations

option_variance = np.sqrt(np.sum((option_payoffs[-1] - option_value)**2) / (num_simulations - 1))
standard_error = option_variance / np.sqrt(num_simulations)

print("Call value is ${0} with SE +/- {1}".format(np.round(option_value, 2), np.round(standard_error, 2)))

# Plot the probability distribution of the option price
x1 = np.linspace(option_value - 3 * standard_error, option_value - 1 * standard_error, 100)
x2 = np.linspace(option_value - 1 * standard_error, option_value + 1 * standard_error, 100)
x3 = np.linspace(option_value + 1 * standard_error, option_value + 3 * standard_error, 100)

s1 = stats.norm.pdf(x1, option_value, standard_error)
s2 = stats.norm.pdf(x2, option_value, standard_error)
s3 = stats.norm.pdf(x3, option_value, standard_error)

plt.fill_between(x1, s1, color='tab:blue', label='> 1 Std Dev')
plt.fill_between(x2, s2, color='cornflowerblue', label='1 Std Dev')
plt.fill_between(x3, s3, color='tab:blue')

plt.plot([option_value, option_value], [0, max(s2) * 1.1], 'k', label='Theoretical Value')
plt.plot([market_option_value, market_option_value], [0, max(s2) * 1.1], 'r', label='Market Value')

plt.ylabel("Probability")
plt.xlabel("Option Price")
plt.legend()
plt.show()
