# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

# Fetch historical data for Bitcoin
stock = 'BTC-USD'
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=365) 
btc_data = pdr.get_data_yahoo(stock, start=startDate, end=endDate)

# Calculate daily log returns
btc_data['Log_Returns'] = np.log(btc_data['Adj Close'] / btc_data['Adj Close'].shift(1))

# Calculate historical volatility (annualized)
historical_volatility = btc_data['Log_Returns'].std() * np.sqrt(252) # I'm assuming 252 trading days in a year here

# Monte Carlo Parameters
initial_investment = 10000  # Total investment amount in USD
initial_price = btc_data['Adj Close'][-1]  # Last available price
mc_sims = 1000  # Number of Monte Carlo simulations
days_in_year = 252  # Trading days

# Simulate daily returns based on historical volatility
def simulate_daily_returns(volatility, days):
    return np.random.normal(0, volatility, days)

# Simulate price paths
def simulate_price_paths(initial_price, daily_returns):
    price_paths = [initial_price]
    for daily_return in daily_returns:
        price_paths.append(price_paths[-1] * (1 + daily_return))
    return np.array(price_paths[1:])

# Monte Carlo Sim
final_values_lump_sum = []
final_values_dca = []

for _ in range(mc_sims):
    daily_returns = simulate_daily_returns(historical_volatility / np.sqrt(days_in_year), days_in_year)
    price_paths = simulate_price_paths(initial_price, daily_returns)
    
    # Lump-sum investment
    final_value_lump_sum = initial_investment * (price_paths[-1] / initial_price)
    final_values_lump_sum.append(final_value_lump_sum)
    
    # DCA investment
    dca_investment = initial_investment / 12
    # Calculate the interval for investments in trading days
    investment_interval = round(days_in_year / 12)
    dca_values = []

    for i in range(12):
        # Calculate the index for the current investment
        investment_index = min(i * investment_interval, len(price_paths) - 1)
        dca_values.append(dca_investment * (price_paths[investment_index] / initial_price))

    final_values_dca.append(sum(dca_values))

# Visualization
sns.set(style="whitegrid")

# Prepare the statements with the average final portfolio values
avg_lump_sum = np.mean(final_values_lump_sum)
avg_dca = np.mean(final_values_dca)
statements = f"Average final Bitcoin portfolio value (Lump Sum): ${avg_lump_sum:,.2f}\n" \
             f"Average final Bitcoin portfolio value (DCA): ${avg_dca:,.2f}"

# Plotting
plt.figure(figsize=(12, 8))
sns.histplot(final_values_lump_sum, bins=50, color="blue", kde=True, label='Lump Sum')
sns.histplot(final_values_dca, bins=50, color="green", kde=True, label='DCA')
plt.legend()
plt.title('Lump Sum vs. DCA Investment Strategies', fontsize=16)
plt.xlabel('Final Portfolio Value', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.text(0.05, 0.95, statements, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# Show the plot
plt.show()