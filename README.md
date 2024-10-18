### Developed by : VINOD KUMAR S
### Reg no: 212222240116
### Date: 
# Ex.No: 6               HOLT WINTERS METHOD

### AIM:

### ALGORITHM:
1.Load and Filter Data: Read the dataset, convert the date column to datetime, and filter for the desired commodity.

2.Set Index and Resample: Set Date as the index and calculate monthly averages for the commodity's prices.

3.Handle Missing Data: Drop any NaN values to ensure a clean dataset for analysis.

4.Split Data: Divide the data into training (all but the last 12 months) and test sets (last 12 months).

5.Fit Holt-Winters Model: Apply the Holt-Winters model with additive trend and seasonality to the training data.

6.Test and Final Forecast: Forecast sales for the test period (last 12 months) and make future predictions (next 12 months).

7.Plot Results: Create separate plots for test predictions and final forecasts, along with the training data.

### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = 'petrol.csv'
data = pd.read_csv(file_path)

# Convert 'Date' to datetime (adjusting for format)
data['Date'] = pd.to_datetime(data['Date'], format='%B %d %Y')

# Select data for 'Delhi' prices
commodity_data = data[['Date', 'Delhi']]

# Set 'Date' as the index
commodity_data.set_index('Date', inplace=True)

# Resample the data to monthly averages
monthly_data = commodity_data['Delhi'].resample('M').mean()

# Drop any NaN values
monthly_data = monthly_data.dropna()

# Split data into training and test sets (we use the last 12 months as test data)
train_data = monthly_data[:-12]
test_data = monthly_data[-12:]

# Fit Holt-Winters model (additive trend and seasonality)
hw_model = ExponentialSmoothing(train_data, 
                                trend='additive', 
                                seasonal='additive', 
                                seasonal_periods=12).fit()

# Test Predictions for the last 12 months
test_predictions = hw_model.forecast(steps=12)

# Drop any NaN values in the test data and predictions for valid comparison
test_data_clean = test_data.dropna()
test_predictions_clean = test_predictions[:len(test_data_clean)]

# Calculate the Mean Squared Error (MSE) between the test data and predictions
mse = mean_squared_error(test_data_clean, test_predictions_clean)
print(f'Test Mean Squared Error: {mse}')

# Final Forecast for the next 12 months beyond the test period
final_forecast = hw_model.forecast(steps=12)

# Plot for Test Predictions
plt.figure(figsize=(10, 6))

# Plot the training data
plt.plot(train_data, label='Training Data', color='blue')

# Plot the actual test data
plt.plot(test_data, label='Test Data', color='orange')

# Plot the test predictions
plt.plot(test_predictions, label='Test Predictions', color='green')

plt.title('Test Predictions (Holt-Winters)')
plt.legend(loc='best')
plt.show()


# Plot for Final Forecast
plt.figure(figsize=(10, 6))

# Plot the training data
plt.plot(train_data, label='Training Data', color='blue')

# Plot the final forecast
plt.plot(final_forecast, label='Final Forecast (Next 12 months)', color='red')

plt.title('Final Predictions (Next 12 Months)')
plt.legend(loc='best')
plt.show()

```

### OUTPUT:


## TEST_PREDICTION
![image](https://github.com/user-attachments/assets/e6200043-d718-45a0-96d3-ee6777bdb7cd)



## FINAL_PREDICTION
![image](https://github.com/user-attachments/assets/0c7dffd1-6b9f-4f3c-860b-1b23e15114f1)


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
