import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import math

st.title('Şube Workload Forecast ve Çalışan Sayısı Prediction ')

branches = [11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 46]
file_path = './Sabancı Servis Verisi.xlsx.csv'  #Buraya kendi file path'ini ekle #
df = pd.read_excel(file_path)
df['Date'] = pd.to_datetime(df['Kaydı Üzerine Alma Tarihi']).dt.date


branches = [branch for branch in branches]

col1, col2, col3 = st.columns([1,1,1])
with col2:
    selected_branch = st.selectbox('Şube', branches, key='page1_branch_select')


branch_data = df[df['Servis Noktası'] == selected_branch].copy()
branch_data['Date'] = pd.to_datetime(branch_data['Date'])
branch_data.set_index('Date', inplace=True)
weekly_data = branch_data.resample('W-Mon').size()
tables_container = st.container()

######  Exponential Smoothing Model  ######

st.subheader('Exponential Smoothing (ETS) Forecast')
train_size = int(len(weekly_data) * 0.8)
train_data, test_data = weekly_data.iloc[:train_size+1], weekly_data.iloc[train_size:]
ets_model = ExponentialSmoothing(train_data, seasonal_periods=12, trend='add', seasonal='add').fit()
forecast_steps = len(test_data) + 5  
forecast_ets = ets_model.forecast(steps=forecast_steps)
last_train_date = train_data.index[-1]
test_dates = pd.date_range(start=train_data.index[-1] + pd.Timedelta(days=7), periods=len(test_data), freq='W-MON')
forecast_ets_test_period = forecast_ets[:len(test_data)]

fig_ets = go.Figure()
fig_ets.add_trace(go.Scatter(x=test_dates, y=forecast_ets_test_period, mode='lines', name='Predict on Test Data', line=dict(dash='dot', color='red')))
fig_ets.add_trace(go.Scatter(x=train_data.index, y=train_data, mode='lines', name='Training Data'))
fig_ets.add_trace(go.Scatter(x=test_data.index, y=test_data, mode='lines', name='Test Data'))
forecast_index_ets = pd.date_range(start=test_data.index[-1], periods=len(forecast_ets), freq='W-MON')[1:]
fig_ets.add_trace(go.Scatter(x=forecast_index_ets, y=forecast_ets[1:], mode='lines', name='Forecast'))
fig_ets.update_layout(title=f'Şube {selected_branch} için Forecast', xaxis_title='Date', yaxis_title='Value')
st.plotly_chart(fig_ets, use_container_width=True)
rmse_ets = np.sqrt(mean_squared_error(test_data, forecast_ets[:len(test_data)]))
next_week_forecast_ets = forecast_ets_test_period[-1]



###### LSTM ######

st.subheader('Long-Short Term Memory (LSTM) Forecast')
def create_dataset(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        v = data[i:(i + time_steps), 0]
        X.append(v)
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(weekly_data.values.reshape(-1, 1))
time_steps = 24
X, y = create_dataset(scaled_data, time_steps)
test_size = int(len(X) * 0.2)
X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0) 
test_predictions = model.predict(X_test)
test_predictions = scaler.inverse_transform(test_predictions)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
input_seq = scaled_data[-time_steps:].reshape(1, time_steps, 1)
forecasted_values = []
for _ in range(15):
    pred = model.predict(input_seq)
    forecasted_values.append(pred[0, 0])
    input_seq = np.append(input_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
forecasted_values = scaler.inverse_transform(np.array(forecasted_values).reshape(-1, 1))

fig = go.Figure()
actual_dates = weekly_data.index
fig.add_trace(go.Scatter(x=actual_dates, y=weekly_data.values, mode='lines', name='Actual Data'))
test_dates = weekly_data.index[-len(test_predictions):] 
fig.add_trace(go.Scatter(x=test_dates, y=test_predictions.flatten(), mode='lines', name='Test Predictions', line=dict(dash='dot', color='red')))
forecast_dates = pd.date_range(start=weekly_data.index[-1], periods=len(forecasted_values), freq='W-MON')[1:] 
fig.add_trace(go.Scatter(x=forecast_dates, y=forecasted_values.flatten(), mode='lines', name='Forecasted Data'))

fig.update_layout(title=f'Şube {selected_branch} için Forecast',
                  xaxis_title='Date', 
                  yaxis_title='Values'
                )

st.plotly_chart(fig, use_container_width=True)
rmse_lstm = np.sqrt(mean_squared_error(y_test_inv, test_predictions))
next_week_forecast_lstm = forecasted_values[0][0]





#######  ARIMA  #######
st.subheader('Autoregressive Integrated Moving Average (ARIMA) Forecast')
branch_data = df[df['Servis Noktası'] == selected_branch].copy()
branch_data['Date'] = pd.to_datetime(branch_data['Date'])
branch_data.set_index('Date', inplace=True)
weekly_data = branch_data.resample('W-Mon').size()
train_size = int(len(weekly_data) * 0.8)
train_data, test_data = weekly_data.iloc[:train_size+1], weekly_data.iloc[train_size:]

auto_model = auto_arima(train_data, start_p=1, start_q=1,
                        max_p=5, max_q=5, d=1,
                        seasonal=False, trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)

fitted_model = auto_model.fit(train_data)
forecast_steps = len(test_data) + 5
forecast = fitted_model.predict(n_periods=forecast_steps)
next_week_forecast_arima = forecast[0]
forecast_index = pd.date_range(start=test_data.index[-1], periods=forecast_steps, freq='W-MON')[1:]

fig = go.Figure()
fig.add_trace(go.Scatter(x=train_data.index, y=train_data, mode='lines', name='Training Data', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=test_data.index, y=test_data, mode='lines', name='Test Data', line=dict(color='green')))
fig.add_trace(go.Scatter(x=forecast_index, y=forecast, mode='lines', name='Forecast', line=dict(color='orange')))
fig.update_layout(title=f'Şube {selected_branch} için Forecast',
                  xaxis_title='Date',
                  yaxis_title='Value')

st.plotly_chart(fig, use_container_width=True)
rmse_arima = np.sqrt(mean_squared_error(test_data, forecast[:len(test_data)]))
average = (next_week_forecast_ets + next_week_forecast_lstm + next_week_forecast_arima) / 3
rmse_df = pd.DataFrame({
    'Model': ['ETS', 'LSTM', 'ARIMA'],
    'RMSE': [rmse_ets, rmse_lstm, rmse_arima]
})

cases_df = pd.DataFrame({
    'Model': ['ETS', 'LSTM', 'ARIMA', 'AVG'],
    'Cases': [next_week_forecast_ets, next_week_forecast_lstm, next_week_forecast_arima, average]
})

def highlight_min(s):
    is_min = s == s.min()
    return ['background-color: green' if v else '' for v in is_min]

styled_rmse_results = rmse_df.style.apply(highlight_min, subset=['RMSE'])\
    .format({'RMSE': "{:.3f}"})\
    .set_properties(**{'text-align': 'center'})\
    .set_table_styles([dict(selector='th', props=[('text-align', 'center')])])

with tables_container:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader('RMSE Karşılaştırması')
        st.table(styled_rmse_results)
        
    with col2:
        st.subheader('Modele Göre Tahmin Edilen Arıza Sayısı')
        st.table(cases_df)

category_proportions = df['Model Kategorisi'].value_counts(normalize=True)
forecasted_total_cases_next_week = average
estimated_cases_by_category = forecasted_total_cases_next_week * category_proportions

estimated_distribution_df = estimated_cases_by_category.reset_index()
estimated_distribution_df.columns = ['Model Kategorisi', 'Gelecek Hafta Tahmini Arıza Sayısı']
estimated_distribution_df.sort_values(by='Model Kategorisi', ascending=True, inplace=True)
estimated_distribution_df.reset_index(drop=True, inplace=True)
st.subheader('Gelecek Hafta İçin Model Kategorisine Göre Arızaların Tahmini Dağılımı')
st.table(estimated_distribution_df)
minutes_per_case = {
    1: 81,
    2: 71,
    3: 47,
    4: 42,
    5: 55,
    6: 39,
    7: 56,
    8: 56,
    9: 56,
    10:56
}
print("Rows causing the error:")
print(estimated_distribution_df[estimated_distribution_df['Model Kategorisi'].isna()])
# Print unique values in the 'Model Kategorisi' column
print("Unique values in 'Model Kategorisi' column:")
print(estimated_distribution_df['Model Kategorisi'].unique())

# Print keys in the `minutes_per_case` dictionary
print("Keys in `minutes_per_case` dictionary:")
print(minutes_per_case.keys())
estimated_distribution_df['Gelecek Hafta Tahmini Dakikalar'] = estimated_distribution_df.apply( lambda row: row['Gelecek Hafta Tahmini Arıza Sayısı'] * minutes_per_case.get(row['Model Kategorisi']), axis=1)

estimated_distribution_df.reset_index(drop=True, inplace=True)
st.subheader('Gelecek Hafta Model Kategorisine Göre Harcanan Tahmini Süre')
st.table(estimated_distribution_df[['Model Kategorisi', 'Gelecek Hafta Tahmini Dakikalar']])

total_estimated_minutes = estimated_distribution_df['Gelecek Hafta Tahmini Dakikalar'].sum()
calculated_number_of_employees = total_estimated_minutes / 2880
calculated_number_of_employees = math.ceil(calculated_number_of_employees)

st.metric(label="Hesaplanan Çalışan Sayısı", value=f"{calculated_number_of_employees} kişi")
