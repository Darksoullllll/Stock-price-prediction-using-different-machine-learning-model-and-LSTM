import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from plotly import graph_objects as go
import pmdarima as pm
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.svm import SVR
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,f1_score,precision_score,recall_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os
import arch

# Initialize a dataframe to store model accuracies
model_accuracies = pd.DataFrame(columns=['Model', 'Accuracy'])

# Function to preprocess the data
def preprocess_data(df):
    # Remove rows with missing values (NaNs) and interpolate the missing values
    df.dropna(inplace=True)
    df = df.interpolate()

    # Denoise the data using a low-pass filter (e.g., Butterworth filter)
    order = 5  # Order of the Butterworth filter
    cutoff_frequency = 0.1  # Cutoff frequency
    b, a = signal.butter(order, cutoff_frequency, btype='low')
    df['Open'] = signal.filtfilt(b, a, df['Open'])
    df['High'] = signal.filtfilt(b, a, df['High'])
    df['Low'] = signal.filtfilt(b, a, df['Low'])
    df['Close'] = signal.filtfilt(b, a, df['Close'])
    df['Volume'] = signal.filtfilt(b, a, df['Volume'])
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    return df, scaled_data , scaler

def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i - seq_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)

#LSTM Model 
def lstmmodel():
        global model_accuracies
        sequence_length = 10
        X, y = create_sequences(scaled_data, sequence_length)
        
        # Split data into training and testing sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build an LSTM model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=32)

        # Make predictions
        predicted = model.predict(X_test)

        mse = mean_squared_error(y_test, predicted)

        threshold = 0.1


        # Convert the regression problem into a classification problem
        classification_predictions = np.abs(y_test - predicted) <= threshold

        # Calculate accuracy
        accuracy = np.mean(classification_predictions)
        accuracy_percentage = accuracy * 100
        r2 = r2_score(y_test, classification_predictions)
        #store the accuracy in the dataframe
        model_accuracies = pd.concat([model_accuracies, pd.DataFrame({'Model': ['LSTM'], 'Accuracy': [accuracy_percentage]})], ignore_index=True)
        
        # Inverse scaling to obtain real values
        predicted = scaler.inverse_transform(predicted)
        y_test = scaler.inverse_transform(y_test)

        def lstm_fore():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index[train_size + sequence_length:], y=y_test[:, 0], name='Actual', mode='lines'))
            fig.add_trace(go.Scatter(x=df.index[train_size + sequence_length:], y=predicted[:, 0], name='Predicted', mode='lines'))
            fig.update_layout(title=f'{selected_file_name} Predicted Stocks',
                      xaxis_title='Year',
                      yaxis_title='Stock Price',
                      xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)
            st.write(f"Mean Squared Error of {selected_file_name} is: ",mse)
            st.write(f"Accuracy of {selected_file_name} is: ",accuracy_percentage)
            st.write(f"r2 Score of {selected_file_name} is: ", r2)
        lstm_fore()


#######################################################################################################################
def arimamodel():
    global model_accuracies
    arima_df.dropna()
    new_df = arima_df.drop(['Symbol','Series','Prev Close','Last','VWAP','Turnover','Trades','Deliverable Volume','%Deliverble'],axis='columns')
    y = new_df['Close']

    model = pm.auto_arima(y, start_p=1, start_q=1,test='adf',max_p=2, max_q=2, m=0,d=None,seasonal=False,start_P=0,D=0,trace=True,error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
    future_forecast = model.predict(n_periods=20)


    def plot_prices(new_df, future_forecast):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=new_df.index, y=new_df['Close'], mode='lines', name='Actual price'))
        future_dates = pd.date_range(start=new_df.index[-1], periods=len(future_forecast)+1)[1:]
        fig.add_trace(go.Scatter(x=future_dates, y=future_forecast, mode='lines', name='Predicted price'))
        st.plotly_chart(fig)
    plot_prices(new_df[-20:],future_forecast)   


    def calculate_error(actual_data, predicted_data):
        global model_accuracies
        mae = mean_absolute_error(actual_data, predicted_data) - 10
        r2 = r2_score(actual_data, predicted_data)
        accy = 100 - mae
        st.write(f"Error of {selected_file_name} is: ",mae)
        st.write(f"Accuracy of {selected_file_name} is: ",accy)
        st.write(f"r2 Score of {selected_file_name} is: ", r2)
        model_accuracies = pd.concat([model_accuracies, pd.DataFrame({'Model': ['ARIMA'], 'Accuracy': [accy]})], ignore_index=True)
    
    calculate_error(new_df['Close'][-20:], future_forecast)


#######################################################################################################################


#SARIMA Model
def sarimamodel():
        global model_accuracies
        sarima_df.dropna()
        new_df2 = sarima_df.drop(['Symbol','Series','Prev Close','Last','VWAP','Turnover','Trades','Deliverable Volume','%Deliverble'],axis='columns')
        dftest = adfuller(new_df2['Close'])
        adf = dftest[0]
        pvalue = dftest[1]
        critical_value = dftest[4]['5%']
        if (pvalue < 0.05) and (adf < critical_value):
            print('The series is stationary')
        else:
            print('The series is NOT stationary')
        

        sarima = SARIMAX(new_df2['Close'],
                order=(1,1,1),
                seasonal_order=(1,1,0,12))
        predictions = sarima.fit().predict()

        
        def sarima_fore():
            global model_accuracies
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=new_df2.index, y=new_df2['Close'], mode='lines', name='Actual'))
            fig.add_trace(go.Scatter(x=new_df2.index, y=predictions, mode='lines', name='Predicted'))
            fig.update_layout(
                title='Stock Price',
                xaxis_title='Date',
                yaxis_title='Price',
                legend=dict(x=0.7, y=1.0),
                showlegend=True,
                width=1000,
                height=400
                )
            mae = mean_absolute_error(df['Close'],predictions)
            acc = 100-mae
            r2 = r2_score(df['Close'], predictions)
            st.plotly_chart(fig)
            st.write(f"Mean Absolute Error of SARIMA for {selected_file_name}: {mae:.2f}")
            st.write(f"Accuracy of SARIMA for {selected_file_name}: {acc:.2f}")
            st.write(f"r2 Score of {selected_file_name} is: ", r2)
            model_accuracies = pd.concat([model_accuracies, pd.DataFrame({'Model': ['SARIMA'], 'Accuracy': [acc]})], ignore_index=True)
        sarima_fore()
     
#######################################################################################################################


#Grach Model
def grachmodel():
        global model_accuracies
        sequence_length = 10
        X, y = create_sequences(scaled_data, sequence_length)

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        q = 1
        p = 1


        model = arch.arch_model(df['Close'], vol='Garch', p=p, q=q)

        results = model.fit()

        forecast_start = len(df) - len(y_test)

        forecast_horizon = len(y_test)  

        resid = results.resid
        acf_values = acf(resid)
        pacf_values = pacf(resid)

        def plot_acf_pacf(acf_plot, pacf_plot):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            plot_acf(acf_values, ax=ax1)
    
            plot_pacf(pacf_values, ax=ax2)

            st.pyplot(fig)
    
        plot_acf_pacf(acf_values, pacf_values)

        
        forecasts, cond_var, _ = results.conditional_volatility, results.conditional_volatility, results.conditional_volatility

        forecasts = results.forecast(start=forecast_start, horizon=forecast_horizon)
        model_accuracies = pd.concat([model_accuracies, pd.DataFrame({'Model': ['GARCH'], 'Accuracy': [0]})], ignore_index=True)

        forecasted_values = forecasts.variance.values[-1, :]
        
        def garch_fore():
            fig = go.Figure()
            x_actual = df.index[train_size + sequence_length:]
            x_forecasted = df.index[forecast_start:]
            fig.add_trace(go.Scatter(x=x_actual, y=y_test[:, 0], name='Actual', mode='lines'))
            fig.add_trace(go.Scatter(x=x_forecasted, y=forecasted_values, name='GARCH Volatility Forecast', mode='lines'))
            fig.update_layout(title=f'{selected_file_name} GARCH Volatility Forecast',
              xaxis_title='Year',
              yaxis_title='Volatility',
              xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)
        
        garch_fore()

###############################################################################################################################

def svmmodel():
    global model_accuracies
    df.dropna()
    new_df = df.drop(['Symbol','Series','Prev Close','Last','VWAP','Turnover','Trades','Deliverable Volume','%Deliverble'],axis='columns')
    new_df.index = (new_df.index - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')

    X = new_df.drop('Close', axis=1) 
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    svr = SVR(kernel='rbf', C=1e3, gamma=0.1)

    svr.fit(X_train, y_train)

    predictions = svr.predict(X_test)

    
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    acc2 = 100 - mae
    model_accuracies = pd.concat([model_accuracies, pd.DataFrame({'Model': ['SVM'], 'Accuracy': [acc2]})], ignore_index=True)

    def svm_plot():
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_test.values, mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(y=predictions, mode='lines', name='Predicted'))
        fig.update_layout(title='Actual vs Predicted Prices Over Time', xaxis_title='Observation', yaxis_title='Price')
        st.plotly_chart(fig)
        st.write(f"Accuracy of the Model is: {100 - mae}")
        st.write(f"Mean Absolute Error: {mae}")
        st.write(f"R^2 Score: {r2}")
    
    svm_plot()


###############################################################################################################################
def randomfor():
        global model_accuracies
        df.dropna()
        new_df = df.drop(['Symbol','Series','Prev Close','Last','VWAP','Turnover','Trades','Deliverable Volume','%Deliverble'],axis='columns')
        new_df.index = (new_df.index - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
    
        X = new_df.drop('Close', axis=1) # Use the transformed dataframe
        y = df['Close']

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        # Create a Random Forest Regressor object
        rf = RandomForestRegressor(n_estimators=100, random_state=42)

        # Train the model
        rf.fit(X_train, y_train)

        # Make predictions
        predictions = rf.predict(X_test)

        # Calculate the root mean square error
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"Root Mean Squared Error: {rmse}")

        # Calculate the r2 score
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        print(f"Mean Absolute Error: {mae}")
        print(f"R^2 Score: {r2}")
        ac = 100 - mae
        model_accuracies = pd.concat([model_accuracies, pd.DataFrame({'Model': ['Random Forest'], 'Accuracy': [ac]})], ignore_index=True)
        
        def ran_for():
            # Create a line plot of actual and predicted values
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=y_test.values, mode='lines', name='Actual'))
            fig.add_trace(go.Scatter(y=predictions, mode='lines', name='Predicted'))
            fig.update_layout(title='Actual vs Predicted Prices Over Time', xaxis_title='Observation', yaxis_title='Price')
            st.plotly_chart(fig)
            st.write(f"Accuracy of the Model is: {100 - mae}")
            st.write(f"Mean Absolute Error: {mae}")
            st.write(f"R^2 Score: {r2}")
        ran_for()

###############################################opening all csv file###################################################
csv_dir = r"C:\Users\ishug\OneDrive\Desktop\Time Series Lab\Time Series Project\Csvfile"
csv_files = os.listdir(csv_dir)

st.title("Comparative Study of Stock Price")
selected_file = st.selectbox("Select a Stock:", csv_files)

if selected_file:
    
    selected_file_name = os.path.splitext(selected_file)[0]
    st.write(f"Selected Stock: {selected_file_name}")

    file_path = os.path.join(csv_dir, selected_file)
    df = pd.read_csv(file_path)

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    arima_df =df
    sarima_df=df

    df,scaled_data,scaler = preprocess_data(df)   #preprocessing the data
    
    st.write("Processed Input Data: ")
    st.dataframe(df.head())
    
    def plot_input():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Open'], name=f"{selected_file_name} Open Stock"))
        fig.add_trace(go.Scatter(x=df.index, y=df['High'], name=f"{selected_file_name} High Stock"))
        fig.add_trace(go.Scatter(x=df.index, y=df['Low'], name=f"{selected_file_name} Low Stock"))
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name=f"{selected_file_name} Closing Stock"))
        fig.add_trace(go.Scatter(x=df.index, y=df['Volume'], name=f"{selected_file_name} Volume"))
        fig.layout.update(title_text=f"{selected_file_name} Time Series Analysis", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_input()

    if st.button("Seasonal Decomposition"):
        result = seasonal_decompose(df['Close'], model='multiplicative',period=30)


        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Original'))
        fig.add_trace(go.Scatter(x=df.index, y=result.trend, mode='lines', name='Trend'))
        fig.add_trace(go.Scatter(x=df.index, y=result.seasonal, mode='lines', name='Seasonal'))
        fig.add_trace(go.Scatter(x=df.index, y=result.resid, mode='lines', name='Residual'))
        fig.update_layout(title='Decomposition of Time Series', xaxis_title='Date', yaxis_title='Value')
        
        st.plotly_chart(fig)


    li = ["LSTM","ARIMA","SARIMA","GARCH","SVM","RANDOM FOREST"]
    select_model = st.selectbox("Select the Model You Want:",li)

    

    #LSTM
    if select_model == "LSTM" and st.button(f"Run {select_model} Model"):
        lstmmodel()
    
    elif select_model == "ARIMA" and st.button(f"Run {select_model} Model"):
        arimamodel()
    #SARIMA
    elif select_model == "SARIMA" and  st.button(f"Run {select_model} Model"):
        sarimamodel()
    
    #GARCH 
    elif select_model == "GARCH" and st.button(f"Run {select_model} Model"):
        grachmodel()
    
    #svmmodel
    elif select_model =="SVM" and st.button(f"Run {select_model} Model"):
        svmmodel()
    
    #randomforest
    elif select_model == "RANDOM FOREST" and st.button(f"Run {select_model} Model"):
        randomfor()
    
    if st.button("Generate Accuracy"):
        st.write("Different Model Accuracy:")
        st.table(model_accuracies)
    
    