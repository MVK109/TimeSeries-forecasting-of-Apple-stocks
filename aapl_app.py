import pandas as pd
import streamlit as st
from plotly import graph_objs as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.title('Apple Stock Price Prediction')
st.text('This app predicts the next 30 days Apple stock price by SARIMA model')
Value = st.slider('Select a Value:', min_value=1, max_value=30,value=1)


dateparse = lambda dates: pd.datetime.strptime(dates, '%d-%m-%Y')
df = pd.read_csv("C:/Users/lavanya/Downloads/AAPL.csv",
                 index_col='Date', parse_dates=['Date'], date_parser=dateparse)

def predict(Value):
    model_sarima = SARIMAX(df["Close"], order=(2,1,2), seasonal_order=(2,1,2,12))
    model_sarima_fit = model_sarima.fit()
    last_date = df.index[-1]
    date_range = pd.date_range(last_date, periods=Value+1, freq='D')[1:]
    prediction = model_sarima_fit.forecast(steps=Value)
    prediction.index = date_range
    return prediction

if st.button('Generate Prediction'):
    prediction = predict(Value)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Actual Stock Price"))
    fig.add_trace(go.Scatter(x=prediction.index, y=prediction, name="Predicted Stock Price",line=dict(color='red', width=4)))
    fig.layout.update(title_text='{} days Stock Price Prediction'.format(Value))

    #fig.layout.update(title_text='{} days Stock Price Prediction with Rangeslider'.format(Value), xaxis_rangeslider_visible=True)
    #fig.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
    st.plotly_chart(fig)

    prediction_df = pd.DataFrame(prediction)
    prediction_df.index.name = 'Date'
    st.table(prediction_df)
    
   









    
    



