from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from pickle import load

@st.cache(allow_output_mutation=True)
def predict_series(minutes_per10):
    model = ARIMA(test['net_power_gen'], order=(0, 0, 0), seasonal_order = (1, 1, 1, 144))
    fitted = model.fit()
    # Forecast
    fc= fitted.forecast(len(test), alpha=0.05)  # 95% conf
# Make as pandas series
    fc_series = pd.Series(fc, index=test.index+len(test))
    return fc_series

@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv("test.csv")
    return data

# load the model and scaler
@st.cache(allow_output_mutation=True)
def load_selected_model():
    #model = load(open('model.pkl', 'rb'))
    model = ARIMAResults.load('model.pkl')
    return model


# this is the main function in which we define our webpage
def main():

    #from PIL import Image
    #image = Image.open('taipower.png')

    # display the front end aspect

    st.markdown("<h1 style='text-align: center; color: grey;'>The Prediction of Chang-Bing Solar Photovoltaic Energy System App</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: grey;'>Yen-Ting Lin and Cheng-Chung Li</h2>",
                unsafe_allow_html=True)
    #st.markdown("<h2 style='text-align: center; color: grey;'>HV-Lab TPRI</h2>", unsafe_allow_html=True)
    st.markdown(
        """<h2 style='display: block; text-align: center;' href="https://tpri.taipower.com.tw/">HV-Lab TPRI</h2>
        """,
        unsafe_allow_html=True,
    )


    #st.title('Transmission Line Fault Type Classification App')
    #st.header('Cheng-Chung Li, Shuo-Fu Hong, Wei-Chih Liang')
    #st.subheader('HV-Lab TPRI')

    #col1, col2 = st.columns(2)

    txt = st.text_area('Text to analyze', '''
       The SARIMA model will predict the future power generation
       of Chang-Bing Solar Photovoltaic Energy System. You can
       input number of days to predict and the predictions will
       be presented in dataframe and line Chart. The parameters
       used in SARIMA model is (0, 0, 0) and (1, 1, 1, 144).
       ''')
    st.write('Notification:', run_sentiment_analysis(txt))

    #col2.image('https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/Taiwan_Power_Company_Seal.svg/220px-Taiwan_Power_Company_Seal.svg.png',
    #         caption='Taipower', width = 466, use_column_width= 'auto')


    st.header('Input the days after November 30, 2022: ')
    days = st.selectbox('Days to Predict: ', ['1', '2', '3', '4', '5', '6', '7'])

    minutes_per10 = days * 144

    #model = load_selected_model()
    data = load_data()

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        #fault_type = ''
        prediction = predict_series(minutes_per10)
        known = list(data['net_power_gen'])
        pred = list(prediction)
        preds = pd.DataFrame(pred, columns =['Power_Predictions'])
        power_gen=known + pred
        final = pd.DataFrame(power_gen, columns =['Power_Generation_Predictions'])
        st.subheader('The Prediction: ')
        st.write('The predicted power generation is: ')
        st.dataframe(preds)
        st.write('The power generation in Nov 27~30, 2022 along with the predictions are: ')
        st.line_chart(final)


        st.balloons()

if __name__ == '__main__':
    main()
