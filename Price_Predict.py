import pandas as pd
import numpy as np
import seaborn as sns
from Data_Analyze_Lib import DataFrame_Analyze
import streamlit as st
import matplotlib.pyplot as plt
import math
import pickle
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
register_matplotlib_converters()
from time import time
import statistics
from pmdarima import auto_arima
from sklearn.metrics import  mean_squared_error
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from statsmodels.tsa.stattools import adfuller
import plotly.express as px
import plotly.graph_objects as go

#Load style
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}<style>',unsafe_allow_html=True)


#Function
def load_data():

    main_data = pd.read_csv("avocado.csv")
    model_table = pd.read_excel("Regression_model.xlsx")
    svr = pickle.load(open('SVR_model.sav','rb'))
    encode = pickle.load(open('encode_model.sav','rb'))
    rb4046 = pickle.load(open('Robust_Scaler_Model/4046.sav','rb'))
    rb4225 = pickle.load(open('Robust_Scaler_Model/4225.sav','rb'))
    rb4770 = pickle.load(open('Robust_Scaler_Model/4770.sav','rb'))
    rbLB = pickle.load(open('Robust_Scaler_Model/Large Bags.sav','rb'))
    rbSB = pickle.load(open('Robust_Scaler_Model/Small Bags.sav','rb'))
    rbXLB = pickle.load(open('Robust_Scaler_Model/XLarge Bags.sav','rb'))
    rbTTB = pickle.load(open('Robust_Scaler_Model/Total Bags.sav','rb'))
    rbTTV = pickle.load(open('Robust_Scaler_Model/Total Volume.sav','rb'))

    return main_data,svr,encode,rb4046,rb4225,rb4770,rbLB,rbSB,rbXLB,rbTTB,rbTTV,model_table


def log_number(x):
    return math.log(x+1)


def predict_price(v4225, v4770, v4046, vLarge_Bags, vXLarge_Bags, vSmall_Bags, region, year, type_, Month, svr,
                  encoder,rb4046,rb4225,rb4770,rbLB,rbSB,rbXLB,rbTTB,rbTTV):
    Total_Volume = v4225 + v4770 + v4046
    Total_Bags = vLarge_Bags + vXLarge_Bags + vSmall_Bags

    def log_number(x):
        return math.log(x + 1)

    def scaler(model,x):
        return model.transform(np.array([x]).reshape(-1,1))[0][0]

    #log
    v4046 = log_number(v4046)
    v4225 = log_number(v4225)
    v4770 = log_number(v4770)
    vLarge_Bags = log_number(vLarge_Bags)
    vSmall_Bags = log_number(vSmall_Bags)
    vXLarge_Bags = log_number(vXLarge_Bags)
    Total_Bags = log_number(Total_Bags)
    Total_Volume = log_number(Total_Volume)
    #encode
    v4046 = scaler(rb4046,v4046)
    v4225 = scaler(rb4225,v4225)
    v4770 = scaler(rb4770,v4770)
    vLarge_Bags = scaler(rbLB,vLarge_Bags)
    vSmall_Bags = scaler(rbSB,vSmall_Bags)
    vXLarge_Bags = scaler(rbXLB,vXLarge_Bags)
    Total_Bags = scaler(rbTTB,Total_Bags)
    Total_Volume = scaler(rbTTV,Total_Volume)




    df_num = pd.DataFrame({'4225': [v4225], '4770': [v4770], '4046': [v4046], 'Large Bags': [vLarge_Bags],
                           'XLarge Bags': [vXLarge_Bags], 'Small Bags': [vSmall_Bags], 'Total Volume': [Total_Volume],
                           'Total Bags': [Total_Bags]})

    df_cat = pd.DataFrame({'region': [region], 'year': [year], 'type': [type_], 'Month': [Month]})
    df_cat['year'] = df_cat['year'].astype(object)
    df_cat['Month'] = df_cat['Month'].astype(object)
    df_cat_encode = pd.DataFrame(encoder.transform(df_cat).toarray(), columns=encoder.get_feature_names_out())
    df = pd.concat([df_num, df_cat_encode], axis=1)

    predict = svr.predict(df)

    return predict

def multi_price_predict(df): #format 4225,4770,4046,LB,XLB,SB,region,year,type,Month
    df['Total Bags'] = df['XLarge Bags'] + df['Small Bags'] + df['Large Bags']
    df['Total Volume'] = df['4225'] + df['4770'] + df['4046']

    num = ['4225','4770','4046','Large Bags','XLarge Bags','Small Bags','Total Volume','Total Bags']
    cat = ['region','year','type','Month']
    scaler = [rb4225,rb4770,rb4046,rbLB,rbXLB,rbSB,rbTTV,rbTTB]

    X_num = df[num].reset_index(drop=True)
    X_cat = df[cat].reset_index(drop=True)

    X_cat['year'] = X_cat['year'].astype(object)
    X_cat['Month'] =  X_cat['Month'].astype(object)

    for i in range(len(num)):
        X_num[num[i]] = X_num[num[i]].apply(lambda x: math.log(x + 1))
        X_num[num[i]] = scaler[i].transform(X_num[[num[i]]])

    X_cat_encode = pd.DataFrame(encode.transform(X_cat).toarray(), columns=encode.get_feature_names_out())
    df2 = pd.concat([X_num, X_cat_encode], axis=1)

    df['predict'] = svr.predict(df2)

    df = df.drop('Total Volume',axis = 1)
    df = df.drop('Total Bags',axis = 1)

    return df


def adf_check(time_series):
    """
    Pass in a time series, returns ADF report
    """
    result = adfuller(time_series)
    st.write('Augmented Dickey-Fuller Test:')
    labels = ['ADF Test Statistic', 'p-value', 'Number of Lags Used', 'Number of Observations Used']

    for value, label in zip(result, labels):
        st.write(label + ' : ' + str(value))

    if result[1] <= 0.05:
        st.write("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
    else:
        st.write("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary \n")

def predict_future(data, to_date, where):
    data2 = data.copy()
    data2 = data.drop('region',axis = 1)
    data2.columns = ['ds','y']
    model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False).add_seasonality(
            name='weekly', period=52, fourier_order=1)
    model.fit(data2)
    # create predict value
    weeks = pd.date_range('2015-01-04', to_date, freq='W').strftime("%Y-%m-%d").tolist()
    future = pd.DataFrame(weeks)
    future.columns = ['ds']
    future['ds'] = pd.to_datetime(future['ds'])
    # predict
    predict = model.predict(future)
    predict['region'] = where
    predict = predict[['ds', 'region', 'yhat']]
    return predict

#Load data
data,svr,encode,rb4046,rb4225,rb4770,rbLB,rbSB,rbXLB,rbTTB,rbTTV,model_table = load_data()
data['Month'] = data['Date'].astype('datetime64[ns]').dt.month
data['Day'] = data['Date'].astype('datetime64[ns]').dt.day
data['Dayofweek'] = data['Date'].astype('datetime64[ns]').dt.dayofweek
countinous = ['AveragePrice','Total Volume','4046','4225','4770','Total Bags','Small Bags','Large Bags','XLarge Bags']
discrete = ['Day','Month','year','type','region']
in_continous = ['Total Volume','4046','4225','4770','Total Bags','Small Bags','Large Bags','XLarge Bags']
in_discrete = discrete


#GUI
menu = ["Business Objective", "Regression", "Time Series","Choose Invest Area"]
choice = st.sidebar.selectbox('Menu', menu)

if choice == "Business Objective":

    choice1 = st.selectbox('Sub Menu',['Overall and introduction','Explore Data Analysis'])
    if choice1 == 'Overall and introduction':
        st.header("I.Introduction")

        st.write('''Bơ “Hass”, một công ty có trụ sở tại Mexico,chuyên sản xuất nhiều loại quả bơ được bán ở Mỹ.
        Họ đã rất thành công trong những năm gần đây và muốn mở rộng. Vì vậy, họ muốn xây dựng mô hình 
        hợp lý để dự đoán giá trung bình của bơ “Hass” ở Mỹ nhằm xem xét việc mở rộng các loại trang trại 
        Bơ đang có cho việc trồng bơ ở các vùng khác.''')

        st.header("II.Deatail")
        st.subheader("a.Business Understanding")

        st.write('''Hiện tại: Công ty kinh doanh quả bơ ở rất nhiều vùng của nước Mỹ với 2 loại bơ là bơ 
        thường và bơ hữu cơ, được đóng gói theo nhiều quy chuẩn (Small/Large/XLarge Bags), và có 3 PLU (Product Look Up)
        khác nhau (4046, 4225, 4770). Nhưng họ chưa có mô hình để dự đoán giá bơ cho việc mở rộng.''')

        st.subheader("b.Data Understanding")

        st.write('''Mục tiêu/ Vấn đề: Xây dựng mô hình dự đoán giá trung bình của bơ “Hass” ở Mỹ 
        => xem xét việc mở rộng sản xuất, kinh doanh.''')

        st.write('''+ Từ mục tiêu/ vấn đề đã xác định: xem xét các dữ liệu mà công ty đang có:
    + Dữ liệu được lấy trực tiếp từ máy tính tiền của các nhà bán lẻ dựa trên doanh số bán lẻ thực tế của bơ Hass.
    + Dữ liệu đại diện cho dữ liệu lấy từ máy quét bán lẻ hàng tuần cho lượng bán lẻ (National retail volume- units) và giá bơ từ tháng 4/2015 đến tháng 3/2018.
    + Giá Trung bình (Average Price) trong bảng phản ánh giá trên một đơn vị (mỗi quả bơ), ngay cả khi nhiều đơn vị (bơ) được bán trong bao.
    + Mã tra cứu sản phẩm - Product Lookup codes (PLU’s) trong bảng chỉ dành cho bơ Hass, không dành cho các sản phẩm khác.''')

        st.write('''+ Toàn bộ dữ liệu được đổ ra và lưu trữ trong tập tin avocado.csv với 18249 record. Với các cột:
    + Date - ngày ghi nhận
    + AveragePrice – giá trung bình của một quả bơ
    + Type - conventional / organic – loại: thông thường/ hữu cơ
    + Region – vùng được bán 
    + Total Volume – tổng số bơ đã bán
    + 4046 – tổng số bơ có mã PLU 4046 đã bán
    + 4225 - tổng số bơ có mã PLU 4225 đã bán
    + 4770 - tổng số bơ có mã PLU 4770 đã bán
    + Total Bags – tổng số túi đã bán
    + Small/Large/XLarge Bags – tổng số túi đã bán theo size''')

        st.header("III.Problem solving")

        st.write("1.Build regression model to predict average hass price")
        st.write("2.Time series model to predict future hass price")
    elif choice1 == 'Explore Data Analysis':
        st.header("I.Single variable analysis")

        st.subheader("a.Central tendency")
        st.dataframe(DataFrame_Analyze(data[countinous]).get_central_tendency())
        st.subheader("b.Dispersion")
        st.dataframe(DataFrame_Analyze(data[countinous]).get_dispersion())

        st.header("II.Bivariable analysis")
        st.write('Using heat map to see correlation of 2 variable:')
        st.write(data[in_continous].corr())
        fig, ax = plt.subplots()
        sns.heatmap(data[in_continous].corr(),ax =ax,annot = True,fmt='.2f',mask = np.triu(data[in_continous].corr()))
        st.write(fig)
        st.write('Using chi2 test to see dependent of 2 categorical variable:')
        st.write(DataFrame_Analyze(data[in_discrete]).table_category_vs_category(alpha=0.05))
        st.header("III.Conclusion")
        st.write("+ All continous variable has high correlation with each others , consider drop.")
        st.write("+ All in continous right skew , consider feature engineering.")
        st.write("+ Almost all category variables are independent.")

elif choice == 'Regression':
    st.sidebar.subheader('I.Model choose')
    st.sidebar.dataframe(model_table)
    st.sidebar.write('* Choose SVR model')
    st.sidebar.subheader('II.Final model overview')
    st.sidebar.code('MSE:0.0109')
    st.sidebar.code('R2 Train:0.9609')
    st.sidebar.code('R2 Test:0.9331')
    choice2 = st.selectbox('Sub Menu',['Single Predict','Multiple Predict'])
    if choice2 == 'Multiple Predict':
        st.markdown("[# Sample upload file](https://drive.google.com/drive/folders/15hujllkpruoy8fCZ27yhRgYz5pq6xob1)")
        data_file = None
        with st.form('Multi predict:'):
            data_file = st.file_uploader('Please upload your file here')
            submitted = st.form_submit_button("Predict")
        if submitted:
            if data_file == None:
                st.write('Please input the file before submitted.')
            else :
                df = pd.read_excel(data_file,sheet_name="Predict_sample",engine = 'openpyxl')
                st.write(multi_price_predict(df))

                if st.download_button(label="Download data as CSV",
                                      data=df.to_csv().encode('utf-8'),
                                      file_name='Price_predict.csv',
                                      mime='text/csv'):
                    st.write('Thanks for downloading!')

    elif choice2 == 'Single Predict':
        with st.form("Single Predict"):
            col1 = st.columns(2)
            #v4046 = col1[0].slider('4046 volume',round(data['4046'].min()),round(data['4046'].max()),round(data['4046'].min()))
            #v4225 = col1[0].slider('4225 volume',round(data['4225'].min()),round(data['4225'].max()),round(data['4225'].min()))
            #v4770 = col1[0].slider('4770 volume',round(data['4770'].min()),round(data['4770'].max()),round(data['4770'].min()))
            #vLB   = col1[1].slider('Large Bags',round(data['Large Bags'].min()),round(data['Large Bags'].max()),round(data['Large Bags'].min()))
            #vSB   = col1[1].slider('Small Bags',round(data['Small Bags'].min()),round(data['Small Bags'].max()),round(data['Small Bags'].min()))
            #vXLB  = col1[1].slider('XLarge Bags',round(data['XLarge Bags'].min()),round(data['XLarge Bags'].max()),round(data['XLarge Bags'].min()))
            v4046 = col1[0].number_input('4046 volume')
            v4225 = col1[0].number_input('4225 volume')
            v4770 = col1[0].number_input('4770 volume')
            vLB   = col1[1].number_input('Large Bags')
            vSB   = col1[1].number_input('Small Bags')
            vXLB  = col1[1].number_input('XLarge Bags')

            col = st.columns(4)
            region = col[0].selectbox('region',data['region'].unique())
            year = col[1].selectbox('year', data['year'].unique())
            type_ = col[2].selectbox('type', data['type'].unique())
            Month = col[3].selectbox('month', data['Month'].unique())
            submitted = st.form_submit_button("Predict")
        if submitted:
            predict = predict_price(v4225, v4770, v4046,  vLB , vXLB , vSB  , region, year, type_, Month, svr,encode, rb4046, rb4225, rb4770, rbLB, rbSB, rbXLB, rbTTB, rbTTV)
            st.write('Predict price of hass is:',round(predict[0],5))
elif choice == 'Time Series':
    choice3 = st.selectbox('Sub Menu',['Explore Data Analysis','Model Arima','Model FbProphet'])
    if choice3 == 'Explore Data Analysis':

        data['Date'] = data['Date'].astype('datetime64[ns]')
        data['region_type'] = data['region'] + '_' + data['type']
        data_organic = data[(data['region'] == 'California') & (data['type'] == 'organic')][['Date', 'AveragePrice']]
        data_conventional = data[(data['region'] == 'California') & (data['type'] == 'conventional')][['Date', 'AveragePrice']]
        st.subheader("I.Data overview")
        st.write("Organic Hass data from California")
        st.write(data_organic)

        st.write("Convetional Hass data from California")
        st.write(data_conventional)

        fig, ax = plt.subplots(2,1,constrained_layout = True)
        sns.lineplot(data=data_organic,x='Date',y='AveragePrice', ax=ax[0])
        ax[0].tick_params(axis='x', rotation=90)
        ax[0].set_title('Organic Hass data from California')
        sns.lineplot(data=data_conventional, x='Date', y='AveragePrice', ax=ax[1])
        ax[1].tick_params(axis='x', rotation=90)
        ax[1].set_title('Convetional Hass data from California')
        st.write(fig)

        data_conventional.index = data_conventional['Date']
        data_conventional = data_conventional.drop('Date', axis=1)
        data_conventional = data_conventional.sort_index()

        data_organic.index = data_organic['Date']
        data_organic = data_organic.drop('Date', axis=1)
        data_organic = data_organic.sort_index()


        st.subheader("II.Data stationary test")
        st.subheader("a.Data organic test")
        adf_check(data_organic)
        st.subheader("b.Data conventional test")
        adf_check(data_conventional)

        st.subheader("III.Decomposition")
        st.subheader("a.Data organic test")
        fig = seasonal_decompose(data_organic, model='additive').plot()
        st.write(fig)
        st.subheader("b.Data conventional test")
        fig = seasonal_decompose(data_conventional, model='additive').plot()
        st.write(fig)



    elif choice3 == 'Model Arima':
        pass
    elif choice3 == 'Model FbProphet':
        with st.form('My form'):
            min_date = datetime(2018,1,1)
            upper_date = datetime(2018,6,1)
            region = st.selectbox("Select area you want to predict:",data['region'].unique())
            type = st.selectbox("Choose type of hass:",data['type'].unique())
            date1 = st.date_input("Choose date you want to start predict:" , min_value = min_date)
            date2 = st.date_input("Choose date you want to predict to:",min_value = upper_date)
            submited = st.form_submit_button('Sumited')
        if submited:
            if date2 <= date1 :
                st.write('Please choose suitable date')
            else :
        # Prepair data
                data_predict = data[(data['region'] == region) & (data['type'] == type)][['Date', 'AveragePrice']].sort_values('Date')
                data_predict2 = data_predict.reset_index(drop = True)
                data_predict2.columns = ['ds', 'y']
                weeks = pd.date_range(date1, date2,freq='W').strftime("%Y-%m-%d").tolist()
                future = pd.DataFrame(weeks)
                future.columns = ['ds']
                future['ds'] = pd.to_datetime(future['ds']).dt.normalize()
        # Train model
                model= Prophet(yearly_seasonality=True,daily_seasonality=False, weekly_seasonality=False)
                model.fit(data_predict2)
        # Predict
                forecast = model.predict(future)[["ds","yhat"]]
                st.write("Predict result of {}:".format(region))

                fig = px.scatter()
                fig.add_scatter(x=forecast['ds'],y = forecast['yhat'],name = 'Data predict')
                fig.add_scatter(x=data_predict2['ds'], y=data_predict2['y'],name = 'Data normal')
                st.write(fig)

                if st.download_button(label="Download data as CSV",
                              data=forecast.to_csv().encode('utf-8'),
                              file_name='Price_predict.csv',
                              mime='text/csv'):
                    st.write('Thanks for downloading!')
if choice == "Choose Invest Area":
    with st.form('My form'):
        area = st.multiselect("Choose Area want to compare:",data['region'].unique(),default = data['region'].unique())
        type = st.selectbox("Choose type of hass:",data['type'].unique())
        predict_by = st.selectbox("Choose predict variable:",['Revenue','AveragePrice','Total Volume'])
        top = st.slider("Number of top area",5,10,5)
        submitted = st.form_submit_button('Submited')
    if submitted:
        if top <= len(area):
            data['ds'] = data['Date'].astype('datetime64[ns]')
            data = data[data['region'].isin(area)]
            data['Revenue'] = data['AveragePrice']*data['Total Volume']
            data = data[['ds',predict_by,'region']]

            place = data['region'].unique()
            final_result = pd.DataFrame({'ds': [], 'region': [], 'yhat': []})
            for i in place:
                result = predict_future(data[data['region'] == i], '2022-01-01', i)
                final_result = pd.concat([final_result, result], axis=0)
            final_result['year'] = final_result['ds'].dt.year
            data_sum = final_result.groupby(['year', 'region']).agg({'yhat': ['mean', 'std']}).reset_index()
            data_sum = data_sum[(data_sum['year'] == 2021) | (data_sum['year'] == 2017)]
            data_sum = pd.pivot_table(data_sum, values=[('yhat', 'mean'), ('yhat', 'std')], index='region', columns='year')[
                'yhat']
            data_sum['mean_diff'] = data_sum[('mean', 2021)] - data_sum[('mean', 2017)]
            data_sum = data_sum.sort_values('mean_diff', ascending=False)
            data_sum_top5 = data_sum.iloc[0:top]
            st.write('Top 5 khu vực có sự gia tăng {} lớn nhất so sanh năm 2017 với 2021:'.format(predict_by))
            st.write(data_sum_top5)
            st.write('Đồ thị :')
            fig = px.line(final_result[final_result['region'].isin(data_sum_top5.index)],x='ds',y = 'yhat',color = 'region')
            st.write(fig)
        else:
            st.write('Number of top need greater than the area need')


















