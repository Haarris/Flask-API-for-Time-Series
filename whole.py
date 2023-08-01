import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, redirect, render_template
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Display a form for uploading a CSV file
    return render_template('upload_form.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    # Load data from CSV file in request
    data = pd.read_csv(request.files['file'])

    # Convert date column to pandas datetime
    data['date'] = pd.to_datetime(data['date'])

    # Set date as index and sort by date
    data.set_index('date', inplace=True)
    data.sort_index(inplace=True)

    # Get number of forecast days from form
    forecast_days = int(request.form['forecast_days'])

    # Redirect to time series calculation route and pass data and forecast_days as parameters
    return redirect('/forecast/ts_calculation?data={}&forecast_days={}'.format(data.to_json(), forecast_days))

@app.route('/forecast/ts_calculation', methods=['GET'])
def ts_calculation():
    # Retrieve data passed as a parameters
    data = pd.read_json(request.args.get('data'))
    forecast_days = int(request.args.get('forecast_days'))
	

    
    ######

    results = adfuller(data["Total"])
    print(results[1])

    if results[1] <= 0.05:
        print("The original time series is already stationary.")
        print("p-value:", results[1])
    else:
        df_diff = data.Total
        d = 0
        while True:
            results = adfuller(df_diff)
            p_value = results[1]
            if p_value <= 0.05:
                break

            df_diff = df_diff.diff().dropna()
            d += 1

        # print the final results of the ADF test
        print("The time series is stationary after {} differences.".format(d))
        print("p-value:", results[1])

    #######
	
    order_aic_bic=[]
    for p in range(5):
        for q in range(5):
            model=ARIMA(data.Total,order=(p,d,q))
            results=model.fit()
            order_aic_bic.append((p,q,results.aic,results.bic))
	
    order_df=pd.DataFrame(order_aic_bic,columns=["p","q","aic","bic"])
	
	
	# Retrieve the rows with the three lowest AIC values
    lowest_aic_rows = order_df.nsmallest(3, "aic")
	
	
	# Loop over the rows and print the p and q values as integers
    for _, row in lowest_aic_rows.iterrows():
        print("p = {}, q = {}".format(int(row["p"]), int(row["q"])))
	
	
    #first_p = int(order_df.sort_values("aic").iloc[0]["p"])
    #print(first_p)
	
    #second_p = int(order_df.sort_values("aic").iloc[1]["p"])
    #print(second_p)
	
    #third_p = int(order_df.sort_values("aic").iloc[2]["p"])
    #print(third_p)
	
    
    #first_q = int(order_df.sort_values("aic").iloc[0]["q"])
    #print(first_q)
	
    #second_q = int(order_df.sort_values("aic").iloc[1]["q"])
    #print(second_q)
	
    #third_q = int(order_df.sort_values("aic").iloc[2]["q"])
    #print(third_q)
	
	

    ######

    first_p, second_p, third_p = order_df.sort_values("aic").iloc[0:3]["p"].astype(int)
    first_q, second_q, third_q = order_df.sort_values("aic").iloc[0:3]["q"].astype(int)
    #####
	
	##############################################################
	
    last_date = data.index.max()
    forecast_start = last_date + pd.DateOffset(days=1)
    forecast_end = forecast_start + pd.DateOffset(days=forecast_days-1)
    print(forecast_end)
	
	
	# Run ARIMA model on data
	
    model = ARIMA(data['Total'], order=(first_p,d,first_q))
    results_a1 = model.fit()
	
    forecast_a1 = results_a1.get_forecast(steps=forecast_days)
    mean_forecast_a1 = forecast_a1.predicted_mean
	
    forecast_a1_all = results_a1.get_forecast(steps=len(data.Total))
    mean_forecast_a1_all = forecast_a1_all.predicted_mean
	
	
	
	
    model = ARIMA(data['Total'], order=(second_p,d,second_q))
    results_a2 = model.fit()
	
    forecast_a2 = results_a2.get_forecast(steps=forecast_days)
    mean_forecast_a2 = forecast_a2.predicted_mean
	
    forecast_a2_all = results_a2.get_forecast(steps=len(data.Total))
    mean_forecast_a2_all = forecast_a2_all.predicted_mean
	
	
	
	
    model = ARIMA(data['Total'], order=(third_p,d,third_q))
    results_a3 = model.fit()
	
    forecast_a3 = results_a3.get_forecast(steps=forecast_days)
    mean_forecast_a3 = forecast_a3.predicted_mean
	
	
    forecast_a3_all = results_a3.get_forecast(steps=len(data.Total))
    mean_forecast_a3_all = forecast_a3_all.predicted_mean
	
	
	
	## SARIMAX
    model1_s1=sm.tsa.statespace.SARIMAX(data.Total,order=(first_p,d,first_q), seasonal_order=(0,d,0,7))
    results1_s1=model1_s1.fit()
	
    forecast1_s1 = results1_s1.get_forecast(steps=forecast_days)
    mean_forecast1_s1 = forecast1_s1.predicted_mean
	
    forecast1_s1_all = results1_s1.get_forecast(steps=len(data.Total))
    mean_forecast1_s1_all = forecast1_s1_all.predicted_mean
	
	
	
    model1_s2=sm.tsa.statespace.SARIMAX(data.Total,order=(second_p,d,second_q), seasonal_order=(0,d,0,7))
    results1_s2=model1_s2.fit()
	
    forecast1_s2 = results1_s2.get_forecast(steps=forecast_days)
    mean_forecast1_s2 = forecast1_s2.predicted_mean
	
    forecast1_s2_all = results1_s2.get_forecast(steps=len(data.Total))
    mean_forecast1_s2_all = forecast1_s2_all.predicted_mean
	
	
	
	
    model1_s3=sm.tsa.statespace.SARIMAX(data.Total,order=(third_p,d,third_q), seasonal_order=(0,d,0,7))
    results1_s3=model1_s3.fit()
	
    forecast1_s3 = results1_s3.get_forecast(steps=forecast_days)
    mean_forecast1_s3 = forecast1_s3.predicted_mean
	
	
    forecast1_s3_all = results1_s3.get_forecast(steps=len(data.Total))
    mean_forecast1_s3_all = forecast1_s3_all.predicted_mean
	
	
	
	
    forecasted_data_a1 = mean_forecast_a1.loc[forecast_start:forecast_end].astype(int).tolist()
    forecasted_data_a2 = mean_forecast_a2.loc[forecast_start:forecast_end].astype(int).tolist()
    forecasted_data_a3 = mean_forecast_a3.loc[forecast_start:forecast_end].astype(int).tolist()
    forecasted_data1_s1 = mean_forecast1_s1.loc[forecast_start:forecast_end].astype(int).tolist()
    forecasted_data1_s2 = mean_forecast1_s2.loc[forecast_start:forecast_end].astype(int).tolist()
    forecasted_data1_s3 = mean_forecast1_s3.loc[forecast_start:forecast_end].astype(int).tolist()
	
	
	# MAE 
	
    def mean_absolute_error(y_true, y_pred):
        errors = [abs(y_true[i] - y_pred[i]) for i in range(len(y_true))]
        return sum(errors) / len(y_true)
	
	
	
    mae1 = mean_absolute_error(data.Total, mean_forecast_a1_all)
    mae1 = int(mae1)
    print("Mean Absolute Error (ARIMA) for first pair of p and q : ", mae1)
	
    #forecasted_data_a22 = forecasted_data_a2[:-6]
    mae2 = mean_absolute_error(data.Total, mean_forecast_a2_all)
    mae2 = int(mae2)
    print("Mean Absolute Error (ARIMA) for second pair of p and q : ", mae2)
	
    #forecasted_data_a33 = forecasted_data_a3[:-6]
    mae3 = mean_absolute_error(data.Total, mean_forecast_a3_all)
    mae3 = int(mae3)
    print("Mean Absolute Error (ARIMA) for third pair of p and q : ", mae3)
	
	
	


    mae4 = mean_absolute_error(data.Total, mean_forecast1_s1_all)
    mae4 = int(mae4)
    print("Mean Absolute Error (SARIMAX) for first pair of p and q : ", mae4)
	
 
    mae5 = mean_absolute_error(data.Total, mean_forecast1_s2_all)
    mae5 = int(mae5)
    print("Mean Absolute Error (SARIMAX) for second pair of p and q : ", mae5)
	
  
    mae6 = mean_absolute_error(data.Total, mean_forecast1_s3_all)
    mae6 = int(mae6)
    print("Mean Absolute Error (SARIMAX) for third pair of p and q : ", mae6)





    #### Graphs
    index=pd.date_range(start=forecast_start, end=forecast_end, freq='D').strftime('%Y-%m-%d')
    
    plt.plot(index,forecasted_data_a1, label="ARIMA", linewidth=2)
    plt.plot(index,forecasted_data1_s1, label="SARIMAX", linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Total call volume')
    plt.legend()
    plt.savefig('static/mygraph_a1.png')
    plt.close()


    plt.plot(index,forecasted_data_a2, label="ARIMA", linewidth=2)
    plt.plot(index,forecasted_data1_s2, label="SARIMAX", linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Total call volume')
    plt.legend()
    plt.savefig('static/mygraph_a2.png')
    plt.close()

    plt.plot(index,forecasted_data_a3, label="ARIMA", linewidth=2)
    plt.plot(index,forecasted_data1_s3, label="SARIMAX", linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Total call volume')
    plt.legend()
    plt.savefig('static/mygraph_a3.png')
    plt.close()



	
    ####
    forecast_data = pd.DataFrame({
        "ARIMA (p={}, q={})".format(first_p, first_q): mean_forecast_a1.astype(int),
        "ARIMA (p={}, q={})".format(second_p, second_q): mean_forecast_a2.astype(int),
        "ARIMA (p={}, q={})".format(third_p, third_q): mean_forecast_a3.astype(int),
        "SARIMAX (p={}, q={})".format(first_p, first_q): mean_forecast1_s1.astype(int),
        "SARIMAX (p={}, q={})".format(second_p, second_q): mean_forecast1_s2.astype(int),
        "SARIMAX (p={}, q={})".format(third_p, third_q): mean_forecast1_s3.astype(int),
    }, index=pd.date_range(start=forecast_start, end=forecast_end, freq='D').strftime('%Y-%m-%d'))



    # Calculate mean absolute error for each model
    mae_data = pd.DataFrame({
        "ARIMA (p={}, q={})".format(first_p, first_q): [mae1],
        "ARIMA (p={}, q={})".format(second_p, second_q): [mae2],
        "ARIMA (p={}, q={})".format(third_p, third_q): [mae3],
        "SARIMAX (p={}, q={})".format(first_p, first_q): [mae4],
        "SARIMAX (p={}, q={})".format(second_p, second_q): [mae5],
        "SARIMAX (p={}, q={})".format(third_p, third_q): [mae6],
    })


    # Concatenate the forecast and MAE dataframes
    forecast_data = pd.concat([forecast_data, mae_data.rename(index={0: 'MAE'})])

    # Render the table in a HTML template
    return render_template('table.html', forecast_data=forecast_data.to_html())



	# Create a list of forecasts
    #forecasts = [forecasted_data_a1, forecasted_data_a2, forecasted_data_a3, forecasted_data1_s1, forecasted_data1_s2, forecasted_data1_s3, mae1, mae2, mae3]
	
    # Convert the list of forecasts to JSON and return as response
    #return jsonify(forecasts)



if __name__ == '__main__':
    app.run(port=8000)
