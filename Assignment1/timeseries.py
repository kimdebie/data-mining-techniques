def timeseries(file):

    '''Implement the ARIMA model for time-series analysis'''

    df = pd.read_csv(file)




    return rsquared, mse


rsquared, mse = timeseries('cleaned_normalized.csv')

print('rsquared: %.2f' % rsquared)
print('mse: %.5f' % mse) # this is low because data is downscaled (we should multiply it by 10 I think?)

python manage.py createsuperuser
loaddata initial_data.json
