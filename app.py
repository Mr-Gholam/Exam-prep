from flask import Flask, render_template
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import statsmodels.api as sm

app = Flask(__name__)

@app.route('/')
def regression():
    df = pd.read_csv('data.csv')
    X = df[['X1', 'X2']]
    y = df['Y']

    # Fit Model
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    residuals = y - predictions

    # Linearity Plot (Actual vs Predicted)
    plt.figure()
    plt.scatter(predictions, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residual Plot (Linearity & Homoscedasticity)')
    plt.savefig('static/residuals.png')
    plt.close()

    # Normality Plot (QQ Plot)
    sm.qqplot(residuals, line='45')
    plt.title('Normality of Residuals')
    plt.savefig('static/normality.png')
    plt.close()

    # Multicollinearity Check (Correlation Matrix)
    corr = X.corr().to_html()

    # R2 Score
    r2 = r2_score(y, predictions)

    return render_template('index.html',
                           coef1=model.coef_[0],
                           coef2=model.coef_[1],
                           intercept=model.intercept_,
                           prediction=model.predict([[6, 7]])[0],
                           r2=r2,
                           corr=corr)

@app.route('/timeseries')
def timeseries():
    df_ts = pd.read_csv('data_ts.csv')
    df_ts['Date'] = pd.to_datetime(df_ts['Date'])

    plt.figure(figsize=(10, 5))
    plt.plot(df_ts['Date'], df_ts['Sales'], marker='o')
    plt.title('Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/timeseries.png')
    plt.close()

    return render_template('timeseries.html')


@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')

if __name__ == '__main__':
    app.run(debug=True)