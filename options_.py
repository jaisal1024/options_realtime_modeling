import pandas as pd
import numpy as np
import math
from scipy.stats import norm
import time
import calendar
from datetime import timedelta, datetime, date

#calendar
c = calendar.Calendar(firstweekday=calendar.SATURDAY)


# HISTORICAL VOLATILITY
def hv_1(historicals, portfolio):
    dates = historicals['date']
    df = {}
    for stock in portfolio:
        sigmas = {}
        h = historicals[stock].join(dates)
        i = timedelta(days = 31)
        j = 1
        while(i+dates[j] <= dates[len(dates)-1]):
            h1 = h[(h['date'] >= dates[j-1]) & (h['date'] <= i+dates[j])]
            h1['volatility'] = h1['close'].pct_change()
            h1[1:]
            sigma_matrix = np.std(h1['volatility'])* np.sqrt(252)
            sigmas[i+dates[j]] = sigma_matrix
            j+=1
        df[stock] = sigmas
    return pd.DataFrame(df)

def hv_2(historicals, portfolio):
    dates = historicals['date']
    df = {}
    for stock in portfolio:
        sigmas = {}
        h = historicals[stock].join(dates)
        i = timedelta(days = 31)
        j = 1
        while(i+dates[j] <= dates[len(dates)-1]):
            h1 = h[(h['date'] >= dates[j-1]) & (h['date'] <= i+dates[j])]
            h1['volatility'] = np.log(h1['close'].shift(1)/h1['close'])
            h1[1:]
            sigma_matrix = np.std(h1['volatility'])* np.sqrt(252)
            sigmas[i+dates[j]] = sigma_matrix
            j+=1
        df[stock] = sigmas
    return pd.DataFrame(df)

def hv_weighted(historicals, portfolio,  weight):
    dates = historicals['date']
    df = {}
    for stock in portfolio:
        sigmas = {}
        h = historicals[stock].join(dates)
        i = timedelta(days = 31)
        j = 1
        while(i+dates[j] <= dates[len(dates)-1]):
            h1 = h[(h['date'] >= dates[j]) & (h['date'] <= i+dates[j])]
            h1['volatility'] = np.log(h1['close']/h1['open'])*weight + np.log(h1['high']/h1['low'])*(1-weight)
            sigma_matrix = np.std(h1['volatility'])* np.sqrt(252)
            sigmas[i+dates[j]] = sigma_matrix
            j+=1
        df[stock] = sigmas
    return pd.DataFrame(df)

def hv_hl(historicals, portfolio):
    dates = historicals['date']
    df = {}
    for stock in portfolio:
        sigmas = {}
        h = historicals[stock].join(dates)
        i = timedelta(days = 31)
        j = 1
        while(i+dates[j] <= dates[len(dates)-1]):
            h1 = h[(h['date'] >= dates[j]) & (h['date'] <= i+dates[j])]
            h1['volatility'] = np.log(h1['high']/h1['low'])
            h1['volatility'] = h1['volatility'].apply(lambda x: 1/(4*np.log(2))*x**2)
            sigma_matrix = np.sqrt(np.sum(h1['volatility'])/len(h1.index)*252)
            sigmas[i+dates[j]] = sigma_matrix
            j+=1
        df[stock] = sigmas
    return pd.DataFrame(df)

def hv_all(stock, hv_1, hv_2, hv_weighted, hv_hl):
    hv_all_ = [pd.DataFrame(hv_1[stock].rename("HV Pct", axis = 'columns'))]
    hv_all_.append(pd.DataFrame(hv_2[stock].rename("HV Log", axis = 'columns')))
    hv_all_.append(pd.DataFrame(hv_weighted[stock].rename("HV Weighted", axis = 'columns')))
    hv_all_.append(pd.DataFrame(hv_hl[stock].rename("HV HL", axis = 'columns')))
    return pd.concat(hv_all_, axis = 1)


# OPTIONS PRICING
def M_(c, default, beta, t):
    x = c*beta*np.sqrt(t)/default
    x = (x-2)/10
    return x/(1+abs(x))+1
def call_(s, x, t, r, q, d1, d2):
    return s*math.exp(-q*t)*norm.cdf(d1)-x*math.exp(-r*t)*norm.cdf(d2)
def put_(s, x, t, r, q, d1, d2):
    return x*math.exp(-r*t)*(1-norm.cdf(d2)) - s*math.exp(-q*t)*(1-norm.cdf(d1))
def d1_(s, x, t, r, q, sigma):
    return (np.log(s/x)+(r-q+(np.power(sigma,2)/2)*t)) / (sigma*np.sqrt(t))
def d2_(d1, sigma, t):
    return d1 - sigma*np.sqrt(t)

# DATE & STRIKE
def third_friday(month, year):
    monthcal = c.monthdatescalendar(year, month)
    return monthcal[2][-1]
def generate_times(start, year, options_dates):
    t = []
    for i in range(6):
        if (i < 3):
            t.append(third_friday(options_dates[(start+i)%len(options_dates)], (year + int((start+i)/len(options_dates)))))
        else:
            t.append(third_friday(options_dates[(start+i+(i-2))%len(options_dates)], (year + int((start+i+(i-2))/len(options_dates)))))
    return t
def t_(today, options_dates):
    t = []
    month = int(today.month-1 + today.day/third_friday(today.month, today.year).day) % 12
    if (month > 9):
        t = generate_times(0, today.year+1, options_dates)
    elif (month > 6):
        t = generate_times(3, today.year, options_dates)
    elif (month > 3):
        t = generate_times(2, today.year, options_dates)
    else:
        t = generate_times(1, today.year, options_dates)
    return t
def strikes_(s, num):
    strikes = {}
    for ticker in s:
        if (s[ticker]>300):
            strikes[ticker] = [((round(s[ticker],-1))+((i-num)*10)) for i in range(2*num)]
        else:
            strikes[ticker] = [((round(s[ticker],-1))+((i-num)*5)) for i in range(2*num)]
    return strikes
def extractQ(p, stocks):
    q_dict = stocks.get_dividends(range = '1y', filter = 'amount')
    q = {}
    for q_s in q_dict:
        sum_ =0
        if (q_dict[q_s]):
            for a in q_dict[q_s]:
                sum_+= a['amount']
        q[q_s] = sum_/p[q_s]
    return q

def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = map(np.uint8, np.array(cmap(k*h)[:3])*255)
        C = list(map(int,C))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale
