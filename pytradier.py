import json
import requests
import pandas as pd
from requests import session
from datetime import date
import re as r

class APIKeyMissingError(Exception):
    pass

class Tradier(object):
    def __init__(self, TRADIER_API_KEY, portfolio = [], application = 'application/json'):
        if TRADIER_API_KEY is None:
            raise APIKeyMissingError(
                "All methods require an API key. See "
                "https://developer.tradier.com/"
                "for how to retrieve an authentication token from Tradier"
            )
        self._session = requests.Session()
        self._url = "https://sandbox.tradier.com"
        self._auth = TRADIER_API_KEY
        self.portfolio = portfolio

        self._session.headers = {}
        self._session.headers['Authorization'] = self._auth
        self._session.headers['Accept'] = application

    def check_expiration(self, expiration, symbol):
        if (type(expiration) == date):
            expiration = expiration.isoformat()
        elif(r.match('^([0-9]{4})-(1[0-2]|0[1-9])-(3[01]|0[1-9]|[12][0-9])$', expiration) is None):
            raise ValueError('Passed an incorrect date type: use ISO YYYY-MM-DD')
        if (expiration not in self.get_option_expiration(symbol)):
            return False
            # raise ValueError('Passed an incorrect expiration date for symbol: {0} at {1}'.format(symbol, expiration))
        return True

    def get_option_expiration(self, symbol):
        endpoint = '{0}/v1/markets/options/expirations'.format(self._url)
        params = {
            "symbol": symbol
        }
        try:
            response = self._session.get(endpoint, params = params).json()["expirations"]["date"]
            return response
        except ValueError:
            raise ValueError

    def get_option_strikes(self, symbol, expiration):
        endpoint = '{0}/v1/markets/options/strikes'.format(self._url)
        params = {
            "symbol": symbol,
            "expiration" : expiration
        }
        self.check_expiration(expiration, symbol)
        try:
            response = self._session.get(endpoint, params = params).json()["strikes"]["strike"]
            return response
        except ValueError:
            raise ValueError

    def get_option_chain(self, symbol, expiration, return_type = 'json'):
        endpoint = '{}/v1/markets/options/chains'.format(self._url)
        params = {
            "symbol": symbol,
            "expiration" : expiration
        }
        self.check_expiration(expiration, symbol)

        try:
            response = self._session.get(endpoint, params = params).json()["options"]["option"]
        except ValueError:
            raise ValueError

        if (return_type == 'pandas'):
            response = pd.DataFrame(response)
        return response
