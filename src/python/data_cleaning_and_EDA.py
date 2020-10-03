from geopy import Nominatim, GoogleV3
from geopy.extra.rate_limiter import RateLimiter
import numpy as np
import pandas as pd
import yaml
from pathlib import Path, PurePath
import datetime
import urllib
from urllib.parse import quote
import simplejson
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from IPython.display import display

# Zarządzanie ścieżkami
cwd = Path.cwd()
path = PurePath(cwd)
config_path = path.parent.joinpath("config").joinpath("config.yaml")

# Wczytanie pliku konfiguracyjnego
with open(config_path, "r") as ymlfile:
    # Using yaml SafeLoader to load configuration into dict
    config = yaml.load(ymlfile, Loader=yaml.SafeLoader)


def obtain_district(location):
    final_location = ", ".join(location.tolist())
    nominatim = Nominatim(user_agent="estate")
    geocoder = RateLimiter(nominatim.geocode)
    geocoded_location = geocoder(final_location)
    if geocoded_location:
        list_of_params = geocoded_location.address.split(", ")
        district_1 = list_of_params[1]
        district_2 = list_of_params[2]
    else:
        district_1 = np.nan
        district_2 = np.nan
    return district_1, district_2


def obtain_localisation(location):
    final_location = ", ".join(location.tolist())
    googlemaps = GoogleV3(user_agent="estate", api_key=config["API_KEY"])
    geocoder = RateLimiter(googlemaps.geocode)
    geocoded_location = geocoder(final_location)
    if geocoded_location:
        latitude, longitude = geocoded_location.latitude, geocoded_location.longitude
    else:
        latitude, longitude = np.nan, np.nan
    return latitude, longitude


def obtain_travel_info_driving(final_location, arrival_time=[2020, 9, 21, 8, 0]):
    city, district, street = final_location.tolist()

    if city and street:
        final_location = quote(f"{city},{street}")
    elif city and district:
        final_location = quote(f"{city},{district}")
    elif city in ["Sopot", "Gdynia"]:
        final_location = quote(city)
    else:
        return np.nan, np.nan

    olivia_business_centre = "54.4032862,18.5686103"
    arrival_time = int(datetime.datetime(*arrival_time).timestamp())
    request_driving_string = rf"https://maps.googleapis.com/maps/api/distancematrix/json?origins=" \
                             rf"{final_location}&destinations={olivia_business_centre}&arrival_time=" \
                             rf"{arrival_time}&mode=driving&key={config['API_KEY']}"

    driving_json = urllib.request.urlopen(request_driving_string)

    travel_info_driving = simplejson.load(driving_json)

    if travel_info_driving["status"] == "OK" and travel_info_driving["rows"][0]["elements"][0]["status"] == "OK":
        travel_time_driving = travel_info_driving["rows"][0]["elements"][0]["duration"]["value"]
        distance_driving = travel_info_driving["rows"][0]["elements"][0]["distance"]["value"]
    else:
        travel_time_driving, distance_driving = np.nan, np.nan

    return travel_time_driving, distance_driving


def obtain_travel_info_transit(final_location, arrival_time=[2020, 9, 28, 8, 0]):
    city, district, street = final_location.tolist()

    if city and street:
        final_location = quote(f"{city},{street}")
    elif city and district:
        final_location = quote(f"{city},{district}")
    elif city in ["Sopot", "Gdynia"]:
        final_location = quote(city)
    else:
        return np.nan, np.nan

    olivia_business_centre = "54.4032862,18.5686103"
    arrival_time = int(datetime.datetime(*arrival_time).timestamp())

    request_transit_string = rf"https://maps.googleapis.com/maps/api/distancematrix/json?origins=" \
                             rf"{final_location}&destinations={olivia_business_centre}&arrival_time=" \
                             rf"{arrival_time}&mode=transit&key={config['API_KEY']}"

    transit_json = urllib.request.urlopen(request_transit_string)

    travel_info_transit = simplejson.load(transit_json)

    if travel_info_transit["status"] == "OK" and travel_info_transit["rows"][0]["elements"][0]["status"] == "OK":
        travel_time_transit = travel_info_transit["rows"][0]["elements"][0]["duration"]["value"]
        distance_transit = travel_info_transit["rows"][0]["elements"][0]["distance"]["value"]
    else:
        travel_time_transit, distance_transit = np.nan, np.nan

    return travel_time_transit, distance_transit


def create_bar_plot(df, variable_name, variable_label, categories=False, category_orders={}):
    fig = go.Figure(px.bar(data_frame=df[variable_name].value_counts(),
                           labels={"index": variable_label,
                                   "value": "Liczność"},
                           category_orders=category_orders,
                           title=f"Liczność dla zmiennej {variable_name}"))
    fig.update_layout(showlegend=False)

    if categories:
        fig.update_layout(**{"xaxis": {"type": "category"}})
    return fig


def calculate_mape(predict, actual):
    predict = np.array(predict)
    actual = np.array(actual)
    score = np.average(np.abs((actual - predict) / actual), axis=0) * 100
    return score


def calculate_metrics(estimator, X_train, y_train, X_test, y_test):
    predict_train = estimator.predict(X_train)
    predict_test = estimator.predict(X_test)

    mape_train = calculate_mape(predict_train, y_train)
    mape_test = calculate_mape(predict_test, y_test)
    rmse_train = mean_squared_error(predict_train, y_train, squared=False)
    rmse_test = mean_squared_error(predict_test, y_test, squared=False)

    scores = {"train": {"MAPE": mape_train, "RMSE": rmse_train},
              "test": {"MAPE": mape_test, "RMSE": rmse_test}}

    results = pd.DataFrame(scores)
    display(results)
