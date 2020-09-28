# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.0
#   kernelspec:
#     display_name: estate
#     language: python
#     name: estate
# ---

# +
import pandas as pd
import numpy as np
import yaml
from pathlib import Path, PurePath
from sqlalchemy import create_engine
import cx_Oracle
import sys
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import ppscore as pps
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

# Zarządzanie ścieżkami
cwd = Path.cwd()
path = PurePath(cwd)
config_path = path.parent.joinpath("config").joinpath("config.yaml")
dtypes_path = path.parent.joinpath("config").joinpath("dtypes_schema.yaml")
outputs_path = path.parent.joinpath("outputs")
functions_path = path.parent.joinpath("src").joinpath("python")

sys.path.append(str(functions_path))
# -

from data_cleaning_and_EDA import obtain_district, obtain_localisation, obtain_travel_info_driving, obtain_travel_info_transit, create_bar_plot

# Wczytanie schematu typów danych
with open(dtypes_path, "r") as ymlfile:
    # Using yaml SafeLoader to load configuration into dict
    dtypes = yaml.load(ymlfile, Loader=yaml.SafeLoader)

# ## 1. Pobranie danych z bazy danych

# Tworzenia połączenia do bazy danych i pobranie zawartości tabeli OTODOM_OFFERS_DETAILS do postaci ramki danych **df**

# +
# Wczytanie pliku konfiguracyjnego
with open(config_path, "r") as ymlfile:
    # Using yaml SafeLoader to load configuration into dict
    config = yaml.load(ymlfile, Loader=yaml.SafeLoader)
    
# Załaduj dane do połączenia z bazą z pliku config
host = config["host"]
port = config["port"]
service_name = config["service_name"]
user = config["user"]
password = config["password"]

# Stwórz połączenie do bazy danych wykorzystując pakiet SQLAlchemy
dns = cx_Oracle.makedsn(host, port, sid=service_name)
connection_string = f'oracle+cx_oracle://{user}:{password}@{dns}'
engine = create_engine(connection_string)
# -

# Pobranie danych z bazy do ramki danych
sql_query = """SELECT * from grzkup_p.otodom_offers_details where id in (SELECT id from (SELECT DISTINCT numer_oferty, id from grzkup_p.otodom_offers_details)) ORDER BY id"""
df = pd.read_sql_query(sql=sql_query, con=engine, index_col="id")

# Zapisz zbiór jako plik csv (backup)
df.to_csv(outputs_path.joinpath("input_dataset.csv"), encoding="windows-1250")

# Wczytaj zbiór z pliku CSV 
df = pd.read_csv(outputs_path.joinpath("input_dataset.csv"), encoding="windows-1250", dtype=dtypes, index_col=0)

# ### 2. Sprawdzenie jakości zbioru + czyszczenie

pd.options.display.max_columns = 40
pd.options.display.max_rows = 500
pd.options.display.max_colwidth = None

# 5 pierwszych rekordów
df.head()

# Wyświetlenie statystyk dla wszystkich zmiennych
df.describe()

# Zmienna "rok_budowy" zawiera błędne wartości na co wskazuje wartość minimalna 80 i maksymalna 20004. 

# Sprawdzenie ofert dla budynków zbudowanych wcześniej niż 1900 rok
df_with_localisation_cleaned[df_with_localisation_cleaned["rok_budowy"] < 1900]["rok_budowy"]

# +
# Sprawdzenie rekordów z najniższymi wartościami - 80 i 1005 rok budowy
df[df["rok_budowy"] < 1100]

# Sprawdzenie rejonu dla ulicy Narcyzowej (bloki mieszkalne) wskazuje, że najprawdopodobniej 
# autor ogłoszenia miał na myśli rok 1980
df[df["rok_budowy"] == 80] = 1980

# W drugim przypadku, z uwagi na możliwość błędnego uzupełnienia wartości, zdecydowano się wyrzucić rekord z rokiem budowy jako 1005
to_drop_index = df[df["rok_budowy"] == 1005].index
df.drop(to_drop_index, axis=0)
# -

to_drop_index

# Sprawdzenie ofert dla budynków, których data budowy jest później niż 2020 rok
df[df["rok_budowy"] > 2020]["rok_budowy"]

# +
# Sprawdzenie rekordu z najniższą wartością - 20004 rok budowy
df[df["rok_budowy"] == 20004]

# Poprawienie wartości na rok 2004
df[df["rok_budowy"] == 20004] = 2004
# -

# Liczba wartości w kolumnie miasto powinny być tylko Gdańsk, Sopot, Gdynia
print(df["miasto"].value_counts())

# Wyświetlenie miasta, dzielnicy, ulicy oraz tytułu dla rekordów z niepoprawną wartością zmiennej miasto
df.loc[~df["miasto"].isin(["Gdańsk", "Gdynia", "Sopot"]), ["miasto", "dzielnica", "ulica", "tytul"]]

# Jak widać, z uwagi na konwencję adresową osoby tworzącej ogłoszenie dane zostały przypisane do błędnych zmiennych. Dla błędnych rekordów zmienna **miasto** zostanie zasilona zmienną **dzielnica** oraz zmienna **ulica** zamieniona na zmienną **miasto**

# Skopiowanie ramki danych przed dalszymi działaniami 
df_corrected = df.copy()
wrong_rows = ~df["miasto"].isin(["Gdańsk", "Gdynia", "Sopot"])

# +
# Zamiana wartości kolumn
df_corrected.loc[wrong_rows, ["ulica"]] = df_corrected.loc[wrong_rows, "miasto"]
df_corrected.loc[wrong_rows, ["miasto"]] = df_corrected.loc[wrong_rows, "dzielnica"]

# Wstawienie wartości nan dla zmianennej "dzielnica" błędnych rekordów
df_corrected.loc[wrong_rows, "dzielnica"] = np.nan

# Sprawdzenie unikalnych wartości zmiennej "miasto"
df_corrected["miasto"].value_counts()
# -

# Skopiowanie ramki danych przed dalszymi działaniami 
df_corrected_with_districts = df_corrected.copy()

# +
# Indeksy rekordów posiadających pustą wartość dla zmiennej dzielnica i jednocześnie posiadających zmienną ulica
missing_district_index = df_corrected_with_districts["dzielnica"].isna() & ~df_corrected_with_districts["ulica"].isna()

# Zgeokoduj adres by uzyskać informacje o dzielnicy i uzupełnij dwie zmienne tymczasowe "dzielnica_temp_1", "dzielnica_temp_2"
df_corrected_with_districts["dzielnica_temp_1"] = np.nan
df_corrected_with_districts["dzielnica_temp_2"] = np.nan

# +
# Wykorzystując zmienne "miasto" i "ulica" zdekoduj lokalizację i pobierz wartości 
# odpowiadające dzielnicy/rejonowi miasta z użyciem funkcji obtain_district

district_temp = df_corrected_with_districts.loc[
    missing_district_index, ["miasto", "ulica"]].apply(obtain_district, axis=1, result_type="expand")

district_temp.columns = ["dzielnica_temp_1", "dzielnica_temp_2"]
df_corrected_with_districts.loc[missing_district_index, ["dzielnica_temp_1", "dzielnica_temp_2"]] = district_temp
# -

df_corrected_with_districts[missing_district_index]

# W przypadku uzyskanych nazw dzielnic rekordy zawierają kombinację:
#  1. osiedle\mniejsza część dzielnicy - dzielnica
#  2. dzielnica - miasto
#  
# Zdecydowano się w przypadku wariantu 1 na wykorzystanie nazwy głównej dzielnicy - zmienna "dzielnica_temp_2". W przeciwnym przypadku brakującą zmienną "dzielnica" uzupełniono zmienną "dzielnica_temp_1".

# Przygotowanie indeksów do uzupełnienia dzielnic
index_district_temp_2 = df_corrected_with_districts["dzielnica"].isna() & ~df_corrected_with_districts["ulica"].isna() & ~df_corrected_with_districts[missing_district_index]["dzielnica_temp_2"].isin(["Gdańsk", "Gdynia", "Sopot"])
index_district_temp_1 = df_corrected_with_districts["dzielnica"].isna() & ~df_corrected_with_districts["ulica"].isna() & df_corrected_with_districts[missing_district_index]["dzielnica_temp_2"].isin(["Gdańsk", "Gdynia", "Sopot"])

# Uzupełnienie zmiennej dzielnice z wykorzystaniem jednej z dwóch wartości uzyskanej z geokodowania
df_corrected_with_districts.loc[index_district_temp_2, "dzielnica"] = df_corrected_with_districts.loc[index_district_temp_2, "dzielnica_temp_2"]
df_corrected_with_districts.loc[index_district_temp_1, "dzielnica"] = df_corrected_with_districts.loc[index_district_temp_1, "dzielnica_temp_1"]

df_corrected_with_districts[missing_district_index]

# Zapis zbioru do pliku CSV
df_corrected_with_districts.to_csv(outputs_path.joinpath("input_dataset_with_districts.csv"), encoding="windows-1250")

# Usunięcie zmiennych tymczasowych "dzielnica_temp_1" oraz "dzielnica_temp_2"
df_with_localisation = df_with_localisation.drop(columns=["dzielnica_temp_1", "dzielnica_temp_2"])

# ### 3. Pobranie danych dodatkowych

# Skopiowanie ramki danych przed dalszymi działaniami 
df_with_localisation = df_corrected_with_districts.copy()

# W celu wizulizacji danych przestrzennych jakimi są oferty zakupu nieruchomości niezbędne jest geokodowanie adresów by uzyskać współrzędne geograficzne. 

# +
# Wybór indeksów dla rekordów posiadających zmienną "miasto" i "ulica"
missing_city_street_index = ~df_corrected_with_districts["miasto"].isna() & ~df_corrected_with_districts["ulica"].isna()

# Współrzędne każdej z nieruchomości będzie pobrana wykorzystując Google API obudowane w funkcji obtain_localisation
localisation_results = df_with_localisation.loc[missing_city_street_index, ["miasto", "ulica"]].apply(obtain_localisation, axis=1, result_type="expand")
localisation_results.columns = ["latitude", "longitude"]
# -

# Wstawienie informacji o współrzędnych do głównej ramki danych
df_with_localisation.loc[missing_city_street_index, ["latitude", "longitude"]] = localisation_results.loc[:, ["latitude", "longitude"]]

df_with_localisation.head()

# Sprawdzenie liczby rekordów dla których udało się pozyskać współrzędne
df_with_localisation["latitude"].isna().value_counts()

# Zapisz rezultat do pliku CSV
df_with_localisation.to_csv(outputs_path.joinpath("input_dataset_with_localisation.csv"), encoding="windows-1250")

# Wczytaj rezultat z pliku CSV
df_with_localisation = pd.read_csv(outputs_path.joinpath("input_dataset_with_localisation.csv"), 
                                   encoding="windows-1250", 
                                   index_col=0,
                                   dtype=dtypes)

# Sprawdzenie brakujących wartości dla każdej ze zmiennych
df_with_localisation.isna().sum()

# Z uwagi na stosunkowo mały udział brakujących wartości dla zmiennej dzielnica, pietro i liczba_pieter zdecydowano się na usunięcie danych rekordów. 
#
# Z uwagi na znaczny udział braku danych - ponad 50 % rekordów - zdecydowano się na odrzucenie zmiennych material_budynku, okna oraz czynsz. 

# +
# Skopiowanie ramki danych przed dalszymi działaniami 
df_with_localisation_cleaned = df_with_localisation.copy()

# Wybór indeksów rekordów zawierających puste wartości dla zmiennych "dzielnica", "pietro", "liczba_pieter" 
# oraz usunięcie rekordów które zawierają puste wartości dla tych zmiennych
missing_pietro_liczba_pieter_index = df_with_localisation_cleaned["dzielnica"].isna() \
| df_with_localisation_cleaned["pietro"].isna() | \
df_with_localisation_cleaned["liczba_pieter"].isna()
df_with_localisation_cleaned = df_with_localisation_cleaned[~missing_pietro_liczba_pieter_index]

# Usunięcie zmiennych "material_budynku", "okna", "czynsz"
df_with_localisation_cleaned = df_with_localisation_cleaned.drop(columns=["material_budynku", "okna", "czynsz"], axis=1)
# -

df_with_localisation_cleaned.isna().sum()

# W kontekście lokalizacji nieruchomości istotny jest czas dojazdu do określonych punktów. Jako cel codziennych podróży wybrano kompleks Olivia Business Centre. Korzystając z Google API w funkcjach **obtain_travel_info_driving** i **obtain_travel_info_transit** zdecydowano się na sprawdzenie czasu dojazdu oraz dystansu dla podróży w poniedziałek z przybyciem do miejsca docelowego o godzinie 8. 
#
# Wartości sprawdzono zarówno dla podróży samochodem osobowym jak i transportem zbiorowym. 

# Pobranie danych o czasie podróży i dystansie do tymczasowej ramki danych travel_info_results_driving
travel_info_result_driving = df_with_localisation.loc[:, ["miasto", "dzielnica", "ulica"]].apply(obtain_travel_info_driving, axis=1, result_type="expand")
travel_info_results_driving.columns = ["czas_auto", "dystans_auto"]

# Pobranie danych o czasie podróży i dystansie do tymczasowej ramki danych travel_info_results_transit
travel_info_result_transit = df_with_localisation.loc[:, ["miasto", "dzielnica", "ulica"]].apply(obtain_travel_info_transit, axis=1, result_type="expand")
travel_info_results_transit.columns = ["czas_zbiorowy", "dystans_zbiorowy"]

# +
# Zasilenie głównej ramki danych nowo utworzonymi zmiennymi 
df_with_localisation_cleaned.loc[:, ["czas_auto", "dystans_auto"]] = travel_info_results.loc[:, ["czas_auto", "dystans_auto"]]

df_with_localisation_cleaned.loc[:, ["czas_zbiorowy", "dystans_zbiorowy"]] = travel_info_results.loc[:, [
    "czas_zbiorowy", "dystans_zbiorowy"]]
# -

df_with_localisation_cleaned.head()

# Zapisz rezultat do pliku CSV
df_with_localisation_cleaned.to_csv(outputs_path.joinpath("input_dataset_with_localisation_cleaned.csv"), encoding="windows-1250")

# Wczytaj rezultat z pliku CSV
df_with_localisation_cleaned = pd.read_csv(outputs_path.joinpath("input_dataset_with_localisation_cleaned.csv"), 
                                   encoding="windows-1250", 
                                   index_col=0)

# ### 4. EDA

df_with_localisation_cleaned.info()

# +
variable_name = "liczba_pokoi"
fig = create_bar_plot(df=df_with_localisation_cleaned, variable_name=variable_name, variable_label="Liczba pokoi")
fig.show()

print(f"Ilość pustych wartości dla zmiennej {variable_name}: {df_with_localisation_cleaned[variable_name].isna().sum()}")

# +
variable_name = "rynek"
fig = create_bar_plot(df=df_with_localisation_cleaned, variable_name=variable_name, variable_label="Typ rynku")
fig.show()

print(f"Ilość pustych wartości dla zmiennej {variable_name}: {df_with_localisation_cleaned[variable_name].isna().sum()}")

# +
variable_name = "zabudowa"
fig = create_bar_plot(df=df_with_localisation_cleaned, variable_name=variable_name, variable_label="Typ zabudowy")
fig.show()

print(f"Ilość pustych wartości dla zmiennej {variable_name}: {df_with_localisation_cleaned[variable_name].isna().sum()}")

# +
variable_name = "pietro"
categories_orders_dict = {"index": ["parter", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", ">_10", "suterena", "poddasze"]}
fig = create_bar_plot(df=df_with_localisation_cleaned, variable_name=variable_name, variable_label="Piętro", categories=True,category_orders=categories_orders_dict)

fig.show()

print(f"Ilość pustych wartości dla zmiennej {variable_name}: {df_with_localisation_cleaned[variable_name].isna().sum()}")

# +
variable_name = "liczba_pieter"
fig = create_bar_plot(df=df_with_localisation_cleaned, variable_name=variable_name, variable_label="Liczba pięter")
fig.show()

print(f"Ilość pustych wartości dla zmiennej {variable_name}: {df_with_localisation_cleaned[variable_name].isna().sum()}")

# +
variable_name = "ogrzewanie"
fig = create_bar_plot(df=df_with_localisation_cleaned, variable_name=variable_name, variable_label="Typ zabudowy")
fig.show()

print(f"Ilość pustych wartości dla zmiennej {variable_name}: {df_with_localisation_cleaned[variable_name].isna().sum()}")

# +
variable_name = "rok_budowy"
min_value = int(df_with_localisation_cleaned[variable_name].min())
max_value = int(df_with_localisation_cleaned[variable_name].max()) + 1
fig = create_bar_plot(df=df_with_localisation_cleaned, variable_name=variable_name, variable_label="Rok budowy", categories=True, category_orders={"index": list(range(min_value, max_value))})
fig.show()

print(f"Ilość pustych wartości dla zmiennej {variable_name}: {df_with_localisation_cleaned[variable_name].isna().sum()}")

# +
variable_name = "stan_wykonczenia"
fig = create_bar_plot(df=df_with_localisation_cleaned, variable_name=variable_name, variable_label="Stan wykończenia")
fig.show()

print(f"Ilość pustych wartości dla zmiennej {variable_name}: {df_with_localisation_cleaned[variable_name].isna().sum()}")

# +
variable_name = "forma_wlasnosci"
fig = create_bar_plot(df=df_with_localisation_cleaned, variable_name=variable_name, variable_label="Forma własności")
fig.show()

print(f"Ilość pustych wartości dla zmiennej {variable_name}: {df_with_localisation_cleaned[variable_name].isna().sum()}")
# -

for feature in ["teren_zamkniety",
                "balkon", 
                "ogrodek", 
                "garaz_miejsce",
                "winda",
                "piwnica",
                "monitoring_ochrona"]:
    variable_name = feature
    fig = create_bar_plot(df=df_with_localisation_cleaned, variable_name=variable_name, variable_label=feature.capitalize())
    fig.show()

# +
# for feature in ["powierzchnia", "cena", "cena_metr", "czas_auto", "czas_zbiorowy", "dystans_auto", "dystans_zbiorowy"]:
#     fig = go.Figure(px.histogram(data_frame=df_with_localisation_cleaned, x=feature, nbins=20))
#     fig.show()
#     fig = go.Figure(px.box(data_frame=df_with_localisation_cleaned, x=feature))
#     fig.show()

for feature in ["powierzchnia", "cena", "cena_metr", "czas_auto", "czas_zbiorowy", "dystans_auto", "dystans_zbiorowy"]:
    fig = plt.figure(figsize=(16,8))
    sns.histplot(data=df_with_localisation_cleaned, x=feature)
    plt.plot()
    fig = plt.figure(figsize=(16,8))
    sns.boxplot(data=df_with_localisation_cleaned, y=feature)
    plt.plot()
# -

for category in ["rynek", "ogrzewanie", "winda", "balkon", "ogrodek", "piwnica", "monitoring_ochrona", "stan_wykonczenia", "teren_zamkniety"]:
    fig = plt.figure(figsize=(16,8))
    sns.boxplot(y=df_with_localisation_cleaned["cena_metr"], x=df_with_localisation_cleaned[category])
    plt.plot()

fig = go.Figure(px.scatter(data_frame=df_with_localisation_cleaned, x="powierzchnia", y="cena_metr", color="rynek"))
fig.show()

fig = go.Figure(ff.create_hexbin_mapbox(
    data_frame=df_with_localisation_cleaned, lat="latitude", lon="longitude",
    nx_hexagon=25, opacity=0.8, zoom=9.4,
    labels={"color": "Średnia cena za metr"},
    color="cena_metr", agg_func=np.mean, color_continuous_scale="Teal",
    show_original_data=True,
    original_data_marker=dict(size=4, opacity=0.7, color="black")
))
fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=18.555, mapbox_center_lat=54.441)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig = go.Figure(ff.create_hexbin_mapbox(
    data_frame=df_with_localisation_cleaned, lat="latitude", lon="longitude",
    nx_hexagon=25, opacity=0.8, zoom=9.4,
    labels={"color": "Czas dojazdu do OBC - transport prywatny [s]"},
    color="czas_auto", agg_func=np.mean, color_continuous_scale="Reds"
))
fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=18.555, mapbox_center_lat=54.441)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig = go.Figure(ff.create_hexbin_mapbox(
    data_frame=df_with_localisation_cleaned, lat="latitude", lon="longitude",
    nx_hexagon=25, opacity=0.8, zoom=9.4,
    labels={"color": "Czas dojazdu do OBC - transport zbiorowy [s]"},
    color="czas_zbiorowy", agg_func=np.mean, color_continuous_scale="Reds"
))
fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=18.555, mapbox_center_lat=54.441)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

# +
fig = go.Figure(px.scatter_mapbox(df_with_localisation_cleaned, 
                                  lat="latitude", 
                                  lon="longitude",     
                                  color="cena_metr", 
                                  size="powierzchnia",
                                  opacity=1,
                                  color_continuous_scale="Reds", 
                                  size_max=15, 
                                  zoom=9.4,
                                  labels={"cena_metr": "Średnia cena za metr", "powierzchnia": "Powierzchnia"}))

fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=18.555, mapbox_center_lat=54.441)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# -

df_with_localisation_cleaned.columns

columns_not_for_modelling = ["id", "id_offers", "ulica", "tytul", "latitude", "longitude", "cena", "numer_oferty"]

predictors_df = pps.predictors(df_with_localisation_cleaned[df_with_localisation_cleaned["miasto"] == "Sopot"].drop(columns=columns_not_for_modelling, axis=1), y="cena_metr")
plt.figure(figsize=(16,8))
sns.barplot(data=predictors_df, x="x", y="ppscore")
plt.xticks(rotation=90)

predictors_df

df_with_localisation_cleaned.head()

df_for_modelling = df_with_localisation_cleaned.copy()

# +
# Enkodowanie zmiennej miasto z użyciem One Hot Encoder
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)

# -

ohe.fit_transform(X=df_for_modelling[["miasto"]])

df_for_modelling[["pierwotny", "wtorny"]] = pd.get_dummies(df_for_modelling["rynek"])

df_for_modelling[["Gdansk", "Gdynia", "Sopot"]] = pd.get_dummies(df_for_modelling["miasto"])

matrix_df = pps.matrix(df_for_modelling[[
    "cena_metr", 
    "powierzchnia", 
    "czas_auto", 
    "dystans_auto", 
    "czas_zbiorowy", 
    "dystans_zbiorowy",
    "rok_budowy",
    "Gdansk",
    "Gdynia",
    "Sopot",
    "pierwotny",
    "wtorny"
]])[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
sns.heatmap(matrix_df, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)

# ### 5. Tworzenie i transformacja zmiennych

df_new_features = df_with_localisation_cleaned.copy()


# +
#Tworzenie zmiennej wiek budynku - różnica między bieżącym rokiem a rokiem budowy, dla budynków w budowie oraz budynków tegorocznych wartość wynosi 0

def calculate_building_age(built_year):
    year = datetime.datetime.now().year
    if built_year > year:
        built_year = year
    return year - built_year

df_new_features["wiek_budynku"] = df_new_features["rok_budowy"].apply(calculate_building_age)

#Tworzennie zmiennej binarnej "w_budowie" dla daty budowy 2021 i później
df_new_features["w_budowie"] = df_new_features["rok_budowy"].apply(lambda x: 1 if (x > datetime.datetime.now().year) else 0)
