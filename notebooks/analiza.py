# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: estate
#     language: python
#     name: estate
# ---

# +
# Import pakietów 

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
import shap
from scipy.stats.distributions import uniform, randint
from scipy.stats import zscore
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold, train_test_split
from yellowbrick.regressor.residuals import residuals_plot, prediction_error

# Zarządzanie ścieżkami
cwd = Path.cwd()
path = PurePath(cwd)
config_path = path.parent.joinpath("config").joinpath("config.yaml")
dtypes_path = path.parent.joinpath("config").joinpath("dtypes_schema.yaml")
outputs_path = path.parent.joinpath("outputs")
functions_path = path.parent.joinpath("src").joinpath("python")

# Dodanie ścieżki do funkcji stworzonych w ramach projektu
sys.path.append(str(functions_path))
# -

# Ustawienia biblioteki pandas
pd.options.display.max_columns = 40
pd.options.display.max_rows = 500
pd.options.display.max_colwidth = None
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Import funkcji stworzonych w ramach projektu
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

# Wstawienie wartości nan dla zmiennej "dzielnica" błędnych rekordów
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

# Wczytaj rezultat z pliku CSV
df_corrected_with_districts = pd.read_csv(outputs_path.joinpath("input_dataset_with_districts.csv"), 
                                   encoding="windows-1250", 
                                   index_col=0,
                                   dtype=dtypes)

df_corrected_with_districts["dzielnica"] = df_corrected_with_districts["dzielnica"].map(district_names_mapping).\
fillna(df_with_localisation_cleaned["dzielnica"])

# ### 3. Pobranie danych dodatkowych

# Skopiowanie ramki danych przed dalszymi działaniami 
df_with_localisation = df_corrected_with_districts.copy()

# Usunięcie zmiennych tymczasowych "dzielnica_temp_1" oraz "dzielnica_temp_2"
df_with_localisation = df_with_localisation.drop(columns=["dzielnica_temp_1", "dzielnica_temp_2"])

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

# Wyświetlenie podglądu ramki danych
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

# Liczba brakujących wartości dla poszczególnych zmiennych
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

# Wyświetlenie podglądu ramki danych 
df_with_localisation_cleaned.head()

# Sprawdzenie wartości unikalnych dla dzielnic każdego z miast.

sorted(df_with_localisation_cleaned[df_with_localisation_cleaned["miasto"] == "Gdańsk"]["dzielnica"].unique())

# Analizując unikalne wartości dzielnic Gdańska zdecydowano się na ujednolicenie nazw dla następujących dzielnic:
# - Żabianka i Żabianka-Wejhera-Jelitkowo-Tysiąclecia - na Żabianka
# - Wrzeszcz i Wrzeszcz Górny - na Wrzeszcz
# - Zaspa i Zaspa-Młyniec - na Zaspa
# - Nowe Ujeścisko na Wzgórze Mickiewicza

sorted(df_with_localisation_cleaned[df_with_localisation_cleaned["miasto"] == "Gdynia"]["dzielnica"].unique())

# Analizując unikalne wartości dzielnic Gdyni zdecydowano się na ujednolicenie nazw dla następujących dzielnic:
# - Witomino i Witomino - Leśniczówka na Witomino
# - Wzgórze Św. Maksymiliana i Wzgórze Świętego Maksymiliana na Wzgórze Św. Maksymiliana

sorted(df_with_localisation_cleaned[df_with_localisation_cleaned["miasto"] == "Sopot"]["dzielnica"].unique())

# Analizując unikalne wartości dzielnic Sopotu zdecydowano się na ujednolicenie nazw dla następujących dzielnic:
# - Górny i Osiedle Mickiewicza na Górny
#
# Dodatkowo okazło się, że występują wartości "województwo pomorskie" 

df_with_localisation_cleaned[df_with_localisation_cleaned["dzielnica"] == "województwo pomorskie"]

# Wyfiltrowane rekordy odpowiadają dzielnicy Karlikowo, która zostanie do nich przyporządkowana. 

# +
# Zamiana wspomnianych wyżej nazw dzielnic 
district_names_mapping = {"Żabianka-Wejhera-Jelitkowo-Tysiąclecia": "Żabianka",
                          "Wrzeszcz Górny": "Wrzeszcz",
                          "Witomino-Leśniczówka": "Witomino",
                          "Wzgórze Świętego Maksymiliana": "Wzgórze Św. Maksymiliana",
                          "Osiedle Mickiewicza": "Górny",
                          "województwo pomorskie": "Karlikowo",
                          "Nowe Ujeścisko": "Wzgórze Mickiewicza",
                          "Zaspa-Młyniec": "Zaspa",
                          "Pustki Cisowskie-Demptowo": "Pustki Cisowskie"
                         }

df_with_localisation_cleaned["dzielnica"] = df_with_localisation_cleaned["dzielnica"].map(district_names_mapping).\
fillna(df_with_localisation_cleaned["dzielnica"])
# -

# Zapisz rezultat do pliku CSV
df_with_localisation_cleaned.to_csv(outputs_path.joinpath("input_dataset_with_localisation_cleaned.csv"), 
                                    encoding="windows-1250")

# Wczytaj rezultat z pliku CSV
df_with_localisation_cleaned = pd.read_csv(outputs_path.joinpath("input_dataset_with_localisation_cleaned.csv"), 
                                           encoding="windows-1250", 
                                           index_col=0)

# ### 4. EDA

# Wyświetl podstawowe informacje o zbiorze
df_with_localisation_cleaned.info()

# +
variable_name = "liczba_pokoi"
fig = create_bar_plot(df=df_with_localisation_cleaned, variable_name=variable_name, variable_label="Liczba pokoi")
fig.show()

print(f"Ilość pustych wartości dla zmiennej {variable_name}: {df_with_localisation_cleaned[variable_name].isna().sum()}")
# -

# Jak można zauważyć, najpopularniejszym układem oferowanych nieruchomości są te 2 i 3 pokojowe.

# +
variable_name = "rynek"
fig = create_bar_plot(df=df_with_localisation_cleaned, variable_name=variable_name, variable_label="Typ rynku")
fig.show()

print(f"Ilość pustych wartości dla zmiennej {variable_name}: {df_with_localisation_cleaned[variable_name].isna().sum()}")
# -

# Zdecydowana większość ofert dotyczy nieruchomości z rynku wtórnego. 

# +
variable_name = "zabudowa"
fig = create_bar_plot(df=df_with_localisation_cleaned, variable_name=variable_name, variable_label="Typ zabudowy")
fig.show()

print(f"Ilość pustych wartości dla zmiennej {variable_name}: {df_with_localisation_cleaned[variable_name].isna().sum()}")
# -

# Najpopularniejszym typem zabudowy wśród ofert są nieruchomości będące częścią bloku mieszkalnego i apartamentowca. Nieruchomości stanowiące niezależne budynki stanowią jedynie odsetek. Dla zmiennej **"zabudowa"** występuje ponad 900 brakujących wartości. 

# +
variable_name = "pietro"
categories_orders_dict = {"index": ["parter", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", ">_10", "suterena", "poddasze"]}
fig = create_bar_plot(df=df_with_localisation_cleaned, variable_name=variable_name, variable_label="Piętro", categories=True,category_orders=categories_orders_dict)

fig.show()

print(f"Ilość pustych wartości dla zmiennej {variable_name}: {df_with_localisation_cleaned[variable_name].isna().sum()}")
# -

# Najczęstszymi typami nieruchomości są te znajdujące się na piętrach 0 - 3, przy czym nieruchomości na pierwszym piętrze występują najczęściej. Wraz ze wzrostem piętra, na jakim znajduje się nieruchomość, maleje liczba ofert. 

# +
variable_name = "liczba_pieter"
fig = create_bar_plot(df=df_with_localisation_cleaned, variable_name=variable_name, variable_label="Liczba pięter")
fig.show()

print(f"Ilość pustych wartości dla zmiennej {variable_name}: {df_with_localisation_cleaned[variable_name].isna().sum()}")
# -

# Zdecydowanie najczęściej występującym typem budynków są budynki 3 i 4 piętrowe. Wyróżniają się również budynki 2 piętrowe (przypuszczalnie kamienice) oraz 10 piętrowe (np. zabudowa wielomieszkalna na Przymorzu)

# +
variable_name = "ogrzewanie"
fig = create_bar_plot(df=df_with_localisation_cleaned, variable_name=variable_name, variable_label="Typ zabudowy")
fig.show()

print(f"Ilość pustych wartości dla zmiennej {variable_name}: {df_with_localisation_cleaned[variable_name].isna().sum()}")
# -

# Najczęstszym typem centralnego ogrzewania jest ogrzewanie miejskie. Dla zmiennej ogrzewanie występuje aż 1088 brakujących wartości. 

# +
variable_name = "rok_budowy"
min_value = int(df_with_localisation_cleaned[variable_name].min())
max_value = int(df_with_localisation_cleaned[variable_name].max()) + 1
fig = create_bar_plot(df=df_with_localisation_cleaned, variable_name=variable_name, variable_label="Rok budowy", categories=True, category_orders={"index": list(range(min_value, max_value))})
fig.show()

print(f"Ilość pustych wartości dla zmiennej {variable_name}: {df_with_localisation_cleaned[variable_name].isna().sum()}")
# -

# Analziując rok budowy nieruchomości można wyróżnić 3 grupy:
# - budynki zabytkowe zbudowane przed 1950 rokiem
# - budynki zbudowane w okresie PRL tzw. wielka płyta - lata 60 - 80 
# - budynki zbudowane po roku 2000 - z największą ilością ofert dla roku budowy 2020
#
# Blisko 400 ofert dotyczy budynków, które są dopiero w budowie - rok budowy późniejszy niż 2020.
#
# Dla 492 rekordów wsytępuje brak wartości dla zmiennej **"rok_budowy"**. 

# +
variable_name = "stan_wykonczenia"
fig = create_bar_plot(df=df_with_localisation_cleaned, variable_name=variable_name, variable_label="Stan wykończenia")
fig.show()

print(f"Ilość pustych wartości dla zmiennej {variable_name}: {df_with_localisation_cleaned[variable_name].isna().sum()}")
# -

# Najwięcej ofert dotyczy nieruchomości gotowych do zamieszkania. Grupa "do_wykończenia" może dotyczyć głównie nieruchomości z rynku pierwotnego. Dla zmiennej **"stan_wyknonczenia"** wykryto 820 brakujących wartości

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

# Zmienne kategoryczne:
# - zdedydowana większość nieruchomości dotyczy nieruchomości nie będących częścią zamkniętego osiedla
# - nieruchmości posiadające balkon stanowią 60 %
# - blisko 500 nieruchomości posiada dostęp do ogródka
# - żadna z ofert nie zawierała informacji o dostępie do garażu / miejsca postojowego - być może jest to wynikiem błędu w procesie pobierania danych ze strony
# - ponad 60 % nieruchomości nie znajduje się w budynku, który jest wyposażony w windę
# - występowanie informacji o przynależności piwnicy stanowi 50 % ogłoszeń
# - blisko 25 % nieruchomości nie posiada monitoringu czy ochrony

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

# Dzięki wykresom boxplot można zauważyć obserwacje odstające w skali całego zbioru. Są to nieruchomości o powierzchni powyżej 120 metrów kwadratowych i cenie za metr 20000. Analizując wykres boxplot oraz histogram dla zmiennej cena, można zauważyć kilka bardzo wysokich ofert - w tym oferta z ceną 16 milionów złotych. 

for category in ["rynek", "ogrzewanie", "winda", "balkon", "ogrodek", "piwnica", "monitoring_ochrona", "stan_wykonczenia", "teren_zamkniety"]:
    fig = plt.figure(figsize=(16,8))
    sns.boxplot(y=df_with_localisation_cleaned["cena_metr"], x=df_with_localisation_cleaned[category])
    plt.plot()

# Po wykresach pudełkowych wskazujących wartość zmiennej celu **cena_metr** oraz rozdział na kategorie:
# - średnia cena za metr jest wyższa dla nieruchomości z rynku wtórnego
# - mimo różnego typu ogrzewania - średnia cena za metr oscyluje wokół tej samej wartości - ok 9000 zł. 
# - średnia cena za metr jest wyższa dla nieruchomości posiadających informację o dostępie do windy
# - minimalnie niższą cenę za metr posiadają nieruchomości bez balkonu
# - nieruchomości, w których ofercie wspomniano o monitoringu/ochronie mają wyższą cenę za cena_metr
# - najwyższą średnią cenę za metr posiadają nieruchomości gotowe do zamieszkania. Najniższą średnią cenę za cena_metr
# wykazują nieruchomości przeznaczone do remontu. 
# - zdecydowanie wyższą średnią cenę za metr posiadają nieruchomości, w których ofercie poinformowano o terenie zamkniętym. 

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

# Najwyższa cena za metr występuje w rejonach stanowiących centrum miast Gdańsk, Gdynia, Sopot. Wyższa cena za metr wiąże się równiez z położeniem w pasie nadmorskim. 

fig = go.Figure(ff.create_hexbin_mapbox(
    data_frame=df_with_localisation_cleaned, lat="latitude", lon="longitude",
    nx_hexagon=25, opacity=0.8, zoom=9.4,
    labels={"color": "Czas dojazdu do OBC - transport prywatny [s]"},
    color="czas_auto", agg_func=np.mean, color_continuous_scale="Reds"
))
fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=18.555, mapbox_center_lat=54.441)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

# Czas dojazdu do kompleksu biurowego Olivia Business Centre najkorzystniej prezentuje się dla dzielnic przylegających do dzielnicy Przymorze - ok 10-12 min. 
#
# Czas przejazdu z południowych dzielnic miasta jest porównywalny do czasu dojazdu z wybranych dzielnic Gdyni. 

fig = go.Figure(ff.create_hexbin_mapbox(
    data_frame=df_with_localisation_cleaned, lat="latitude", lon="longitude",
    nx_hexagon=25, opacity=0.8, zoom=9.4,
    labels={"color": "Czas dojazdu do OBC - transport zbiorowy [s]"},
    color="czas_zbiorowy", agg_func=np.mean, color_continuous_scale="Reds"
))
fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=18.555, mapbox_center_lat=54.441)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

# Porównując czas dojazdu samochodem z czasem dojazdu korzystając z transportu zbiorowego można zauważyć problem dostępności południowych dzielnic Gdańska oraz Wyspy Sobieszewskiej.

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

# Najdroższą nieruchomością (cena za metr) jest nieruchomosć przy ulicy Orłowskiej w Gdyni, natomiast największą powierzchnię, jednocześnie z bardzo wysoką ceną za metr, charakteryzuje się nieruchomość w Śródmieściu Gdyni, w pobliżu Skweru Kościuszki. 

# Wyświetlenie podglądu ramki danych
df_with_localisation_cleaned.head()

# ### 5. Tworzenie i transformacja zmiennych

# +
# Wskazanie kolumn, które nie będą wykorzystane podczas modelowania
columns_not_for_modelling = ["id", "id_offers", 
                             "ulica", "tytul", 
                             "latitude", "longitude", 
                             "cena", "numer_oferty", 
                             "garaz_miejsce"]

df_for_modelling = df_with_localisation_cleaned.drop(columns = columns_not_for_modelling, axis=1).copy()
# -

# Podgląd ramki danych
df_for_modelling.head()

# Wyświetlenie podstawowych informacji o zbiorze
df_for_modelling.info()

# Z uwagi na dużą liczbę brakujących danych przy jednocześnie małej wariancji wartości zmiennych **zabudowa**, **ogrzewanie** oraz **forma_wlasnosci** zecydowano się je usunąć ze zbioru. 
#
# Dla zmiennych **rok_budowy**, **stan_wykonczenia** oraz **czas_zbiorowy** zecydowano się usunąć rekordy zawierające brakujące dane

# +
# Odrzucenie zmiennych z dużą ilością brakujących wartości
df_for_modelling = df_for_modelling.drop(columns=["zabudowa", "ogrzewanie", "forma_wlasnosci"], axis=1)

# Wybór indeksów dla obserwacji, które posiaają braki danych dla wybranych zmiennych
missing_data_index = ~df_for_modelling["rok_budowy"].isna() & ~df_for_modelling["stan_wykonczenia"].isna() & \
~df_for_modelling["czas_zbiorowy"].isna()

# Usunięcie rekordów z pustymi wartościami 
df_for_modelling = df_for_modelling[missing_data_index]
# -

# Wyświetlenie podstawowych informacji o zbiorze
df_for_modelling.info()


# +
# Tworzenie zmiennej wiek budynku - różnica między bieżącym rokiem a rokiem budowy, 
# dla budynków w budowie oraz budynków tegorocznych wartość wynosi 0

def calculate_building_age(built_year):
    year = datetime.datetime.now().year
    if built_year > year:
        built_year = year
    return year - built_year

df_for_modelling["wiek_budynku"] = df_for_modelling["rok_budowy"].apply(calculate_building_age)

# Tworzennie zmiennej binarnej "w_budowie" dla daty budowy 2021 i później
df_for_modelling["w_budowie"] = df_for_modelling["rok_budowy"].apply(lambda x: 1 if (x > datetime.datetime.now().year) else 0)

# Usunięcie zmiennej źródłowej - rok_budowy
df_for_modelling = df_for_modelling.drop(columns="rok_budowy", axis=1)

# +
# Tworzenie nowej zmiennej - dzielnica nadmorska na bazie nazw dzielnic sąsiadujących z pasem nadmorskim

districts_at_see = ['Brzeźno', 'Jelitkowo', 'Krakowiec-Górki Zachodnie', 'Przymorze', 'Stogi', 'Wyspa Sobieszewska', 
                    'Babie Doły', 'Orłowo', 'Redłowo', 'Wzgórze Św. Maksymiliana', 'Śródmieście', 'Kamienna Góra', 'Oksywie',
                    'Karlikowo', 'Kamienny Pototok', 'Centrum', 'Dolny']

df_for_modelling["nad_morzem"] = df_for_modelling["dzielnica"].apply(lambda x: 1 if x in districts_at_see else 0)
df_for_modelling = df_for_modelling.drop(columns=["dzielnica"], axis=1)

# +
# Enkodowanie zmiennej miasto oraz rynek z użyciem One Hot Encoder oraz usunięcie zmiennej oryginalnej
df_for_modelling = df_for_modelling.join(pd.get_dummies(df_for_modelling["miasto"]))

df_for_modelling = df_for_modelling.join(pd.get_dummies(df_for_modelling["rynek"], prefix="rynek"))
df_for_modelling = df_for_modelling.drop(columns="rynek_wtorny", axis=1)

df_for_modelling = df_for_modelling.join(pd.get_dummies(df_for_modelling["stan_wykonczenia"]))

df_for_modelling = df_for_modelling.drop(columns=["rynek", "miasto", "stan_wykonczenia"], axis=1)

# +
# Zamiana wartości pietro na zmienną liczbową
dict_pietro = {"parter": 0, "suterena": 0, ">_10": 15}

df_for_modelling["pietro"] = df_for_modelling["pietro"].map(dict_pietro).fillna(df_for_modelling["pietro"])

index_poddasze = df_for_modelling["pietro"] == "poddasze"

df_for_modelling.loc[index_poddasze, "pietro"] = df_for_modelling.loc[
    index_poddasze, "liczba_pieter"].apply(lambda liczba_pieter: liczba_pieter if liczba_pieter <= 10 else 15)

df_for_modelling["pietro"] = pd.to_numeric(df_for_modelling["pietro"]).astype("int8")
# -

# Wyświetlenie wariancji dla zmiennych
df_for_modelling.var()

# Wyświetlenie podglądu ramki danych
df_for_modelling.head()

# # 6. Modelowanie

# Podział na zbior zmiennych X oraz zbiór zmiennej celu y - cena_metr
X = df_for_modelling.drop(columns="cena_metr", axis=1).copy()
y = df_for_modelling["cena_metr"]

X.head()

y.head()

# Podział na zbiór treningowy i testowy (niestosowany do nauki modelu, a do jego testowania) stanowiący 15 % obserwacji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=12345)

print(f"Rozmiar zbioru treningowego: {X_train.shape}")
print(f"Rozmiar zbioru testowego: {X_test.shape}")

# Siatka parametrów dla estymatora LightGBM użyta podczas poszukiwania hyperparametrów 
random_search_grid = {"learning_rate": uniform(0.0001, 0.1),
                      "n_estimators": randint(150, 350),
                     "min_data_in_leaf": randint(10, 50),
                     "max_bin": randint(40, 200),
                      "bagging_fraction": uniform(0.2, 0.5),
                      "bagging_freq": randint(4, 10),
                      "boosting_type": ["dart", "gbdt"]
                     }

# Przygotowanie obiektów estymatora oraz metody do szukania hyperparametrów
lgbm_model = lgbm.LGBMRegressor(n_jobs=-1, n_iter=200)
random_search = RandomizedSearchCV(estimator=lgbm_model, 
                                   param_distributions=random_search_grid, 
                                   n_iter=50, 
                                   cv=4, 
                                   verbose=50, 
                                   scoring="neg_root_mean_squared_error")

# Uruchomienie procedury wyszukiwania hyperparametrów
random_search.fit(X_train, y_train)

# Wyświetlenie rezultatów dla poszczególnych prób hyperparametrów
pd.DataFrame(random_search.cv_results_).sort_values("rank_test_score")

# Przypisanie najlepszego, wytrenowanego modelu do zmiennej
best_model = random_search.best_estimator_

# Obliczenie metryk MAPE i RMSE korzystając z funkcji calculate_metrics
calculate_metrics(best_model, X_train, y_train, X_test, y_test)

# Wykres reszt
residuals_plot(best_model, X_train, y_train, X_test, y_test)

# Wykres błędów predykcji
prediction_error(best_model, X_train, y_train, X_test, y_test)

# Próba usunięcia wartości odstających dla zmiennej celu
df_for_modelling_outliers_index = df_for_modelling.copy()
df_for_modelling_outliers_index["zscore"] = zscore(df_for_modelling_outliers_index["cena_metr"])

# Wskazanie obserwacji, których moduł wartości zscore przekracza 3 - wartości odstające
outliers_indices = df_for_modelling_outliers_index["zscore"].abs() >= 3

# +
display(df_for_modelling_outliers_index[outliers_indices])

print(f"Liczba obserwacji odstających: {outliers_indices.sum()}")
# -

# Pominięcie obserwacji odstających i ponowna faza modelowania
df_for_modelling_outliers_index = df_for_modelling_outliers_index[~outliers_indices].drop(columns="zscore", axis=1)

# +
# Podział na zbior zmiennych X oraz zbiór zmiennej celu y
X_without_outliers = df_for_modelling_outliers_index.drop(columns="cena_metr", axis=1).copy()
y_without_outliers = df_for_modelling_outliers_index["cena_metr"]

# Podział na nowy zbiór uczący i testowy
X_train_wo, X_test_wo, y_train_wo, y_test_wo = train_test_split(X_without_outliers, 
                                                                y_without_outliers, 
                                                                test_size=0.15, 
                                                                random_state=12345)

print(f"Rozmiar zbioru treningowego: {X_train_wo.shape}")
print(f"Rozmiar zbioru testowego: {X_test_wo.shape}")

lgbm_model = lgbm.LGBMRegressor(n_jobs=-1, n_iter=200)
random_search_without_outliers = RandomizedSearchCV(estimator=lgbm_model, 
                                                    param_distributions=random_search_grid, 
                                                    n_iter=50, 
                                                    cv=4, 
                                                    verbose=50, 
                                                    scoring="neg_root_mean_squared_error")

random_search_without_outliers.fit(X_train_wo, y_train_wo)
# -

# Wyświetlenie rezultatów dla poszczególnych prób hyperparametrów
pd.DataFrame(random_search_without_outliers.cv_results_).sort_values("rank_test_score")

# Zwrócenie najlepszego modelu do zmiennej
best_model_wo = random_search_without_outliers.best_estimator_

# Obliczenie metryk dla zbioru uczącego i testowego
calculate_metrics(best_model_wo, X_train_wo, y_train_wo, X_test_wo, y_test_wo)

# Wykres reszt
residuals_plot(best_model_wo, X_train_wo, y_train_wo, X_test_wo, y_test_wo)

# Wykres błędów predykcji
prediction_error(best_model_wo, X_train_wo, y_train_wo, X_test_wo, y_test_wo)

# Interpretacja wyników modelu z wykorzystaniem wartości SHAP
shap_values = shap.TreeExplainer(best_model).shap_values(X_test_wo)

shap.summary_plot(shap_values, X_test_wo)

# Usunięcie wartości odstających poprawiło wyniki modelu - podniesienie R2 o blisko 2 punkty procentowe, redukcja błędów predykcji. 
#
# Z wykresu reszt oraz błędów predykcji można zauważyć, że model słabiej radzi sobie z ofertami o wysokiej cenie za metr - powyżej 15 tysięcy złotych. 
#
# Z wykresu wartości SHAP można zauważyć m.in:
# - im starszy budynek tym bardziej negatywny wpływ na cenę jednostkową
# - dłuższy dojazd transportem zbiorowym wpływa negatywnie na cenę jednostkową
# - położenie w dzielnicach nadmorskich pozytywnie wpływa na cenę
# - nieruchomości do remontu mają niższą cenę za metr
# - nieruchomości, które nie są w Gdańsku mają wyższą cenę za metr
