import pandas as pd
import numpy as np
from selenium import webdriver
from bs4 import BeautifulSoup
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager
import time
from datetime import datetime
import random
import yaml
from pathlib import Path, PurePath
from unidecode import unidecode
import re
from sqlalchemy import create_engine
import cx_Oracle
from sqlalchemy.types import Numeric

# Zarządzanie ścieżkami
cwd = Path.cwd()
path = PurePath(cwd)
config_path = path.joinpath("config").joinpath("config.yaml")
outputs_path = path.joinpath("outputs")

# Wczytanie pliku konfiguracyjnego
with open(config_path, "r") as ymlfile:
    # Using yaml SafeLoader to load configuration into dict
    config = yaml.load(ymlfile, Loader=yaml.SafeLoader)


def collect_offers_urls(config, base_url):
    """
    Funkcja służy do pobrania adresów url pojedynczych ofert w ramach zdeklarowanego w postaci zmiennej "base_url"
    filtru.

    Parameters
    ----------
    config : dict
        Słownik zawierający dane do stworzenia połączenia z bazą danych Oracle.
    base_url
        Adres bazowy wskazujący wyniki wyszukiwania/filtrowania.
    Returns
    -------
    Zapis adresów pojedynczych ofert do bazy danych - tabela OTODOM_OFFERS, ramka danych (pandas DataFrame) z
    adresami ofert
    """
    # Sciezka do webdrivera uzywanego przez Selenium
    driver = webdriver.Chrome(ChromeDriverManager().install())
    # Przejdź do strony bazowej
    driver.get(base_url)

    # Przygotuj pustą ramkę danych do przechowywania rezultatów
    df_websites = pd.DataFrame(columns=["url", "scraped", "date"])
    time.sleep(5)

    # Znajdź liczbę stron z ofertami
    num_of_pages = int(driver.find_element_by_xpath('//*[@id="pagerForm"]/ul/li[5]/a').text)

    # Załaduj dane do połączenia z bazą ze zmiennej config
    host = config["host"]
    port = config["port"]
    service_name = config["service_name"]
    user = config["user"]
    password = config["password"]

    # Stwórz połączenie z bazą danych Oracle
    dsn_tns = cx_Oracle.makedsn(host, port, service_name=service_name)
    conn = cx_Oracle.connect(user=user, password=password, dsn=dsn_tns)
    c = conn.cursor()

    # Przeiteruj po wszystkich odsłonach strony bazowej i pobierz adresy pojedynczych ofert
    for page in tqdm(range(1, num_of_pages)):
        print(f"Page {page} out of {num_of_pages}")
        driver.get(rf"{base_url}&page={page}")
        # Odczekaj losową liczbę sekund przed przejściem do kolejnej strony - by ograniczyć ryzyko zablokowania
        # adresu IP z uwagi na automatyczne wykorzystanie przeglądarki
        wait = random.randint(3, 7)
        content = driver.page_source
        soup = BeautifulSoup(content, features="html.parser")
        print(f"Waiting for {wait} seconds")
        time.sleep(wait)
        for header in tqdm(soup.find_all(attrs={"class": "offer-item-header"})):
            url = header.h3.a["href"]
            print("Found the URL:", url)
            timestamp = str(datetime.now())
            df_websites = df_websites.append({"url": url, "scraped": 0, "date": timestamp}, ignore_index=True)
            # Zapisz rekord do bazy danych
            c.execute(f"INSERT INTO otodom_offers (url, scraped, scraping_date) VALUES ('{url}', 0, TIMESTAMP '"
                      f"{timestamp}')")
        # Wykonaj kwerendy dla ostatnio zczytanej odsłony
        conn.commit()
    # Zamknij połączenie z bazą
    conn.close()

    # Zapisz ramkę danych do pliku płaskiego CSV
    df_websites.to_csv(outputs_path.joinpath("offers_address.csv"))
    return df_websites


def scrape_one_offer_details(offer_id, offer_url, driver, df_layout):
    """
    Pobierze informacje szczegółowe dla strony oferty

    Parameters
    ----------
    offer_id : int
        Numer id oferty z tabeli OTODOM_OFFERS
    offer_url : url as str
        Adres url strony offerty
    driver : Selenium webdriver object
        Obiekt przeglądarki Selenium
    df_layout : pandas DataFrame
        Pusta ramka danych stanowiąca wzór ramki danych dla pojedynczej oferty
    Returns
    -------
    Ramka danych (pandas DataFrame) zawierająca cechy pojedynczej oferty
    """

    # Przejdź do strony oferty
    print("Getting url")
    driver.get(offer_url)
    # By uniknąć zablokowania IP - odczekaj losową ilość sekund
    wait = random.randint(2, 4)
    print(f"Waiting for {wait} seconds")
    time.sleep(wait)

    # Przeszukaj elementy dla pojedynczego ogłoszenia

    # Lokalizacja
    localisation = driver.find_element_by_xpath('//*[@id="root"]/article/header/div[1]/div/div/div/a').text
    localisation_list = localisation.split(", ")
    # Wyodrębnij miasto, dzielnicę i ulicę
    if len(localisation_list) == 3:
        city, district, street = localisation_list
    elif len(localisation_list) == 2:
        city, district = localisation_list
        street = np.nan
        if district == "pomorskie":
            district = np.nan
    elif len(localisation_list) == 1:
        city = localisation_list[0]
        district, street = np.nan
    elif len(localisation_list) == 0:
        city, district, street = np.nan
    else:
        city = localisation_list[0]
        district = localisation_list[1]
        street = localisation_list[-1]

    # Pobierz cechy opisujące nieruchomość, zdekoduj polskie znaki, pozbądź się jednostek
    details_list = unidecode(
        driver.find_element_by_xpath('//*[@id="root"]/article/div[3]/div[1]/section[1]/div/ul').text). \
        replace(" zl", "").replace(" m2", "").replace(",", ".").split("\n")

    # Mapa nazw kolumn związanych z docelową tabelą bazy danych OTODOM_OFFERS_DETAILS
    columns_map = {"Powierzchnia": "POWIERZCHNIA",
                   "Liczba pokoi": "LICZBA_POKOI",
                   "Rynek": "RYNEK",
                   "Rodzaj zabudowy": "ZABUDOWA",
                   "Pietro": "PIETRO",
                   "Liczba pieter": "LICZBA_PIETER",
                   "Material budynku": "MATERIAL_BUDYNKU",
                   "Okna": "OKNA",
                   "Ogrzewanie": "OGRZEWANIE",
                   "Rok budowy": "ROK_BUDOWY",
                   "Stan wykonczenia": "STAN_WYKONCZENIA",
                   "Czynsz": "CZYNSZ",
                   "Forma wlasnosci": "FORMA_WLASNOSCI",
                   "teren zamknięty": "TEREN_ZAMKNIETY",
                   "balkon": "BALKON",
                   "ogródek": "OGRODEK",
                   "garaż / miejsce parkingowe": "GARAZ_MIEJSCE",
                   "winda": "WINDA",
                   "piwnica": "PIWNICA",
                   "monitoring / ochrona": "MONITORING_OCHRONA"
                   }

    # Zamień listę detali oferty w słownik
    details_dict = {columns_map.get(elem.split(":")[0]): elem.split(": ")[1].replace(" ", "_") for elem in
                    details_list if columns_map.get(elem.split(":")[0])}

    # Zamień zmienne liczbowe
    for detail_key in ["LICZBA_PIETER", "LICZBA_POKOI", "ROK_BUDOWY", "CZYNSZ"]:
        if details_dict.get(detail_key):
            details_dict[detail_key] = int(details_dict[detail_key])

    # Wyodrębnij cenę i cenę jednostkową
    price = int(driver.find_element_by_xpath('//*[@id="root"]/article/header/div[2]/div[1]/div[2]').text.rsplit(" zł")[
                    0].replace(" ", ""))
    unit_price = int(
        driver.find_element_by_xpath('//*[@id="root"]/article/header/div[2]/div[2]/div').text.rsplit(" zł/m²")[
            0].replace(" ", ""))
    # Pobierz informacje dodatkowe o ofercie i zamień je na słownik
    try:
        additional_info_list = driver.find_element_by_xpath(
            '//*[@id="root"]/article/div[3]/div[1]/section[3]/div/ul').text.split("\n")
        additional_info_dict = {columns_map[additional_info]: 1 for additional_info in additional_info_list if
                                columns_map.get(additional_info)}
    except:
        additional_info_list = ["TEREN_ZAMKNIETY", "BALKON", "OGRODEK", "GARAZ_MIEJSCE", "WINDA", "PIWNICA",
                                "MONITORING_OCHRONA"]
        additional_info_dict = {additional_info: 0 for additional_info in additional_info_list}
    # Pobierz tytuł oraz numer oferty
    title = driver.find_element_by_xpath('//*[@id="root"]/article/header/div[1]/div/div/h1').text
    offer_no = int(re.search("\d+", driver.find_element_by_xpath(
        '//*[@id="root"]/article/div[3]/div[1]/div[3]/div/div[1]').text)[0])

    # Stwórz słownik z podstawowymi informacjami o ofercie
    basic_info_dict = {
        "ID_OFFERS": offer_id,
        "MIASTO": city,
        "DZIELNICA": district,
        "ULICA": street,
        "TYTUL": title,
        "CENA": price,
        "CENA_METR": unit_price,
        "NUMER_OFERTY": offer_no
    }

    # Przygoutj kopię layoutu by przechować informacje o ofercie w ramce danych
    offer_df = df_layout.copy()
    # Połącz wszystkie 3 słowniki z cechami oferty
    features_dict = {**basic_info_dict, **details_dict, **additional_info_dict}
    # Zasil ramkę danych
    offer_df = offer_df.append(features_dict, ignore_index=True)
    # Dostosowanie cech zmiennych - dostosowanie do typów zmiennych w tabeli OTODOM_OFFERS_DETAILS
    offer_df.loc[:, "POWIERZCHNIA"] = offer_df.loc[:, "POWIERZCHNIA"].astype("float")
    offer_df.loc[:, ["TEREN_ZAMKNIETY", "BALKON", "OGRODEK", "GARAZ_MIEJSCE", "WINDA", "PIWNICA",
                     "MONITORING_OCHRONA"]] = offer_df.loc[:, ["TEREN_ZAMKNIETY", "BALKON", "OGRODEK",
                                                               "GARAZ_MIEJSCE", "WINDA", "PIWNICA",
                                                               "MONITORING_OCHRONA"]].fillna(0)
    return offer_df


def scrape_offers(config, limit_offers_to_scrape=None):
    """
    Funkcja do pobrania inofrmacji o ofertach, jedna po drugiej, korzystając z funkcji scrape_one_offer_details dla
    każdego z adresu oferty

    Parameters
    ----------
    config : dict
        Słownik zawierający dane do stworzenia połączenia z bazą danych Oracle.
    limit_offers_to_scrape : int, None, optional, default=None
        Liczba ofert, które należy pobrać w ramach jednej sesji, gdy None
    Returns
    -------
    Zapis informacji o ofercie do tabeli OTODOM_OFFERS_DETAILS
    """

    driver = webdriver.Chrome(ChromeDriverManager().install())
    offers_df = pd.DataFrame(columns=["ID_OFFERS",
                                      "MIASTO",
                                      "DZIELNICA",
                                      "ULICA",
                                      "TYTUL",
                                      "POWIERZCHNIA",
                                      "CENA",
                                      "CENA_METR",
                                      "LICZBA_POKOI",
                                      "RYNEK",
                                      "ZABUDOWA",
                                      "PIETRO",
                                      "LICZBA_PIETER",
                                      "MATERIAL_BUDYNKU",
                                      "OKNA",
                                      "OGRZEWANIE",
                                      "ROK_BUDOWY",
                                      "STAN_WYKONCZENIA",
                                      "CZYNSZ",
                                      "FORMA_WLASNOSCI",
                                      "NUMER_OFERTY",
                                      "TEREN_ZAMKNIETY",
                                      "BALKON",
                                      "OGRODEK",
                                      "GARAZ_MIEJSCE",
                                      "WINDA",
                                      "PIWNICA",
                                      "MONITORING_OCHRONA"])

    df_layout = offers_df.copy()

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
    engine.connect()
    # Pobierz adresy ofert, które nie zostały jeszcze pobrane - status 0 w tabeli OTODOM_OFFERS
    if limit_offers_to_scrape:
        select_offers_query = f"SELECT id, url FROM otodom_offers WHERE scraped = 0 AND ROWNUM <=" \
                              f" {limit_offers_to_scrape} " \
                              f"ORDER BY id"
    else:
        select_offers_query = f"SELECT id, url FROM otodom_offers WHERE scraped = 0 ORDER BY id"

    url_df = pd.read_sql_query(select_offers_query, engine)
    num_of_offers = len(url_df)
    # Iteracyjnie pobierz dane dla każdej z ofert
    for no_row in tqdm(range(num_of_offers)):
        offer_id, offer_url = url_df.iloc[no_row]
        print(f"Scraping offer with id {offer_id} using {offer_url} url.")
        try:
            offer_df = scrape_one_offer_details(offer_id, offer_url, driver, df_layout)
        except:
            # Jeśli próba wejścia na stronę się nie powiedzie ustaw status 99 w tabeli OTODOM_OFFERS i przejdź do
            # kolejnej iteracji
            print(f"Skipping {offer_id} offer with {offer_url} url.")
            timestamp = str(datetime.now())
            engine.execute(f"UPDATE otodom_offers SET scraped = 99, scraping_date = TIMESTAMP '{timestamp}' WHERE id = "
                           f"{offer_id}")
            continue

        # Zapisz rekord do bazy danych i uaktualnij status "scraped" do 1 w tabeli OTODOM_OFFERS
        offer_df.to_sql("otodom_offers_details", engine, if_exists="append", index=False, dtype={"CZYNSZ": Numeric(),
                                                                                                 "LICZBA_PIETER": Numeric(),
                                                                                                 "LICZBA_POKOI": Numeric(),
                                                                                                 "ROK_BUDOWY": Numeric()})
        print("Saved to db")
        timestamp = str(datetime.now())
        engine.execute(f"UPDATE otodom_offers SET scraped = 1, scraping_date = TIMESTAMP '{timestamp}' WHERE id = "
                       f"{offer_id}")
        offers_df = offers_df.append(offer_df, ignore_index=True)

    return offers_df


base_url = r"https://www.otodom.pl/sprzedaz/mieszkanie/?locations%5B0%5D%5Bregion_id%5D=11&" \
           r"locations%5B0%5D%5Bsubregion_id%5D=439&locations%5B0%5D%5Bcity_id%5D=40&locations%5B1%5D%5Bregion_id%5D=" \
           r"11&locations%5B1%5D%5Bsubregion_d%5D=278&locations%5B1%5D%5Bcity_id%5D=206&locations%5B2%5D%5Bregion_id%" \
           r"5D=11&locations%5B2%5D%5Bsubregion_id%5D=280&locations%5B2%5D%5Bcity_id%5D=208"


# Użyj by pobrać listę adresów url dla stron ofert spośród wyfiltrowanych kryteriów w ramach bazowego linka (base_url)
df_websites = collect_offers_urls(config, base_url)

# Użyj by pobrać dane szczegółówe dla wybranej liczby ofert
offers_df = scrape_offers(config, limit_offers_to_scrape=None)
