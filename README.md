## Projekt analizy ofert nieruchomości z serwisu OTODOM dla Gdańska, Gdyni i Sopotu

Projekt zaliczeniowy w ramach przedmiotu Bazy Danych i Hurtownie Danych w ramach studiów podyplomowych Inżynieria
 Danych - Data Science
 
### Zawartość repozytorium
- skrypt do pobrania danych z portalu otodom.pl
- dedykowane funkcje wykorzystane podczas analizy
- skrypty SQL wykorzystane do stworzenia bazy danych
- pliki konfiguracyjne
- raport w formacie HTML [Pobierz archiwum zip z raportem](https://github.com/Cookiee-monster/estate_analysis_project/raw/master/notebooks/analiza_raport.zip)
- opis pobrania danych [opis.md](https://github.com/Cookiee-monster/estate_analysis_project/blob/master/Opis.md)
- plik analiza.py możliwy do otwarcia m.in w Jupyter Notebook


### Stworzenie środowiska

1. Instalacja środowiska Anaconda
2. Konfiguracja środowiska bazowego 
    ```
    pip install "plotly==4.5.4" "ipywidgets>=7.2"
    jupyter nbextension enable --py widgetsnbextension
    jupyter nbextension enable --py plotlywidget
    ```
3. Skolonowanie repozytorium do wybranej lokalizacji korzystając z:
    - korzystając z HTTPS `git clone https://github.com/Cookiee-monster/estate_analysis_project.git`
    - korzystając z SSH `git clone git@github.com:Cookiee-monster/estate_analysis_project.git`
4. Stworzenie środowiska korzystając z konsoli (znajdując się w folderze sklonowanego repozytorium
 estate_analysis_project) anaconda_prompt, gitbash lub innej wybranej korzystjąc z polecenia:
`conda create --name estate python=3.8`
5. Po stworzeniu środowiska należy je aktywować korzystając z terminala: `conda activate estate`
6. Instalacja pakietów korzystając z pliku requirements.txt: `pip install -r requirements.txt'
7. Rejestracja kernella w Jupyter Notebook:
`python -m ipykernel install --user --name estate`
8. Pobranie **instantclient_19_8** ze stron Oracle oraz ustawienie ścieżki do pliku exe do zmiennej systemowej **PATH**

