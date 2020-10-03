### Założenia

##### 1. Zebranie adresów URL odsłon ofert
Pozyskanie zbioru danych o ofertach nieruchomości zostało wykonane poprzez scraping stron serwisu otodom.pl

Pierwszym etapem było pobranie listy adresów szczegółowych ofert z wszystkich wyfiltrowanych odsłon stron po (oferty z
 Gdańska, Gdyni i Sopotu) spośród wszystkich ofert serwisu:
![image](https://user-images.githubusercontent.com/43417324/94979766-18a00080-0525-11eb-8c26-111144463a12.png)

Poszczególne adresy URL szczegółowych ofert zostały pobrane i zapisane w bazie danych w tabeli OTODOM_OFFERS wraz z dat
ą scrapowania oraz statusem 0 (oferta nie została jeszcze poddana scrapingowi)

![image](https://user-images.githubusercontent.com/43417324/94979960-49346a00-0526-11eb-8b72-eeb50f7976bd.png)

##### 2. Pobranie informacji z ofert

Proces pobierania informacji o danej ofercie poprzedzony jest pobranie adresu URL z tabeli OTODOM_OFFERS. 

![image](https://user-images.githubusercontent.com/43417324/94980123-2eaec080-0527-11eb-9534-8b8be32c0e66.png)
![image](https://user-images.githubusercontent.com/43417324/94980220-e348e200-0527-11eb-9cdd-68f77150866d.png)
1. Tytuł oferty
2. Lokalizacja
3. Cena
4. Cena za metr
5. Szczegóły nieruchomości
6. Informacje dodatkowe

W ramach informacji dodatkowych brano pod uwagę tylko informacje o:
`"TEREN_ZAMKNIETY", "BALKON", "OGRODEK", "GARAZ_MIEJSCE", "WINDA", "PIWNICA", "MONITORING_OCHRONA"`
Gdy informacja się pojawiała przypisywano jej wartość 1, przy braku wartość 0. Uznano, że głównym celem oferty jest
 jak najlepsze zareklamowanie nieruchomości więc w gestii osoby sprzedającej jest podanie wszelkich atrybutów/cech
  nieruchomości mogącej wyróżnić daną ofertę - stąd wcześniej opisane założenie. 
  
 Dane, po przetworzeniu np. pozbycie się jednostek, rozbicie lokalizacji na miasto, dzielnicę i miasto itp. trafiały
  do tabeli OTODOM_OFFERS_DETAILS
  
  ![image](https://user-images.githubusercontent.com/43417324/94980319-dd9fcc00-0528-11eb-90f7-0cfaa3b79ecf.png)

W przypadku udanego pobrania danych ze strony oferty, data pobrania danych oraz status (1) został zaktualizowany w
 tabeli OTODOM_OFFERS. W przeciwnym przypadku status został zaktualizowany jako 99.
 
##### 3. Wykorzystanie zbioru danych do analizy
Przygotowany zbiór został pobrany do pamięci Jupyter Notebook, korzystając z bibliotek pandas, sqlalchemy oraz
 cx_Oracle, gdzie został poddany analizie.
 
##### 4. Proces analizy 
Całość analizy, przetwarzania danych oraz modelowania zostało przedstawione w raporcie.html oraz pliku analiza.py