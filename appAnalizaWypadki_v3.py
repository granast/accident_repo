# app_static.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# --- Konfiguracja strony Streamlit ---
st.set_page_config(layout="wide", page_title="Analiza Wypadków Drogowych UK (Statyczna)")

# --- Pasek boczny nawigacji ---
st.sidebar.title("Nawigacja")
section = st.sidebar.radio(
    "Wybierz sekcję analizy:",
    (
        "Wprowadzenie",
        "Opis Przygotowania Danych",
        "Analiza Wstępna Kierowców",
        "Analiza Związku: Miejsce Zamieszkania vs Lokalizacja Wypadku",
        "Opis Modelowania ML",
        "Ocena Modeli",
        "Ważność Cech (XGBoost)",
        "Analiza Kluczowych Cech (Chi-kwadrat)",
        "Wnioski i Podsumowanie"
    )
)

# --- Wyświetlanie wybranej sekcji ---

if section == "Wprowadzenie":
    st.title("Analiza związku między miejscem zamieszkania kierowcy a prawdopodobieństwem udziału w wypadku drogowym na terenach wiejskich")

    st.header("I. Temat")
    st.markdown("""
    **"Analiza związku między miejscem zamieszkania kierowcy a prawdopodobieństwem udziału w wypadku drogowym na terenach wiejskich."**
    """)

    st.subheader("1.1 Cel pracy:")
    st.markdown("""
    - Zbadanie, czy istnieje związek między miejscem zamieszkania kierowcy (wiejskim lub miejskim) a prawdopodobieństwem jego udziału w wypadku drogowym na terenie wiejskim oraz identyfikacja kluczowych czynników wpływających na przewidywanie lokalizacji wypadku, z wykorzystaniem modeli uczenia maszynowego.
    """)

    st.subheader("1.2 Pytania badawcze:")
    st.markdown("""
    - Czy miejsce zamieszkania kierowcy (miejskie vs. niemiejskie) wpływa na prawdopodobieństwo udziału w wypadku drogowym na terenie wiejskim?
    - Jakie z wybranych cech kontekstowych (np. typ drogi, warunki oświetleniowe, kontrola skrzyżowań) mają największy wpływ na prawdopodobieństwo wystąpienia wypadku na terenie wiejskim?
    - Czy modele uczenia maszynowego (XGBoost, RandomForest) mogą skutecznie przewidzieć lokalizację wypadku na podstawie miejsca zamieszkania kierowcy i cech kontekstowych?
    """)

    st.subheader("1.3 Hipoteza badawcza:")
    st.markdown("""
    - Kierowcy z obszarów miejskich są bardziej narażeni na udział w wypadkach drogowych na terenach wiejskich niż kierowcy z obszarów wiejskich.
    - Specyficzne cechy, takie jak drogi jednopasmowe, brak oświetlenia ulicznego oraz niekontrolowane skrzyżowania, znacząco zwiększają ryzyko wypadku na terenie wiejskim.
    - Modele uczenia maszynowego (XGBoost, RandomForest) nie osiągają wysokiej skuteczności w przewidywaniu lokalizacji wypadku (wiejskiej vs. miejskiej) na podstawie miejsca zamieszkania kierowcy i cech kontekstowych.
    """)

elif section == "Opis Przygotowania Danych":
    st.title("II. Dane i Metodyka - Opis Przygotowania Danych")

    st.header("1. Źródła danych")
    st.markdown("""
    - Dane pochodzą z oficjalnych brytyjskich baz danych (Department for Transport - data.gov.uk) dotyczących wypadków drogowych z lat 2021-2023 na terenie UK.
    - Tabele (`casualties`, `vehicles`, `accidents`) zawierające dane m.in. o ofiarach (wiek, miejsce zamieszkania), informacje o pojazdach i kierowcach (np. obszar zamieszkania, odległość od miejsca wypadku) oraz kontekst wypadków (warunki pogodowe, typ drogi) zostały połączone w tabelę `data` po kluczu `accident_index`.
    - Statystyki dotyczą wyłącznie wypadków z obrażeniami ciała na drogach publicznych, które są zgłaszane policji, a następnie rejestrowane przy użyciu formularza zgłaszania kolizji `STATS19`.
    - **Przewodnik** po statystykach dotyczących wypadków drogowych: [link](https://www.gov.uk/guidance/road-accident-and-safety-statistics-guidance)
    - **Zestawy danych** do pobrania: [link](https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-accidents-safety-data)
    """)

    st.header("2. Opis użytych zmiennych")
    st.markdown("""
    W analizie wykorzystano następujące zmienne, które opisują okoliczności wypadków drogowych, charakterystyki kierowców, pojazdów oraz poszkodowanych:
    - `road_type` – Rodzaj drogi, na której doszło do wypadku. Kategorie obejmują: rondo (1), ulica jednokierunkowa (2), droga dwujezdniowa (3), droga jednojezdniowa (6), droga dojazdowa (7), nieznana (9), ulica jednokierunkowa/droga dojazdowa (12) lub brak danych (-1).
    - `light_conditions` – Warunki oświetlenia w czasie wypadku. Kategorie: światło dzienne (1), ciemność z działającym oświetleniem (4), ciemność z niedziałającym oświetleniem (5), ciemność bez oświetlenia (6), ciemność z nieznanym stanem oświetlenia (7) lub brak danych (-1).
    - `junction_detail` – Szczegóły dotyczące skrzyżowania w miejscu wypadku. Obejmuje: brak skrzyżowania w promieniu 20 metrów (0), rondo (1), mini-rondo (2), skrzyżowanie typu T lub rozwidlenie (3), droga dojazdowa (5), skrzyżowanie czteroramienne (6), skrzyżowanie z więcej niż 4 ramionami (7), prywatny wjazd (8), inne skrzyżowanie (9), nieznane (99) lub brak danych (-1).
    - `junction_control` – Rodzaj kontroli ruchu na skrzyżowaniu. Kategorie: brak skrzyżowania w promieniu 20 metrów (0), osoba upoważniona (1), sygnalizacja świetlna (2), znak stopu (3), ustąp pierwszeństwa lub brak kontroli (4), nieznane (9) lub brak danych (-1).
    - `driver_home_area_type` – Typ obszaru zamieszkania kierowcy. Obejmuje: obszar miejski (1), małe miasto (2), obszar wiejski (3) lub brak danych (-1).
    - `accident_year` – Rok, w którym doszło do wypadku.
    - `age_of_casualty` – Wiek osoby poszkodowanej w wypadku. Wartość -1 oznacza brak danych.
    - `driver_distance_banding` – Odległość miejsca wypadku od miejsca zamieszkania kierowcy. Kategorie: do 5 km (1), 5,001–10 km (2), 10,001–20 km (3), 20,001–100 km (4), powyżej 100 km (5) lub brak danych (-1).
    - `weather_conditions` – Warunki pogodowe w czasie wypadku. Kategorie: dobra pogoda bez silnego wiatru (1), deszcz bez silnego wiatru (2), śnieg bez silnego wiatru (3), dobra pogoda z silnym wiatrem (4), deszcz z silnym wiatrem (5), śnieg z silnym wiatrem (6), mgła (7), inne (8), nieznane (9) lub brak danych (-1).
    - `urban_or_rural_area` – Typ obszaru, w którym doszło do wypadku: miejski (1), wiejski (2), nieprzypisany (3) lub brak danych (-1).
    - `casualty_type` – Typ poszkodowanego w wypadku, np.: pieszy (0), rowerzysta (1), motocyklista (2–5, 23, 97, 103–106), pasażer
    taksówki (8), pasażer samochodu (9), pasażer busa (10–11), jeździec konny (16), inne typy pojazdów (17–21, 90, 98–99, 108–110, 113) lub brak danych (-1).
    - `speed_limit` – Ograniczenie prędkości na drodze w miejscu wypadku. Wartości w milach na godzinę, np. 30, 60; 99 oznacza nieznane (zgłoszone przez uczestnika), a -1 brak danych.
    - `driver_imd_decile` – Poziom deprywacji społeczno-ekonomicznej kierowcy według indeksu IMD (ang. Index of Multiple Deprivation). Skala od 1 (najbardziej deprywowany 10%) do 10 (najmniej deprywowany 10%) lub brak danych (-1).
    - `age_of_vehicle` – Wiek pojazdu w latach w momencie wypadku. Wartość -1 oznacza brak danych.
    - `age_of_driver` – Wiek kierowcy w momencie wypadku. Wartość -1 oznacza brak danych.
    - `number_of_casualties` – Liczba osób poszkodowanych w wyniku wypadku.
    - `skidding_and_overturning` – Informacja o poślizgu lub przewróceniu pojazdu. Kategorie: brak (0), poślizg (1), poślizg i przewrócenie (2), wyłamanie (3), wyłamanie i przewrócenie (4), przewrócenie (5), nieznane (9) lub brak danych (-1).
    """)

    st.header("3. Opis kroków przygotowania danych")
    st.markdown("""
    W oryginalnej analizie przeprowadzono następujące kroki przygotowania danych (nie są one wykonywane w tej statycznej wersji):

    - **Oczyszczenie danych:** Zastąpiono wartości `-1` i `99` na `NaN` w kluczowych kolumnach, a następnie usunięto wiersze z brakami w tych kolumnach.
    - **Przekształcenie czasu:** Z kolumny `time` wyodrębniono godzinę i utworzono nową kolumnę `hour_of_day`.
    - **Przygotowanie zmiennych kategorycznych:**
      - Dla `driver_home_area_type` zsumowano wartości 2 (small town) i 3 (rural) w jedną etykietę o wartości 2, aby uprościć dane.
      - Stworzono zmienną binarną `is_urban_driver` (1 = kierowca z obszaru miejskiego, gdy `driver_home_area_type` = 1; 0 w przeciwnym razie).
      - Stworzono zmienną docelową `is_rural_accident` (1 = wypadek na terenie wiejskim, gdy `urban_or_rural_area` = 2; 0 w przeciwnym razie).
    - **Normalizacja:** Kolumnę `speed_limit` znormalizowano za pomocą `StandardScaler`, tworząc `speed_limit_normalized`.
    - **Binowanie wieku:** Kolumny `age_of_casualty` i `age_of_driver` podzielono na 5 przedziałów wiekowych (≤17 lat, 18-25 lat, 26-40 lat, 41-60 lat, >60 lat), tworząc odpowiednio `age_of_casualty_binned` i `age_of_driver_binned`.
    - **Inżynieria cech:**
      - `urban_driver_speed` jako iloczyn `is_urban_driver` i `speed_limit_normalized`.
      - `is_rush_hour` na podstawie `hour_of_day` (1, jeśli godzina należy do godzin szczytu: 7:00-9:00 lub 15:00-18:00; 0 w przeciwnym razie).
      - `distance_speed_interaction` jako iloczyn `driver_distance_banding` i `urban_driver_speed`.
    - **Wybór cech:** Ustalono listę `selected_features`, obejmującą: `is_urban_driver`, `road_type`, `light_conditions`, `junction_detail`, `junction_control`, `driver_distance_banding`, `weather_conditions`, `is_rush_hour`, `age_of_driver_binned`, `age_of_casualty_binned`, `distance_speed_interaction`, `speed_limit_normalized`, `driver_imd_decile`, `hour_of_day`, `number_of_casualties`, `urban_driver_speed`, `skidding_and_overturning`, `casualty_type`.
    - **Kodowanie kategoryczne:** Zmienne kategoryczne z `selected_features` (`road_type`, `light_conditions`, `junction_detail`, `junction_control`, `age_of_casualty_binned`, `driver_distance_banding`, `is_rush_hour`, `weather_conditions`, `age_of_driver_binned`, `skidding_and_overturning`, `casualty_type`) zakodowano metodą zero-jedynkową (one-hot encoding) z użyciem `pd.get_dummies`.
    - **Dodatkowe cechy po kodowaniu:**
      - `important_driver_distance` jako zmienna binarna (1, jeśli `driver_distance_banding_4.0` lub `driver_distance_banding_3.0` > 0, czyli dystans > 20 km; 0 w przeciwnym razie).
      - `urban_driver_long_distance` jako iloczyn `is_urban_driver` i `important_driver_distance`.
      - `urban_driver_no_junction_control` jako iloczyn `is_urban_driver` i `junction_control_4.0` (jeśli taka kolumna istnieje po kodowaniu, co oznacza brak kontroli ruchu na skrzyżowaniu).
    - **Podział danych:** Dane podzielono na zbiory:
      - Treningowy + walidacyjny (80%) i testowy (20%) z zachowaniem stratyfikacji.
      - Następnie zbiór treningowy + walidacyjny podzielono na treningowy (60% całości) i walidacyjny (20% całości), również ze stratyfikacją.
    - **Balansowanie danych:** Zastosowano SMOTE na zbiorze treningowym, aby zrównoważyć klasy zmiennej docelowej `is_rural_accident`.
    - **Rozmiary zbiorów danych po przetworzeniu:** 
      - Zbiór treningowy (po SMOTE): 228388 rekordów / Zbiór walidacyjny: 54611 rekordów / Zbiór testowy: 54611 rekordów.

    Celem było przygotowanie danych (X) i zmiennej docelowej (y, czyli `is_rural_accident`) do modelowania poprzez oczyszczenie, transformację i stworzenie nowych cech, uwzględniając również typ uczestnika wypadku (`casualty_type`).

    *Ta wersja aplikacji jedynie **prezentuje** wyniki uzyskane po tych krokach.*
    """)

elif section == "Analiza Wstępna Kierowców":
    st.title("Analiza Wstępna: Charakterystyka Kierowców w Wypadkach (Wyniki Statyczne)")

    # --- Dane statyczne ---
    total_accidents_static = 273053

    # Tabela 1: Proporcje kierowców
    driver_origin_data = {
        'Pochodzenie': ['Miejski', 'Niemiejski', 'Suma'],
        'Liczba': [222719, 50334, 273053],
        'Procent': [81.6, 18.4, 100.0]
    }
    driver_origin_display = pd.DataFrame(driver_origin_data)

    # Tabela 2: Rozkład wg lat
    driver_stats_data = {
        'Rok': [2021, 2022, 2023],
        'Niemiejski': [15908, 17419, 17007],
        'Procent Niemiejski': [17.8, 18.7, 18.9],
        'Miejski': [73686, 75877, 73156],
        'Procent Miejski': [82.2, 81.3, 81.1],
        'Suma': [89594, 93296, 90163]
    }
    driver_stats_display = pd.DataFrame(driver_stats_data)

    # --- Wyświetlanie w Streamlit ---
    st.subheader("Tabela 1: Proporcje kierowców według miejsca zamieszkania")
    st.dataframe(driver_origin_display.style.format({'Liczba': '{:,.0f}', 'Procent': '{:.1f}%'}))
    st.markdown("""
    **Komentarz:** Kierowcy z obszarów miejskich dominują w ogólnej liczbie wypadków (81.6%), co może odzwierciedlać większą populację miejską lub częstsze korzystanie z dróg.
    """)

    st.subheader("Tabela 2: Rozkład kierowców według miejsca zamieszkania w latach 2021-2023")
    st.dataframe(driver_stats_display.style.format({
        'Niemiejski': '{:,.0f}', 'Procent Niemiejski': '{:.1f}%',
        'Miejski': '{:,.0f}', 'Procent Miejski': '{:.1f}%',
        'Suma': '{:,.0f}'
    }))
    st.markdown("""
    **Komentarz:** Proporcje pozostają stosunkowo stałe w latach 2021-2023, z lekkim wzrostem udziału kierowców niemiejskich w 2023 roku (18.9%), co sugeruje stabilność trendów w czasie.
    """)

    st.subheader("Wizualizacje (Odtworzone)")

    # --- Odtworzenie wykresów Matplotlib na podstawie danych statycznych ---
    fig_mpl = plt.figure(figsize=(12, 10))
    gs = fig_mpl.add_gridspec(2, 2, height_ratios=[1, 1.2])

    # Wykres 1: Całkowita liczba wypadków
    ax1 = fig_mpl.add_subplot(gs[0, 0])
    bars1 = ax1.bar(['Wszystkie wypadki'], [total_accidents_static], color='#93c47d')
    ax1.set_title('Całkowita liczba analizowanych wypadków')
    ax1.set_ylabel('Liczba')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height/2, f'{int(height):,} (100%)', ha='center', va='center', fontsize=10, color='black')

    # Wykres 2: Proporcje kierowców
    ax2 = fig_mpl.add_subplot(gs[0, 1])
    driver_origin_plot = driver_origin_display[driver_origin_display['Pochodzenie'] != 'Suma'].set_index('Pochodzenie')
    bottom_val = 0
    colors = {'Niemiejski': '#1f77b4', 'Miejski': '#ff7f0e'}
    order = ['Niemiejski', 'Miejski']
    for origin_type in order:
        if origin_type in driver_origin_plot.index:
            value = driver_origin_plot.loc[origin_type, 'Liczba']
            percentage = driver_origin_plot.loc[origin_type, 'Procent']
            bar = ax2.bar(['Kierowcy'], [value], bottom=[bottom_val], color=colors[origin_type], label=origin_type)
            text_y = bottom_val + value / 2
            ax2.text(0, text_y, f"{int(value):,}\n({percentage:.1f}%)", ha='center', va='center', fontsize=10, color='white')
            bottom_val += value

    ax2.set_title('Proporcje kierowców wg miejsca zamieszkania')
    ax2.set_ylabel('Liczba kierowców')
    ax2.set_xticks([])
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.set_ylim(0, total_accidents_static * 1.1)

    # Wykres 3: Rozkład kierowców według lat
    ax3 = fig_mpl.add_subplot(gs[1, :])
    bar_width = 0.35
    x = np.arange(len(driver_stats_display['Rok']))
    rects1 = ax3.bar(x - bar_width/2, driver_stats_display['Niemiejski'], bar_width, label='Niemiejski', color='#1f77b4')
    rects2 = ax3.bar(x + bar_width/2, driver_stats_display['Miejski'], bar_width, label='Miejski', color='#ff7f0e')

    ax3.set_title('Rozkład kierowców wg miejsca zamieszkania w latach')
    ax3.set_xlabel('Rok')
    ax3.set_ylabel('Liczba kierowców')
    ax3.set_xticks(x)
    ax3.set_xticklabels(driver_stats_display['Rok'])
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{int(height):,}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    autolabel(rects1, ax3)
    autolabel(rects2, ax3)

    fig_mpl.suptitle('Analiza kierowców w wypadkach drogowych (2021-2023)', fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    st.pyplot(fig_mpl)

elif section == "Analiza Związku: Miejsce Zamieszkania vs Lokalizacja Wypadku":
    st.title("Analiza Związku: Miejsce Zamieszkania Kierowcy a Lokalizacja Wypadku (Wyniki Statyczne)")

    # --- Dane statyczne ---
    contingency_data = {
        'Wypadek Miejski': [15893, 174431],
        'Wypadek Wiejski': [34441, 48288]
    }
    contingency_table = pd.DataFrame(contingency_data, index=['Niemiejski', 'Miejski'])

    location_stats_data = {
        'Wypadki Miejskie (%)': [31.6, 78.3],
        'Wypadki Wiejskie (%)': [68.4, 21.7]
    }
    location_stats = pd.DataFrame(location_stats_data, index=['Niemiejski', 'Miejski'])

    chi2_stat = 42475.60
    p_value_chi2 = 0.0
    dof_chi2 = 1
    phi_stat = 0.394
    strength = "Umiarkowany (φ = 0.3–0.5)"
    conclusion = f"Odrzucamy hipotezę zerową (H₀). Istnieje statystycznie istotny związek (p < 0.0001)."
    alpha = 0.05

    expected_data = {
        'Wypadek Miejski': [35083.9, 155240.1],
        'Wypadek Wiejski': [15250.1, 67478.9]
    }
    expected_df = pd.DataFrame(expected_data, index=['Niemiejski', 'Miejski'])

    # --- Wyświetlanie w Streamlit ---
    st.subheader("Tabela Kontyngencji (Obserwowane Liczby)")
    st.dataframe(contingency_table.style.format("{:,.0f}"))

    st.subheader("Tabela Procentowa Lokalizacji Wypadków wg Pochodzenia Kierowcy")
    st.dataframe(location_stats.style.format("{:.1f}%"))

    st.subheader("Wyniki Testu Chi-kwadrat Niezależności")
    st.markdown(f"""
    - **Statystyka chi-kwadrat (χ²):** {chi2_stat:.2f}
    - **Wartość p (p-value):** {p_value_chi2:.4e} (bardzo bliska 0)
    - **Stopnie swobody (dof):** {dof_chi2}
    - **Współczynnik Phi (φ):** {phi_stat:.3f}
    - **Interpretacja siły związku (Phi):** {strength}
    - **Wniosek (poziom istotności α = {alpha}):** {conclusion}
    """)

    st.subheader("Tabela Oczekiwana (Gdyby nie było związku)")
    st.dataframe(expected_df.style.format("{:,.1f}"))

    # --- Odtworzenie Wykresu Plotly ---
    st.subheader("Wykres: Procent Wypadków Miejskich i Wiejskich wg Pochodzenia Kierowcy")
    location_stats_plot = location_stats.reset_index().rename(columns={'index': 'Pochodzenie Kierowcy'})
    location_stats_melted = location_stats_plot.melt(
        id_vars='Pochodzenie Kierowcy',
        var_name='Typ Obszaru Wypadku',
        value_name='Procent Wypadków'
    )
    location_stats_melted['Typ Obszaru Wypadku'] = location_stats_melted['Typ Obszaru Wypadku'].str.replace(' (%)', '')

    fig_plotly = px.bar(location_stats_melted,
                         x='Pochodzenie Kierowcy',
                         y='Procent Wypadków',
                         color='Typ Obszaru Wypadku',
                         title='Procent Wypadków Miejskich i Wiejskich<br>wg Miejsca Zamieszkania Kierowcy',
                         labels={'Procent Wypadków': 'Procent Wypadków (%)', 'Typ Obszaru Wypadku': 'Lokalizacja Wypadku'},
                         color_discrete_map={'Wypadki Miejskie': '#1f77b4', 'Wypadki Wiejskie': '#ff7f0e'},
                         barmode='group',
                         text='Procent Wypadków'
                         )
    fig_plotly.update_layout(yaxis_ticksuffix='%', yaxis_title='Procent Wypadków (%)', xaxis_title='Pochodzenie Kierowcy')
    fig_plotly.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig_plotly, use_container_width=True)

    # --- Podsumowanie i wnioski z tej sekcji ---
    st.subheader("Interpretacja i Wnioski z Analizy Związku")
    st.markdown("""
    **Kluczowe obserwacje:**
    - **Kierowcy z obszarów niemiejskich**: Znacznie częściej uczestniczą w wypadkach na terenach wiejskich (68.4%) niż miejskich (31.6%).
    - **Kierowcy z miast**: Dominują w wypadkach na terenach miejskich (78.3%), a rzadziej uczestniczą w wypadkach na terenach wiejskich (21.7%).

    **Wyniki testu chi-kwadrat:**
    - Test wykazał **statystycznie istotny związek** (p < 0.0001) między miejscem zamieszkania kierowcy a lokalizacją wypadku.
    - Siła tego związku, mierzona współczynnikiem Phi (φ ≈ 0.394), jest **umiarkowana**. Oznacza to, że miejsce zamieszkania jest ważnym, ale nie jedynym czynnikiem determinującym lokalizację wypadku. Inne zmienne, takie jak warunki drogowe czy prędkość, również odgrywają rolę.

    **Wnioski:**
    1. **Istnienie związku**: Potwierdzono wyraźny związek. Kierowcy częściej ulegają wypadkom w środowisku odpowiadającym ich miejscu zamieszkania (miejscy w miastach, niemiejscy na wsiach). Jednak kierowcy niemiejscy wykazują ponad trzykrotnie wyższe prawdopodobieństwo udziału w wypadkach wiejskich (68.4%) niż miejscy (21.7%).
    2. **Weryfikacja hipotezy**: Wyniki **nie potwierdzają** hipotezy, że kierowcy miejscy są bardziej narażeni na wypadki na terenach wiejskich. Przeciwnie, kierowcy niemiejscy dominują w tej kategorii w swoich grupach.
    3. **Praktyczne znaczenie**: Umiarkowana siła związku sugeruje potrzebę dalszej analizy z wykorzystaniem modeli ML, aby zidentyfikować dodatkowe czynniki wpływające na ryzyko wypadków wiejskich.
    """)

elif section == "Opis Modelowania ML":
    st.title("Modelowanie Uczenia Maszynowego - Opis")
    st.header("Cel: Przewidywanie, czy wypadek zdarzy się na terenie wiejskim (`is_rural_accident` = 1)")

    st.subheader("Wybrane Modele:")
    st.markdown("- **XGBoost Classifier:** Wydajny model gradient boostingowy.")
    st.markdown("- **Random Forest Classifier:** Zespół drzew decyzyjnych.")

    st.subheader("Opis Procesu (z oryginalnej analizy):")
    st.markdown("""
    1. Dane zostały podzielone na zbiory: treningowy (60%), walidacyjny (20%) i testowy (20%).
    2. Zastosowano **SMOTE** na zbiorze treningowym, aby zrównoważyć klasy.
    3. Modele zostały wytrenowane na zbalansowanym zbiorze treningowym z użyciem określonych hiperparametrów (przykładowe poniżej).
    4. Ocena modeli odbyła się na **niezmienionych** (niezbalansowanych) zbiorach walidacyjnym i testowym.

    *Ta statyczna wersja aplikacji nie trenuje modeli, jedynie prezentuje wcześniej uzyskane wyniki.*
    """)

    st.subheader("Przykładowe Hiperparametry Użyte w Analizie:")
    st.code("""
# XGBoost
params_xgb = {
    'random_state': 42, 'scale_pos_weight': 1, 'max_depth': 9,
    'n_estimators': 269, 'learning_rate': 0.06, 'reg_alpha': 0.1,
    'reg_lambda': 1.9, 'subsample': 0.8, 'colsample_bytree': 0.6,
}

# RandomForest
params_rf = {
    'random_state': 42, 'n_estimators': 229, 'max_depth': 14,
    'min_samples_split': 54, 'min_samples_leaf': 26, 'n_jobs': -1,
    'max_features': 'sqrt', 'criterion': 'entropy', 'bootstrap': False,
}
    """, language='python')

elif section == "Ocena Modeli":
    st.title("Ocena Modeli Uczenia Maszynowego (Wyniki Statyczne)")
    st.markdown("Ocena przeprowadzona na zbiorach **walidacyjnym** i **testowym** (bez SMOTE). Próg decyzyjny: 0.5.")

    # --- Statyczne Wyniki ---
    report_val_xgb_static = """
                  precision    recall  f1-score   support

           0       0.91      0.93      0.92     38065
           1       0.83      0.79      0.81     16546

    accuracy                           0.89     54611
   macro avg       0.87      0.86      0.86     54611
weighted avg       0.89      0.89      0.89     54611
    """
    auc_val_xgb_static = 0.9400

    report_val_rf_static = """
                  precision    recall  f1-score   support

           0       0.92      0.88      0.90     38065
           1       0.76      0.83      0.79     16546

    accuracy                           0.87     54611
   macro avg       0.84      0.86      0.85     54611
weighted avg       0.87      0.87      0.87     54611
    """
    auc_val_rf_static = 0.9338

    report_test_xgb_static = """
                  precision    recall  f1-score   support

           0       0.91      0.93      0.92     38065
           1       0.84      0.78      0.81     16546

    accuracy                           0.89     54611
   macro avg       0.87      0.86      0.86     54611
weighted avg       0.89      0.89      0.89     54611
    """
    auc_test_xgb_static = 0.9400

    report_test_rf_static = """
                  precision    recall  f1-score   support

           0       0.92      0.89      0.90     38065
           1       0.76      0.83      0.79     16546

    accuracy                           0.87     54611
   macro avg       0.84      0.86      0.85     54611
weighted avg       0.87      0.87      0.87     54611
    """
    auc_test_rf_static = 0.9327

    # --- Wyświetlanie wyników ---
    st.subheader("Wyniki na Zbiorze Walidacyjnym")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**XGBoost**")
        st.text(f"AUC-ROC: {auc_val_xgb_static:.4f}")
        st.text("Raport Klasyfikacji:")
        st.code(report_val_xgb_static)
    with col2:
        st.markdown("**Random Forest**")
        st.text(f"AUC-ROC: {auc_val_rf_static:.4f}")
        st.text("Raport Klasyfikacji:")
        st.code(report_val_rf_static)

    st.subheader("Wyniki na Zbiorze Testowym (Ostateczna Ocena)")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**XGBoost**")
        st.text(f"AUC-ROC: {auc_test_xgb_static:.4f}")
        st.text("Raport Klasyfikacji:")
        st.code(report_test_xgb_static)
    with col4:
        st.markdown("**Random Forest**")
        st.text(f"AUC-ROC: {auc_test_rf_static:.4f}")
        st.text("Raport Klasyfikacji:")
        st.code(report_test_rf_static)

    # --- Krzywe ROC (Odtworzone - przykładowe dane) ---
    st.subheader("Krzywe ROC (Zbiór Testowy - Wykres Ilustracyjny)")
    fpr_xgb_static = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.5, 1])
    tpr_xgb_static = np.array([0, 0.6, 0.8, 0.88, 0.92, 0.96, 1])
    fpr_rf_static = np.array([0, 0.07, 0.15, 0.25, 0.35, 0.55, 1])
    tpr_rf_static = np.array([0, 0.55, 0.75, 0.85, 0.90, 0.94, 1])

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr_xgb_static, y=tpr_xgb_static, mode='lines', name=f'XGBoost (AUC ≈ {auc_test_xgb_static:.4f})'))
    fig_roc.add_trace(go.Scatter(x=fpr_rf_static, y=tpr_rf_static, mode='lines', name=f'Random Forest (AUC ≈ {auc_test_rf_static:.4f})'))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Losowy Klasyfikator', line=dict(dash='dash')))

    fig_roc.update_layout(
        title='Krzywa ROC - Zbiór Testowy (Ilustracja)',
        xaxis_title='False Positive Rate (FPR)',
        yaxis_title='True Positive Rate (TPR)',
        legend_title='Model',
        xaxis=dict(range=[0.0, 1.0]),
        yaxis=dict(range=[0.0, 1.05])
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    st.caption("Uwaga: Krzywa ROC jest ilustracją opartą na przykładowych danych dla tej wersji statycznej.")

    # --- Krzywa Uczenia się (Learning Curve) ---
    st.subheader("Krzywa Uczenia się (F1-score) - XGBoost")
    train_sizes = np.array([18271, 36542, 54813, 73084, 91355, 109626, 127897, 146168, 164439, 182710])
    f1_train = np.array([np.nan, 0.85596769, 0.84882457, 0.84334605, 0.83608956, 0.83447398, 0.84805978, 0.87375429, 0.90109748, 0.91796358])
    f1_val = np.array([np.nan, 0.80674855, 0.82824203, 0.8356456, 0.83552517, 0.83828277, 0.84275766, 0.88375834, 0.89837232, 0.90064811])

    valid_indices = ~np.isnan(f1_train) & ~np.isnan(f1_val)
    train_sizes = train_sizes[valid_indices]
    f1_train = f1_train[valid_indices]
    f1_val = f1_val[valid_indices]

    fig_learning = plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, f1_train, label='F1-score XGBoost (trening)', color='blue', marker='o')
    plt.plot(train_sizes, f1_val, label='F1-score XGBoost (walidacja)', color='cyan', marker='o')
    plt.fill_between(train_sizes, f1_train - 0.01, f1_train + 0.01, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, f1_val - 0.01, f1_val + 0.01, alpha=0.1, color='cyan')

    plt.title('Krzywa uczenia (F1-score) - XGBoost', fontsize=14)
    plt.xlabel('Rozmiar zbioru treningowego', fontsize=12)
    plt.ylabel('F1-score', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig_learning)

    st.markdown("""
    **Interpretacja:**
    - **XGBoost**: Lepszy od RandomForest pod względem AUC-ROC (0.9400 vs 0.9327) i F1-score dla klasy wiejskiej (0.81 vs 0.79). Wyższy balans precision-recall.
    - **Krzywa uczenia**: F1-score rośnie z rozmiarem danych, stabilizując się na poziomie 0.90 (walidacja) i 0.92 (trening), co wskazuje na dobrą generalizację.
    - **Wniosek**: XGBoost wybrano do dalszej analizy ze względu na wyższą skuteczność i stabilność.
    """)

elif section == "Ważność Cech (XGBoost)":
    st.title("Ważność Cech według Modelu XGBoost (Wyniki Statyczne)")
    st.markdown("Pokazuje, które cechy miały największy wpływ na predykcje modelu XGBoost w oryginalnej analizie.")

    # --- Statyczne Dane Ważności Cech (Top 12) ---
    feature_importance_data = {
        'Cecha': [
            'speed_limit_normalized',
            'urban_driver_speed',
            'is_urban_driver',
            'distance_speed_interaction',
            'junction_detail_1.0',
            'road_type_6',
            'junction_control_4.0',
            'light_conditions_6.0',
            'casualty_type_9.0',
            'important_driver_distance',
            'urban_driver_long_distance',
            'skidding_and_overturning_9.0'
        ],
        'Ważność': [0.1827, 0.1711, 0.0566, 0.0543, 0.0291, 0.0268, 0.0242, 0.0231, 0.0170, 0.0166, 0.0148, 0.0132]
    }
    top_features = pd.DataFrame(feature_importance_data)

    st.subheader("Top 12 najważniejszych cech")
    st.dataframe(top_features.style.format({'Ważność': '{:.4f}'}))

    # Wizualizacja
    st.subheader("Wykres Ważności Cech (Odtworzony)")
    fig_imp = plt.figure(figsize=(10, 8))
    plt.barh(top_features['Cecha'], top_features['Ważność'], color='skyblue')
    plt.xlabel('Ważność (Importance)')
    plt.ylabel('Cecha')
    plt.title('Ważność Cech (XGBoost) - Top 12')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig_imp)

    st.markdown("""
    **Interpretacja:**
    - **speed_limit_normalized**: Wyższe limity prędkości (typowe dla dróg wiejskich) są kluczowym predyktorem.
    - **urban_driver_speed**: Kierowcy z miast jeżdżący szybciej na wsiach są bardziej narażeni.
    - **is_urban_driver**: Pochodzenie kierowcy ma istotny wpływ.
    - **distance_speed_interaction**: Dłuższe trasy z wyższą prędkością zwiększają ryzyko.
    - **junction_detail_1.0**: Skrzyżowania typu Y są ryzykowne na wsiach.
    - **road_type_6**: Drogi jednopasmowe dominują w wypadkach wiejskich.
    - **junction_control_4.0**: Brak kontroli ruchu zwiększa ryzyko.
    - **light_conditions_6.0**: Ciemność bez oświetlenia to istotny czynnik.
    """)

elif section == "Analiza Kluczowych Cech (Chi-kwadrat)":
    st.title("Szczegółowa Analiza Kluczowych Cech vs Lokalizacja Wypadku (Test Chi-kwadrat - Wyniki Statyczne)")

    # --- Statyczne wyniki testów Chi-kwadrat ---
    chi2_results_data = {
        'Cecha': [
            'is_urban_driver', 'road_type', 'junction_control', 'junction_detail',
            'important_driver_distance', 'light_conditions', 'casualty_type'
        ],
        'chi2': [42475.6, 349.1, 9180.7, 2669.3, 13160.5, 13593.7, 8562.5],
        'p_value': [0.0, 1.543e-76, 0.0, 0.0, 0.0, 0.0, 0.0],
        'V': [0.394, 0.036, 0.183, 0.099, 0.220, 0.223, 0.177],
        'Interpretacja': ['Umiarkowany związek', 'Słaby związek', 'Umiarkowany związek', 'Słaby związek', 'Umiarkowany związek', 'Umiarkowany związek', 'Umiarkowany związek']
    }
    results_df = pd.DataFrame(chi2_results_data).set_index('Cecha')

    st.subheader("Wyniki Testów Chi-kwadrat dla Kluczowych Cech")
    st.dataframe(results_df.style.format({
        'chi2': '{:.1f}',
        'p_value': '{:.1e}',
        'V': '{:.3f}'
    }))

    st.markdown("""
    **Interpretacja:**
    - Wszystkie cechy wykazują statystycznie istotny związek (p < 0.05) z lokalizacją wypadku.
    - **is_urban_driver** ma najsilniejszy związek (V=0.394), co potwierdza jego kluczową rolę.
    - **important_driver_distance**, **light_conditions** i **junction_control** mają umiarkowany wpływ (V > 0.18).
    - **road_type** i **junction_detail** wykazują słabszy związek (V < 0.1), ale nadal są istotne.
    """)

elif section == "Wnioski i Podsumowanie":
    st.title("Wnioski Końcowe i Podsumowanie Analizy")

    st.header("Podsumowanie Wyników")
    st.markdown("""
    1. **Związek miejsca zamieszkania z lokalizacją wypadku:**
       - Istnieje **statystycznie istotny, umiarkowany związek** (φ = 0.394, p < 0.0001).
       - Kierowcy niemiejscy mają ponad trzykrotnie wyższe prawdopodobieństwo wypadków wiejskich (68.4%) niż miejscy (21.7%).
       - Kierowcy miejscy dominują w wypadkach miejskich (78.3%).
       - Hipoteza o większym ryzyku kierowców miejskich na wsiach **nie potwierdzona**.

    2. **Skuteczność modelowania:**
       - XGBoost osiągnął AUC-ROC 0.9400, RandomForest 0.9327, co obala hipotezę o niskiej skuteczności modeli ML.
       - Modele dobrze przewidują lokalizację wypadku na podstawie miejsca zamieszkania i cech kontekstowych.

    3. **Kluczowe czynniki:**
       - **speed_limit_normalized**, **urban_driver_speed**, **is_urban_driver**, **distance_speed_interaction**, **junction_detail_1.0**, **road_type_6**, **junction_control_4.0**, **light_conditions_6.0** to najważniejsze cechy według XGBoost.
       - Potwierdzono hipotezę o wpływie dróg jednopasmowych, braku oświetlenia i niekontrolowanych skrzyżowań.

    4. **Odpowiedzi na pytania badawcze:**
       - Miejsce zamieszkania wpływa na lokalizację wypadku.
       - Kluczowe cechy: prędkość, typ drogi, oświetlenie, kontrola skrzyżowań.
       - Modele ML skutecznie przewidują lokalizację.
    """)

    st.header("Ograniczenia Analizy")
    st.markdown("""
    - Dane dotyczą tylko zgłoszonych wypadków z obrażeniami.
    - Brak danych o natężeniu ruchu i dokładnych trasach.
    - Korelacja nie oznacza przyczynowości.
    """)

    st.header("Rekomendacje")
    st.markdown("""
    - Prewencja na drogach jednopasmowych i niekontrolowanych skrzyżowaniach wiejskich.
    - Edukacja kierowców miejskich nt. jazdy na wsiach.
    - Dalsze badania z uwzględnieniem doświadczenia kierowcy i natężenia ruchu.
    """)