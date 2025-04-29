import streamlit as st
import folium
from geopy.distance import geodesic
import random
import math
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
from io import BytesIO

# East cutoff at Tyumen longitude
CUTOFF_LON = 65.534

st.set_page_config(layout="wide")
st.title("Управление фабриками, точками и охват населения")

# --- Загрузка и подготовка данных городов ---
@st.cache_data
def load_cities(path="cities.xlsx"):
    try:
        df = pd.read_excel(path)
    except FileNotFoundError:
        st.error(f"Не найден файл: {path}")
        return pd.DataFrame()
    required = ['Город', 'Широта', 'Долгота', 'Население']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Отсутствуют колонки: {missing}")
        return pd.DataFrame()
    df = df.dropna(subset=['Широта', 'Долгота'])
    df['Широта'] = pd.to_numeric(df['Широта'], errors='coerce')
    df['Долгота'] = pd.to_numeric(df['Долгота'], errors='coerce')
    df['Население'] = (df['Население'].astype(str)
                       .str.replace(r"\D", "", regex=True)
                       .replace("", "0").astype(int))
    return df.reset_index(drop=True)

cities_df = load_cities()
if cities_df.empty:
    st.stop()

# --- Векторизация расстояний ---
coords_rad = np.deg2rad(cities_df[['Широта', 'Долгота']].values)
pop_arr    = cities_df['Население'].values
R_EARTH    = 6371.0

def haversine_dist(lat_r, lon_r):
    dlat = coords_rad[:,0] - lat_r
    dlon = coords_rad[:,1] - lon_r
    a = np.sin(dlat/2)**2 + np.cos(lat_r)*np.cos(coords_rad[:,0])*np.sin(dlon/2)**2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))

# --- Генерация случайных точек внутри круга ---
def gen_points(center, count, radius):
    lat0, lon0 = center
    pts = []
    for _ in range(count):
        r     = radius * math.sqrt(random.random())
        theta = random.random() * 360
        dest  = geodesic(kilometers=r).destination((lat0, lon0), theta)
        pts.append((round(dest.latitude,6), round(dest.longitude,6)))
    return pts

# --- Жадный подбор PM-фабрик ---
def select_pm_locations(df, existing_centers, radius_km, k):
    idxs    = df.index[df['Долгота'] <= CUTOFF_LON]
    covered = np.zeros(len(df), dtype=bool)
    for lat, lon in existing_centers:
        lat_r, lon_r = np.deg2rad((lat, lon))
        covered |= (haversine_dist(lat_r, lon_r) <= radius_km)
    selected = []
    for _ in range(k):
        best_idx, best_gain = None, 0
        for i in idxs:
            if covered[i]: continue
            lat_r, lon_r = np.deg2rad((df.at[i,'Широта'], df.at[i,'Долгота']))
            new_cov = (~covered) & (haversine_dist(lat_r, lon_r) <= radius_km)
            gain    = pop_arr[new_cov].sum()
            if gain > best_gain:
                best_gain, best_idx = gain, i
        if best_idx is None:
            break
        selected.append((df.at[best_idx,'Город'], (df.at[best_idx,'Широта'], df.at[best_idx,'Долгота'])))
        lat_r, lon_r = np.deg2rad((df.at[best_idx,'Широта'], df.at[best_idx,'Долгота']))
        covered    |= (haversine_dist(lat_r, lon_r) <= radius_km)
    return selected

# --- Инициализация состояния фабрик ---
if 'factory_states' not in st.session_state:
    st.session_state.factory_states = {
        'Липецк': {'center':(52.6032,39.5703),'type':'ПФ','radius':500,
                   'retailer_count':5,'retailer_radius':50,'retailer_pts':[],
                   'horeca_count':100,'horeca_pts':[]},
        'Ростов': {'center':(47.2224364,39.7187866),'type':'ПФ','radius':500,
                   'retailer_count':5,'retailer_radius':50,'retailer_pts':[],
                   'horeca_count':100,'horeca_pts':[]},
        'Ирбит':  {'center':(57.6667,63.0500),'type':'ПФ','radius':500,
                   'retailer_count':5,'retailer_radius':50,'retailer_pts':[],
                   'horeca_count':100,'horeca_pts':[]},
    }

# --- Sidebar: автодобавление PM и оптимизация радиусов по населению ---
with st.sidebar.expander('Автоматическое PM и оптимизация', expanded=True):
    default_radius = st.number_input('Радиус новых PM (км)', 50, 500, 500)
    pm_k           = st.number_input('Число новых PM-фабрик', 0, 20, 8)
    if st.button('Добавить PM-фабрики'):
        existing = [fs['center'] for fs in st.session_state.factory_states.values()]
        for lat, lon, name in [(55.7558,37.6173,'Москва'),(59.9343,30.3351,'Санкт-Петербург')]:
            key = f'PM - {name}'
            if key not in st.session_state.factory_states:
                st.session_state.factory_states[key] = {
                    'center':(lat,lon),'type':'ПМ','radius':default_radius,
                    'retailer_count':5,'retailer_radius':50,'retailer_pts':[],
                    'horeca_count':100,'horeca_pts':[]}
                existing.append((lat,lon))
        additions = select_pm_locations(cities_df, existing, default_radius, pm_k)
        for city, coord in additions:
            key = f'PM - {city}'
            if key not in st.session_state.factory_states:
                st.session_state.factory_states[key] = {
                    'center':coord,'type':'ПМ','radius':default_radius,
                    'retailer_count':5,'retailer_radius':50,'retailer_pts':[],
                    'horeca_count':100,'horeca_pts':[]}
    if st.button('Подобрать оптимальные радиусы'):
        global_covered = np.zeros(len(cities_df), dtype=bool)
        results = {}
        for name, fs in st.session_state.factory_states.items():
            best_r, best_pop, best_mask = 0, -1, None
            lat0, lon0 = fs['center']
            lat_r0, lon_r0 = np.deg2rad((lat0, lon0))
            for r in range(50, 501, 50):
                mask = haversine_dist(lat_r0, lon_r0) <= r
                new_cover = mask & (~global_covered)
                pop_cov   = cities_df.loc[new_cover, 'Население'].sum()
                if pop_cov > best_pop:
                    best_pop, best_r, best_mask = pop_cov, r, mask
            fs['radius'] = best_r
            global_covered |= best_mask if best_mask is not None else False
            results[name] = (best_r, best_pop)
        st.write('### Оптимальные радиусы (дополнительное покрытие):')
        for n,(r,p) in results.items(): st.write(f"{n}: {r} км, +{p:,} чел.")

# --- Sidebar: управление фабриками и параметрами ---
with st.sidebar.expander('Управление фабриками', expanded=False):
    new_n   = st.text_input('Название новой фабрики')
    new_lat = st.number_input('Широта', format='%.6f')
    new_lon = st.number_input('Долгота', format='%.6f')
    if st.button('Добавить фабрику'):
        if new_n and new_n not in st.session_state.factory_states:
            st.session_state.factory_states[new_n] = {
                'center':(new_lat,new_lon),'type':'ПМ','radius':default_radius,
                'retailer_count':5,'retailer_radius':50,'retailer_pts':[],
                'horeca_count':100,'horeca_pts':[]
            }
        else:
            st.warning('Неверное имя')
    to_del = st.multiselect('Удалить фабрики', list(st.session_state.factory_states.keys()))
    if st.button('Удалить выбранные'):
        for d in to_del: st.session_state.factory_states.pop(d, None)
    names = list(st.session_state.factory_states.keys())
    sel   = st.selectbox('Текущая фабрика', names)
    fs    = st.session_state.factory_states[sel]
    with st.form('params_form'):
        r  = st.slider('Радиус', 50, 1000, value=fs['radius'])
        rc = st.number_input('Ритейлеры', 0, 20, value=fs['retailer_count'])
        hc = st.number_input('HORECA', 0, 500, value=fs['horeca_count'])
        if st.form_submit_button('Сохранить'):
            fs.update({'radius':r,'retailer_count':rc,'horeca_count':hc})

# --- Sidebar: экспорт данных в Excel (уникальные потребители) ---
if st.sidebar.button('Экспорт Excel'):
    buffer = BytesIO()
    rows   = []
    for name, fs in st.session_state.factory_states.items():
        sup_lat, sup_lon = fs['center']
        sup_type         = fs['type']
        for typ, pts in [('HORECA', fs['horeca_pts']), ('Retail', fs['retailer_pts'])]:
            for lat, lon in pts:
                dist_km = geodesic(fs['center'], (lat, lon)).km
                rows.append({
                    'consumer_key': (typ, lat, lon),
                    'Поставщик': name,
                    'Тип_поставщика': sup_type,
                    'Широта_поставщика': sup_lat,
                    'Долгота_поставщика': sup_lon,
                    'Тип_потребителя': typ,
                    'Широта_потребителя': lat,
                    'Долгота_потребителя': lon,
                    'Расстояние_км': round(dist_km, 2)
                })
    lipetsk_pf_name = 'Липецк'
    if lipetsk_pf_name in st.session_state.factory_states:
        lipetsk_fs = st.session_state.factory_states[lipetsk_pf_name]
        # Убедимся, что Липецк это ПФ
        if lipetsk_fs['type'] == 'ПФ':
            lipetsk_center = lipetsk_fs['center']
            lipetsk_lat, lipetsk_lon = lipetsk_center
            lipetsk_radius = lipetsk_fs['radius']  # Радиус Липецка

            # Итерируем по всем фабрикам, чтобы найти ПМ
            for pm_name, pm_fs in st.session_state.factory_states.items():
                # Нас интересуют только фабрики типа ПМ
                if pm_fs['type'] == 'ПМ':
                    pm_center = pm_fs['center']
                    pm_lat, pm_lon = pm_center

                    # Рассчитываем расстояние от Липецка до текущей ПМ фабрики
                    dist_lipetsk_to_pm_km = geodesic(lipetsk_center, pm_center).km

                    # Добавляем строку для связи Липецк -> ПМ
                    rows.append({
                        'consumer_key': ('PF_to_PM', pm_name),  # Уникальный ключ для этой связи
                        'Поставщик': lipetsk_pf_name,  # Поставщик - Липецк
                        'Тип_поставщика': 'ПФ',  # Тип - ПФ
                        'Радиус_поставщика': lipetsk_radius,  # Радиус Липецка
                        'Широта_поставщика': lipetsk_lat,
                        'Долгота_поставщика': lipetsk_lon,
                        'Тип_потребителя': pm_name,  # Указываем, что потребитель - ПМ
                        'Широта_потребителя': pm_lat,  # Координаты ПМ фабрики
                        'Долгота_потребителя': pm_lon,  # Координаты ПМ фабрики
                        'Расстояние_км': round(dist_lipetsk_to_pm_km, 2)  # Расстояние Липецк -> ПМ
                        # Можно добавить имя ПМ фабрики как отдельный столбец при необходимости
                        # 'Имя_потребителя_ПМ': pm_name
                    })
        else:
            st.sidebar.warning(f"Фабрика '{lipetsk_pf_name}' найдена, но не является ПФ.")
    else:
        st.sidebar.warning(f"Фабрика '{lipetsk_pf_name}' не найдена в данных. Связи ПФ->ПМ не добавлены.")
    df_rows   = pd.DataFrame(rows)
    idx       = df_rows.groupby('consumer_key')['Расстояние_км'].idxmin()
    df_export = df_rows.loc[idx].drop(columns=['consumer_key']).reset_index(drop=True)
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_export.to_excel(writer, index=False, sheet_name='Data')
    buffer.seek(0)
    st.sidebar.download_button(label='Скачать Excel', data=buffer,
        file_name='export.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

# --- Сводная информация ---
all_cov = set()
for fs in st.session_state.factory_states.values():
    # --- Новая логика генерации HORECA (Вариант А: Повторное использование городов) ---
    lat_r, lon_r = np.deg2rad(fs['center'])
    mask = haversine_dist(lat_r, lon_r) <= fs['radius']

    retailer_radius_val = fs.get('retailer_radius', 50)
    fs['retailer_pts'] = gen_points(fs['center'], fs['retailer_count'], retailer_radius_val)
    # 1. Находим города в радиусе обслуживания фабрики
    cities_in_radius_df = cities_df.loc[mask]

    horeca_points = [] # Список для хранения новых координат HORECA
    target_count = fs['horeca_count'] # Сколько точек нужно сгенерировать
    num_available_cities = len(cities_in_radius_df)

    # 2. Обработка случаев, когда генерация невозможна или не нужна
    if num_available_cities == 0 or target_count == 0:
        # Если нет доступных городов-якорей или не нужно генерировать точки
        fs['horeca_pts'] = [] # Устанавливаем пустой список
    else:
        # 3. Выбираем города-"якоря" для генерации точек.
        #    Нам нужно выбрать ровно target_count якорей.
        #    Если target_count > num_available_cities, то некоторые города
        #    будут выбраны повторно, так как мы используем replace=True.
        #    Если target_count <= num_available_cities, будут выбраны target_count уникальных городов.
        sampled_city_anchors_df = cities_in_radius_df.sample(
            n=target_count,   # Выбираем столько якорей, сколько точек нам нужно
            replace=True,     # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Разрешаем повторный выбор одного и того же города
            random_state=42   # Для воспроизводимости результатов при тех же параметрах
        )

        # 4. Задаем малый радиус для размещения точки *возле* города (в км)
        #    Вы можете настроить это значение
        near_city_radius_km = 10

        # 5. Для КАЖДОГО выбранного якоря (даже если это один и тот же город несколько раз)
        #    генерируем ОДНУ точку HORECA рядом с ним.
        for _, city_row in sampled_city_anchors_df.iterrows():
            city_center = (city_row['Широта'], city_row['Долгота'])
            try:
                # Генерируем одну точку рядом с центром текущего города-якоря
                generated_point = gen_points(center=city_center, count=1, radius=near_city_radius_km)[0]
                horeca_points.append(generated_point)
            except IndexError:
                 # Этот блок на случай, если gen_points вдруг не вернет точку
                 print(f"Warning: Could not generate HORECA point near {city_row['Город']} ({city_center})")
                 # Можно решить, добавлять ли здесь точку с координатами самого города или просто пропустить
                 # Пропуск - самый безопасный вариант.
                 pass

        # 6. Сохраняем сгенерированные точки. Их будет ровно target_count,
        #    если gen_points всегда успешно генерировал точку.
        fs['horeca_pts'] = horeca_points
    # --- Конец новой логики HORECA (Вариант А) ---
# --- Конец новой логики HORECA ---

# --- Сводная информация в Sidebar ---
st.sidebar.markdown('---') # Разделитель
st.sidebar.markdown('## Общая сводка (Западнее Урала)')

# Отбираем только активные фабрики (западнее среза) для сводки
active_factories = {name: fs for name, fs in st.session_state.factory_states.items() if fs['center'][1] <= CUTOFF_LON}

# Считаем общее покрытие городов и населения
all_covered_cities_mask = np.zeros(len(cities_df), dtype=bool)
total_horeca_points = 0
total_retailer_points = 0

for name, fs in active_factories.items():
    # Суммируем сгенерированные точки для активных фабрик
    total_horeca_points += len(fs.get('horeca_pts', [])) # Используем .get для надежности
    total_retailer_points += len(fs.get('retailer_pts', [])) # Используем .get для надежности

    # Обновляем маску покрытия городов
    lat_r, lon_r = np.deg2rad(fs['center'])
    all_covered_cities_mask |= (haversine_dist(lat_r, lon_r) <= fs['radius'])

# Расчет итоговых метрик покрытия
covered_cities = cities_df.loc[all_covered_cities_mask]
unique_covered_cities_count = len(covered_cities)
unique_covered_population = covered_cities['Население'].sum()

# Считаем количество фабрик ПФ и ПМ
pf_count = sum(1 for fs in active_factories.values() if fs["type"]=="ПФ")
pm_count = sum(1 for fs in active_factories.values() if fs["type"]=="ПМ")

# Выводим информацию в сайдбар
st.sidebar.write(f'Уникальных городов в покрытии: {unique_covered_cities_count}')
st.sidebar.write(f'Уникальное население в покрытии: {unique_covered_population:,.0f} чел.')
st.sidebar.write(f'Всего сгенерировано HORECA точек: {total_horeca_points}')
st.sidebar.write(f'Всего сгенерировано Retail точек: {total_retailer_points}')
st.sidebar.write(f'Фабрик ПФ (активных): {pf_count}')
st.sidebar.write(f'Фабрик ПМ (активных): {pm_count}')
# --- Конец блока Сводной информации ---





first_center = next(iter(st.session_state.factory_states.values()))['center']
map_obj = folium.Map(location=first_center, zoom_start=5, tiles='cartodbpositron')
folium.PolyLine(locations=[[-90, CUTOFF_LON], [90, CUTOFF_LON]], color='black', weight=2, dash_array='5').add_to(map_obj)
for name, fs in st.session_state.factory_states.items():
    if fs['center'][1] > CUTOFF_LON: continue
    folium.Marker(fs['center'], popup=f"{name} ({fs['type']})",
                  icon=folium.Icon(color='black' if fs['type']=='ПФ' else 'purple', prefix='fa', icon='industry')).add_to(map_obj)
    folium.Circle(fs['center'], radius=fs['radius']*1000, color='red', weight=2, fill=False).add_to(map_obj)
    for lat, lon in fs['horeca_pts']:
        folium.CircleMarker((lat,lon), radius=4, color='orange', fill=True, fill_opacity=0.8).add_to(map_obj)
    for lat, lon in fs['retailer_pts']:
        folium.CircleMarker((lat,lon), radius=5, color='blue', fill=True, fill_opacity=0.7).add_to(map_obj)
components.html(map_obj._repr_html_(), height=700, width=900)
