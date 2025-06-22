from dataclasses import fields

from flask import Flask
from flask import render_template
import geopandas as gpd
import folium
from folium import plugins
import pandas as pd
import os
import json
import cx_Oracle
from sqlalchemy import create_engine, text, Integer, Float, Numeric
import numpy as np


app = Flask(__name__)


def get_map_data():
    try:
        with open('dataset/lec12_seoul_geo_sigugun.json', 'r', encoding='utf-8') as f:
            geo_data = json.load(f)
        df = pd.read_csv("dataset/gangnamgu_final.csv")
        print(df.head())

        m = folium.Map(location=[37.562225, 126.978555], tiles="OpenStreetMap", zoom_start=11)

        print("GeoJSON 데이터 타입:", type(geo_data))
        print("GeoJSON 키:", list(geo_data.keys()) if isinstance(geo_data, dict) else "딕셔너리가 아님")

        print(geo_data)
        mc = plugins.MarkerCluster()
        folium.GeoJson(
            geo_data,
            style_function=lambda feature: {
                'fillColor': 'lightblue',
                'color': 'blue',
                'weight': 2,
                'fillOpacity': 0.3
            },
            tooltip=folium.GeoJsonTooltip(fields=['name'], labels=False)
        ).add_to(m)

        geo_list = []
        name_list = []

        for i in range(len(df)):
            lat = df.iloc[i]['위도']
            lng = df.iloc[i]['경도']
            sname = df.iloc[i]['대지위치']

            if pd.notna(lat) and pd.notna(lng):
                geo_list.append([lat, lng])  # 튜플 대신 리스트 사용
                name_list.append(sname)


        if geo_list:
            plugins.MarkerCluster(geo_list, popups=name_list).add_to(m)

        m.get_root().height = "400px"
        map_html_str = m.get_root()._repr_html_()

        return map_html_str

    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return f"<div style='color:red;'>지도 생성 중 오류: {str(e)}</div>"

# 메인페이지 라우트
@app.route('/')
def main_app():
    cached_map_html = 0
    if cached_map_html == 0 :
        map_html_str = get_map_data()  # 최초 1회만 실행
        cached_map_html = 1
    return render_template('index.html', map_html=map_html_str)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9999)