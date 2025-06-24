from dataclasses import fields
import geopandas as gpd
import folium
import requests
from folium import plugins
import os
import json
import csv
from io import StringIO
from flask import Flask, render_template, jsonify, request
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from requests.exceptions import Timeout
import pickle


app = Flask(__name__)

SHORT_CACHE_PATH = r'C:\AIDC\SA_AI_SEMI_PROJECT\data\cache_short_term_weather.csv'
MID_CACHE_PATH   = r'C:\AIDC\SA_AI_SEMI_PROJECT\data\cache_mid_term_weather.csv'

# --- 1) 앱 시작 시 한 번만 메모리에 로드 ---
short_cache = pd.read_csv(
    SHORT_CACHE_PATH,
    usecols=['datetime','temp','ws','wd','humidity']
)
mid_cache   = pd.read_csv(
    MID_CACHE_PATH,
    usecols=['datetime','sky_code','prep','wf']
)

# --- 2) 수정 시간 트래킹을 위한 초기화 ---
last_short_mtime = Path(SHORT_CACHE_PATH).stat().st_mtime
last_mid_mtime   = Path(MID_CACHE_PATH).stat().st_mtime

@app.before_request
def refresh_cache_if_needed():
    global short_cache, mid_cache, last_short_mtime, last_mid_mtime
    # 단기 캐시 갱신
    cur = Path(SHORT_CACHE_PATH).stat().st_mtime
    if cur != last_short_mtime:
        short_cache = pd.read_csv(SHORT_CACHE_PATH, usecols=short_cache.columns)
        last_short_mtime = cur
    # 중기 캐시 갱신
    cur = Path(MID_CACHE_PATH).stat().st_mtime
    if cur != last_mid_mtime:
        mid_cache = pd.read_csv(MID_CACHE_PATH, usecols=mid_cache.columns)
        last_mid_mtime = cur

# --- 1. 모델 로드 ---
try:
    fire_models = {
        'rf': joblib.load('models/rf_model.pkl'),
        'lgbm': joblib.load('models/lgbm_model.pkl'),
        'xgb': joblib.load('models/xgb_model.pkl'),
    }
    drown_models = {
        'kmeans': joblib.load('models/kmeans_model.pkl')
    }
    print("앙상블 모델 4개 로드 완료.")
except FileNotFoundError as e:
    print(f"[오류] 모델 파일을 찾을 수 없습니다: {e}")
    models = {}

# --- 2. 모델 학습에 사용된 최종 특성 목록 (순서 중요!) ---
# fire0622.ipynb 분석을 통해 확인된, 원-핫 인코딩 후의 최종 컬럼 목록입니다.
FIRE_MODEL_FEATURES = [
    'ground_nof', 'bstory_cnt', 'totar', 'bottom_area', 'time_unit_tmprt',
    'time_unit_ws', 'time_unit_wd', 'time_unit_humidity', 'spt_frstt_dist',
    'strc_etc', 'strc_brick', 'strc_block', 'strc_wood', 'strc_steel',
    'strc_src', 'strc_rc'
]
DROWN_MODEL_FEATURES = [
    'struc', 'bstory_cnt', 'ground_nof', 'height(m)',
    '1f_flo', 'old', 'flood_point_count', 'cluster'
]
print(f"모델이 학습한 최종 특성 목록: {FIRE_MODEL_FEATURES}")


# --- 3. 데이터 전처리 함수 정의 ---
def preprocess_for_prediction(df_raw, train_columns):
    """
    예측할 데이터를 모델 학습 때와 동일하게 전처리하고, 컬럼을 정렬합니다.
    """
    df = df_raw.copy()

    # 1. 컬럼 이름 문제 해결: 깨진 이름 -> 올바른 이름
    # '癤풿round_nof' 컬럼이 존재할 경우에만 이름 변경
    if '癤풿round_nof' in df.columns:
        df.rename(columns={'癤풿round_nof': 'ground_nof'}, inplace=True)

    # 2. 건물구조(buld_strctr) 원-핫 인코딩
    # Notebook(107셀)에서 사용한 숫자 -> 텍스트 매핑
    mapping_dict = {
        7.0: 'rc', 2.0: 'brick', 4.0: 'wood', 5.0: 'steel',
        1.0: 'etc', 3.0: 'block', 6.0: 'src'
    }
    # 'buld_strctr'이 숫자 코드로 되어 있다고 가정하고 텍스트로 변환
    if 'buld_strctr' in df.columns:
        df['구조명'] = df['buld_strctr'].map(mapping_dict)

        # 텍스트로 변환된 '구조명' 컬럼을 원-핫 인코딩
        # prefix='strc'로 하여 strc_rc, strc_brick 등으로 컬럼 생성
        dummies = pd.get_dummies(df['구조명'], prefix='strc')
        df = pd.concat([df, dummies], axis=1)
        # 원본 컬럼들은 삭제
        df.drop(['buld_strctr', '구조명'], axis=1, inplace=True, errors='ignore')

    # 3. 로그 변환 (Notebook 130셀)
    log_cols = [
        'ground_nof', 'bstory_cnt', 'totar', 'bottom_area', 'spt_frstt_dist'
    ]
    for col in log_cols:
        if col in df.columns:
            # 음수 값이 없도록 0 미만은 0으로 처리
            df[col] = np.log1p(df[col].clip(lower=0))

    # 4. 최종 컬럼 정렬 및 맞추기 (가장 중요!)
    # 모델 학습에 사용된 컬럼 순서대로 데이터프레임을 재구성합니다.
    # 없는 컬럼은 0으로 채우고(fill_value=0), 불필요한 컬럼은 버립니다.
    df_aligned = df.reindex(columns=train_columns, fill_value=0)

    return df_aligned


# --- 4. 서버 시작 시 데이터 로드 및 전처리 ---
try:
    gangnam_fire_raw = pd.read_csv('data/gangnam_fire_data.csv', encoding='cp949')
    print("강남구 화재 데이터 로드 완료.")

    gangnam_df_addr = pd.read_csv('data/gangnam_str_info.csv')
    print("강남구 건물 데이터 로드 완료.")

    gangnam_drown_raw = pd.read_csv('data/gangnam_drown_data.csv')
    print("강남구 침수 데이터 로드 완료.")

    # 전처리 함수를 호출하여 예측에 사용할 최종 데이터프레임 생성
    gangnam_fire_processed = preprocess_for_prediction(gangnam_fire_raw, FIRE_MODEL_FEATURES)
    gangnam_drown_processed = preprocess_for_prediction(gangnam_drown_raw, DROWN_MODEL_FEATURES)
    print("강남구 데이터 전처리 완료.")
    print(f"전처리 후 화재 예측에 사용될 컬럼: {gangnam_fire_processed.columns.tolist()}")
    print(f"전처리 후 침수 예측에 사용될 컬럼: {gangnam_drown_processed.columns.tolist()}")

except Exception as e:
    print(f"[오류] 데이터 로드 또는 전처리 중 오류 발생: {e}")
    gangnam_fire_processed = None
    gangnam_drown_processed = None

# 메인페이지 라우트
@app.route('/')
def main_app():
    return render_template('index.html')

@app.route('/api/buildings')
def get_buildings():
    # 1) lat/lon 컬럼만 추출하고 이름 바꾸기
    coords = gangnam_fire_raw[['lat', 'lon']].dropna(subset=['lat', 'lon']).copy()
    coords.rename(columns={'lon': 'lng'}, inplace=True)
    # 2) 같은 인덱스로 gangnam_df_addr의 '대지위치' 병합
    # (coords와 gangnam_df_addr가 동일한 순서/길이여야 합니다)
    coords['address'] = gangnam_df_addr.loc[coords.index, '대지위치'].values
    # 3) JSON 응답: lat, lng, address 포함
    buildings = coords[['lat', 'lng', 'address']].to_dict('records')
    return jsonify(buildings)
    # # 원본 데이터에서 지도 표시에 필요한 정보만 추출
    # df = gangnam_df_raw.rename(columns={'lon': 'lng'})
    # buildings = df[['lat', 'lng']].to_dict('records')
    # return jsonify(buildings)


@app.route('/api/predict', methods=['POST'])
def predict():
    if gangnam_fire_processed is None or not fire_models:
        return jsonify({'error': '서버 데이터 또는 모델이 준비되지 않았습니다.'}), 500

    data = request.json
    building_index = data.get('index')

    latest_weather = short_cache.iloc[0]

    inp = gangnam_fire_processed.iloc[[building_index]].copy()
    # 예측 데이터에 기상 정보 입력
    inp['time_unit_tmprt']    = latest_weather['temp']
    inp['time_unit_ws']       = latest_weather['ws']
    inp['time_unit_wd']       = latest_weather['wd']
    inp['time_unit_humidity'] = latest_weather['humidity']

    fire_prediction_input = inp[FIRE_MODEL_FEATURES]

    try:
        # 미리 전처리된 데이터에서 해당 건물의 특성(feature)을 가져옴
        fire_prediction_input = gangnam_fire_processed.iloc[[building_index]].copy()

        # 예측에 필요한 컬럼만 다시 한번 확인 (순서 보장)
        fire_prediction_input = fire_prediction_input[FIRE_MODEL_FEATURES]

        # 각 모델로부터 예측값 받기
        fire_predictions = [model.predict(fire_prediction_input)[0] for model in fire_models.values()]

        # 앙상블 (평균). boxcox 변환된 값이므로 그대로 평균냅니다.
        final_prediction_boxcox = np.mean(fire_predictions)

        # Box-Cox 역변환 수행 (Notebook 156, 163셀 참조)
        # 람다(lambda) 값은 학습 때 사용된 값과 동일해야 합니다.
        LAMBDA_VAL = 0.2728  # Notebook 163셀에서 확인된 값
        y_pred_original = np.exp(final_prediction_boxcox * LAMBDA_VAL) - 1 if LAMBDA_VAL == 0 else (
           final_prediction_boxcox * LAMBDA_VAL + 1) ** (
               1 / LAMBDA_VAL)

        # Z-Score 변환 (Notebook 163셀) - 검증세트의 통계량을 써야하나, 여기선 간략화
        # 실제 운영시에는 학습 데이터 전체의 평균/표준편차를 저장해두고 사용해야 합니다.
        # 여기서는 설명을 위해 임시값으로 대체합니다.
        fire_score_raw = np.clip(50 + 10 * ((y_pred_original - 0.15) / 0.2), 0, 100)

        try:
            pos = gangnam_drown_raw.index[gangnam_drown_raw['index'] == building_index].tolist()[0]
            drown_input = gangnam_drown_processed.iloc[[pos]].copy()
            drown_input = drown_input.reindex(columns=DROWN_MODEL_FEATURES, fill_value=0)
            drown_preds = [m.predict(drown_input)[0] for m in drown_models.values()]
            drown_score_raw = int(drown_preds[0])
        except (IndexError, KeyError):
            # 인덱스 범위 초과 또는 컬럼 누락 시 flood risk level = 0
            drown_score_raw = 0

        tornado_score_raw = 10
        earthquake_score_raw = 10

        return jsonify({
        'raw': {
            'tornado': tornado_score_raw,
            'drown': drown_score_raw,
            'earthquake': earthquake_score_raw,
            'fire': fire_score_raw
        }
        })

    except Exception as e:
        return jsonify({'error': f"예측 오류: {e}"}), 500

@app.route('/api/weather')
def api_weather():
    latest_short = short_cache.iloc[0]
    latest_mid   = mid_cache.iloc[0]

    return jsonify({
        'short_time': latest_short['datetime'],
        'temp':      latest_short['temp'],
        'ws':        latest_short['ws'],
        'wd':        latest_short['wd'],          # 풍향
        'humidity':  latest_short['humidity'],

        'mid_time': latest_mid['datetime'],
        'sky_code': latest_mid['sky_code'],
        'wf': latest_mid['wf'],
        'prep': int(latest_mid['prep'])  # 0=없음, 1=비,2=비/눈,3=눈,4=소나기
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9999)