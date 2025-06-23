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
from datetime import datetime

app = Flask(__name__)

# --- 1. 모델 로드 ---
try:
    models = {
        'rf': joblib.load('models/rf_model.pkl'),
        'lgbm': joblib.load('models/lgbm_model.pkl'),
        'xgb': joblib.load('models/xgb_model.pkl')
    }
    print("앙상블 모델 3개 로드 완료.")
except FileNotFoundError as e:
    print(f"[오류] 모델 파일을 찾을 수 없습니다: {e}")
    models = {}

# --- 2. 모델 학습에 사용된 최종 특성 목록 (순서 중요!) ---
# fire0622.ipynb 분석을 통해 확인된, 원-핫 인코딩 후의 최종 컬럼 목록입니다.
FINAL_MODEL_FEATURES = [
    'ground_nof', 'bstory_cnt', 'totar', 'bottom_area', 'time_unit_tmprt',
    'time_unit_ws', 'time_unit_wd', 'time_unit_humidity', 'spt_frstt_dist',
    'strc_etc', 'strc_brick', 'strc_block', 'strc_wood', 'strc_steel',
    'strc_src', 'strc_rc'
]
print(f"모델이 학습한 최종 특성 목록: {FINAL_MODEL_FEATURES}")


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
    gangnam_df_raw = pd.read_csv('data/강남구0622.csv', encoding='cp949')
    print("강남구 원본 데이터 로드 완료.")

    gangnam_df_addr = pd.read_csv('data/gangnamgu_final.csv')

    # 전처리 함수를 호출하여 예측에 사용할 최종 데이터프레임 생성
    gangnam_df_processed = preprocess_for_prediction(gangnam_df_raw, FINAL_MODEL_FEATURES)
    print("강남구 데이터 전처리 완료.")
    print(f"전처리 후 예측에 사용될 컬럼: {gangnam_df_processed.columns.tolist()}")

except Exception as e:
    print(f"[오류] 데이터 로드 또는 전처리 중 오류 발생: {e}")
    gangnam_df_processed = None

# 메인페이지 라우트
@app.route('/')
def main_app():
    return render_template('index.html')

@app.route('/api/buildings')
def get_buildings():
    # 1) lat/lon 컬럼만 추출하고 이름 바꾸기
    coords = gangnam_df_raw[['lat', 'lon']].dropna(subset=['lat', 'lon']).copy()
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
    if gangnam_df_processed is None or not models:
        return jsonify({'error': '서버 데이터 또는 모델이 준비되지 않았습니다.'}), 500

    data = request.json
    building_index = data.get('index')

    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    tm = now.strftime('%Y%m%d%H') + '00'
    API_KEY = 'HbpfhxhxR9O6X4cYcUfT0Q'
    url = (
        'https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php'
        f'?tm={tm}&stn=108&help=0&authKey={API_KEY}'
    )
    text = requests.get(url).text
    line = next((l for l in text.splitlines() if l and not l.startswith('#')), None)
    if not line:
        return jsonify({'error': '기상청 데이터 없음'}), 500
    tok = line.split()
    WD = float(tok[2]);
    WS = float(tok[3])
    TA = float(tok[11]);
    HM = float(tok[14])

    # 2) 전처리된 프레임에서 해당 행만 뽑아 날씨 컬럼 채우기
    gangnam_df_processed['time_unit_tmprt']    = TA
    gangnam_df_processed['time_unit_ws']       = WS
    gangnam_df_processed['time_unit_wd']       = WD
    gangnam_df_processed['time_unit_humidity'] = HM
    try:
        # 미리 전처리된 데이터에서 해당 건물의 특성(feature)을 가져옴
        prediction_input = gangnam_df_processed.iloc[[building_index]]

        # 예측에 필요한 컬럼만 다시 한번 확인 (순서 보장)
        prediction_input = prediction_input[FINAL_MODEL_FEATURES]

        print(prediction_input['time_unit_tmprt'])
        print(gangnam_df_processed['time_unit_tmprt'])
        # 각 모델로부터 예측값 받기
        predictions = [model.predict(prediction_input)[0] for model in models.values()]

        # 앙상블 (평균). boxcox 변환된 값이므로 그대로 평균냅니다.
        final_prediction_boxcox = np.mean(predictions)

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

        tornado_score_raw = 10
        drown_score_raw = 10
        earthquake_score_raw = 10

        total_tornado_score = 25
        total_drown_score = 25
        total_earthquake_score = 25
        total_fire_score = 25

        return jsonify({
        'raw': {
            'tornado': tornado_score_raw,
            'drown': drown_score_raw,
            'earthquake': earthquake_score_raw,
            'fire': fire_score_raw
        },
        # ② total 점수: pie 차트 표시용 (합 = 1 혹은 100)
        'total': {
            'tornado': total_tornado_score,
            'drown': total_drown_score,
            'earthquake': total_earthquake_score,
            'fire': total_fire_score
        }
        })

    except Exception as e:
        return jsonify({'error': f"예측 오류: {e}"}), 500

@app.route('/api/weather')
def api_weather():
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    today = now.date()
    tm  = now.strftime('%Y%m%d%H') + '00'
    print("▶ 요청 시각 TM =", tm)

    API_KEY = 'HbpfhxhxR9O6X4cYcUfT0Q'
    cur_url = (
      f'https://apihub.kma.go.kr/api/typ01/url/'
      f'kma_sfctm2.php?tm={tm}&stn=108&help=0&authKey={API_KEY}'
    )
    cur_res = requests.get(cur_url)
    print("▶ [1] 단기 실황 HTTP 상태:", cur_res.status_code)

    lines = cur_res.text.splitlines()
    for i, line in enumerate(lines[:5]):
        print(f"   {i:02d}:", line)

    line = next((l for l in lines if l and not l.startswith('#')), None)
    toks = line.split()
    temp, ws, humidity = float(toks[11]), float(toks[3]), float(toks[14])
    print("▶ 단기 실황:", temp, "℃", ws, "m/s", humidity, "%")

    # 2) 예보 API로 SKY 코드만 가져오기
    fc_url = (
        'https://apihub.kma.go.kr/api/typ01/url/fct_afs_dl.php'
        '?reg=11B10101'   # 서울
        '&disp=1'         # CSV
        '&help=0'         # 도움말 없이
        f'&authKey={API_KEY}'
    )
    fc_res = requests.get(fc_url)
    print("▶ [2] 중기 예보 HTTP 상태:", fc_res.status_code)
    fc_lines = fc_res.text.splitlines()
    for i, line in enumerate(fc_lines[:5]):
        print(f"   예보 {i:02d}:", line)

    data_lines = fc_lines[2:-1]

    reader = csv.reader(data_lines)
    rows = [row for row in reader if row[0] == '11B10101']
    for i, l in enumerate(data_lines):
        print(f"{i:02d}:", l)

    # 1) 문자열 → 리스트 변환
    rows = [line.split(',') for line in data_lines]

    # 2) 오늘 날짜, 현재 시각 구하기
    now   = datetime.now()
    today = now.date()

    # 3) TM_EF(index 2) 파싱해서 (datetime, row) 튜플 리스트 생성
    forecasts = []
    for r in rows:
        # 예: '202506231200'
        tm_ef = datetime.strptime(r[2], '%Y%m%d%H%M')
        forecasts.append((tm_ef, r))

    # 4) 오늘 날짜이고, now 이전 것만 필터
    candidates = [(t, r) for t, r in forecasts if t.date() == today and t <= now]
    if not candidates:
        raise ValueError("오늘 이전 예보가 없습니다.")

    # 5) 가장 최근 예보 선택
    best_time, best_row = max(candidates, key=lambda x: x[0])

    print("선택된 TM_EF:", best_time)
    print("해당 예보 데이터:", best_row)

    # 헤더 + 첫 번째 데이터 행 뽑기
    # 수동으로 정의한 헤더
    header = [
        'REG_ID','TM_FC','TM_EF','MOD','NE','STN','C','MAN_ID',
        'MAN_FC','W1','T','W2','TA','ST','SKY','PREP','WF'
    ]

    sky_code = best_row[ header.index('SKY') ]  # --> 'DB01' 같은 코드

    # 3) 아이콘 매핑
    sky_map = {
      'DB01': ('sun', '맑음'),
      'DB02': ('cloud-sun', '구름조금'),
      'DB03': ('cloud', '구름많음'),
      'DB04': ('cloud-showers-heavy', '흐림'),
    }
    icon, desc = sky_map.get(sky_code, ('cloud', '정보없음'))
    print("▶ 매핑된 아이콘, 설명:", icon, desc)

    return jsonify({
        'temp':      temp,
        'ws':        ws,
        'humidity':  humidity,
        'icon':      icon,
        'desc':      desc
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9999)