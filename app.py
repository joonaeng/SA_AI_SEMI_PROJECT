from flask import Flask, render_template, jsonify, request
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# --- 1. 모델 로드 ---
try:
    models = {
        'rf': joblib.load('rf_model.pkl'),
        'lgbm': joblib.load('lgbm_model.pkl'),
        'xgb': joblib.load('xgb_model.pkl')
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

    # 전처리 함수를 호출하여 예측에 사용할 최종 데이터프레임 생성
    gangnam_df_processed = preprocess_for_prediction(gangnam_df_raw, FINAL_MODEL_FEATURES)
    print("강남구 데이터 전처리 완료.")
    print(f"전처리 후 예측에 사용될 컬럼: {gangnam_df_processed.columns.tolist()}")

except Exception as e:
    print(f"[오류] 데이터 로드 또는 전처리 중 오류 발생: {e}")
    gangnam_df_processed = None


# --- 5. Flask 라우트 정의 ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/buildings')
def get_buildings():
    # 원본 데이터에서 지도 표시에 필요한 정보만 추출
    df = gangnam_df_raw.rename(columns={'lon': 'lng'})
    buildings = df[['lat', 'lng']].to_dict('records')
    return jsonify(buildings)


@app.route('/api/predict', methods=['POST'])
def predict():
    if gangnam_df_processed is None or not models:
        return jsonify({'error': '서버 데이터 또는 모델이 준비되지 않았습니다.'}), 500

    data = request.json
    building_index = data.get('index')

    try:
        # 미리 전처리된 데이터에서 해당 건물의 특성(feature)을 가져옴
        prediction_input = gangnam_df_processed.iloc[[building_index]]

        # 예측에 필요한 컬럼만 다시 한번 확인 (순서 보장)
        prediction_input = prediction_input[FINAL_MODEL_FEATURES]

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
        final_score = np.clip(50 + 10 * ((y_pred_original - 0.15) / 0.2), 0, 100)

        return jsonify({'prediction': final_score})

    except Exception as e:
        return jsonify({'error': f"예측 오류: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True)