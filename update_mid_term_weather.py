import requests
import pandas as pd
import os
from datetime import datetime, timezone, timedelta

CACHE_FILE = '/mnt/c/AIDC/SA_AI_SEMI_PROJECT/data/cache_mid_term_weather.csv'
API_KEY = 'HbpfhxhxR9O6X4cYcUfT0Q'
BASE_URL = 'https://apihub.kma.go.kr/api/typ01/url/fct_afs_dl.php'

KST = timezone(timedelta(hours=9))

def fetch_mid_weather(API_KEY):
    # 1) tm 파라미터 없이 호출 → 모든 예보를 받아옴
    url = f"{BASE_URL}?reg=11B10101&disp=1&help=0&authKey={API_KEY}"
    res = requests.get(url, timeout=5)
    res.raise_for_status()

    # 2) CSV 본문에서 데이터 라인만 추출
    lines = res.text.splitlines()[2:-1]
    rows = [line.split(',') for line in lines if line.startswith('11B10101')]

    # 3) 현재 시각 계산 (KST)
    now = datetime.now(KST)

    # 4) TM_EF 컬럼으로 파싱한 뒤 now 이전 것만 필터
    forecasts = []
    for r in rows:
        tm_ef = datetime.strptime(r[2], '%Y%m%d%H%M').replace(tzinfo=KST)
        if tm_ef <= now:
            forecasts.append((tm_ef, r))

    if not forecasts:
        return None

    # 5) 가장 최신(가장 큰 tm_ef) 예보 선택
    best_tm, best_row = max(forecasts, key=lambda x: x[0])

    # 6) SKY 코드 추출
    header = ['REG_ID','TM_FC','TM_EF','MOD','NE','STN','C','MAN_ID',
              'MAN_FC','W1','T','W2','TA','ST','SKY','PREP','WF']
    sky_code = best_row[header.index('SKY')]
    prep_code = best_row[header.index('PREP')]
    wf_text = best_row[header.index('WF')].strip('"')

    return {
        'datetime': best_tm.strftime('%Y-%m-%d %H:%M:%S'),
        'sky_code': sky_code,
        'prep': prep_code,
        'wf': wf_text
    }

def update_mid_weather_with_fallback():
    try:
        data = fetch_mid_weather(API_KEY)
        if data:
            pd.DataFrame([data]).to_csv(CACHE_FILE, index=False)
            print(f"▶ [ MID ] 데이터 업데이트 성공: {data['datetime']}")
        else:
            raise ValueError("[ MID ] 데이터 없음")
    except Exception as e:
        print(f"▶ [ MID ] 데이터 업데이트 실패: {e}")
        if not os.path.exists(CACHE_FILE):
            print("⚠️ [ MID ] 기존 캐시 파일 없음. 확인 필요.")

if __name__ == "__main__":
    update_mid_weather_with_fallback()
