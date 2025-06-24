import requests
import pandas as pd
import os
from datetime import datetime

CACHE_FILE = '/mnt/c/AIDC/SA_AI_SEMI_PROJECT/data/cache_short_term_weather.csv'

def fetch_weather_data(tm, API_KEY):
    url = f'https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php?tm={tm}&stn=108&authKey={API_KEY}'
    res = requests.get(url, timeout=5)
    res.raise_for_status()

    lines = res.text.splitlines()
    line = next((l for l in lines if l and not l.startswith('#')), None)
    if not line:
        return None

    tok = line.split()
    weather_data = {
        'datetime': datetime.strptime(tok[0], '%Y%m%d%H%M').strftime('%Y-%m-%d %H:%M:%S'),
        'temp': float(tok[11]),
        'ws': float(tok[3]),
        'wd': float(tok[2]),
        'humidity': float(tok[14])
    }

    return weather_data

def update_weather_with_fallback():
    API_KEY = 'HbpfhxhxR9O6X4cYcUfT0Q'
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    attempts = 0
    max_attempts = 5
    data = None

    while attempts < max_attempts:
        tm = (now - pd.Timedelta(hours=attempts)).strftime('%Y%m%d%H') + '00'
        print(f"▶ 데이터 요청 시각: {tm}")
        try:
            data = fetch_weather_data(tm, API_KEY)
            if data:
                print(f"▶ 성공적으로 데이터를 가져왔습니다: {data['datetime']}")
                break
            else:
                print("▶ 해당 시각 데이터 없음, 이전 시각 데이터 조회 중...")
        except Exception as e:
            print(f"▶ 데이터 요청 실패({tm}): {e}")
        attempts += 1

    if data:
        # 최신 데이터를 캐시에 저장
        pd.DataFrame([data]).to_csv(CACHE_FILE, index=False)
        print(f"▶ 캐시 업데이트 성공: {data['datetime']}")
    else:
        print("▶ 데이터가 없어서 기존 캐시 유지")
        if not os.path.exists(CACHE_FILE):
            print("⚠️ 기존 캐시파일도 없습니다. 확인 필요.")

if __name__ == "__main__":
    update_weather_with_fallback()
