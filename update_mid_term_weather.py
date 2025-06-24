import requests
import pandas as pd
import os
from datetime import datetime, timezone, timedelta

CACHE_FILE = '/mnt/c/AIDC/SA_AI_SEMI_PROJECT/data/cache_mid_term_weather.csv'

KST = timezone(timedelta(hours=9))

def fetch_mid_weather(API_KEY):
    url = (
        f'https://apihub.kma.go.kr/api/typ01/url/fct_afs_dl.php'
        '?reg=11B10101&disp=1&help=0'
        f'&authKey={API_KEY}'
    )
    res = requests.get(url, timeout=5)
    res.raise_for_status()

    now = datetime.now(KST)
    print("✅ 현재 시각 (KST):", now)

    fc_lines = res.text.splitlines()[2:-1]
    rows = [line.split(',') for line in fc_lines if line.startswith('11B10101')]
    print("✅ rows 개수:", len(rows))

    forecasts = []
    for r in rows:
        raw = r[2]
        try:
            tm_naive = datetime.strptime(raw, '%Y%m%d%H%M')
            tm_ef = tm_naive.replace(tzinfo=KST)
            forecasts.append((tm_ef, r))
            print(f"🕒 TM_EF: {tm_ef} <= 현재시각? {'✅' if tm_ef <= now else '❌'}")
        except Exception as e:
            print(f"⚠️ 파싱 실패: {raw}, 에러: {e}")

    candidates = [f for f in forecasts if f[0] <= now]
    print(f"✅ 필터 통과 개수: {len(candidates)}")
    if not candidates:
        return None

    best_time, best_row = max(candidates, key=lambda x: x[0])

    header = [
        'REG_ID', 'TM_FC', 'TM_EF', 'MOD', 'NE', 'STN', 'C', 'MAN_ID',
        'MAN_FC', 'W1', 'T', 'W2', 'TA', 'ST', 'SKY', 'PREP', 'WF'
    ]

    sky_code = best_row[header.index('SKY')]

    return {
        'datetime': best_time.strftime('%Y-%m-%d %H:%M:%S'),
        'sky_code': sky_code
    }

def update_mid_weather_with_fallback():
    API_KEY = 'HbpfhxhxR9O6X4cYcUfT0Q'

    try:
        data = fetch_mid_weather(API_KEY)
        if data:
            pd.DataFrame([data]).to_csv(CACHE_FILE, index=False)
            print(f"▶ 중기기상 데이터 업데이트 성공: {data['datetime']}")
        else:
            raise ValueError("중기기상 데이터 없음")
    except Exception as e:
        print(f"▶ 중기기상 데이터 업데이트 실패: {e}")
        if not os.path.exists(CACHE_FILE):
            print("⚠️ 기존 캐시파일 없음. 확인 필요.")

if __name__ == "__main__":
    update_mid_weather_with_fallback()
