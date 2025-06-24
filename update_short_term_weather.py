import requests
import pandas as pd
import os
from datetime import datetime, timedelta

CACHE_FILE = '/mnt/c/AIDC/SA_AI_SEMI_PROJECT/data/cache_short_term_weather.csv'
API_KEY = 'HbpfhxhxR9O6X4cYcUfT0Q'
BASE_URL = 'https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php'

def fetch_latest():
    """
    tm 파라미터 없이 요청해서 기상청이 제공하는 '가장 최신' 관측값을 내려받습니다.
    """
    url = f"{BASE_URL}?stn=108&authKey={API_KEY}"
    res = requests.get(url, timeout=5)
    res.raise_for_status()

    lines = [l for l in res.text.splitlines() if l and not l.startswith('#')]
    if not lines:
        raise ValueError("데이터가 없습니다")
    tok = lines[0].split()

    return {
        'datetime': datetime.strptime(tok[0], '%Y%m%d%H%M').strftime('%Y-%m-%d %H:%M:%S'),
        'temp':      float(tok[11]),
        'ws':        float(tok[3]),
        'wd':        float(tok[2]),
        'humidity':  float(tok[14])
    }

def fetch_by_tm(tm):
    """
    tm 파라미터로 특정 시각 관측값을 요청합니다.
    tm 예시: '202506241300'
    """
    url = f"{BASE_URL}?tm={tm}&stn=108&authKey={API_KEY}"
    res = requests.get(url, timeout=5)
    res.raise_for_status()

    lines = [l for l in res.text.splitlines() if l and not l.startswith('#')]
    if not lines:
        return None
    tok = lines[0].split()

    return {
        'datetime': datetime.strptime(tok[0], '%Y%m%d%H%M').strftime('%Y-%m-%d %H:%M:%S'),
        'temp':      float(tok[11]),
        'ws':        float(tok[3]),
        'wd':        float(tok[2]),
        'humidity':  float(tok[14])
    }

def update_weather_with_fallback():
    """
    1) tm 없이 최신값 요청
    2) 실패하면 1시간 전 tm으로 한 번 더 시도
    3) 성공 시 캐시 파일에 기록
    """
    data = None
    now = datetime.now()

    # 1) 최신값 시도
    try:
        data = fetch_latest()
        print(f"▶ [SHORT] 최신값 요청 성공:  {now.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"▶ [SHORT] 최신 요청 실패: {e}")
        # 2) 1시간 전 tm으로 한 번 보충 조회
        tm1 = (datetime.now() - timedelta(hours=1)).strftime('%Y%m%d%H%M')
        print(f"▶ [SHORT] 1시간 전 시간으로 재시도: {tm1}")
        try:
            data = fetch_by_tm(tm1)
            if data:
                print(f"▶ [SHORT] 1시간 전 데이터 성공: {data['datetime']}")
            else:
                print("▶ [SHORT] 1시간 전에도 데이터 없음 → 캐시 유지")
                return
        except Exception as e2:
            print(f"▶ [SHORT] 1시간 전 재시도 실패: {e2} → 캐시 유지")
            return

    # 3) 캐시에 저장
    pd.DataFrame([data]).to_csv(CACHE_FILE, index=False)
    print(f"▶ [SHORT] 캐시 업데이트 완료: {data['datetime']}")

if __name__ == "__main__":
    update_weather_with_fallback()
