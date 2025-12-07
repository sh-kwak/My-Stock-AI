import streamlit as st
import io
import requests
import pandas as pd
import json
import time
import numpy as np
import FinanceDataReader as fdr 
import matplotlib.pyplot as plt
import koreanize_matplotlib

# [한글 깨짐 방지] - Windows 터미널에서 문제 발생 시 주석 처리
# sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
# sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# =============================================================================
# [설정 영역]
# =============================================================================
# ★★★ 선생님의 실제 App Key와 Secret을 입력해주세요 ★★★
# 로컬 테스트용 기본값 (배포 시에는 Streamlit Secrets 사용 권장)
DEFAULT_APP_KEY = "PSTmwr8yGJqGMn86dWiwRVjCeQa54QtEoskT"
DEFAULT_APP_SECRET = "RCPnw1rZVbs3jYdKwV6/5k5Rky+LCRJgO7s2oVc8kHKGFEubiiErLhf0w73m6XMBmtfetmY2P2EKxAC4Lyw/T/00h852W8Eoy6aZ187lIIY3KojtvwL3w86bL4vfDbbEWbKK0q2A2bpW0lJzlax5C/+0f6ptedDiInhyDRP16+DulwdUH30="

try:
    APP_KEY = st.secrets["APP_KEY"]
    APP_SECRET = st.secrets["APP_SECRET"]
except:
    # Secrets가 없으면 코드 상단 변수 사용 (테스트용)
    APP_KEY = DEFAULT_APP_KEY
    APP_SECRET = DEFAULT_APP_SECRET

BASE_URL = "https://openapi.koreainvestment.com:9443"

# [설정] 분석할 종목 개수 (시가총액 상위 N개)
TOP_N = 100 

# =============================================================================
# [1] 인증 (Auth)
# =============================================================================
def get_access_token():
    url = f"{BASE_URL}/oauth2/tokenP"
    headers = {"content-type": "application/json"}
    body = {"grant_type": "client_credentials", "appkey": APP_KEY, "appsecret": APP_SECRET}
    try:
        res = requests.post(url, headers=headers, data=json.dumps(body))
        res.raise_for_status()
        return res.json()["access_token"]
    except Exception as e:
        print(f"[인증 실패] {e}")
        return None

# =============================================================================
# [NEW] 실시간 우량주 리스트 가져오기 (FinanceDataReader)
# =============================================================================
def get_top_stocks(limit=100):
    print(f"\n[시스템] 시가총액 상위 {limit}개 종목 리스트를 업데이트합니다...")
    
    try:
        # KOSPI/KOSDAQ 개별 호출 대신 'KRX' 통합 호출 사용 (안정성 향상)
        df_total = fdr.StockListing('KRX')
        
        # 시가총액(Marcap) 순으로 정렬하여 상위 N개만 자름
        # 데이터프레임 컬럼명이 다를 수 있으니 확인: 보통 'Marcap' 사용
        df_top = df_total.sort_values(by='Marcap', ascending=False).head(limit)
        
        # 우리가 쓰는 형식 [('코드', '이름'), ...] 으로 변환
        stock_list = []
        for idx, row in df_top.iterrows():
            stock_list.append((str(row['Code']), row['Name']))
        
        print(f"[시스템] 리스트 확보 완료! (1위 {stock_list[0][1]} ~ {limit}위 {stock_list[-1][1]})")
        return stock_list
    except Exception as e:
        print(f"[오류] 종목 리스트 가져오기 실패: {e}")
        print("[시스템] 백업용 수동 리스트(Top 10)를 사용합니다.")
        # 백업용 하드코딩 리스트 (2025년 기준 주요 시총 상위 10 종목)
        backup_list = [
            ('005930', '삼성전자'), ('000660', 'SK하이닉스'), ('373220', 'LG에너지솔루션'),
            ('207940', '삼성바이오로직스'), ('005380', '현대차'), ('000270', '기아'),
            ('068270', '셀트리온'), ('005490', 'POSCO홀딩스'), ('035420', 'NAVER'),
            ('006400', '삼성SDI')
        ]
        return backup_list[:limit]
def get_stock_data(stock_code, stock_name, access_token):
    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price"
    headers = {
        "content-type": "application/json", "authorization": f"Bearer {access_token}",
        "appkey": APP_KEY, "appsecret": APP_SECRET, "tr_id": "FHKST01010100"
    }
    params = {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": stock_code}
    try:
        res = requests.get(url, headers=headers, params=params)
        res.raise_for_status()
        data = res.json()
        if data['rt_cd'] != '0': return None
        output = data['output']
        return {
            "code": stock_code, "name": stock_name,
            "price": float(output.get('stck_prpr', 0)),
            "eps": float(output.get('eps', 0)),
        }
    except: return None
# =============================================================================
# [NEW] 시장 분석: KOSPI 추세 확인 (FinanceDataReader)
# =============================================================================
def check_market_trend():
    """
    KOSPI 지수의 60일 이동평균선 여부를 확인합니다.
    Return: (is_bull_market, message)
    """
    try:
        # KOSPI 지수 (심볼 'KS11')
        df = fdr.DataReader('KS11', '2023-01-01') # 넉넉하게 조회
        if len(df) < 60: return True, "데이터 부족"
        
        recent_close = df['Close'].iloc[-1]
        ma60 = df['Close'].rolling(window=60).mean().iloc[-1]
        
        if recent_close < ma60:
            return False, "하락장(Bear)"
        else:
            return True, "상승장(Bull)"
    except:
        return True, "시장 분석 실패"

# =============================================================================
# [NEW] 기술적 분석: RSI & 20일 이평선 (KIS API)
# =============================================================================
def calculate_rsi(prices, period=14):
    """
    가격 리스트(최신순 아님, 시간순이어야 함)를 받아 RSI를 계산합니다.
    """
    delta = pd.Series(prices).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] # 최근 RSI 리턴

def get_technical_indicators(stock_code, access_token):
    """
    최근 60일치 일봉 데이터를 가져와 MA20 및 RSI(14)를 계산합니다.
    Return: (ma20_price, is_bull_trend, rsi_value)
    """
    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
    headers = {
        "content-type": "application/json", "authorization": f"Bearer {access_token}",
        "appkey": APP_KEY, "appsecret": APP_SECRET, "tr_id": "FHKST01010400"
    }
    # 넉넉하게 60일치 요청
    params = {
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": stock_code,
        "fid_period_div_code": "D",
        "fid_org_adj_prc": "1"
    }
    try:
        res = requests.get(url, headers=headers, params=params)
        res.raise_for_status()
        data = res.json()
        if data['rt_cd'] != '0': return None, False, 50.0
        
        # 일봉 리스트 (API는 최신순으로 줌 -> 시간순으로 뒤집어야 RSI 계산 편함)
        daily_prices_desc = [float(x['stck_clpr']) for x in data['output']]
        daily_prices_asc = daily_prices_desc[::-1] # 시간순 정렬
        
        if len(daily_prices_desc) < 20:
            return None, False, 50.0
            
        # 1. MA20 계산 (최신 20일)
        ma20 = sum(daily_prices_desc[:20]) / 20.0
        current_price = daily_prices_desc[0]
        is_bull = current_price >= ma20
        
        # 2. RSI 계산 (판다스 활용)
        rsi_val = 50.0
        if len(daily_prices_asc) > 15:
            rsi_val = calculate_rsi(daily_prices_asc)
            if pd.isna(rsi_val): rsi_val = 50.0
            
        return ma20, is_bull, rsi_val
        
    except Exception as e:
        return None, False, 50.0

# =============================================================================
# [NEW] 수급 분석: KIS API 활용 (크롤링 X -> API O)
# =============================================================================
def get_supply_score(stock_code, access_token): # token이 필요합니다!
    """
    KIS API를 통해 최근 5일간 외국인/기관 순매수 추이를 분석합니다.
    """
    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-investor"
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "FHKST01010900" # 투자자별 매매동향(일별) TR ID
    }
    params = {
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": stock_code
    }
    
    try:
        res = requests.get(url, headers=headers, params=params)
        res.raise_for_status()
        data = res.json()
        
        if data['rt_cd'] != '0': return 0, "API오류"
        
        # 최근 5일치 데이터만 확인
        daily_data = data['output'][:5]
        
        inst_buy_count = 0
        for_buy_count = 0
        
        for row in daily_data:
            # 외국인 순매수 (prsn_ntby_qty: 개인, frgn: 외국인)
            # API 응답 필드 확인 필요. 보통 frgn_ntby_qty
            if int(row.get('frgn_ntby_qty', 0)) > 0:
                for_buy_count += 1
            # 기관 순매수 (orgn_ntby_qty)
            if int(row.get('orgn_ntby_qty', 0)) > 0:
                inst_buy_count += 1
                
        score = 0
        msg_parts = []
        
        if for_buy_count >= 3:
            score += 1
            msg_parts.append(f"외인{for_buy_count}일")
            
        if inst_buy_count >= 3:
            score += 1
            msg_parts.append(f"기관{inst_buy_count}일")
            
        return score, "/".join(msg_parts) if msg_parts else "수급약함"
        
    except Exception as e:
        return 0, "에러"

# =============================================================================
# [NEW] 실적 모멘텀: 이익 추정치 상향 여부 (Naver)
# =============================================================================
def get_earnings_momentum(stock_code):
    """
    현재 EPS 추정치가 1개월 전/3개월 전보다 상향되었는지 확인합니다.
    Return: (is_improving, message)
    """
    try:
        url = f"https://finance.naver.com/item/main.naver?code={stock_code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        text = requests.get(url, headers=headers).text
        dfs = pd.read_html(io.StringIO(text), encoding='euc-kr')
        
        # '투자의견 목표주가' 테이블 찾기 (보통 dfs[3] 근처)
        trend_df = None
        for df in dfs:
            if '현재' in str(df.columns) and '1개월전' in str(df.columns):
                trend_df = df
                break
                
        if trend_df is None: return False, "데이터 없음"
        
        # 인덱스 설정 (EPS, PER 등이 인덱스로 옴)
        trend_df = trend_df.set_index(trend_df.columns[0])
        
        # 'EPS' 포함된 행 찾기
        target_row = None
        for idx in trend_df.index:
            if 'EPS' in str(idx):
                target_row = idx
                break
                
        if target_row:
            try:
                current_eps = float(str(trend_df.loc[target_row, '현재']).replace(',',''))
                month_ago_eps = float(str(trend_df.loc[target_row, '1개월전']).replace(',',''))
                
                # [판단 로직] 현재 추정치가 1개월 전보다 높으면 모멘텀 있음
                if current_eps > month_ago_eps:
                    return True, "이익전망 상향중"
                else:
                    return False, "이익전망 하향/횡보"
            except:
                return False, "데이터 오류"
                
        return False, "EPS 데이터 없음"
        
    except Exception as e:
        return False, "분석 실패"

# =============================================================================
# [NEW] 재무 데이터: KIS API (분기별 실적)
# =============================================================================
def get_quarterly_financials_from_api(stock_code, access_token):
    """
    한국투자증권 API로 최근 4분기 재무 데이터를 조회합니다.
    """
    url = f"{BASE_URL}/uapi/domestic-stock/v1/finance/financial-ratio"
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "FHKST66430200"
    }
    params = {
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": stock_code
    }
    
    try:
        res = requests.get(url, headers=headers, params=params)
        res.raise_for_status()
        data = res.json()
        
        if data['rt_cd'] != '0':
            return None
            
        output = data.get('output', [])
        
        result = {
            'quarters': [],     # 결산년월
            'eps': [],          # EPS
            'sales': [],        # 매출액
            'op_profit': [],    # 영업이익
            'roe': [],          # ROE
            'per': [],          # PER (추가)
            'pbr': []           # PBR (추가)
        }
        
        if isinstance(output, list):
            # 0번째가 최신 데이터 (User Spec)
            for item in output[:4]:
                result['quarters'].append(item.get('stac_yymm'))
                result['eps'].append(item.get('eps'))
                result['sales'].append(item.get('sale_account'))
                result['op_profit'].append(item.get('op_prfi'))
                result['roe'].append(item.get('roe_val'))
                result['per'].append(item.get('per'))
                result['pbr'].append(item.get('pbr'))
                
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"  [API 오류] 네트워크 에러: {e}")
        return None
    except KeyError as e:
        print(f"  [API 오류] 응답 구조 오류: {e}")
        return None
    except Exception as e:
        print(f"  [API 오류] 예상치 못한 오류: {e}")
        print(f"  [API 오류] 응답 내용: {res.text[:200] if 'res' in locals() else '응답 없음'}")
        return None

def get_consensus_from_api(stock_code, access_token):
    """
    증권사 투자의견 컨센서스를 조회합니다.
    """
    url = f"{BASE_URL}/uapi/domestic-stock/v1/finance/invest-opinion"
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "FHKST66430300"
    }
    params = {
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": stock_code
    }
    
    try:
        res = requests.get(url, headers=headers, params=params)
        res.raise_for_status()
        data = res.json()
        
        if data['rt_cd'] != '0':
            return None
            
        output = data.get('output', [])
        if not output:
             return None
             
        # 결과가 리스트인 경우 첫 번째 항목 사용 (또는 단일 딕셔너리)
        item = output[0] if isinstance(output, list) else output
        
        # 안전한 형변환 헬퍼
        def to_float(val):
            try: return float(val)
            except: return 0.0
            
        def to_int(val):
            try: return int(val)
            except: return 0

        result = {
            'target_price': to_float(item.get('stck_nttp')),
            'current_year_eps': to_float(item.get('stck_fcam_tr_pbnt')),
            'next_year_eps': to_float(item.get('stck_fcam_ntby_pbnt')),
            'analyst_count': to_int(item.get('hval_cnst_co_shtn')),
            'buy_count': to_int(item.get('invt_opnn_cls_code_1')),
            'hold_count': to_int(item.get('invt_opnn_cls_code_2'))
        }
        
        return result

    except Exception as e:
        return None

def get_roe_from_api(stock_code, access_token):
    """
    최근 4분기 평균 ROE를 계산합니다.
    Return: float 또는 None
    """
    try:
        quarterly_data = get_quarterly_financials_from_api(stock_code, access_token)
        if quarterly_data is None or 'roe' not in quarterly_data:
            return None
            
        roe_list = quarterly_data['roe']
        valid_roes = []
        
        for r in roe_list:
            try:
                # API 데이터가 문자열일 수 있으므로 변환
                if r:
                    val = float(str(r).replace(',', ''))
                    if val > 0:
                        valid_roes.append(val)
            except:
                continue
                
        if not valid_roes:
            return None
            
        return float(np.mean(valid_roes))
        
    except Exception as e:
        return None

def analyze_eps_trend(quarterly_data):
    """
    분기 EPS 추세를 분석하여 점수와 메시지를 반환합니다.
    Return: (score: int, message: str)
    """
    try:
        if quarterly_data is None or 'eps' not in quarterly_data:
            return 0, "데이터 부족"
            
        # 데이터 정제 및 형변환
        raw_eps = quarterly_data['eps'][:4]
        eps_list = []
        for e in raw_eps:
            try:
                # 문자열 제거 및 공백 처리
                clean_e = str(e).replace(',', '').strip()
                if clean_e and clean_e != '-':
                    eps_list.append(float(clean_e))
                else:
                    eps_list.append(0.0)
            except:
                eps_list.append(0.0)

        # 0이 아닌 유효 데이터만 사용하는 것이 좋으나, 연속성을 위해 포함하되
        # 데이터가 너무 적으면 분석 불가 처리
        if len(eps_list) < 3:
            return 0, "데이터 부족"

        # 시간순 정렬 (Oldest -> Latest) : API는 보통 최신순으로 줌
        eps_chrono = eps_list[::-1]

        # 선형 회귀 기울기 계산
        x = np.arange(len(eps_chrono))
        slope = np.polyfit(x, eps_chrono, 1)[0]
        avg_eps = np.mean(eps_chrono)

        # 평균 EPS가 너무 작거나 음수면 트렌드 강도 계산 왜곡됨 -> 예외처리 필요하지만
        # user 로직 따름: avg_eps > 0 일때만 나눔
        trend_strength = (slope / avg_eps) if avg_eps > 0 else 0

        # QoQ 성장률 계산
        qoq_list = []
        for i in range(1, len(eps_chrono)):
            prev = eps_chrono[i-1]
            curr = eps_chrono[i]
            # 분모가 0이거나 매우 작을 때 처리
            if abs(prev) > 1: # 1원 미만이면 성장률 의미 없음
                qoq = (curr - prev) / abs(prev)
                qoq_list.append(qoq)
        
        avg_qoq = np.mean(qoq_list) if qoq_list else 0.0

        # 점수 부여
        msg = ""
        score = 0
        qoq_pct = avg_qoq * 100
        
        if trend_strength > 0.10:
            score = 80
            msg = f"강한 성장세 (QoQ +{qoq_pct:.1f}%)"
        elif trend_strength > 0.05:
            score = 50
            msg = f"성장세 (QoQ +{qoq_pct:.1f}%)"
        elif trend_strength > -0.05:
            score = 0
            msg = "횡보"
        elif trend_strength > -0.10:
            score = -50
            msg = f"둔화 (QoQ {qoq_pct:.1f}%)"
        else:
            score = -80
            msg = f"실적 악화 (QoQ {qoq_pct:.1f}%)"
            
        return score, msg
        
    except Exception as e:
        return 0, "분석 실패"

def predict_eps_smart(stock_code, stock_name, current_eps, access_token):
    """
    네이버 컨센서스를 우선 활용한 EPS 예측
    """
    try:
        # 1. 네이버에서 컨센서스 EPS 가져오기
        naver_eps, _, _ = get_naver_financial_info(stock_code, stock_name)
        
        # 2. 신뢰도 평가
        if naver_eps and naver_eps > 0:
            # 네이버 컨센서스가 있으면 사용
            
            # 현재 EPS와 비교하여 신뢰도 결정
            if current_eps > 0:
                deviation = abs(naver_eps - current_eps) / current_eps
                
                if deviation < 0.2:  # 20% 이내 차이
                    confidence = 80
                    message = "네이버 컨센서스 (신뢰도 높음)"
                    return (naver_eps, confidence, message)
                
                elif deviation < 0.5:  # 50% 이내 차이
                    # 혼합 사용
                    blended = (naver_eps * 0.6) + (current_eps * 0.4)
                    confidence = 65
                    message = "네이버 60% + 현재 40% 혼합"
                    return (blended, confidence, message)
                
                else:  # 50% 이상 차이 (의심스러움)
                    # 보수적으로 현재 EPS 사용
                    confidence = 45
                    message = f"편차 과대({deviation*100:.0f}%) → 현재 EPS 사용"
                    return (current_eps, confidence, message)
            
            else:
                # 현재 EPS가 0이거나 음수면 네이버 컨센서스 신뢰
                confidence = 70
                message = "네이버 컨센서스 채택"
                return (naver_eps, confidence, message)
        
        else:
            # 네이버 컨센서스가 없으면 현재 EPS 사용
            confidence = 50
            message = "현재 EPS 유지 (컨센서스 없음)"
            return (current_eps, confidence, message)
            
    except Exception as e:
        print(f"  [예측 오류] {e}")
        return (current_eps, 40, "예측 실패 - 현재 EPS 사용")

def calculate_target_per_advanced(stock_code, stock_name, base_per, access_token):
    """
    ROE와 업종 특성을 반영하여 목표 PER을 계산합니다.
    Return: float (최종 목표 PER)
    """
    try:
        # 1. ROE 조회 및 가중치 적용
        roe = get_roe_from_api(stock_code, access_token)
        adjusted_per = base_per
        
        if roe:
            if roe >= 20:
                adjusted_per = adjusted_per * 1.3
                print(f"  [ROE] {roe:.1f}% → PER +30%")
            elif roe >= 15:
                adjusted_per = adjusted_per * 1.15
                print(f"  [ROE] {roe:.1f}% → PER +15%")
            elif roe >= 10:
                pass  # 변화 없음
            elif roe < 8:
                adjusted_per = adjusted_per * 0.85
                print(f"  [ROE] {roe:.1f}% → PER -15%")
        
        # 2. 업종 상한선 적용 (보수적 밸류에이션)
        sector_caps = {
            '반도체': 18, '전자': 18, 'SK하이닉스': 18, '삼성전자': 18,
            '자동차': 10, '현대차': 10, '기아': 10,
            '은행': 7, '금융': 7, 'KB': 7, '신한': 7,
            '통신': 9, 'KT': 9, 'SK텔레콤': 9,
            '바이오': 35, '셀트리온': 35, '제약': 35,
            '게임': 20, '엔씨': 20, '크래프톤': 20,
            'IT': 25, 'NAVER': 25, '카카오': 25,
            '화학': 12, 'LG화학': 12
        }
        
        # 종목명에서 키워드 매칭
        for keyword, cap in sector_caps.items():
            if keyword in stock_name:
                if adjusted_per > cap:
                    print(f"  [업종 상한] {keyword} PER {adjusted_per:.1f} → {cap}")
                    adjusted_per = cap
                break
                
        return adjusted_per
        
    except Exception as e:
        print(f"  [PER 계산 오류] {e}")
        return base_per

# =============================================================================
# [3] 네이버 크롤링 & 스마트 PER
# =============================================================================
def get_naver_financial_info(stock_code, stock_name=""):
    try:
        url = f"https://finance.naver.com/item/main.naver?code={stock_code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        # [수정] 인코딩 안정성 강화
        res = requests.get(url, headers=headers)
        try:
            text = res.content.decode('euc-kr')
        except UnicodeDecodeError:
            text = res.content.decode('euc-kr', 'replace')
            
        dfs = pd.read_html(io.StringIO(text))
        
        # 기업실적분석 표 찾기 (동적 로직)
        fin_df = None
        for df in dfs:
            if not df.empty:
                # 첫번째 컬럼에 'EPS(원)'이 포함되어 있는지 확인
                col_vals = df.iloc[:, 0].astype(str).values
                if any('EPS(원)' in val for val in col_vals):
                    fin_df = df
                    break
        
        # 못 찾았으면 fallback
        if fin_df is None: 
            fin_df = dfs[4] if len(dfs)>4 else (dfs[3] if len(dfs)>3 else None)
            
        if fin_df is None: return None, 12.0, 12.0

        fin_df = fin_df.set_index(fin_df.columns[0])
        
        # 1. 예상 EPS
        target_col = None
        for col in fin_df.columns:
            if 'E' in str(col): target_col = col; break
        
        consensus_eps = None
        if target_col:
            try:
                # 인덱스 이름에 'EPS(원)' 포함된 행 찾기
                eps_idx = [idx for idx in fin_df.index if 'EPS(원)' in str(idx)][0]
                val = fin_df.loc[eps_idx, target_col]
                if pd.notna(val): consensus_eps = float(val)
            except: pass

        # 2. 과거 PER 가중평균
        per_history = []
        try:
            per_idx = [idx for idx in fin_df.index if 'PER(배)' in str(idx)][0]
            
            # [금융공학] 이상치 제거 (Trimmed Data) 설정
            # 기본적으로 PER 50배 넘으면 '비정상' 데이터로 간주하고 제거
            outlier_threshold = 50.0
            if '바이오' in stock_name or '셀트리온' in stock_name:
                outlier_threshold = 100.0 # 바이오는 100배까지 인정

            # 최근 4개년도 확인
            for col in fin_df.columns[:4]:
                val = fin_df.loc[per_idx, col]
                if pd.notna(val):
                    if isinstance(val, str): val = float(val.replace(',',''))
                    # 음수거나, 이상치(Threshold)를 넘어가면 제외
                    if val > 0 and val <= outlier_threshold:
                        per_history.append(float(val))
        except: pass
        
        my_hist_per = 12.0
        if len(per_history) >= 1:
            # [금융공학] 평균(Average) -> 중간값(Median)
            # 이상치가 있어도 중간값은 흔들리지 않음
            my_hist_per = np.median(per_history)

        # 3. 업종 PER
        sector_per = my_hist_per
        for df in dfs:
            if '동일업종 PER' in str(df) or (not df.empty and '동일업종 PER' in str(df.columns)):
                try:
                    # 보통 (1,1) 또는 (0,1) 위치에 있음, 구조에 따라 다름
                   if df.shape[1] > 1:
                        val = df.iloc[0, 1]
                        if isinstance(val, str): val = float(val.replace('배','').replace(',',''))
                        sector_per = val
                        break
                except: pass

        return consensus_eps, my_hist_per, sector_per
    except Exception as e: 
        return None, 12.0, 12.0

def get_fair_value_chart_figure(df):
    """
    Streamlit용 차트 Figure 객체를 반환합니다.
    """
    try:
        # [수정] koreanize_matplotlib 사용으로 복잡한 폰트 설정 제거
        plt.rcParams['axes.unicode_minus'] = False 
        
        # 2. 데이터 준비 (상위 10개만)
        chart_df = df.head(10).copy()
        
        names = chart_df['종목명'].tolist()
        current_prices = chart_df['현재가'].tolist()
        fair_values = chart_df['적정주가'].tolist()
        
        # 3. 그래프 그리기
        x = np.arange(len(names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, current_prices, width, label='현재가', color='gray')
        rects2 = ax.bar(x + width/2, fair_values, width, label='적정주가', color='red', alpha=0.7)
        
        # 4. 꾸미기
        ax.set_ylabel('주가 (원)')
        ax.set_title('저평가 우량주 Top 10 분석 (현재가 vs 적정주가)')
        ax.set_xticks(x)
        # 한글 폰트 문제로 깨질 수 있으니 Streamlit에서는 차라리 영어로 하거나... 
        # 일단 그대로 둠
        ax.set_xticklabels(names, rotation=15)
        ax.legend()
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        print(f"[오류] 차트 생성 실패: {e}")
        return None

def send_telegram_message(message):
    """ 텍스트 메시지를 보냅니다. """
    bot_token = "8297423754:AAHiYrE2XenVrBBwbQ_azWZmX0VI4abZOaA"
    chat_id = "34839919"
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {'chat_id': chat_id, 'text': message}
        requests.post(url, data=data)
    except Exception as e:
        print(f"[텔레그램 오류] {e}")

def send_telegram_photo(photo_path):
    """ 저장된 차트 이미지를 보냅니다. """
    bot_token = "8297423754:AAHiYrE2XenVrBBwbQ_azWZmX0VI4abZOaA"
    chat_id = "34839919"
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        with open(photo_path, 'rb') as f:
            requests.post(url, data={'chat_id': chat_id}, files={'photo': f})
    except Exception as e:
        print(f"[이미지 전송 오류] {e}")

def analyze_stock_item(code, name, token, is_bull_market):
    """
    개별 종목을 분석하여 결과 딕셔너리를 반환합니다.
    조건에 맞지 않으면 None을 반환합니다.
    """
    try:
        stock_info = get_stock_data(code, name, token)
        if not stock_info: return None

        # [NEW] 퀀트 3박자 분석 + 기술적 지표(RSI)
        ma20, is_bull_trend, rsi = get_technical_indicators(code, token)
        supply_score, supply_msg = get_supply_score(code, token) 
        is_improving, mom_msg = get_earnings_momentum(code)

        # [1] 스마트 EPS 예측 (API 기반)
        predicted_eps, eps_confidence, eps_msg = predict_eps_smart(
            code, name, stock_info['eps'], token
        )

        # [신뢰도 필터] 30점 미만이면 제외
        if eps_confidence < 30:
            return None

        if predicted_eps <= 0: 
            return None

        # [2] 기본 PER (네이버 백업)
        _, my_hist_per, sector_per = get_naver_financial_info(code, name)
        used_sector_per = sector_per if sector_per > 0 else my_hist_per
        base_per = (my_hist_per * 0.6) + (used_sector_per * 0.4)

        # [3] 동적 PER 계산 (ROE + 업종 반영)
        final_target_per = calculate_target_per_advanced(
            code, name, base_per, token
        )
        
        # [안전장치] 목표 PER가 비정상적으로 높으면 '제외(Skip)' 합니다.
        limit_per = 30.0
        if '바이오' in name or '셀트리온' in name:
            limit_per = 60.0
            
        if final_target_per > limit_per:
            return None
        
        target_price = predicted_eps * final_target_per
        price = stock_info['price']
        
        upside = 0
        if price > 0:
            upside = ((target_price - price) / price) * 100

        # [NEW] 시장 상황(KOSPI) 반영: 하락장일 경우 기준 상향
        if not is_bull_market:
            if upside < 40: 
                return None

        # 기본 의견 (Valuation) + 수급 점수 반영
        if upside >= 30 and supply_score >= 2 and rsi < 70:
            signal = "Strong Buy (★★★)"
        elif upside >= 30:
            signal = "Strong Buy (★)"
        elif upside >= 15: 
            signal = "Buy"
        elif upside >= 0: 
            signal = "Hold"
        else: 
            signal = "Sell"

        # [NEW] 기술적 필터 (Timing & RSI)
        if not is_bull_trend:
            if rsi < 30: 
                signal = "Buy (과매도)" 
            elif "Buy" in signal:
                signal = "Hold (하락세)"
        
        # 2. 과열 구간 (RSI > 70)이면 매수 보류
        if rsi > 70 and "Buy" in signal:
            signal = "Wait (과열)"
            
        return {
            "종목명": name,
            "현재가": int(price),
            "적정주가": int(target_price),
            "괴리율(%)": round(upside, 2),
            "의견": signal,
            "수급": supply_msg,
            "RSI": round(rsi, 1),
            "EPS신뢰도": int(eps_confidence),
            "목표PER": round(final_target_per, 2),
            "발굴점수": int(eps_confidence) * (upside / 100)
        }
    except Exception as e:
        return None

# =============================================================================
# Streamlit App Logic
# =============================================================================

# 페이지 설정
st.set_page_config(
    page_title="Korea Stock Fair Value Analyzer",
    page_icon="📈",
    layout="wide"
)

# 제목 및 설명
st.title("📈 Korea Stock Fair Value Analyzer")
st.markdown("""
**AI 기반 한국 주식 적정주가 분석기**입니다.
KIS API와 네이버 금융 데이터를 활용하여 저평가 우량주를 발굴합니다.
""")

# 사이드바 설정
st.sidebar.header("설정 (Configuration)")
stock_count = st.sidebar.number_input(
    "분석할 종목 수 (Top N)", 
    min_value=10, 
    max_value=500, 
    value=50, 
    step=10,
    help="시가총액 상위 N개 종목을 분석합니다."
)

# 실행 버튼
if st.button("🚀 분석 시작 (Start Analysis)"):
    
    # 1. 초기화 및 준비
    status_text = st.empty()
    progress_bar = st.progress(0)
    result_area = st.container()
    
    try:
        # 1-1. 시장 추세 확인
        status_text.text("📡 시장 추세(Market Trend)를 분석 중입니다...")
        is_bull_market, market_msg = check_market_trend()
        
        if is_bull_market:
            st.success(f"시장 상황: {market_msg} (상승장)")
        else:
            st.warning(f"시장 상황: {market_msg} (하락장 - 보수적 기준 적용)")
            
        # 1-2. 토큰 발급
        status_text.text("🔑 API 토큰을 발급받고 있습니다...")
        token = get_access_token()
        if not token:
            st.error("API 토큰 발급 실패. 앱 키/시크릿을 확인하세요.")
            st.stop()
            
        # 1-3. 종목 리스트 확보
        status_text.text(f"📋 시가총액 상위 {stock_count}개 종목을 가져오는 중...")
        stock_list = get_top_stocks(limit=stock_count)
        
        if not stock_list:
            st.error("종목 리스트를 가져오지 못했습니다.")
            st.stop()
            
        # 2. 분석 루프
        results = []
        total_stocks = len(stock_list)
        
        for i, (code, name) in enumerate(stock_list):
            # 진행상황 업데이트
            progress = (i + 1) / total_stocks
            progress_bar.progress(progress)
            status_text.text(f"🔍 [{i+1}/{total_stocks}] {name} ({code}) 분석 중...")
            
            # 개별 종목 분석
            result = analyze_stock_item(code, name, token, is_bull_market)
            
            if result:
                results.append(result)
            
            # API 제한 고려 (0.8초 대기)
            time.sleep(0.5) 
            
        # 3. 결과 처리
        status_text.text("✅ 분석 완료! 결과를 정리하고 있습니다...")
        progress_bar.progress(1.0)
        
        if results:
            df = pd.DataFrame(results)
            
            # 정렬 (발굴점수 기준)
            df_sorted = df.sort_values(by="발굴점수", ascending=False)
            
            # 주요 컬럼만 선택
            display_cols = ["종목명", "현재가", "적정주가", "괴리율(%)", "의견", "수급", "RSI", "EPS신뢰도", "목표PER", "발굴점수"]
            final_df = df_sorted[display_cols].reset_index(drop=True)
            
            # 3-1. 결과 테이블 표시
            st.subheader(f"🏆 분석 결과 (Top {len(final_df)})")
            
            # 스타일링
            st.dataframe(
                final_df.style.format({
                    "현재가": "{:,}원",
                    "적정주가": "{:,}원",
                    "괴리율(%)": "{:.2f}%",
                    "RSI": "{:.1f}",
                    "목표PER": "{:.2f}",
                    "발굴점수": "{:.1f}"
                }).background_gradient(subset=['괴리율(%)'], cmap="Reds"),
                use_container_width=True
            )
            
            # 3-2. 차트 표시
            st.subheader("📊 Top 10 시각화")
            fig = get_fair_value_chart_figure(final_df)
            if fig:
                st.pyplot(fig)
            else:
                st.warning("차트를 생성할 수 없습니다.")
                
            # 3-3. CSV 다운로드 버튼
            csv = final_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="💾 결과 CSV 다운로드",
                data=csv,
                file_name="korea_value_stocks_web.csv",
                mime="text/csv",
            )
            
        else:
            st.warning("조건에 맞는 저평가 우량주를 찾지 못했습니다.")
            
    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
