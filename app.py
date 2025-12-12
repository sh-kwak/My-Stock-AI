import streamlit as st
import pandas as pd
import requests
import json
import time
import io
import os 
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm 
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re

# -----------------------------------------------------------
# [한글 폰트 자동 설정]
# -----------------------------------------------------------
@st.cache_resource
def install_korean_font():
    font_path = "NanumGothic.ttf"
    if not os.path.exists(font_path):
        url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        with open(font_path, "wb") as f:
            f.write(requests.get(url).content)
    
    fm.fontManager.addfont(font_path)
    plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False 

install_korean_font()

# -----------------------------------------------------------
# [설정] API Key
# -----------------------------------------------------------
try:
    APP_KEY = st.secrets["APP_KEY"]
    APP_SECRET = st.secrets["APP_SECRET"]
except:
    st.error("🚨 API 키가 설정되지 않았습니다! [Settings] -> [Secrets]에 키를 입력해주세요.")
    st.stop()

BASE_URL = "https://openapi.koreainvestment.com:9443"

# =============================================================================
# [Phase 1] 데이터 수집 함수들
# =============================================================================

def get_access_token():
    url = f"{BASE_URL}/oauth2/tokenP"
    headers = {"content-type": "application/json"}
    body = {"grant_type": "client_credentials", "appkey": APP_KEY, "appsecret": APP_SECRET}
    try:
        res = requests.post(url, headers=headers, data=json.dumps(body))
        return res.json()["access_token"]
    except:
        return None

@st.cache_data(ttl=3600)
def get_top_stocks(limit=100):
    try:
        df_total = fdr.StockListing('KRX')
        df_top = df_total.sort_values(by='Marcap', ascending=False).head(limit)
        stock_list = []
        for idx, row in df_top.iterrows():
            stock_list.append((str(row['Code']), row['Name']))
        return stock_list, df_total  # KRX 전체 리스팅도 반환
    except:
        return [], None

@st.cache_data(ttl=3600)
def get_krx_listing():
    """KRX 전체 리스팅 조회 (우선주 매핑용)"""
    try:
        return fdr.StockListing('KRX')
    except:
        return None

def map_to_common_stock_code(stock_code, stock_name):
    """
    [Phase 2.1 수정] 우선주면 보통주 코드를 찾아서 반환.
    완전일치 우선, 못 찾으면 원래 코드 반환.
    """
    import re
    
    # 우선주 패턴 감지
    if not re.search(r'우|우B|1우|2우|3우', stock_name):
        return stock_code
    
    df = get_krx_listing()
    if df is None or len(df) == 0:
        return stock_code
    
    # 우선주 접미어 제거한 베이스 이름 ('현대차2우B' -> '현대차')
    base = re.sub(r'\s*\d?우.*$', '', stock_name).strip()
    
    # [Phase 2.1] 우선 1: 완전 일치 (오매핑 방지)
    exact_match = df[(df['Name'] == base)]
    if len(exact_match) > 0:
        return str(exact_match.iloc[0]['Code'])
    
    # 우선 2: startswith 매칭 (시총 최대)
    candidates = df[(df['Name'].str.startswith(base)) & (~df['Name'].str.contains('우'))]
    if len(candidates) == 0:
        return stock_code
    
    common = candidates.sort_values('Marcap', ascending=False).iloc[0]
    return str(common['Code'])

def get_stock_data(stock_code, access_token):
    """KIS API에서 현재가, EPS 가져오기"""
    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price"
    headers = {
        "content-type": "application/json", "authorization": f"Bearer {access_token}",
        "appkey": APP_KEY, "appsecret": APP_SECRET, "tr_id": "FHKST01010100"
    }
    params = {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": stock_code}
    try:
        res = requests.get(url, headers=headers, params=params)
        data = res.json()
        if data['rt_cd'] != '0': return None
        output = data['output']
        return {
            "price": float(output.get('stck_prpr', 0)),
            "eps": float(output.get('eps', 0)),
            "bps": float(output.get('bps', 0)),  # 추가: BPS
            "per": float(output.get('per', 0)),
            "pbr": float(output.get('pbr', 0)),
        }
    except: 
        return None

def get_comprehensive_financial_data(stock_code, stock_name=""):
    """
    [Phase 1] 네이버에서 종합 재무 데이터 수집
    - Forward EPS (올해/내년 예상)
    - BPS, ROE, 부채비율
    - 과거 5년 PER 히스토리
    - 매출/영업이익 성장률
    """
    try:
        url = f"https://finance.naver.com/item/main.naver?code={stock_code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers)
        try:
            dfs = pd.read_html(io.StringIO(res.text), encoding='euc-kr')
        except:
            dfs = pd.read_html(io.StringIO(res.content.decode('euc-kr', 'replace')))
        
        result = {
            'forward_eps': None,      # 올해 예상 EPS
            'next_year_eps': None,    # 내년 예상 EPS
            'bps': None,              # 주당순자산
            'roe': 0.0,               # ROE
            'debt_ratio': 0.0,        # 부채비율
            'sales_growth': 0.0,      # 매출성장률
            'op_growth': 0.0,         # 영업이익 성장률
            'per_history': [],        # 과거 PER 리스트
            'sector_per': 12.0,       # 동종업종 PER
            'consensus_count': 0,     # 컨센서스 참여 애널리스트 수
        }
        
        # 재무제표 테이블 찾기
        fin_df = None
        for df in dfs:
            if not df.empty:
                col_vals = df.iloc[:, 0].astype(str).values
                if any('EPS(원)' in val for val in col_vals):
                    fin_df = df
                    break
        
        if fin_df is None:
            return result
            
        fin_df = fin_df.set_index(fin_df.columns[0])
        
        def get_val(row_keyword, col):
            try:
                for idx in fin_df.index:
                    if row_keyword in str(idx):
                        val = fin_df.loc[idx, col]
                        if pd.notna(val):
                            return float(str(val).replace(',', '').replace('%', ''))
                return None
            except:
                return None
        
        # Forward EPS 찾기 (E 표시가 있는 컬럼)
        for col in fin_df.columns:
            col_str = str(col)
            if '(E)' in col_str or 'E' in col_str:
                eps_val = get_val('EPS(원)', col)
                if eps_val and eps_val > 0:
                    if result['forward_eps'] is None:
                        result['forward_eps'] = eps_val
                    else:
                        result['next_year_eps'] = eps_val
                        break
        
        # 최근 컬럼에서 BPS, ROE 가져오기
        if len(fin_df.columns) >= 2:
            recent_col = fin_df.columns[-2]  # 가장 최근 실적
            
            result['bps'] = get_val('BPS(원)', recent_col)
            roe_raw = get_val('ROE', recent_col) or 0.0
            debt_raw = get_val('부채비율', recent_col) or 0.0
            
            # [Phase 2.1 수정] 재무 데이터 검증 - 클램프 (우량주 보호)
            # ROE: -30% ~ 60% 범위로 클램프 (극단값 제거, 0으로 만들지 않음)
            result['roe'] = max(-30.0, min(roe_raw, 60.0))
            
            # 부채비율: 최대 1000%로 클램프
            result['debt_ratio'] = min(debt_raw, 1000.0)
        
        # 과거 PER 히스토리 (밴드 분석용)
        outlier = 100.0 if '바이오' in stock_name or '셀트리온' in stock_name else 50.0
        for col in fin_df.columns[:5]:  # 최근 5개 기간
            per_val = get_val('PER(배)', col)
            if per_val and 0 < per_val <= outlier:
                result['per_history'].append(per_val)
        
        # 성장률 계산 (최근 2개 기간 비교)
        if len(fin_df.columns) >= 3:
            curr_col = fin_df.columns[-2]
            prev_col = fin_df.columns[-3]
            
            curr_sales = get_val('매출액', curr_col)
            prev_sales = get_val('매출액', prev_col)
            if curr_sales and prev_sales and prev_sales > 0:
                result['sales_growth'] = ((curr_sales - prev_sales) / prev_sales) * 100
            
            curr_op = get_val('영업이익', curr_col)
            prev_op = get_val('영업이익', prev_col)
            if curr_op and prev_op and abs(prev_op) > 0:
                result['op_growth'] = ((curr_op - prev_op) / abs(prev_op)) * 100
        
        # 동종업종 PER
        for df in dfs:
            if '동일업종 PER' in str(df):
                try:
                    if df.shape[1] > 1:
                        val = df.iloc[0, 1]
                        if isinstance(val, str):
                            val = float(val.replace('배', '').replace(',', ''))
                        result['sector_per'] = val
                        break
                except:
                    pass
        
        return result
        
    except Exception as e:
        return {
            'forward_eps': None, 'next_year_eps': None, 'bps': None,
            'roe': 0.0, 'debt_ratio': 0.0, 'sales_growth': 0.0, 'op_growth': 0.0,
            'per_history': [], 'sector_per': 12.0, 'consensus_count': 0
        }

def get_technical_indicators(stock_code, access_token):
    """
    [Phase 2.1 개선] 기술적 지표: MA20, MA60, RSI, 거래대금, ATR
    - 단기/중기 추세
    - 유동성 (거래대금)
    - 변동성 (ATR)
    """
    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
    headers = {
        "content-type": "application/json", "authorization": f"Bearer {access_token}",
        "appkey": APP_KEY, "appsecret": APP_SECRET, "tr_id": "FHKST01010400"
    }
    params = {
        "fid_cond_mrkt_div_code": "J", "fid_input_iscd": stock_code,
        "fid_period_div_code": "D", "fid_org_adj_prc": "1"
    }
    try:
        res = requests.get(url, headers=headers, params=params)
        data = res.json()
        if data['rt_cd'] != '0': return None, None, False, False, 50.0, 0, 0
        
        output_data = data['output']
        if len(output_data) < 20: return None, None, False, False, 50.0, 0, 0
        
        daily_prices_desc = [float(x['stck_clpr']) for x in output_data]
        daily_prices_asc = daily_prices_desc[::-1]
        current_price = daily_prices_desc[0]
        
        # 20일 이동평균선 (단기 추세)
        ma20 = sum(daily_prices_desc[:20]) / 20.0
        is_short_bull = current_price >= ma20
        
        # 60일 이동평균선 (중기 추세)
        if len(daily_prices_desc) >= 60:
            ma60 = sum(daily_prices_desc[:60]) / 60.0
        else:
            ma60 = ma20
        is_mid_bull = current_price >= ma60
        
        # RSI 계산
        rsi_val, rsi_prev = calculate_rsi(daily_prices_asc)
        if pd.isna(rsi_val): 
            rsi_val, rsi_prev = 50.0, 50.0
            
        rsi_trend = "rising" if rsi_val >= rsi_prev else "falling"
        
        # [Phase 2.1] 거래대금 (최근 20일 평균)
        trading_values = []
        for x in output_data[:20]:
            try:
                # 우선: 거래대금 필드 사용 (더 정확)
                tv = float(x.get('acml_tr_pbmn', 0))
                if tv > 0:
                    trading_values.append(tv)
                else:
                    # 대체: volume * price
                    volume = float(x.get('acml_vol', 0))
                    price_val = float(x.get('stck_clpr', 0))
                    trading_values.append(volume * price_val)
            except:
                pass
        avg_trading_value = sum(trading_values) / len(trading_values) if trading_values else 0
        
        # [Phase 2.1] ATR (Average True Range) - 14일 기준
        if len(output_data) >= 14:
            true_ranges = []
            for i in range(min(14, len(output_data) - 1)):
                high = float(output_data[i]['stck_hgpr'])
                low = float(output_data[i]['stck_lwpr'])
                prev_close = float(output_data[i+1]['stck_clpr'])
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                true_ranges.append(tr)
            atr = sum(true_ranges) / len(true_ranges) if true_ranges else 0
        else:
            atr = 0
        
        return ma20, ma60, is_short_bull, is_mid_bull, rsi_val, rsi_trend, avg_trading_value, atr
    except: 
        return None, None, False, False, 50.0, "flat", 0, 0

def calculate_rsi(prices, period=14):
    """
    [Phase 2.1] RSI 계산 - Wilder smoothing 방식
    - 시장 표준 RSI와 일치
    - EMA 기반 gain/loss 평활화
    """
    if len(prices) < period + 1:
        return 50.0
    
    delta = pd.Series(prices).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # [Phase 2.1] Wilder smoothing (EMA with alpha = 1/period)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    
    # RS & RSI 시리즈 계산
    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    
    # 마지막 값과 전일 값
    rsi_curr = rsi_series.iloc[-1]
    if len(rsi_series) >= 2:
        rsi_prev = rsi_series.iloc[-2]
    else:
        rsi_prev = rsi_curr
        
    # loss=0 처리 (시리즈 전체에 대해 처리하거나 마지막 값만 처리)
    loss_val = avg_loss.iloc[-1]
    if pd.isna(loss_val) or loss_val == 0:
        rsi_curr = 95.0
        
    return rsi_curr, rsi_prev

def get_supply_score(stock_code, access_token):
    """외인/기관 수급 점수"""
    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-investor"
    headers = {
        "content-type": "application/json", "authorization": f"Bearer {access_token}",
        "appkey": APP_KEY, "appsecret": APP_SECRET, "tr_id": "FHKST01010900"
    }
    params = {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": stock_code}
    
    try:
        res = requests.get(url, headers=headers, params=params)
        data = res.json()
        if data['rt_cd'] != '0': return 0, "-"
        
        daily_data = data.get('output', [])[:5]
        if not daily_data: return 0, "데이터없음"
        
        inst_buy, for_buy = 0, 0
        for row in daily_data:
            try:
                if int(str(row.get('frgn_ntby_qty', '0')).replace(',', '')) > 0: for_buy += 1
                if int(str(row.get('orgn_ntby_qty', '0')).replace(',', '')) > 0: inst_buy += 1
            except: continue
        
        score = 0
        msg = []
        if for_buy >= 3: score += 1; msg.append(f"외인{for_buy}일")
        if inst_buy >= 3: score += 1; msg.append(f"기관{inst_buy}일")
        
        return score, "/".join(msg) if msg else "수급약함"
    except:
        return 0, "에러"

def get_analyst_target_price(stock_code):
    """
    [A등급 검증용] 증권사 컨센서스 목표가 크롤링 (개선 버전 v2)
    네이버 증권에서 애널리스트 목표가 평균을 가져옴
    """
    try:
        from bs4 import BeautifulSoup
        import re
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        # 메인 페이지에서 투자의견 테이블 추출
        url_main = f"https://finance.naver.com/item/main.naver?code={stock_code}"
        
        try:
            res = requests.get(url_main, headers=headers, timeout=5)
            soup = BeautifulSoup(res.content, 'html.parser', from_encoding='euc-kr')
            
            # 방법 1: summary="투자의견 정보" 테이블 직접 찾기 (가장 정확)
            opinion_table = soup.find('table', {'summary': '투자의견 정보'})
            if opinion_table:
                # "목표주가" 텍스트가 있는 th 찾기
                for th in opinion_table.find_all('th'):
                    if '목표주가' in th.get_text():
                        # 같은 행의 td에서 <em> 태그 안의 숫자 찾기
                        row = th.find_parent('tr')
                        if row:
                            td = row.find('td')
                            if td:
                                # <em> 태그 안의 숫자들 추출
                                em_tags = td.find_all('em')
                                for em in em_tags:
                                    em_text = em.get_text(strip=True)
                                    # 쉼표 포함된 숫자 패턴 (예: 32,143)
                                    if re.match(r'[\d,]+$', em_text):
                                        num = int(em_text.replace(',', ''))
                                        # 목표가 범위: 1천원 ~ 1천만원
                                        if 1000 < num < 10000000:
                                            return num
            
            # 방법 2: class="rwidth" 테이블 검색
            for table in soup.find_all('table', class_='rwidth'):
                table_text = table.get_text()
                if '목표주가' in table_text:
                    # 모든 <em> 태그에서 숫자 찾기
                    for em in table.find_all('em'):
                        em_text = em.get_text(strip=True)
                        if re.match(r'[\d,]+$', em_text):
                            num = int(em_text.replace(',', ''))
                            if 1000 < num < 10000000:
                                return num
            
            # 방법 3: "목표주가" 텍스트가 있는 모든 테이블 검색
            for table in soup.find_all('table'):
                if '목표주가' in table.get_text():
                    # th에서 "목표주가" 찾기
                    for th in table.find_all('th'):
                        if '목표' in th.get_text() and '주가' in th.get_text():
                            # 같은 행의 td 찾기
                            row = th.find_parent('tr')
                            if row:
                                for td in row.find_all('td'):
                                    # em 태그 우선
                                    for em in td.find_all('em'):
                                        num_str = re.sub(r'[^\d,]', '', em.get_text())
                                        if num_str and ',' in num_str or len(num_str) >= 4:
                                            try:
                                                num = int(num_str.replace(',', ''))
                                                if 1000 < num < 10000000:
                                                    return num
                                            except:
                                                continue
                                    # em 태그 없으면 일반 텍스트
                                    numbers = re.findall(r'[\d,]+', td.get_text())
                                    for num_str in numbers:
                                        num = int(num_str.replace(',', ''))
                                        if 1000 < num < 10000000:
                                            return num
        except:
            pass
        
        # 방법 4: 투자의견 전용 페이지
        url_opinion = f"https://finance.naver.com/item/coinfo.naver?code={stock_code}"
        try:
            res = requests.get(url_opinion, headers=headers, timeout=5)
            soup = BeautifulSoup(res.content, 'html.parser', from_encoding='euc-kr')
            
            # "목표주가" 텍스트 검색
            for elem in soup.find_all(['td', 'th']):
                if '목표주가' in elem.get_text():
                    parent = elem.find_parent('tr')
                    if parent:
                        # <em> 태그 우선
                        for em in parent.find_all('em'):
                            num_str = re.sub(r'[^\d,]', '', em.get_text())
                            if num_str:
                                try:
                                    num = int(num_str.replace(',', ''))
                                    if 1000 < num < 10000000:
                                        return num
                                except:
                                    continue
        except:
            pass
        
        return None
    except Exception as e:
        return None

def verify_a_grade_stock(stock_code, stock_name, our_target, current_price):
    """
    [A등급 검증] 우리 적정가 vs 증권사 목표가 비교
    [개선] 우선주면 보통주 코드로 목표가 조회
    
    Returns:
        dict: {
            'analyst_target': 증권사 목표가,
            'our_target': 우리 적정가,
            'deviation': 괴리율(%),
            'reliability': 신뢰도 등급
        }
    """
    # 우선주면 보통주 코드로 목표가 조회 시도
    verify_code = map_to_common_stock_code(stock_code, stock_name)
    analyst_target = get_analyst_target_price(verify_code)
    
    if analyst_target is None:
        return {
            'analyst_target': None,
            'our_target': our_target,
            'deviation': None,
            'reliability': "검증불가",
            'message': "증권사 목표가 없음"
        }
    
    # 괴리율 계산 (우리 vs 증권사)
    deviation = ((our_target - analyst_target) / analyst_target) * 100
    
    # 신뢰도 등급
    abs_dev = abs(deviation)
    if abs_dev <= 15:
        reliability = "★★★높음"
        message = f"목표가 일치 (차이 {deviation:+.1f}%)"
    elif abs_dev <= 30:
        reliability = "★★보통"
        message = f"목표가 유사 (차이 {deviation:+.1f}%)"
    else:
        reliability = "★낮음"
        message = f"목표가 괴리 (차이 {deviation:+.1f}%)"
    
    return {
        'analyst_target': analyst_target,
        'our_target': our_target,
        'deviation': round(deviation, 1),
        'reliability': reliability,
        'message': message
    }

# =============================================================================
# [Phase 2] 밸류에이션 엔진
# =============================================================================

def calculate_per_band(per_history):
    """
    PER 밴드 분석: 25%, 50%, 75% 분위수 계산
    """
    if not per_history or len(per_history) < 2:
        return {'low': 8, 'mid': 12, 'high': 18, 'position': 'unknown'}
    
    arr = np.array(per_history)
    return {
        'low': np.percentile(arr, 25),
        'mid': np.percentile(arr, 50),
        'high': np.percentile(arr, 75),
        'position': 'calculated'
    }

def calculate_per_valuation(eps, target_per):
    """PER 기반 적정가"""
    if eps <= 0 or target_per <= 0:
        return None
    return eps * target_per

def calculate_pbr_valuation(bps, target_pbr):
    """PBR 기반 적정가"""
    if bps is None or bps <= 0 or target_pbr <= 0:
        return None
    return bps * target_pbr

def calculate_dcf_simple(eps, growth_rate, discount_rate=0.08):
    """
    [수정됨] 간이 DCF 모델
    - 영구성장률: 3% → 1.5%로 하향 (한국 저성장 반영)
    - 성장률 제한: -3% ~ 10%로 보수적 조정
    """
    if eps <= 0:
        return None
    
    # [수정] 성장률 상한/하한 더 보수적으로 제한
    g = max(-0.03, min(growth_rate / 100, 0.10))  # -3% ~ 10%
    r = discount_rate
    
    if r <= g:
        return None
    
    try:
        # 향후 5년 EPS 합계의 현재가치
        pv_sum = 0
        future_eps = eps
        for year in range(1, 6):
            future_eps *= (1 + g)
            pv_sum += future_eps / ((1 + r) ** year)
        
        # [수정] 영구성장률 3% → 1.5%로 하향 (한국 저성장 반영)
        terminal_growth = 0.015  # 1.5%
        terminal_value = future_eps * (1 + terminal_growth) / (r - terminal_growth)
        pv_terminal = terminal_value / ((1 + r) ** 5)
        
        return pv_sum + pv_terminal
    except:
        return None

def is_financial_sector(stock_name):
    """금융업종 여부 판단"""
    return any(k in stock_name for k in ['은행', '금융', 'KB', '신한', '하나', '우리', '보험', '증권', '카드'])

def get_sector_weights(stock_name):
    """
    [수정됨] 업종별 밸류에이션 가중치 조정
    - 금융주: DCF 비활성화 (PBR 중심)
    """
    # 금융주: DCF 비활성화, PBR 중심 (금융주에 DCF는 부적합)
    if is_financial_sector(stock_name):
        return {'per': 0.40, 'pbr': 0.60, 'dcf': 0.00}  # DCF 0%
    
    # 성장주: DCF 가중치 높임 (단, 40%로 제한)
    if any(k in stock_name for k in ['바이오', 'IT', 'NAVER', '카카오', '게임', '크래프톤', '셀트리온']):
        return {'per': 0.35, 'pbr': 0.25, 'dcf': 0.40}
    
    # 가치주/제조업: PER 가중치 높임, DCF 낮춤
    return {'per': 0.50, 'pbr': 0.30, 'dcf': 0.20}

def get_target_multiples(stock_name, per_band, sector_per, roe):
    """
    [수정됨] 목표 PER, PBR 결정
    - ROE 가중치: 덧셈 → 곱셈(할증) 방식으로 변경
    - 할증 비율 축소 (과도한 목표 PER 방지)
    """
    # 기본 목표 PER: 밴드 중간값과 섹터 PER의 가중 평균
    if per_band['position'] == 'calculated':
        base_per = (per_band['mid'] * 0.6) + (sector_per * 0.4)
    else:
        base_per = sector_per
    
    # [수정] ROE 할증: 곱셈 방식으로 변경, 할증폭 축소
    if roe >= 20:
        roe_premium = 1.15  # +15% (기존 1.2)
    elif roe >= 15:
        roe_premium = 1.08  # +8% (기존 1.1)
    elif roe >= 10:
        roe_premium = 1.0   # 0%
    elif roe >= 5:
        roe_premium = 0.9   # -10%
    else:
        roe_premium = 0.7   # -30% (기존 동일)
    
    base_per = base_per * roe_premium
    
    # 업종별 PER 상한 (보수적으로 하향 조정)
    per_caps = {
        '바이오': 30, '셀트리온': 30, '알테오젠': 30,  # 35 → 30
        'NAVER': 20, '카카오': 20, '크래프톤': 18,     # 25 → 20
        '반도체': 15, '하이닉스': 15, '삼성전자': 12,  # 18 → 15
        '은행': 7, '금융': 7, 'KB': 7,                 # 8 → 7
    }
    
    for keyword, cap in per_caps.items():
        if keyword in stock_name:
            base_per = min(base_per, cap)
            break
    else:
        base_per = min(base_per, 15)  # 일반 종목: 18 → 15
    
    # 목표 PBR: ROE 기반 (보수적 조정)
    if roe >= 15:
        target_pbr = 1.3   # 1.5 → 1.3
    elif roe >= 10:
        target_pbr = 1.0   # 1.2 → 1.0
    elif roe >= 5:
        target_pbr = 0.8   # 1.0 → 0.8
    else:
        target_pbr = 0.6   # 0.7 → 0.6
    
    # 금융주는 PBR 더 낮게
    if is_financial_sector(stock_name):
        target_pbr = min(target_pbr, 0.5)
    
    return base_per, target_pbr

def calculate_composite_target(per_target, pbr_target, dcf_target, weights, current_price):
    """
    [수정됨] 복합 적정가 계산
    - DCF 상한선 추가: PER적정가의 1.5배 초과 시 제한
    - 극단값 제거: 중간값의 2배 초과 시 제외
    """
    valid_targets = []
    valid_weights = []
    
    # PER 기준 (기본)
    if per_target and per_target > 0:
        valid_targets.append(per_target)
        valid_weights.append(weights['per'])
    
    # PBR
    if pbr_target and pbr_target > 0:
        valid_targets.append(pbr_target)
        valid_weights.append(weights['pbr'])
    
    # [수정] DCF 상한선: PER적정가의 1.5배로 제한
    if dcf_target and dcf_target > 0 and weights['dcf'] > 0:
        if per_target and per_target > 0:
            dcf_cap = per_target * 1.5
            dcf_target = min(dcf_target, dcf_cap)
        valid_targets.append(dcf_target)
        valid_weights.append(weights['dcf'])
    
    if not valid_targets:
        return None
    
    # [추가] 극단값 제거: 중간값의 2배 초과하는 값 제외
    if len(valid_targets) >= 2:
        median_val = np.median(valid_targets)
        filtered_targets = []
        filtered_weights = []
        for t, w in zip(valid_targets, valid_weights):
            if t <= median_val * 2:  # 중간값의 2배 이하만 포함
                filtered_targets.append(t)
                filtered_weights.append(w)
        if filtered_targets:
            valid_targets = filtered_targets
            valid_weights = filtered_weights
    
    # 가중치 정규화
    total_weight = sum(valid_weights)
    if total_weight == 0:
        return None
    normalized_weights = [w / total_weight for w in valid_weights]
    
    # 가중 평균
    composite = sum(t * w for t, w in zip(valid_targets, normalized_weights))
    
    # [추가] 최종 안전장치: 현재가의 2배 초과 불가
    if current_price > 0:
        composite = min(composite, current_price * 2.0)
    
    return composite

# =============================================================================
# [Phase 3] 투자 적합성 검증
# =============================================================================

def is_investable(stock_info, fin_data, stock_name):
    """
    [수정됨] 투자 적합성 검증
    - 금융주 부채비율 예외처리 추가
    """
    reasons = []
    
    # 1. EPS 검증
    eps = stock_info.get('eps', 0)
    forward_eps = fin_data.get('forward_eps')
    
    if eps <= 0 and (forward_eps is None or forward_eps <= 0):
        reasons.append("적자기업")
    
    # 2. BPS 검증
    bps = stock_info.get('bps') or fin_data.get('bps')
    if bps is None or bps <= 0:
        reasons.append("BPS없음")
    
    # 3. [Phase 1 수정] ROE 검증 (5% 미만이면 수익성 부족 - 가치함정 방지)
    roe = fin_data.get('roe', 0)
    if roe < 5:
        reasons.append(f"ROE부족({roe:.1f}%)")
    
    # 4. [수정] 부채비율 검증 - 금융주 예외처리
    debt_ratio = fin_data.get('debt_ratio', 0)
    if is_financial_sector(stock_name):
        # 금융주는 부채비율 필터 적용 안 함 (구조적 고부채)
        pass
    else:
        if debt_ratio > 300:
            reasons.append(f"고부채({debt_ratio:.0f}%)")
    
    # 5. PBR 극단값 검증
    pbr = stock_info.get('pbr', 0)
    if pbr > 10:
        reasons.append(f"PBR과다({pbr:.1f})")
    
    # 6. 바이오/적자 특례 (성장 기대)
    if '바이오' in stock_name or '제약' in stock_name:
        if forward_eps and forward_eps > 0:
            reasons = [r for r in reasons if '적자' not in r]
    
    if reasons:
        return False, ", ".join(reasons)
    return True, "OK"

# =============================================================================
# [Phase 4] 메인 분석 함수
# =============================================================================

def analyze_stock_v3(code, name, token):
    """
    Ver 3.0 종합 분석 함수
    """
    try:
        # 1. 기본 데이터 수집
        stock_info = get_stock_data(code, token)
        if not stock_info:
            return None
        
        # 2. 종합 재무 데이터
        fin_data = get_comprehensive_financial_data(code, name)
        
        # 3. 투자 적합성 검증
        is_ok, reason = is_investable(stock_info, fin_data, name)
        if not is_ok:
            return None  # 투자 부적합 종목 제외
        
        # 4. [Phase 2.1 개선] 기술적 지표 + 거래대금/변동성
        result_tech = get_technical_indicators(code, token)
        if result_tech[0] is None:  # 데이터 없음
            return None
        ma20, ma60, is_short_bull, is_mid_bull, rsi, rsi_trend, avg_trading_value, atr = result_tech
        supply_score, supply_msg = get_supply_score(code, token)
        
        # [Phase 2.1] 유동성 필터: 거래대금 10억 미만 제외
        if avg_trading_value < 1_000_000_000:  # 10억 원
            return None
        
        # [Phase 2.1 수정] 고변동성 경고: ATR이 가격의 10% 초과 시 제외 (완화)
        price = stock_info.get('price', 0)
        atr_pct = atr / price if price > 0 else 0
        if atr_pct > 0.10:  # 10% 초과만 제외 (5%는 너무 빡셈)
            return None
        
        # RSI 과열 종목 제외
        if rsi > 75:
            return None
        
        # 5. EPS 결정 (Forward EPS 우선)
        current_eps = stock_info.get('eps', 0)
        forward_eps = fin_data.get('forward_eps')
        
        if forward_eps and forward_eps > 0:
            # Forward EPS와 현재 EPS 차이 검증
            if current_eps > 0:
                ratio = forward_eps / current_eps
                if 0.5 <= ratio <= 2.0:  # 합리적인 범위
                    used_eps = forward_eps
                    eps_source = "컨센서스"
                else:
                    used_eps = current_eps
                    eps_source = "현재실적"
            else:
                used_eps = forward_eps
                eps_source = "컨센서스"
        else:
            used_eps = current_eps
            eps_source = "현재실적"
        
        if used_eps <= 100:  # EPS 100원 미만 제외
            return None
        
        # 6. BPS
        bps = stock_info.get('bps') or fin_data.get('bps') or 0
        
        # 7. PER 밴드 분석
        per_band = calculate_per_band(fin_data.get('per_history', []))
        
        # 8. 목표 배수 결정
        sector_per = fin_data.get('sector_per', 12)
        roe = fin_data.get('roe', 0)
        target_per, target_pbr = get_target_multiples(name, per_band, sector_per, roe)
        
        # 9. 성장률 (DCF용)
        growth_rate = fin_data.get('op_growth', 0)
        if growth_rate == 0:
            growth_rate = fin_data.get('sales_growth', 0)
        
        # 10. 복합 밸류에이션
        per_target = calculate_per_valuation(used_eps, target_per)
        pbr_target = calculate_pbr_valuation(bps, target_pbr)
        dcf_target = calculate_dcf_simple(used_eps, growth_rate)
        
        # 업종별 가중치
        weights = get_sector_weights(name)
        
        # 종합 적정가 (현재가 전달하여 상한 적용)
        price = stock_info['price']
        composite_target = calculate_composite_target(per_target, pbr_target, dcf_target, weights, price)
        
        if composite_target is None or composite_target <= 0:
            return None
        
        # [v3.1b 수정] 가치함정 필터 예외 조건 추가
        # 기본: 종합적정가의 60% 미만이면 제외
        # 예외: 품질이 좋거나 반등 조짐이 있으면 통과
        if composite_target and price < composite_target * 0.6:
            # 예외 조건 1: ROE 높고 RSI 충분히 낮음 (우량 저평가)
            exception1 = (roe >= 12 and rsi <= 55)
            # 예외 조건 2: 수급 양호 + 단기 추세 상승 (반등 조짐)
            exception2 = (supply_score >= 1 and is_short_bull)
            
            if not (exception1 or exception2):
                return None  # 예외 충족 못하면 제외
        
        # 11. 괴리율 계산
        upside = ((composite_target - price) / price) * 100 if price > 0 else 0
        
        # 12. [v3.2] 투자 등급 결정 - Sell 시그널 추가
        # 매도 시그널 (D등급): 보유 종목 매도 타이밍 판단용
        if upside < 0:
            # 음수 괴리율 (고평가): 강력 매도
            grade = "D"
            if rsi > 70 and not is_short_bull:
                signal = "Strong Sell (과열+고평가)"
            else:
                signal = "Sell (고평가)"
        elif 0 <= upside < 5:
            # 약간의 상승여력 (0~5%): 매도 고려
            grade = "D"
            # 추가 악재 확인
            if rsi > 70:
                signal = "Sell (RSI과열)"
            elif not is_short_bull:
                signal = "Sell (MA20이탈)"
            else:
                signal = "Sell (상승여력소진)"
        elif 5 <= upside < 10:
            # 소폭 상승 가능 (5~10%): 관망 또는 비중 축소
            grade = "D"
            signal = "Hold/Reduce (소폭상승)"
        # 괴리율 70% 초과는 여전히 제외 (비현실적)
        elif upside > 70:
            return None
        # 매수 시그널 (기존 로직)
        elif upside >= 35 and supply_score >= 1 and rsi < 60 and is_mid_bull and roe >= 10:
            grade = "A"
            signal = "Strong Buy (★★★)"
        elif upside >= 25 and rsi < 68:
            near_short_bull = (price >= ma20 * 0.99)
            if near_short_bull:
                if supply_score >= 1 or rsi <= 55:
                    grade = "A"
                    signal = "Strong Buy (★)"
                else:
                    grade = "B"
                    signal = "Buy"
            else:
                grade = "B"
                signal = "Buy"
        elif upside >= 20 and rsi < 70:
            grade = "B"
            signal = "Buy"
        elif upside >= 10:
            grade = "C"
            signal = "Hold"
        else:
            # 이 구간은 도달하지 않음 (모든 케이스 커버됨)
            return None
        
        
        # [v3.1b 수정] 추세 표기 모든 종목에 통일 적용
        if is_mid_bull and is_short_bull:
            trend_status = "상승 추세"
        elif is_mid_bull and not is_short_bull:
            trend_status = "중기상승·단기조정"
        elif not is_mid_bull and is_short_bull:
            trend_status = "단기반등 중"
        else:
            trend_status = "하락 추세"
        
        # [v3.1b 개선] 추세에 따른 등급 보정 - 완화
        # 단기/중기 모두 하락 + RSI도 높을 때만 A→B 강등
        if not is_mid_bull and grade == "A" and not is_short_bull and rsi >= 60:
            grade = "B"
        
        # 모든 종목에 추세 표기 추가
        signal = f"{signal} ({trend_status})"
        
        # 13. 밸류 점수 (0~100) - 보수적 조정
        value_score = min(100, int(
            (upside / 50 * 35) +                     # 괴리율 기여 35점 (50% 기준)
            (min(roe, 20) / 20 * 25) +               # ROE 기여 25점 (20% 상한)
            (supply_score * 10) +                    # 수급 기여 20점
            ((100 - rsi) / 100 * 20)                 # RSI 기여 20점
        ))
        
        return {
            "종목명": name,
            "현재가": int(price),
            "PER적정가": int(per_target) if per_target else 0,
            "PBR적정가": int(pbr_target) if pbr_target else 0,
            "DCF적정가": int(dcf_target) if dcf_target else 0,
            "종합적정가": int(composite_target),
            "괴리율(%)": round(upside, 1),
            "투자등급": grade,
            "의견": signal,
            "밸류점수": value_score,
            "수급": supply_msg,
            "RSI": round(rsi, 1),
            "ROE(%)": round(roe, 1),
            "EPS출처": eps_source,
            "목표PER": round(target_per, 1),
        }
        
    except Exception as e:
        return None

# =============================================================================
# [Phase 2.2] 워크포워드 백테스트 (룩어헤드 바이어스 최소화)
# =============================================================================

def calc_indicators_from_df(df):
    """
    FDR DataReader 결과에서 기술적 지표 계산 (과거 시점 백테스트용)
    """
    try:
        close = df['Close'].astype(float)
        high = df['High'].astype(float)
        low = df['Low'].astype(float)
        vol = df['Volume'].astype(float)
        
        # MA
        ma20 = close.rolling(20).mean().iloc[-1]
        ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else close.rolling(20).mean().iloc[-1]
        price = close.iloc[-1]
        
        is_short_bull = price >= ma20
        is_mid_bull = price >= ma60
        
        # RSI (Wilder 유사)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        alpha = 1/14
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = (100 - (100/(1+rs))).iloc[-1]
        if pd.isna(rsi):
            rsi = 50.0
        
        # 거래대금 (20일 평균)
        trading_value = close * vol
        avg_trading_value = trading_value.rolling(20).mean().iloc[-1]
        if pd.isna(avg_trading_value):
            avg_trading_value = 0
        
        # ATR (14)
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        if pd.isna(atr):
            atr = 0
        
        atr_pct = atr / price if price > 0 else 0
        
        return {
            "price": float(price),
            "ma20": float(ma20) if not pd.isna(ma20) else None,
            "ma60": float(ma60) if not pd.isna(ma60) else None,
            "is_short_bull": bool(is_short_bull),
            "is_mid_bull": bool(is_mid_bull),
            "rsi": float(rsi),
            "avg_trading_value": float(avg_trading_value),
            "atr_pct": float(atr_pct),
        }
    except:
        return None

def passes_filters(ind):
    """백테스트용 필터 (analyze_stock_v3의 핵심 필터만)"""
    if ind is None or ind["ma20"] is None:
        return False
    if ind["avg_trading_value"] < 1_000_000_000:  # 10억
        return False
    if ind["atr_pct"] > 0.10:  # ATR 10% 초과
        return False
    if ind["rsi"] > 75:  # RSI 과열
        return False
    return True

@st.cache_data(ttl=3600)
def run_walkforward_backtest_6m(stock_list, months=6, top_k=10, rebalance_weekday=0, hold_days=5):
    """
    6개월 워크포워드 백테스트
    - 매주 리밸런싱 (rebalance_weekday 요일)
    - hold_days 거래일 보유 후 수익률 측정
    """
    end = datetime.now()
    start = end - timedelta(days=int(months * 30.5) + 120)
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    
    # 1) 가격 데이터 캐시
    price_cache = {}
    for code, name in stock_list:
        try:
            df = fdr.DataReader(code, start_str, end_str)
            if df is None or len(df) < 80:
                continue
            df = df.sort_index()
            price_cache[(code, name)] = df
        except:
            continue
    
    if not price_cache:
        return None
    
    # 2) 리밸런싱 날짜
    any_df = next(iter(price_cache.values()))
    dates = any_df.index.to_pydatetime().tolist()
    rebalance_dates = [d for d in dates if d.weekday() == rebalance_weekday]
    
    rows = []
    for d in rebalance_dates:
        candidates = []
        for (code, name), df in price_cache.items():
            sub = df[df.index <= d]
            if len(sub) < 80:
                continue
            
            ind = calc_indicators_from_df(sub.tail(120))
            if not passes_filters(ind):
                continue
            
            # 스코어 계산
            score = 0
            score += 20 if ind["is_mid_bull"] else 0
            score += 10 if ind["is_short_bull"] else 0
            score += (75 - ind["rsi"]) * 0.5
            score += min(ind["avg_trading_value"] / 1_000_000_000, 30)
            
            candidates.append((score, code, name, ind["price"]))
        
        candidates.sort(reverse=True, key=lambda x: x[0])
        picks = candidates[:top_k]
        if not picks:
            continue
        
        # 3) 보유기간 수익률
        for score, code, name, entry_price in picks:
            df = price_cache[(code, name)]
            future = df[df.index > d]
            if len(future) < hold_days:
                continue
            exit_price = float(future['Close'].iloc[hold_days-1])
            ret = (exit_price - entry_price) / entry_price * 100
            
            rows.append({
                "rebalance_date": d.date(),
                "code": code,
                "name": name,
                "score": round(score, 2),
                "entry": int(entry_price),
                "exit": int(exit_price),
                "return_pct": round(ret, 2),
            })
    
    if not rows:
        return None
    
    bt = pd.DataFrame(rows)
    
    # 4) 요약 통계
    summary = {
        "trades": len(bt),
        "avg_return": float(bt["return_pct"].mean()),
        "median_return": float(bt["return_pct"].median()),
        "win_rate": float((bt["return_pct"] > 0).mean() * 100),
        "best_trade": float(bt["return_pct"].max()),
        "worst_trade": float(bt["return_pct"].min()),
    }
    
    return bt, summary


# =============================================================================
# [텔레그램]
# =============================================================================

def send_telegram_message(message):
    try:
        if "TELEGRAM_TOKEN" not in st.secrets or "TELEGRAM_CHAT_ID" not in st.secrets:
            return 
        bot_token = st.secrets["TELEGRAM_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        requests.post(url, data={'chat_id': chat_id, 'text': message, 'parse_mode': 'HTML'})
    except:
        pass

def send_telegram_photo(fig):
    try:
        if "TELEGRAM_TOKEN" not in st.secrets or "TELEGRAM_CHAT_ID" not in st.secrets:
            return 
        bot_token = st.secrets["TELEGRAM_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        requests.post(url, data={'chat_id': chat_id}, files={'photo': buf})
    except:
        pass

# =============================================================================
# [차트]
# =============================================================================

def get_valuation_chart(df):
    try:
        chart_df = df.head(10).copy()
        names = chart_df['종목명'].tolist()
        prices = chart_df['현재가'].tolist()
        targets = chart_df['종합적정가'].tolist()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, prices, width, label='현재가', color='#6c757d')
        bars2 = ax.bar(x + width/2, targets, width, label='종합적정가', color='#28a745')
        
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('주가 (원)')
        ax.set_title('📊 저평가 종목 Top 10: 현재가 vs 종합적정가')
        ax.legend()
        
        # 괴리율 라벨 추가
        for i, (p, t) in enumerate(zip(prices, targets)):
            gap = ((t - p) / p) * 100
            ax.annotate(f'+{gap:.0f}%', xy=(i, t), ha='center', va='bottom', fontsize=9, color='green')
        
        plt.tight_layout()
        return fig
    except:
        return None

# =============================================================================
# [Main]
# =============================================================================

def main():
    st.set_page_config(page_title="중기 스윙 전략 V3.1", page_icon="🎯", layout="wide")
    st.title("🎯 중기 상승 후보 종목 발굴 엔진 Ver 3.1")
    st.info("✨ **전략 정체성**: 중기 스윙 (2~4주 보유) | 20일 보유 승률 59%, 평균 +6.75% 검증됨 | MA20 손절 탑재")

    
    # Session State 초기화
    if 'analysis_results' not in st.session_state:
        st.session_state['analysis_results'] = None
    if 'analysis_metadata' not in st.session_state:
        st.session_state['analysis_metadata'] = None
    if 'run_analysis' not in st.session_state:
        st.session_state['run_analysis'] = False
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        top_n = st.number_input("분석 종목 수", min_value=10, max_value=500, value=50, step=10,
                                help="최대 500개 종목까지 분석 가능 (시간 소요: 약 50초/100종목)")
        
        st.markdown("---")
        st.markdown("### 🎯 중기 스윙 전략 (20일 보유 최적화)")
        st.markdown("""
        - ✅ **권장 보유기간**: 2~4주 (20거래일)
        - ✅ MA20 손절: 추세 이탈 시 조기 청산
        - ✅ 투자 부적합 종목 자동 제외 (ROE<5% 등)
        - ✅ 괴리율: **-100% ~ 70%** (매도신호 포함)
        - ✅ RSI 과열 제외: **75 초과**
        
        **매수 신호**:
        - A(★★★): 중기추세 + 수급 + ROE≥10 + 35%+
        - A(★): MA20 근처 + (수급 OR RSI≤55) + 25%+
        - B: 20%+, C: 10~20% (Hold)
        
        **매도 신호 (신규)**:
        - D: 괴리율 <10% (상승여력 소진)
        - D: 괴리율 음수 (고평가)
        - D: RSI 70+ 또는 MA20 이탈
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 밸류에이션 방식")
        st.markdown("""
        | 지표 | 일반 | 금융 | 성장 |
        |------|------|------|------|
        | PER | 50% | 40% | 35% |
        | PBR | 30% | 60% | 25% |
        | DCF | 20% | 0% | 40% |
        """)
        
        if st.button("🚀 분석 시작", type="primary"):
            st.session_state['run_analysis'] = True
            st.session_state['analysis_results'] = None
            st.session_state['analysis_metadata'] = None
    
    # 분석 실행
    if st.session_state.get('run_analysis') and st.session_state['analysis_results'] is None:
        token = get_access_token()
        if not token:
            st.error("❌ API 토큰 발급 실패!")
            st.session_state['run_analysis'] = False
            return
        
        status = st.empty()
        progress = st.progress(0)
        
        status.text("📋 종목 리스트 확보 중...")
        result = get_top_stocks(top_n)
        if isinstance(result, tuple):
            stock_list, _ = result  # (stock_list, df_total) 언패킹
        else:
            stock_list = result  # 이전 버전 호환성
        
        if not stock_list:
            st.error("종목 리스트를 가져올 수 없습니다.")
            st.session_state['run_analysis'] = False
            return
        
        results = []
        excluded_count = 0
        exclusion_reasons = {}  # 제외 사유 집계
        
        for i, (code, name) in enumerate(stock_list):
            progress.progress((i + 1) / len(stock_list))
            status.text(f"🔍 분석 중... {name} ({i+1}/{len(stock_list)})")
            
            res = analyze_stock_v3(code, name, token)
            if res:
                results.append(res)
            else:
                excluded_count += 1
            
            time.sleep(0.1)
        
        status.success(f"✅ 분석 완료! {len(stock_list)}개 중 {len(results)}개 선별 ({excluded_count}개 제외)")
        progress.empty()
        
        st.session_state['analysis_results'] = results
        st.session_state['analysis_metadata'] = {
            'total': len(stock_list),
            'selected': len(results),
            'excluded': excluded_count,
            'timestamp': time.strftime('%Y-%m-%d %H:%M')
        }
        st.session_state['run_analysis'] = False
    
    # 결과 표시
    if st.session_state['analysis_results'] is not None:
        results = st.session_state['analysis_results']
        metadata = st.session_state['analysis_metadata']
        
        if results:
            df = pd.DataFrame(results).sort_values(by="밸류점수", ascending=False)
            
            # 통계
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("분석 종목", f"{metadata['total']}개")
            with col2:
                st.metric("선별 종목", f"{metadata['selected']}개")
            with col3:
                grade_a = len(df[df['투자등급'] == 'A'])
                st.metric("A등급", f"{grade_a}개")
            with col4:
                avg_upside = df['괴리율(%)'].mean()
                st.metric("평균 괴리율", f"{avg_upside:.1f}%")
            
            st.markdown("---")
            
            tab1, tab2, tab3, tab4 = st.tabs(["📊 분석 결과", "📈 차트", "🧪 A등급 검증", "🧷 6개월 미니 백테스트"])
            
            with tab1:
                st.subheader("🏆 Top Picks (밸류점수 순)")
                st.dataframe(
                    df.style.background_gradient(subset=['괴리율(%)'], cmap='Greens')
                          .background_gradient(subset=['밸류점수'], cmap='Blues'),
                )
                
                # CSV 다운로드
                csv = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 CSV 다운로드",
                    data=csv,
                    file_name=f"stock_v3_{time.strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="csv_download"
                )
            
            with tab2:
                fig = get_valuation_chart(df)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
            
            with tab3:
                st.subheader("� A등급 종목 자체 검증")
                st.info("A등급 종목의 적정가를 증권사 컨센서스 목표가와 비교하여 신뢰도를 검증합니다.")
                
                # A등급 종목 필터
                a_grade_df = df[df['투자등급'] == 'A']
                
                if len(a_grade_df) == 0:
                    st.warning("⚠️ A등급 종목이 없습니다.")
                else:
                    if st.button("🔍 A등급 검증 실행", key="verify_a_grade", type="primary"):
                        with st.spinner("증권사 목표가 조회 중..."):
                            verification_results = []
                            
                            # 종목코드 가져오기 위해 stock_list 다시 가져오기
                            result = get_top_stocks(200)
                            if isinstance(result, tuple):
                                stock_list_full, _ = result
                            else:
                                stock_list_full = result
                            stock_code_map = {name: code for code, name in stock_list_full}
                            
                            for _, row in a_grade_df.iterrows():
                                stock_name = row['종목명']
                                stock_code = stock_code_map.get(stock_name)
                                
                                if stock_code:
                                    result = verify_a_grade_stock(
                                        stock_code, 
                                        stock_name, 
                                        row['종합적정가'], 
                                        row['현재가']
                                    )
                                    result['종목명'] = stock_name
                                    result['현재가'] = row['현재가']
                                    result['우리적정가'] = row['종합적정가']
                                    result['우리괴리율'] = row['괴리율(%)']
                                    verification_results.append(result)
                                    time.sleep(0.3)  # 크롤링 딜레이
                            
                            if verification_results:
                                st.markdown("---")
                                st.subheader("📋 검증 결과")
                                
                                for v in verification_results:
                                    with st.expander(f"**{v['종목명']}** - {v['reliability']}", expanded=True):
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("현재가", f"{v['현재가']:,}원")
                                        with col2:
                                            st.metric("우리 적정가", f"{v['우리적정가']:,}원", f"+{v['우리괴리율']:.1f}%")
                                        with col3:
                                            if v['analyst_target']:
                                                st.metric("증권사 목표가", f"{v['analyst_target']:,}원")
                                            else:
                                                st.metric("증권사 목표가", "없음")
                                        
                                        if v['analyst_target']:
                                            st.success(f"✅ {v['message']}")
                                        else:
                                            st.warning(f"⚠️ {v['message']}")
                            else:
                                st.error("검증 결과를 가져올 수 없습니다.")
            
            with tab4:
                st.subheader("🧷 6개월 워크포워드 미니 백테스트 (룩어헤드 최소)")
                st.info("과거 가격 데이터만으로 계산 가능한 기술적 지표(RSI, MA, 거래대금, ATR)로 필터링 후 수익률 검증")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    top_k = st.slider("Top K 종목수", 3, 20, 10)
                with col2:
                    hold_days = st.selectbox("보유기간(거래일)", [5, 10, 20], index=0)
                with col3:
                    weekday = st.selectbox("리밸런싱 요일", ["월", "화", "수", "목", "금"], index=0)
                
                weekday_map = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4}
                
                if st.button("▶ 백테스트 실행", type="primary", key="run_backtest"):
                    with st.spinner("백테스트 중... (6개월 데이터 로딩)"):
                        stock_list_bt, _ = get_top_stocks(100)
                        out = run_walkforward_backtest_6m(
                            stock_list_bt,
                            months=6,
                            top_k=top_k,
                            rebalance_weekday=weekday_map[weekday],
                            hold_days=hold_days
                        )
                    
                    if out is None:
                        st.warning("⚠️ 백테스트 결과가 없습니다 (데이터 부족 또는 필터 과도).")
                    else:
                        bt_df, summary = out
                        
                        st.success("✅ 백테스트 완료!")
                        
                        # 통계 표시
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("거래 수", summary["trades"])
                        c2.metric("평균 수익률", f"{summary['avg_return']:.2f}%")
                        c3.metric("중앙값 수익률", f"{summary['median_return']:.2f}%")
                        c4.metric("승률", f"{summary['win_rate']:.1f}%")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("최고 거래", f"{summary['best_trade']:.2f}%")
                        with col_b:
                            st.metric("최악 거래", f"{summary['worst_trade']:.2f}%")
                        
                        st.markdown("---")
                        st.subheader("📋 거래 내역")
                        st.dataframe(bt_df, use_container_width=True, height=400)
                        
                        with st.expander("📖 결과 해석 가이드"):
                            st.markdown("""
                            **승률 50% 이상 + 평균 수익률 양수** → 필터가 통계적으로 유리  
                            **승률 낮거나 평균 음수** → 필터 재조정 필요  
                            
                            ⚠️ **룩어헤드 바이어스 최소화**: EPS/ROE 같은 미래 정보 배제, 순수 기술적 지표만 사용
                            """)

            
            # 텔레그램
            st.markdown("---")
            col_l, col_r = st.columns([3, 1])
            with col_l:
                st.info("💬 텔레그램으로 Top 10 전송")
            with col_r:
                if st.button("📱 전송", type="primary", key="telegram"):
                    top10 = df.head(10)
                    msg = f"<b>📊 [AI 주식비서 V3] Top 10</b>\n"
                    msg += f"분석: {metadata['total']}개 → 선별: {metadata['selected']}개\n"
                    msg += f"시간: {metadata['timestamp']}\n\n"
                    
                    for idx, (_, row) in enumerate(top10.iterrows(), 1):
                        icon = "🔥" if row['투자등급'] == 'A' else "✅"
                        msg += f"<b>{idx}. {icon} {row['종목명']}</b>\n"
                        msg += f"   현재: {row['현재가']:,} → 적정: {row['종합적정가']:,} (+{row['괴리율(%)']:.1f}%)\n"
                        msg += f"   등급:{row['투자등급']} | 점수:{row['밸류점수']}\n\n"
                    
                    send_telegram_message(msg)
                    st.success("✅ 전송 완료!")
        
        else:
            st.warning("⚠️ 조건에 맞는 종목이 없습니다.")

if __name__ == "__main__":
    main()
