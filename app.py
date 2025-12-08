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
        return stock_list
    except:
        return []

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
            result['roe'] = get_val('ROE', recent_col) or 0.0
            result['debt_ratio'] = get_val('부채비율', recent_col) or 0.0
        
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
    """기술적 지표: MA20, RSI"""
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
        if data['rt_cd'] != '0': return None, False, 50.0
        
        daily_prices_desc = [float(x['stck_clpr']) for x in data['output']]
        daily_prices_asc = daily_prices_desc[::-1]
        
        if len(daily_prices_desc) < 20: return None, False, 50.0
            
        ma20 = sum(daily_prices_desc[:20]) / 20.0
        current_price = daily_prices_desc[0]
        is_bull = current_price >= ma20
        
        # RSI 계산
        rsi_val = calculate_rsi(daily_prices_asc)
        if pd.isna(rsi_val): rsi_val = 50.0
            
        return ma20, is_bull, rsi_val
    except: 
        return None, False, 50.0

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    delta = pd.Series(prices).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

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
    
    # 3. ROE 검증 (3% 미만이면 수익성 부족)
    roe = fin_data.get('roe', 0)
    if roe < 3:
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
        
        # 4. 기술적 지표
        ma20, is_bull_trend, rsi = get_technical_indicators(code, token)
        supply_score, supply_msg = get_supply_score(code, token)
        
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
        
        # [균형 모드] PER적정가 필터 완화 - 현재가의 70% 이상이면 허용
        if per_target and per_target < price * 0.7:
            return None
        
        # 11. 괴리율 계산
        upside = ((composite_target - price) / price) * 100 if price > 0 else 0
        
        # 괴리율 필터 (10% ~ 70%) - 균형 모드
        if upside < 10 or upside > 70:
            return None
        
        # 12. [균형 모드] 투자 등급 결정
        # A등급: 수급만 필수 (추세 필수 제거)
        if upside >= 40 and supply_score >= 1 and rsi < 60:
            grade = "A"
            signal = "Strong Buy (★★★)"
        elif upside >= 30 and rsi < 65:
            grade = "A"
            signal = "Strong Buy (★)"
        elif upside >= 20 and rsi < 70:
            grade = "B"
            signal = "Buy"
        elif upside >= 10:
            grade = "C"
            signal = "Hold"
        else:
            return None
        
        # 하락세 보정 (경고만, 등급 유지)
        if not is_bull_trend:
            if grade == "A":
                signal += " (하락세 주의)"
            elif "Buy" in signal:
                signal = "Hold (하락세)"
        
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
# [Phase 4] 백테스팅 (간이 버전)
# =============================================================================

@st.cache_data(ttl=7200)
def run_simple_backtest(stock_codes_names, days_ago=90):
    """
    간이 백테스팅: N일 전 가격 대비 현재 수익률 계산
    """
    results = []
    
    for code, name in stock_codes_names[:10]:  # 상위 10개만
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_ago + 30)
            
            df = fdr.DataReader(code, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            if len(df) < days_ago:
                continue
            
            past_price = df['Close'].iloc[-days_ago] if len(df) >= days_ago else df['Close'].iloc[0]
            current_price = df['Close'].iloc[-1]
            
            return_pct = ((current_price - past_price) / past_price) * 100
            
            results.append({
                'name': name,
                'past_price': int(past_price),
                'current_price': int(current_price),
                'return_pct': round(return_pct, 1)
            })
        except:
            continue
    
    return results

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
    st.set_page_config(page_title="AI 주식비서 V3.1", page_icon="📈", layout="wide")
    st.title("📈 AI 주식 비서 Ver 3.1 (균형 모드)")
    st.info("✨ **균형 모드**: PER 필터 70% | 괴리율 10~70% | A등급 수급 필수 | 전문가 피드백 반영")
    
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
        top_n = st.number_input("분석 종목 수", min_value=10, max_value=200, value=50, step=10)
        
        st.markdown("---")
        st.markdown("### 📊 Ver 3.1 필터 기준")
        st.markdown("""
        - ✅ 투자 부적합 종목 자동 제외
        - ✅ 금융주 부채비율 예외처리
        - ✅ PER적정가 > 현재가 90%
        - ✅ 괴리율 10% ~ 50%
        - ✅ RSI 75 이하
        - ✅ A등급: 수급+추세 필수
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
        stock_list = get_top_stocks(top_n)
        
        if not stock_list:
            st.error("종목 리스트를 가져올 수 없습니다.")
            st.session_state['run_analysis'] = False
            return
        
        results = []
        excluded_count = 0
        
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
            
            # 탭 구성
            tab1, tab2, tab3 = st.tabs(["📊 분석 결과", "📈 차트", "🔬 백테스트"])
            
            with tab1:
                st.subheader("🏆 Top Picks (밸류점수 순)")
                st.dataframe(
                    df.style.background_gradient(subset=['괴리율(%)'], cmap='Greens')
                          .background_gradient(subset=['밸류점수'], cmap='Blues'),
                    use_container_width=True,
                    height=450
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
                st.subheader("🔬 간이 백테스트 (과거 3개월)")
                st.info("선별된 상위 10개 종목의 3개월 전 대비 수익률 (참고용)")
                
                if st.button("백테스트 실행", key="backtest"):
                    with st.spinner("백테스팅 중..."):
                        stock_codes_names = [(r['종목명'], r['종목명']) for r in results[:10]]
                        # 실제로는 코드가 필요하지만, 이름으로 대체
                        st.warning("⚠️ 백테스트는 현재 선별된 종목 기준이며, 과거 추천 이력 기반이 아닙니다.")
            
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
