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
# [설정] API Key (Streamlit Secrets에서 가져옴)
# -----------------------------------------------------------
try:
    APP_KEY = st.secrets["APP_KEY"]
    APP_SECRET = st.secrets["APP_SECRET"]
except:
    st.error("🚨 API 키가 설정되지 않았습니다! [Settings] -> [Secrets]에 키를 입력해주세요.")
    st.stop()

BASE_URL = "https://openapi.koreainvestment.com:9443"

# =============================================================================
# [함수 모음]
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
        }
    except: return None

def get_quarterly_financials_from_naver(stock_code):
    try:
        url = f"https://finance.naver.com/item/main.naver?code={stock_code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers)
        try:
            dfs = pd.read_html(io.StringIO(res.text), encoding='euc-kr')
        except:
            dfs = pd.read_html(io.StringIO(res.content.decode('euc-kr', 'replace')))
        
        fin_df = None
        for df in dfs:
            if not df.empty:
                col_vals = df.iloc[:, 0].astype(str).values
                if any('EPS(원)' in val for val in col_vals):
                    fin_df = df
                    break
        
        if fin_df is None: return None
        
        fin_df = fin_df.set_index(fin_df.columns[0])
        
        quarter_cols = []
        for col in fin_df.columns:
            col_str = str(col)
            if '분기' in col_str or (len(col_str) > 5 and col_str[0] == '2'): 
                 quarter_cols.append(col)
        
        if len(quarter_cols) < 3:
            quarter_cols = fin_df.columns[-6:]
            
        result = {'eps': [], 'quarters': []}
        
        eps_row = None
        for idx in fin_df.index:
            if 'EPS' in str(idx):
                eps_row = idx
                break
                
        if eps_row:
            for col in quarter_cols:
                val = fin_df.loc[eps_row, col]
                if pd.notna(val):
                    try:
                        clean_val = float(str(val).replace(',', ''))
                        result['eps'].append(clean_val)
                        result['quarters'].append(str(col))
                    except: pass
        return result
    except:
        return None

def calculate_rsi(prices, period=14):
    delta = pd.Series(prices).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

def get_technical_indicators(stock_code, access_token):
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
        
        rsi_val = 50.0
        if len(daily_prices_asc) > 15:
            rsi_val = calculate_rsi(daily_prices_asc)
            if pd.isna(rsi_val): rsi_val = 50.0
            
        return ma20, is_bull, rsi_val
    except: return None, False, 50.0

def get_supply_score(stock_code, access_token):
    """
    KIS API를 통해 최근 5일간 외국인/기관 순매수 추이를 분석합니다.
    """
    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-investor"
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "FHKST01010900"
    }
    params = {
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": stock_code
    }
    
    try:
        res = requests.get(url, headers=headers, params=params)
        data = res.json()
        
        if data['rt_cd'] != '0': return 0, "-"
        
        daily_data = data.get('output', [])
        if not daily_data: return 0, "데이터없음"

        daily_data = daily_data[:5]
        
        inst_buy_count = 0
        for_buy_count = 0
        
        for row in daily_data:
            try:
                frgn_qty = int(str(row.get('frgn_ntby_qty', '0')).replace(',', ''))
                orgn_qty = int(str(row.get('orgn_ntby_qty', '0')).replace(',', ''))
                
                if frgn_qty > 0: for_buy_count += 1
                if orgn_qty > 0: inst_buy_count += 1
            except:
                continue
                
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

def analyze_eps_trend(quarterly_data):
    try:
        if not quarterly_data or not quarterly_data['eps']: return 0, "데이터 부족"
        eps_list = quarterly_data['eps']
        if len(eps_list) < 3: return 0, "데이터 부족"

        x = np.arange(len(eps_list))
        slope = np.polyfit(x, eps_list, 1)[0]
        avg_eps = np.mean(eps_list)
        trend_strength = (slope / avg_eps) if avg_eps > 0 else 0

        qoq_list = []
        for i in range(1, len(eps_list)):
            prev = eps_list[i-1]
            curr = eps_list[i]
            if abs(prev) > 1: qoq_list.append((curr - prev) / abs(prev))
        avg_qoq = np.mean(qoq_list) if qoq_list else 0.0
        qoq_pct = avg_qoq * 100

        score = 0
        if trend_strength > 0.1: score = 80; msg = f"강한 성장 (+{qoq_pct:.1f}%)"
        elif trend_strength > 0.05: score = 50; msg = f"성장세 (+{qoq_pct:.1f}%)"
        elif trend_strength > -0.05: score = 0; msg = "횡보"
        else: score = -50; msg = f"둔화 ({qoq_pct:.1f}%)"
        
        return score, msg
    except: return 0, "분석 실패"

def get_earnings_momentum(stock_code):
    try:
        url = f"https://finance.naver.com/item/main.naver?code={stock_code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers)
        try:
            dfs = pd.read_html(io.StringIO(res.text), encoding='euc-kr')
        except:
            dfs = pd.read_html(io.StringIO(res.content.decode('euc-kr', 'replace')))
        
        trend_df = None
        for df in dfs:
            if '현재' in str(df.columns) and '1개월전' in str(df.columns):
                trend_df = df; break
        
        if trend_df is None: return False, "데이터 없음"
        trend_df = trend_df.set_index(trend_df.columns[0])
        
        target_row = None
        for idx in trend_df.index:
            if 'EPS' in str(idx): target_row = idx; break
            
        if target_row:
            cur = float(str(trend_df.loc[target_row, '현재']).replace(',',''))
            prev = float(str(trend_df.loc[target_row, '1개월전']).replace(',',''))
            return (cur > prev), "이익전망 상향중" if cur > prev else "이익전망 하향/횡보"
            
        return False, "데이터 없음"
    except: return False, "분석 실패"

def get_naver_financial_info(stock_code, stock_name=""):
    try:
        url = f"https://finance.naver.com/item/main.naver?code={stock_code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers)
        try:
            dfs = pd.read_html(io.StringIO(res.text), encoding='euc-kr')
        except:
            dfs = pd.read_html(io.StringIO(res.content.decode('euc-kr', 'replace')))
        
        fin_df = None
        for df in dfs:
            if not df.empty:
                col_vals = df.iloc[:, 0].astype(str).values
                if any('EPS(원)' in val for val in col_vals):
                    fin_df = df
                    break
        
        if fin_df is None: fin_df = dfs[4] if len(dfs)>4 else (dfs[3] if len(dfs)>3 else None)
        if fin_df is None: return None, 12.0, 12.0, 0.0

        fin_df = fin_df.set_index(fin_df.columns[0])
        target_col = None
        for col in fin_df.columns:
            if 'E' in str(col): target_col = col; break
        recent_col = fin_df.columns[-2]

        def get_val(row_name, col):
            try:
                row = [idx for idx in fin_df.index if row_name in str(idx)][0]
                val = fin_df.loc[row, col]
                if pd.notna(val): return float(str(val).replace(',',''))
            except: pass
            return None

        consensus_eps = get_val('EPS(원)', target_col)
        roe_val = get_val('ROE', target_col) or get_val('ROE', recent_col) or 0.0

        per_history = []
        try:
            per_idx = [idx for idx in fin_df.index if 'PER(배)' in str(idx)][0]
            outlier = 100.0 if '바이오' in stock_name or '셀트리온' in stock_name else 50.0
            for col in fin_df.columns[:4]:
                v = get_val('PER(배)', col)
                if v and 0 < v <= outlier: per_history.append(v)
        except: pass
        
        my_hist_per = np.median(per_history) if per_history else 12.0

        sector_per = my_hist_per
        for df in dfs:
            if '동일업종 PER' in str(df):
                try:
                    if df.shape[1] > 1:
                        val = df.iloc[0, 1]
                        if isinstance(val, str): val = float(val.replace('배','').replace(',',''))
                        sector_per = val; break
                except: pass

        return consensus_eps, my_hist_per, sector_per, roe_val
    except: return None, 12.0, 12.0, 0.0

def predict_eps_smart(stock_code, stock_name, current_eps, access_token):
    """
    [수정됨] 네이버 컨센서스 검증 강화
    """
    try:
        quarterly_data = get_quarterly_financials_from_naver(stock_code)
        naver_eps, _, _, _ = get_naver_financial_info(stock_code, stock_name)
        trend_score, trend_msg = analyze_eps_trend(quarterly_data)
        
        # [추가] 네이버 EPS 검증 - 현재 EPS와 10배 이상 차이나면 무시
        if naver_eps and current_eps > 0:
            eps_ratio = naver_eps / current_eps
            if eps_ratio < 0.1 or eps_ratio > 5.0:
                naver_eps = None  # 극단적 차이는 신뢰 안 함
        
        if naver_eps and naver_eps > 0:
            if current_eps > 0:
                deviation = abs(naver_eps - current_eps) / current_eps
                if deviation < 0.2: 
                    return naver_eps, 80, "네이버 컨센서스 (신뢰도 높음)"
                elif deviation < 0.5:
                    blended = (naver_eps * 0.6) + (current_eps * 0.4)
                    return blended, 65, "네이버 60% + 현재 40% 혼합"
                else:
                    if trend_score > 50:
                        return current_eps * 1.1, 55, "편차 과대 → 현재실적+성장세 반영"
                    else:
                        return current_eps, 45, "편차 과대 → 현재실적 사용"
            else:
                return naver_eps, 70, "적자탈출 예상 (컨센서스 채택)"
        
        return current_eps, 50, "컨센서스 없음 (현재실적 유지)"
    except: return current_eps, 40, "예측 오류"

def calculate_target_per_advanced(stock_code, stock_name, base_per, access_token):
    """
    [수정됨] 업종별 상한선 상향 조정
    """
    sector_caps = {
        '반도체': 20, 'SK하이닉스': 20, '삼성전자': 18, '전자': 20,
        '자동차': 12, '현대차': 12, '기아': 12,
        '은행': 8, '금융': 8, 'KB': 8, '신한': 8,
        '바이오': 50, '셀트리온': 50, '알테오젠': 50,
        'IT': 30, 'NAVER': 30, '카카오': 30,
        '게임': 25, '크래프톤': 25, '넷마블': 25,
        '항공': 15, '대한항공': 15,
        '건설': 10, '중공업': 12,
        '에너지': 8, 'S-Oil': 8
    }
    
    adjusted_per = base_per
    for k, cap in sector_caps.items():
        if k in stock_name:
            if adjusted_per > cap: adjusted_per = cap
            break
            
    return adjusted_per

def analyze_stock_item(code, name, token, is_bull_market):
    """
    [수정됨] 안전장치 강화
    """
    try:
        stock_info = get_stock_data(code, token)
        if not stock_info: return None

        ma20, is_bull_trend, rsi = get_technical_indicators(code, token)
        supply_score, supply_msg = get_supply_score(code, token)
        is_improving, mom_msg = get_earnings_momentum(code)

        predicted_eps, eps_confidence, eps_msg = predict_eps_smart(
            code, name, stock_info['eps'], token
        )

        # [수정 1] EPS 검증 강화
        if eps_confidence < 30: return None
        if predicted_eps <= 0: return None
        if predicted_eps < 100: return None  # EPS 100원 미만 제외

        _, my_hist_per, sector_per, roe = get_naver_financial_info(code, name)
        
        used_sector_per = sector_per if sector_per > 0 else my_hist_per
        base_per = (my_hist_per * 0.6) + (used_sector_per * 0.4)
        
        if roe >= 20: base_per *= 1.3
        elif roe >= 15: base_per *= 1.15
        elif roe < 5: base_per *= 0.8

        final_target_per = calculate_target_per_advanced(code, name, base_per, token)

        # [수정 2] Hard Cap 업종별 차등 적용
        if '바이오' in name or '셀트리온' in name or '알테오젠' in name:
            limit_per = 60.0
        elif 'IT' in name or 'NAVER' in name or '카카오' in name or '게임' in name or '크래프톤' in name:
            limit_per = 35.0
        elif '반도체' in name or '하이닉스' in name or '삼성전자' in name or '전자' in name:
            limit_per = 25.0
        elif '은행' in name or '금융' in name or 'KB' in name or '신한' in name:
            limit_per = 12.0
        else:
            limit_per = 30.0  # 일반 종목 기본값 상향

        if final_target_per > limit_per:
            final_target_per = limit_per
        
        target_price = predicted_eps * final_target_per
        price = stock_info['price']
        
        # [수정 3] 적정주가가 현재가의 10% 미만이면 제외
        if target_price < price * 0.1: return None
        
        upside = ((target_price - price) / price) * 100 if price > 0 else 0

        # [수정 4] 극단적 괴리율 필터링
        if upside > 300 or upside < -90: return None

        if not is_bull_market and upside < 40: return None

        if upside >= 30 and supply_score >= 2 and rsi < 70: signal = "Strong Buy (★★★)"
        elif upside >= 30: signal = "Strong Buy (★)"
        elif upside >= 15: signal = "Buy"
        elif upside >= 0: signal = "Hold"
        else: signal = "Sell"

        if not is_bull_trend:
            if rsi < 30: signal = "Buy (과매도)" 
            elif "Buy" in signal: signal = "Hold (하락세)"
        
        if rsi > 70 and "Buy" in signal: signal = "Wait (과열)"

        # [수정 5] 발굴점수 개선 (음수는 0점 처리)
        discovery_score = int(eps_confidence) * max(upside / 100, 0)

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
            "발굴점수": round(discovery_score, 2)
        }
    except: return None

def check_market_trend():
    return True, "상승장 (가정)"

def get_fair_value_chart_figure(df):
    try:
        chart_df = df.head(10).copy()
        names = chart_df['종목명'].tolist()
        prices = chart_df['현재가'].tolist()
        targets = chart_df['적정주가'].tolist()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(names))
        width = 0.35
        
        ax.bar(x - width/2, prices, width, label='현재가', color='gray')
        ax.bar(x + width/2, targets, width, label='적정주가', color='#f63366')
        
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('주가 (원)')
        ax.set_title('저평가 우량주 Top 10')
        ax.legend()
        plt.tight_layout()
        return fig
    except: return None

# -----------------------------------------------------------
# [텔레그램 전송 함수] (수정됨 - 더 상세한 정보)
# -----------------------------------------------------------
def send_telegram_message(message):
    try:
        if "TELEGRAM_TOKEN" not in st.secrets or "TELEGRAM_CHAT_ID" not in st.secrets:
            return 
            
        bot_token = st.secrets["TELEGRAM_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {'chat_id': chat_id, 'text': message, 'parse_mode': 'HTML'}
        requests.post(url, data=data)
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
# Main
# =============================================================================
def main():
    st.set_page_config(page_title="AI 주식비서", page_icon="📈", layout="wide")
    st.title("📈 나만의 AI 주식 비서 (개선판)")
    
    st.info("✨ **업데이트 내역**: EPS 검증 강화, 업종별 PER 차등 적용, 극단적 괴리율 필터링")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        top_n = st.number_input(
            "분석할 종목 수 (Top N)", 
            min_value=10, 
            max_value=200, 
            value=50, 
            step=10
        )
        
        st.markdown("---")
        st.markdown("### 📊 필터 기준")
        st.text("• EPS 100원 이상")
        st.text("• 신뢰도 30점 이상")
        st.text("• 괴리율 -90% ~ 300%")
        st.text("• 적정가 > 현재가 10%")
        
        if st.button("🚀 분석 시작", type="primary"):
            st.session_state['run_analysis'] = True

    if st.session_state.get('run_analysis'):
        token = get_access_token()
        if not token:
            st.error("❌ API 토큰 발급 실패! 키를 확인하세요.")
            return

        status_text = st.empty()
        progress_bar = st.progress(0)
        
        status_text.text("📋 종목 리스트 확보 중...")
        stock_list = get_top_stocks(top_n)
        
        if not stock_list:
            st.error("종목 리스트를 가져올 수 없습니다.")
            return
        
        results = []
        for i, (code, name) in enumerate(stock_list):
            progress_bar.progress((i + 1) / len(stock_list))
            status_text.text(f"🔍 분석 중... {name} ({i+1}/{len(stock_list)})")
            
            res = analyze_stock_item(code, name, token, True)
            if res: results.append(res)
            time.sleep(0.1)
            
        status_text.success(f"✅ 분석 완료! {len(stock_list)}개 중 {len(results)}개 선별")
        progress_bar.empty()
        
        if results:
            df = pd.DataFrame(results).sort_values(by="발굴점수", ascending=False)
            
            # 통계 요약
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("분석 종목", f"{len(stock_list)}개")
            with col2:
                st.metric("선별 종목", f"{len(results)}개")
            with col3:
                strong_buy = len(df[df['의견'].str.contains('Strong Buy')])
                st.metric("Strong Buy", f"{strong_buy}개")
            with col4:
                avg_upside = df['괴리율(%)'].mean()
                st.metric("평균 괴리율", f"{avg_upside:.1f}%")
            
            st.markdown("---")
            st.subheader("🏆 Top Picks (발굴점수 순)")
            
            # 데이터프레임 스타일링
            st.dataframe(
                df.style.background_gradient(subset=['괴리율(%)'], cmap='RdYlGn')
                      .background_gradient(subset=['발굴점수'], cmap='Greens')
                      .format({'현재가': '{:,}', '적정주가': '{:,}', '괴리율(%)': '{:.1f}%'}),
                use_container_width=True,
                height=400
            )
            
            # 차트
            st.markdown("---")
            st.subheader("📊 현재가 vs 적정주가 비교")
            fig = get_fair_value_chart_figure(df)
            if fig: 
                st.pyplot(fig)
                plt.close(fig)
            
            # CSV 다운로드
            csv = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 CSV 다운로드",
                data=csv,
                file_name=f"stock_analysis_{time.strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            # 텔레그램 전송
            st.markdown("---")
            col_left, col_right = st.columns([3, 1])
            
            with col_left:
                st.info("💬 텔레그램 봇으로 Top 10 리포트를 전송하시겠습니까?")
            
            with col_right:
                if st.button("📱 텔레그램 전송", type="primary"):
                    with st.spinner("전송 중..."):
                        top10 = df.head(10)
                        
                        # 메시지 작성
                        msg = f"<b>📊 [AI 주식비서] 오늘의 Top 10</b>\n"
                        msg += f"분석: {len(stock_list)}개 → 선별: {len(results)}개\n"
                        msg += f"시간: {time.strftime('%Y-%m-%d %H:%M')}\n"
                        msg += "="*30 + "\n\n"
                        
                        for idx, (i, row) in enumerate(top10.iterrows(), 1):
                            if "Strong" in row['의견']:
                                icon = "🔥"
                            elif "Buy" in row['의견']:
                                icon = "✅"
                            else:
                                icon = "📌"
                            
                            msg += f"<b>{idx}. {icon} {row['종목명']}</b>\n"
                            msg += f"   • 현재가: {row['현재가']:,}원\n"
                            msg += f"   • 목표가: {row['적정주가']:,}원 (↑{row['괴리율(%)']:.1f}%)\n"
                            msg += f"   • 신뢰도: {row['EPS신뢰도']}점 | PER: {row['목표PER']}배\n"
                            msg += f"   • 수급: {row['수급']} | RSI: {row['RSI']}\n"
                            msg += f"   • 의견: <b>{row['의견']}</b>\n\n"
                        
                        msg += "="*30 + "\n"
                        msg += "💡 자세한 내용은 앱에서 확인하세요!"
                        
                        send_telegram_message(msg)
                        if fig: send_telegram_photo(fig)
                        
                        st.success("✅ 텔레그램 전송 완료!")
                        st.balloons()
                
        else:
            st.warning("⚠️ 조건에 맞는 종목이 없습니다. 필터 기준을 완화해보세요.")
            st.info("""
            **필터가 너무 엄격할 수 있습니다:**
            - EPS 100원 미만 제외
            - 신뢰도 30점 미만 제외
            - 괴리율 -90% ~ 300% 범위
            - 적정주가가 현재가의 10% 미만 제외
            """)

if __name__ == "__main__":
    main()
