import streamlit as st
import pandas as pd
import requests
import json
import time
import io
import os # 파일 확인용
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # 폰트 관리
import streamlit as st

# -----------------------------------------------------------
# [한글 폰트 자동 설정] (koreanize_matplotlib 대체)
# -----------------------------------------------------------
@st.cache_resource
def install_korean_font():
    # 폰트 파일이 없으면 다운로드 (나눔고딕)
    font_path = "NanumGothic.ttf"
    if not os.path.exists(font_path):
        url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        with open(font_path, "wb") as f:
            f.write(requests.get(url).content)
    
    # 폰트 등록
    fm.fontManager.addfont(font_path)
    plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

# 폰트 설정 실행
install_korean_font()

# -----------------------------------------------------------
# [설정] API Key (오직 Streamlit Secrets에서만 가져옴)
# -----------------------------------------------------------

try:
    APP_KEY = st.secrets["APP_KEY"]
    APP_SECRET = st.secrets["APP_SECRET"]
except:
# Secrets가 없으면 경고 메시지를 띄우고 앱을 중단합니다.
    st.error("🚨 API 키가 설정되지 않았습니다!")
    st.info("Streamlit Cloud의 [Settings] -> [Secrets] 메뉴에 키를 입력해주세요.")
    st.stop() # 더 이상 실행하지 않음

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
    """ 네이버 증권 재무제표 크롤링 """
    try:
        url = f"https://finance.naver.com/item/main.naver?code={stock_code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers)
        # 인코딩 처리 강화
        try:
            dfs = pd.read_html(io.StringIO(res.text), encoding='euc-kr')
        except:
            dfs = pd.read_html(io.StringIO(res.content.decode('euc-kr', 'replace')))
        
        fin_df = None
        for df in dfs:
            if not df.empty:
                # 데이터프레임 값을 문자열로 변환하여 검색
                df_str = df.astype(str)
                if '매출액' in df_str.iloc[:, 0].values and '영업이익' in df_str.iloc[:, 0].values:
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
        
        daily = data['output'][:5]
        inst, frgn = 0, 0
        for row in daily:
            if int(row.get('frgn_ntby_qty', 0)) > 0: frgn += 1
            if int(row.get('orgn_ntby_qty', 0)) > 0: inst += 1
            
        score = 0
        msg = []
        if frgn >= 3: score+=1; msg.append(f"외인{frgn}일")
        if inst >= 3: score+=1; msg.append(f"기관{inst}일")
        return score, "/".join(msg) if msg else "수급약함"
    except: return 0, "에러"

def analyze_eps_trend(quarterly_data):
    try:
        if not quarterly_data or not quarterly_data['eps']: return 0, "데이터 부족"
        eps_list = quarterly_data['eps']
        if len(eps_list) < 3: return 0, "데이터 부족"

        # 추세 계산 (기울기)
        x = np.arange(len(eps_list))
        slope = np.polyfit(x, eps_list, 1)[0]
        avg_eps = np.mean(eps_list)
        trend_strength = (slope / avg_eps) if avg_eps > 0 else 0

        # QoQ
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

def get_earnings_momentum(stock_code):
    try:
        url = f"https://finance.naver.com/item/main.naver?code={stock_code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers)
        dfs = pd.read_html(io.StringIO(res.text), encoding='euc-kr')
        
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

def predict_eps_smart(stock_code, stock_name, current_eps, access_token):
    try:
        # 1. 분기 실적 (네이버 크롤링)
        quarterly_data = get_quarterly_financials_from_naver(stock_code)
        
        # 2. 컨센서스 (네이버)
        naver_eps, _, _, _ = get_naver_financial_info(stock_code, stock_name)
        
        # 3. 추세 점수
        trend_score, trend_msg = analyze_eps_trend(quarterly_data)
        
        # 4. 종합 판단
        if naver_eps and naver_eps > 0:
            if current_eps > 0:
                deviation = abs(naver_eps - current_eps) / current_eps
                if deviation < 0.2: 
                    return naver_eps, 80, "네이버 컨센서스 (신뢰도 높음)"
                elif deviation < 0.5:
                    blended = (naver_eps * 0.6) + (current_eps * 0.4)
                    return blended, 65, "네이버 60% + 현재 40% 혼합"
                else:
                    # 너무 차이나면 보수적으로 현재 실적 사용하되, 추세가 좋으면 가산
                    if trend_score > 50:
                        return current_eps * 1.1, 55, "편차 과대 → 현재실적+성장세 반영"
                    else:
                        return current_eps, 45, "편차 과대 → 현재실적 사용"
            else:
                return naver_eps, 70, "적자탈출 예상 (컨센서스 채택)"
        
        return current_eps, 50, "컨센서스 없음 (현재실적 유지)"
        
    except: return current_eps, 40, "예측 오류"

def calculate_target_per_advanced(stock_code, stock_name, base_per, access_token):
    # 업종별 CAP 등 기존 로직 유지
    sector_caps = {
        '반도체': 18, 'SK하이닉스': 18, '삼성전자': 18,
        '자동차': 10, '현대차': 10, '기아': 10,
        '은행': 7, '금융': 7, 'KB': 7, '신한': 7,
        '바이오': 40, '셀트리온': 40, 
        'IT': 25, 'NAVER': 25, '카카오': 25
    }
    
    adjusted_per = base_per
    
    # 키워드 매칭
    for k, cap in sector_caps.items():
        if k in stock_name:
            if adjusted_per > cap: adjusted_per = cap
            break
            
    return adjusted_per

def analyze_stock_item(code, name, token, is_bull_market):
    try:
        stock_info = get_stock_data(code, token)
        if not stock_info: return None

        ma20, is_bull_trend, rsi = get_technical_indicators(code, token)
        supply_score, supply_msg = get_supply_score(code, token)
        is_improving, mom_msg = get_earnings_momentum(code)

        predicted_eps, eps_confidence, eps_msg = predict_eps_smart(
            code, name, stock_info['eps'], token
        )

        if eps_confidence < 30 or predicted_eps <= 0: return None

        _, my_hist_per, sector_per, roe = get_naver_financial_info(code, name)
        
        used_sector_per = sector_per if sector_per > 0 else my_hist_per
        base_per = (my_hist_per * 0.6) + (used_sector_per * 0.4)
        
        # ROE 가중치
        if roe >= 20: base_per *= 1.3
        elif roe >= 15: base_per *= 1.15
        elif roe < 5: base_per *= 0.8

        final_target_per = calculate_target_per_advanced(code, name, base_per, token)
        
        target_price = predicted_eps * final_target_per
        price = stock_info['price']
        upside = ((target_price - price) / price) * 100 if price > 0 else 0

        # 하락장 보수적 기준
        if not is_bull_market and upside < 40: return None

        # 의견
        if upside >= 30 and supply_score >= 2 and rsi < 70: signal = "Strong Buy (★★★)"
        elif upside >= 30: signal = "Strong Buy (★)"
        elif upside >= 15: signal = "Buy"
        elif upside >= 0: signal = "Hold"
        else: signal = "Sell"

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
    except: return None

def check_market_trend():
    # 간단히 KOSPI 2000 이상이면 상승장으로 가정 (실제로는 지수 조회 필요)
    return True, "상승장 (가정)"

def get_fair_value_chart_figure(df):
    try:
        # Streamlit에서는 기본 폰트 사용 (한글 깨짐 방지는 koreanize_matplotlib가 처리)
        chart_df = df.head(10).copy()
        names = chart_df['종목명'].tolist()
        prices = chart_df['현재가'].tolist()
        targets = chart_df['적정주가'].tolist()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(names))
        width = 0.35
        
        ax.bar(x - width/2, prices, width, label='Current', color='gray')
        ax.bar(x + width/2, targets, width, label='Target', color='#f63366')
        
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45)
        ax.legend()
        plt.tight_layout()
        return fig
    except: return None

# -----------------------------------------------------------
# [텔레그램 전송 기능]
# -----------------------------------------------------------
def send_telegram_message(message):
    """ 텍스트 메시지를 보냅니다. """
    # 사용자별 봇 설정을 위해 st.secrets 사용 권장하나, 여기서는 하드코딩된 값 사용
    bot_token = "8297423754:AAHiYrE2XenVrBBwbQ_azWZmX0VI4abZOaA"
    chat_id = "34839919"
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {'chat_id': chat_id, 'text': message}
        res = requests.post(url, data=data)
        
        if res.status_code != 200:
            print(f"[텔레그램 오류] Status: {res.status_code}, Response: {res.text}")
            st.error(f"텔레그램 전송 실패 (Code {res.status_code}): {res.text}")
        else:
            print("[텔레그램] 메시지 전송 성공")
            
    except Exception as e:
        print(f"[텔레그램 오류] {e}")
        st.error(f"텔레그램 전송 중 예외 발생: {e}")

def send_telegram_photo(photo_path):
    """ 저장된 차트 이미지를 보냅니다. """
    bot_token = "8297423754:AAHiYrE2XenVrBBwbQ_azWZmX0VI4abZOaA"
    chat_id = "34839919"
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        with open(photo_path, 'rb') as f:
            res = requests.post(url, data={'chat_id': chat_id}, files={'photo': f})
            
        if res.status_code != 200:
            print(f"[텔레그램 이미지 오류] Status: {res.status_code}, Response: {res.text}")
            st.error(f"이미지 전송 실패 (Code {res.status_code}): {res.text}")
        else:
            print("[텔레그램] 이미지 전송 성공")
            
    except Exception as e:
        print(f"[이미지 전송 오류] {e}")
        st.error(f"이미지 전송 중 예외 발생: {e}")

# =============================================================================
# Main
# =============================================================================
def main():
    st.set_page_config(page_title="AI 주식비서", page_icon="📈", layout="wide")
    st.title("📈 나만의 AI 주식 비서")
    
    with st.sidebar:
        st.header("Settings")
        top_n = st.slider("분석 종목 수", 10, 100, 20)
        use_telegram = st.checkbox("텔레그램 알림 받기", value=True)
        if st.button("🚀 분석 시작"):
            st.session_state['run_analysis'] = True

    if st.session_state.get('run_analysis'):
        token = get_access_token()
        if not token:
            st.error("API 토큰 발급 실패! 키를 확인하세요.")
            return

        status = st.empty()
        progress = st.progress(0)
        
        status.text("리스트 확보 중...")
        stock_list = get_top_stocks(top_n)
        
        results = []
        for i, (code, name) in enumerate(stock_list):
            progress.progress((i + 1) / len(stock_list))
            status.text(f"Analyzing {name}...")
            
            res = analyze_stock_item(code, name, token, True)
            if res: results.append(res)
            time.sleep(0.1)
            
        status.success("완료!")
        progress.empty()
        
        if results:
            df = pd.DataFrame(results).sort_values(by="발굴점수", ascending=False)
            st.subheader("🏆 Top Picks")
            st.dataframe(df.style.background_gradient(subset=['괴리율(%)'], cmap='RdYlGn'), use_container_width=True)
            
            fig = get_fair_value_chart_figure(df)
            if fig: st.pyplot(fig)
            
            # 텔레그램 전송
            if use_telegram:
                st.info("텔레그램으로 결과 전송 중...")
                try:
                    msg_text = f"🚀 [AI 주식비서] 분석 완료!\n총 {len(results)}개 유망 종목 발견\n\n"
                    # 상위 5개만 텍스트로 요약
                    for i, r in enumerate(results[:5]):
                        emoji = "🥇" if i==0 else ("🥈" if i==1 else "🥉" if i==2 else "🔹")
                        msg_text += f"{emoji} {r['종목명']} ({r['의견']})\n   목표가:{r['적정주가']:,}원 (괴리율:{r['괴리율(%)']}%)\n"
                    
                    send_telegram_message(msg_text)
                    
                    if fig:
                        img_path = "chart_temp.png"
                        fig.savefig(img_path)
                        send_telegram_photo(img_path)
                        st.success("텔레그램 전송 완료!")
                except Exception as e:
                    st.error(f"텔레그램 전송 실패: {e}")
        else:
            st.warning("조건에 맞는 종목이 없습니다.")

if __name__ == "__main__":
    main()

