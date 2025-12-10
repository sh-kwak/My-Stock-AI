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
    st.error("🚨 API 키가 설정되지 않았습니다!")
    st.stop()

BASE_URL = "https://openapi.koreainvestment.com:9443"

# =============================================================================
# [데이터 수집 함수들]
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
        return [(str(row['Code']), row['Name']) for _, row in df_top.iterrows()]
    except:
        return []

def get_stock_data(stock_code, access_token):
    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price"
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "FHKST01010100"
    }
    params = {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": stock_code}
    try:
        res = requests.get(url, headers=headers, params=params)
        data = res.json()
        if data['rt_cd'] != '0':
            return None
        output = data['output']
        return {
            "price": float(output.get('stck_prpr', 0)),
            "eps": float(output.get('eps', 0)),
            "bps": float(output.get('bps', 0)),
            "per": float(output.get('per', 0)),
            "pbr": float(output.get('pbr', 0)),
        }
    except:
        return None

def get_naver_data(stock_code, stock_name=""):
    """네이버에서 재무 데이터 수집"""
    try:
        url = f"https://finance.naver.com/item/main.naver?code={stock_code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=5)
        
        try:
            dfs = pd.read_html(io.StringIO(res.text), encoding='euc-kr')
        except:
            dfs = pd.read_html(io.StringIO(res.content.decode('euc-kr', 'replace')))
        
        result = {
            'forward_eps': None,
            'roe': 0.0,
            'per_history': [],
            'sector_per': 12.0,
        }
        
        # 재무제표 찾기
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
        
        # Forward EPS
        for col in fin_df.columns:
            if '(E)' in str(col) or 'E' in str(col):
                eps_val = get_val('EPS(원)', col)
                if eps_val and eps_val > 0:
                    result['forward_eps'] = eps_val
                    break
        
        # ROE
        if len(fin_df.columns) >= 2:
            recent_col = fin_df.columns[-2]
            result['roe'] = get_val('ROE', recent_col) or 0.0
        
        # PER 히스토리
        outlier = 100.0 if '바이오' in stock_name or '셀트리온' in stock_name else 50.0
        for col in fin_df.columns[:5]:
            per_val = get_val('PER(배)', col)
            if per_val and 0 < per_val <= outlier:
                result['per_history'].append(per_val)
        
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
        
    except:
        return {
            'forward_eps': None,
            'roe': 0.0,
            'per_history': [],
            'sector_per': 12.0,
        }

def get_technical_indicators(stock_code, access_token):
    """RSI 계산"""
    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "FHKST01010400"
    }
    params = {
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": stock_code,
        "fid_period_div_code": "D",
        "fid_org_adj_prc": "1"
    }
    try:
        res = requests.get(url, headers=headers, params=params)
        data = res.json()
        if data['rt_cd'] != '0':
            return None, False, 50.0
        
        daily_prices_desc = [float(x['stck_clpr']) for x in data['output']]
        daily_prices_asc = daily_prices_desc[::-1]
        
        if len(daily_prices_desc) < 20:
            return None, False, 50.0
            
        ma20 = sum(daily_prices_desc[:20]) / 20.0
        current_price = daily_prices_desc[0]
        is_bull = current_price >= ma20
        
        # RSI
        if len(daily_prices_asc) > 15:
            delta = pd.Series(daily_prices_asc).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_val = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
        else:
            rsi_val = 50.0
            
        return ma20, is_bull, rsi_val
    except:
        return None, False, 50.0

def get_supply_score(stock_code, access_token):
    """외인/기관 수급"""
    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-investor"
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {access_token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "FHKST01010900"
    }
    params = {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": stock_code}
    
    try:
        res = requests.get(url, headers=headers, params=params)
        data = res.json()
        if data['rt_cd'] != '0':
            return 0, "-"
        
        daily_data = data.get('output', [])[:5]
        if not daily_data:
            return 0, "데이터없음"
        
        inst_buy, for_buy = 0, 0
        for row in daily_data:
            try:
                if int(str(row.get('frgn_ntby_qty', '0')).replace(',', '')) > 0:
                    for_buy += 1
                if int(str(row.get('orgn_ntby_qty', '0')).replace(',', '')) > 0:
                    inst_buy += 1
            except:
                continue
        
        score = 0
        msg = []
        if for_buy >= 3:
            score += 1
            msg.append(f"외인{for_buy}일")
        if inst_buy >= 3:
            score += 1
            msg.append(f"기관{inst_buy}일")
        
        return score, "/".join(msg) if msg else "수급약함"
    except:
        return 0, "에러"

def get_analyst_target_price(stock_code):
    """네이버 증권사 목표가 크롤링"""
    try:
        url = f"https://finance.naver.com/item/main.naver?code={stock_code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=5)
        html = res.text
        
        import re
        
        # 방법 1: 목표주가 직후 <em>숫자</em>
        pattern1 = r'목표주가.*?<em>([\d,]+)</em>'
        match1 = re.search(pattern1, html, re.DOTALL)
        
        if match1:
            price_str = match1.group(1).replace(',', '')
            try:
                price = int(price_str)
                if 1000 <= price <= 5000000:
                    return price
            except:
                pass
        
        # 방법 2: 투자의견 테이블 내부
        pattern2 = r'투자의견.*?</table>'
        table_match = re.search(pattern2, html, re.DOTALL)
        
        if table_match:
            table_html = table_match.group(0)
            em_numbers = re.findall(r'<em>([\d,]+)</em>', table_html)
            
            for num_str in em_numbers:
                try:
                    num = int(num_str.replace(',', ''))
                    if 1000 <= num <= 5000000 and num > 100:
                        return num
                except:
                    continue
        
        return None
        
    except:
        return None

# =============================================================================
# [분석 함수]
# =============================================================================

def analyze_stock_simple(code, name, token):
    """간소화된 분석 함수"""
    try:
        # 기본 데이터
        stock_info = get_stock_data(code, token)
        if not stock_info or stock_info['price'] <= 0:
            return None
        
        # 네이버 데이터
        naver_data = get_naver_data(code, name)
        
        # 기술적 지표
        ma20, is_bull_trend, rsi = get_technical_indicators(code, token)
        supply_score, supply_msg = get_supply_score(code, token)
        
        # RSI 과열 제외
        if rsi > 75:
            return None
        
        # EPS 결정
        current_eps = stock_info['eps']
        forward_eps = naver_data['forward_eps']
        
        eps_source = "현재"
        if forward_eps and forward_eps > 0 and current_eps > 0:
            ratio = forward_eps / current_eps
            if 0.5 <= ratio <= 2.0:
                used_eps = forward_eps
                eps_source = "컨센서스"
            else:
                used_eps = current_eps
        elif forward_eps and forward_eps > 0:
            used_eps = forward_eps
            eps_source = "컨센서스"
        else:
            used_eps = current_eps
        
        # EPS 필터
        if used_eps <= 100:
            return None
        
        # EPS 상한 (비정상 값 제외)
        if '바이오' in name or '제약' in name:
            eps_limit = 50000
        elif '반도체' in name or '하이닉스' in name:
            eps_limit = 40000
        else:
            eps_limit = 30000
        
        if used_eps > eps_limit:
            return None
        
        # PER 계산
        per_history = naver_data['per_history']
        if per_history:
            hist_per = np.median(per_history)
        else:
            hist_per = 12.0
        
        sector_per = naver_data['sector_per']
        base_per = (hist_per * 0.6) + (sector_per * 0.4)
        
        # ROE 할증
        roe = naver_data['roe']
        if roe >= 20:
            base_per *= 1.15
        elif roe >= 15:
            base_per *= 1.08
        elif roe < 5:
            base_per *= 0.85
        
        # 업종 상한
        per_caps = {
            '바이오': 30, '셀트리온': 30,
            'NAVER': 20, '카카오': 20, '게임': 18,
            '반도체': 15, '하이닉스': 15, '삼성전자': 12,
            '은행': 7, '금융': 7,
        }
        
        for keyword, cap in per_caps.items():
            if keyword in name:
                base_per = min(base_per, cap)
                break
        else:
            base_per = min(base_per, 15)
        
        # 적정가
        target_price = used_eps * base_per
        price = stock_info['price']
        
        # 적정가 필터 (현재가의 70% 이상)
        if target_price < price * 0.7:
            return None
        
        # 최종 상한 (현재가의 1.7배)
        target_price = min(target_price, price * 1.7)
        
        # 괴리율
        upside = ((target_price - price) / price) * 100
        
        # 괴리율 필터 (10~50%)
        if upside < 10 or upside > 50:
            return None
        
        # 등급
        if upside >= 25 and (supply_score >= 1 or is_bull_trend) and rsi < 65:
            grade = "A"
            signal = "Strong Buy (★★★)"
        elif upside >= 20 and rsi < 70:
            grade = "A"
            signal = "Strong Buy (★)"
        elif upside >= 15:
            grade = "B"
            signal = "Buy"
        else:
            grade = "C"
            signal = "Hold"
        
        # 하락세 보정
        if not is_bull_trend and "Buy" in signal:
            signal += " (하락세)"
        
        # 밸류 점수
        value_score = min(100, int(
            (upside / 50 * 40) +
            (min(roe, 20) / 20 * 25) +
            (supply_score * 10) +
            ((100 - rsi) / 100 * 25)
        ))

        # -----------------------------------------------
        # [추가] 투자 가치 설명 및 추천 매매가 로직
        # -----------------------------------------------
        reasons = []
        # 1. 밸류에이션 관점
        if upside >= 30:
            reasons.append("📉 현저한 저평가 (괴리율 30% 이상)")
        elif upside >= 20:
            reasons.append("📉 저평가 매력 (상승여력 충분)")
        
        # 2. 수급 관점
        if supply_score >= 1:
            reasons.append("💰 메이저(외인/기관) 수급 유입 중")
        
        # 3. 펀더멘털 관점
        if roe >= 10:
            reasons.append("💎 견조한 수익성 (ROE 10% 이상)")
        if eps_source == "컨센서스":
            reasons.append("📈 실적 성장 기대 (Forward EPS 사용)")
            
        # 4. 기술적 관점
        if is_bull_trend:
            reasons.append("📈 상승 추세 (20일선 위)")
        elif rsi <= 40:
            reasons.append("ea 과매도 구간 (기술적 반등 기대)")

        reason_text = " + ".join(reasons) if reasons else "저평가 매력 보유"

        # 추천 매매가 (단기 스윙 기준)
        # 매수: 현재가 ~ 현재가 -2% 구간 / 매도: 적정주가
        buy_price = f"{int(price * 0.98):,} ~ {int(price):,}원"
        sell_price = f"{int(target_price):,}원"

        return {
            "종목명": name,
            "현재가": int(price),
            "적정주가": int(target_price),
            "괴리율(%)": round(upside, 1),
            "투자등급": grade,
            "의견": signal,
            "밸류점수": value_score,
            "수급": supply_msg,
            "RSI": round(rsi, 1),
            "ROE(%)": round(roe, 1),
            "EPS출처": eps_source,
            "목표PER": round(base_per, 1),
            # 추가된 필드
            "분석사유": reason_text,
            "매수가": buy_price,
            "매도가": sell_price
        }
        
    except:
        return None

# =============================================================================
# [차트]
# =============================================================================

def get_chart(df):
    try:
        chart_df = df.head(10).copy()
        names = chart_df['종목명'].tolist()
        prices = chart_df['현재가'].tolist()
        targets = chart_df['적정주가'].tolist()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(names))
        width = 0.35
        
        ax.bar(x - width/2, prices, width, label='현재가', color='#6c757d')
        ax.bar(x + width/2, targets, width, label='적정주가', color='#28a745')
        
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('주가 (원)')
        ax.set_title('저평가 종목 Top 10')
        ax.legend()
        
        plt.tight_layout()
        return fig
    except:
        return None

# -----------------------------------------------------------
# [텔레그램 전송 함수]
# -----------------------------------------------------------
def send_telegram_message(message):
    try:
        if "TELEGRAM_TOKEN" not in st.secrets or "TELEGRAM_CHAT_ID" not in st.secrets:
            return False, "설정 파일에 텔레그램 정보가 없습니다."
            
        bot_token = st.secrets["TELEGRAM_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {'chat_id': chat_id, 'text': message, 'parse_mode': 'HTML'}
        res = requests.post(url, data=data)
        
        if res.status_code == 200:
            return True, "전송 성공"
        else:
            return False, f"전송 실패 ({res.status_code})"
    except Exception as e:
        return False, str(e)

# =============================================================================
# [Main]
# =============================================================================

def main():
    st.set_page_config(page_title="AI 주식비서 V3.2", page_icon="📈", layout="wide")
    st.title("📈 AI 주식 비서 Ver 3.2 (보수 모드)")
    st.info("✨ **보수 모드**: 괴리율 10~50% | EPS 상한 | PER 상한 | 현재가 1.7배 상한")
    
    # Session State
    if 'results' not in st.session_state:
        st.session_state['results'] = None
    if 'run' not in st.session_state:
        st.session_state['run'] = False
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        top_n = st.number_input("분석 종목 수", 10, 200, 50, 10)
        
        st.markdown("---")
        st.markdown("### 📊 필터 기준")
        st.markdown("""
        - EPS 100원 이상
        - 괴리율 10% ~ 50%
        - RSI 75 이하
        - 적정가 > 현재가 70%
        """)
        
        if st.button("🚀 분석 시작", type="primary"):
            st.session_state['run'] = True
            st.session_state['results'] = None
    
    # 분석 실행
    if st.session_state.get('run') and st.session_state['results'] is None:
        token = get_access_token()
        if not token:
            st.error("❌ API 토큰 발급 실패!")
            st.session_state['run'] = False
            return
        
        status = st.empty()
        progress = st.progress(0)
        
        status.text("📋 종목 리스트 확보 중...")
        stock_list = get_top_stocks(top_n)
        
        if not stock_list:
            st.error("종목 리스트를 가져올 수 없습니다.")
            st.session_state['run'] = False
            return
        
        results = []
        for i, (code, name) in enumerate(stock_list):
            progress.progress((i + 1) / len(stock_list))
            status.text(f"🔍 {name} ({i+1}/{len(stock_list)})")
            
            res = analyze_stock_simple(code, name, token)
            if res:
                results.append(res)
            
            time.sleep(0.1)
        
        status.success(f"✅ 완료! {len(stock_list)}개 중 {len(results)}개 선별")
        progress.empty()
        
        st.session_state['results'] = results
        st.session_state['run'] = False
    
    # 결과 표시
    if st.session_state['results'] is not None:
        results = st.session_state['results']
        
        if results:
            df = pd.DataFrame(results).sort_values(by="밸류점수", ascending=False)
            
            # 텔레그램 전송 UI
            with st.container():
                col_btn, col_msg = st.columns([1, 4])
                with col_btn:
                    if st.button("📱 텔레그램으로 요약 전송"):
                        with st.spinner("전송 중..."):
                            top5 = df.head(5)
                            msg = f"📈 <b>[AI 주식비서] 추천 Top 5</b>\n({datetime.now().strftime('%Y-%m-%d')})\n\n"
                            
                            for _, row in top5.iterrows():
                                icon = "🔥" if row['투자등급'] == 'A' else "✅"
                                msg += f"{icon} <b>{row['종목명']}</b> ({row['투자등급']})\n"
                                msg += f"   현재가: {row['현재가']:,}원\n"
                                msg += f"   적정가: {row['적정주가']:,}원\n"
                                msg += f"   괴리율: +{row['괴리율(%)']}%\n\n"
                            
                            msg += "※ 본 정보는 투자 참고용입니다."
                            
                            success, res_msg = send_telegram_message(msg)
                            if success:
                                st.success("✅ 전송 완료!")
                            else:
                                st.error(f"❌ 전송 실패: {res_msg}")

            # 통계
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("선별 종목", f"{len(results)}개")
            with col2:
                grade_a = len(df[df['투자등급'] == 'A'])
                st.metric("A등급", f"{grade_a}개")
            with col3:
                avg_upside = df['괴리율(%)'].mean()
                st.metric("평균 괴리율", f"{avg_upside:.1f}%")
            with col4:
                avg_score = df['밸류점수'].mean()
                st.metric("평균 점수", f"{avg_score:.0f}점")
            
            st.markdown("---")
            
            # [추가된 섹션] A등급 상세 리포트
            st.subheader("🏆 A등급 종목 상세 투자 리포트")
            a_grade_stocks = df[df['투자등급'] == 'A']
            
            if not a_grade_stocks.empty:
                for idx, row in a_grade_stocks.iterrows():
                    with st.expander(f"📌 {row['종목명']} ({row['의견']})", expanded=True):
                        st.markdown(f"**💡 투자 포인트:** {row['분석사유']}")
                        
                        col_buy, col_sell, col_info = st.columns(3)
                        with col_buy:
                            st.info(f"**🔵 추천 매수가**\n\n{row['매수가']}")
                        with col_sell:
                            st.error(f"**🔴 목표 매도가**\n\n{row['매도가']}")
                        with col_info:
                            st.success(f"**수익 기대율**\n\n+{row['괴리율(%)']}%")
            else:
                st.info("현재 기준 A등급(강력 매수) 종목이 포착되지 않았습니다.")

            st.markdown("---")

            tab1, tab2, tab3, tab4 = st.tabs(["📊 전체 결과", "📈 차트", "🔍 목표가 검증", "🐛 디버그"])
            
            with tab1:
                st.dataframe(
                    df.style.background_gradient(subset=['괴리율(%)'], cmap='Greens')
                          .background_gradient(subset=['밸류점수'], cmap='Blues'),
                    use_container_width=True,
                    height=450
                )
                
                csv = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    "📥 CSV 다운로드",
                    csv,
                    f"stock_{time.strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
            
            with tab2:
                fig = get_chart(df)
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
            
            with tab3:
                st.subheader("🔍 증권사 목표가 vs 우리 적정가")
                
                top10 = df.head(10)
                
                if st.button("🔍 검증 실행", key="verify"):
                    with st.spinner("조회 중..."):
                        stock_list = get_top_stocks(200)
                        code_map = {name: code for code, name in stock_list}
                        
                        for _, row in top10.iterrows():
                            name = row['종목명']
                            code = code_map.get(name)
                            
                            if code:
                                analyst_target = get_analyst_target_price(code)
                                
                                with st.expander(f"**{name}** ({row['투자등급']}등급)"):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("현재가", f"{row['현재가']:,}원")
                                    with col2:
                                        st.metric("우리 적정가", f"{row['적정주가']:,}원", f"+{row['괴리율(%)']:.1f}%")
                                    with col3:
                                        if analyst_target:
                                            st.metric("증권사 목표가", f"{analyst_target:,}원")
                                            
                                            # 괴리율 계산
                                            dev = ((row['적정주가'] - analyst_target) / analyst_target) * 100
                                            if abs(dev) <= 15:
                                                st.success(f"✅ 일치 (차이 {dev:+.1f}%)")
                                            elif abs(dev) <= 30:
                                                st.info(f"ℹ️ 유사 (차이 {dev:+.1f}%)")
                                            else:
                                                st.warning(f"⚠️ 괴리 (차이 {dev:+.1f}%)")
                                        else:
                                            st.metric("증권사 목표가", "없음")
                                            st.warning("⚠️ 컨센서스 없음")
                                
                                time.sleep(0.5)
            
            with tab4:
                st.subheader("🐛 HTML 디버그")
                
                stock_list = get_top_stocks(100)
                names = [n for c, n in stock_list]
                
                selected = st.selectbox("종목 선택", names)
                
                if st.button("🔍 HTML 확인"):
                    code = None
                    for c, n in stock_list:
                        if n == selected:
                            code = c
                            break
                    
                    if code:
                        st.write(f"**종목코드: {code}**")
                        
                        try:
                            url = f"https://finance.naver.com/item/main.naver?code={code}"
                            res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
                            
                            import re
                            pattern = r'투자의견.*?</table>'
                            match = re.search(pattern, res.text, re.DOTALL)
                            
                            if match:
                                table = match.group(0)[:800]
                                st.success("✅ 테이블 발견!")
                                st.code(table, language='html')
                                
                                em_tags = re.findall(r'<em>([^<]+)</em>', table)
                                st.write("**<em> 태그:**", em_tags)
                                
                                numbers = []
                                for em in em_tags:
                                    clean = em.replace(',', '').strip()
                                    if clean.replace('.', '').isdigit():
                                        try:
                                            num = int(float(clean))
                                            if num > 100:
                                                numbers.append(num)
                                        except:
                                            pass
                                
                                if numbers:
                                    st.write("**숫자 후보:**", numbers)
                                    st.write(f"**목표가: {max(numbers):,}원**")
                            else:
                                st.error("❌ 테이블 없음")
                        except Exception as e:
                            st.error(f"오류: {e}")
        
        else:
            st.warning("조건에 맞는 종목이 없습니다.")

if __name__ == "__main__":
    main()
