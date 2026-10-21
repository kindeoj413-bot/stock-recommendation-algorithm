"""주식 추천 알고리즘 — 누구나 사용할 수 있는 국내 주식 탐색 MVP."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core import (
    ALL_FIELDS, EXPERIENCE, SYMBOLIC, combined_score, grade, indicators,
    normalized_rating_csv, rating_score, signal_explanation,
    strategy_summary, technical_score, technical_universe_backtest, validate_ohlcv
)

st.set_page_config(page_title='주식 추천 알고리즘', page_icon='🔎', layout='wide', initial_sidebar_state='expanded')
ROOT = Path(__file__).parent
KST = ZoneInfo('Asia/Seoul')
STOCKS = {
    '005930.KS': ('삼성전자', '코스피', 'IT·반도체'),
    '000660.KS': ('SK하이닉스', '코스피', 'IT·반도체'),
    '035420.KS': ('NAVER', '코스피', '플랫폼'),
    '035720.KS': ('카카오', '코스피', '플랫폼'),
    '005380.KS': ('현대차', '코스피', '자동차'),
    '000270.KS': ('기아', '코스피', '자동차'),
    '068270.KS': ('셀트리온', '코스피', '바이오'),
    '105560.KS': ('KB금융', '코스피', '금융'),
    '055550.KS': ('신한지주', '코스피', '금융'),
    '207940.KS': ('삼성바이오로직스', '코스피', '바이오'),
    '373220.KS': ('LG에너지솔루션', '코스피', '2차전지'),
    '006400.KS': ('삼성SDI', '코스피', '2차전지'),
    '051910.KS': ('LG화학', '코스피', '화학'),
    '012330.KS': ('현대모비스', '코스피', '자동차'),
    '005490.KS': ('POSCO홀딩스', '코스피', '철강'),
    '086790.KS': ('하나금융지주', '코스피', '금융'),
    '247540.KQ': ('에코프로비엠', '코스닥', '2차전지'),
    '086520.KQ': ('에코프로', '코스닥', '2차전지'),
    '196170.KQ': ('알테오젠', '코스닥', '바이오'),
    '091990.KQ': ('셀트리온헬스케어 (과거 종목)', '코스닥', '참고용 제외'),
    '028300.KQ': ('HLB', '코스닥', '바이오'),
    '293490.KQ': ('카카오게임즈', '코스닥', '게임'),
    '067310.KQ': ('하나마이크론', '코스닥', '반도체'),
    '035900.KQ': ('JYP Ent.', '코스닥', '엔터'),
}
# Delisted/merged tickers are intentionally not in the selectable universe.
STOCKS.pop('091990.KQ')
DEFAULT = ['005930.KS','000660.KS','035420.KS','005380.KS','105560.KS','247540.KQ','196170.KQ','035900.KQ']

st.markdown("""<style>
.block-container {max-width: 1440px; padding-top:1.3rem; padding-bottom:3rem;}
html, body, [data-testid="stApp"] {font-family: 'Pretendard', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;}
[data-testid="stMetric"] {background: #f5f9fc; border: 1px solid #deebf0; border-radius: 13px; padding: 14px 17px;}
[data-testid="stMetricLabel"] p {font-size: 0.88rem;}
[data-testid="stSidebar"] {border-right: 1px solid #e4edf2;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:0.65rem;}
section[data-testid="stMain"] h1 {letter-spacing:-1.0px; margin-bottom:0.1rem;}
.stTabs [data-baseweb="tab-list"] {gap: 8px; flex-wrap: wrap;}
.stTabs [data-baseweb="tab"] {border-radius: 9px; padding:9px 15px;}
[data-testid="stAlert"] {border-radius:12px;}
.kstep{padding:12px 16px;border-radius:12px;background:#f0f6fa;border:1px solid #dce9f0;color:#235066;font-size:0.95rem;}
.khint{color:#536877;font-size:.92rem;}
</style>""", unsafe_allow_html=True)

@st.cache_data(ttl=300, show_spinner=False)
def market_daily(symbol: str):
    import yfinance as yf
    data = yf.Ticker(symbol).history(period='2y', interval='1d', auto_adjust=True)
    return validate_ohlcv(data)

@st.cache_data(ttl=60, show_spinner=False)
def market_intraday(symbol: str):
    import yfinance as yf
    data = yf.Ticker(symbol).history(period='1d', interval='1m', auto_adjust=True)
    if data is None or data.empty or 'Close' not in data.columns:
        return pd.DataFrame()
    raw = data[['Close','Volume']].copy()
    idx = pd.DatetimeIndex(raw.index)
    if idx.tz is not None:
        raw.index = idx.tz_convert(KST)
    return raw

@st.cache_data(ttl=300, show_spinner=False)
def market_benchmark(ticker: str):
    import yfinance as yf
    return validate_ohlcv(yf.Ticker(ticker).history(period='2y', interval='1d', auto_adjust=True))

@st.cache_data(show_spinner=False)
def demo_daily(symbol: str):
    rng = np.random.default_rng(2006 + sum(map(ord, symbol)))
    idx = pd.bdate_range(end=pd.Timestamp.now(KST).date(), periods=510)
    chg = rng.normal(0.00035, 0.0185, len(idx))
    close = 30000 * np.exp(np.cumsum(chg))
    volume = rng.integers(200000, 2400000, len(idx))
    open_p = close * (1 + rng.normal(0, 0.004, len(idx)))
    hi = np.maximum(open_p, close) * 1.012
    lo = np.minimum(open_p, close) * 0.988
    return validate_ohlcv(pd.DataFrame({'Open':open_p,'High':hi,'Low':lo,'Close':close,'Volume':volume}, index=idx))


def name_of(sym):
    return STOCKS.get(sym, (sym, '직접 입력', '미지정'))[0]


def ticker_name(sym):
    return f'{name_of(sym)} · {sym.split(".")[0]}'


def label_of(score):
    if pd.isna(score): return '데이터 없음'
    if score >= 80: return '강세 후보'
    if score >= 65: return '관심 후보'
    if score >= 50: return '중립·관망'
    return '약세·주의'


def fmt_num(v, ndigits=0, unit=''):
    return f'{v:,.{ndigits}f}{unit}' if pd.notna(v) else '—'


def import_ratings(upload):
    if upload is None:
        return pd.DataFrame(), '미업로드'
    return normalized_rating_csv(upload.getvalue()), '사용자 업로드 (출처 미검증)'


def merged_ratings(upload_df, is_demo):
    if is_demo:
        source = normalized_rating_csv((ROOT/'data'/'sample_ratings.csv').read_bytes())
    else:
        source = upload_df.copy()
    # Session state overrides any uploaded values for that symbol. These are
    # ALWAYS identified as subjective user input, never a verified survey.
    for ticker, value in st.session_state.get('my_ratings', {}).items():
        code = ticker.split('.')[0]
        for field, score in value.items():
            source.loc[code, field] = score
    return source


if 'my_ratings' not in st.session_state:
    st.session_state.my_ratings = {}

with st.sidebar:
    st.markdown('## 📊 주식 추천 알고리즘')
    st.caption('누구나 쉽게 사용하는 국내 주식 추천 MVP')
    st.divider()
    st.subheader('① 분석할 종목')
    basket = st.selectbox('빠른 종목 선택', ['대표 종목 8개', '코스피 중심', '코스닥 중심', '직접 선택'], help='이 화면은 사전에 등록한 대표 종목을 조회합니다. 전체 상장종목 자동 스캔은 미지원입니다.')
    if basket == '대표 종목 8개':
        basis = DEFAULT
    elif basket == '코스피 중심':
        basis = [x for x, meta in STOCKS.items() if meta[1] == '코스피'][:12]
    elif basket == '코스닥 중심':
        basis = [x for x, meta in STOCKS.items() if meta[1] == '코스닥']
    else:
        basis = []
    symbols = st.multiselect('종목 추가·제외', options=list(STOCKS), default=basis,
                             format_func=ticker_name, help='네트워크 요청량을 고려해 최대 16종목까지 분석합니다.')
    st.caption(f'선택: {len(symbols)}종목 (최대 16종목)')
    st.subheader('② 분석 방식')
    mode = st.radio('주가 데이터', ['온라인 시세', '체험용 데모'], horizontal=True,
                    help='온라인: 외부 제공사 일봉 데이터. 체험용 데모: 무작위 생성 데이터로 절대 투자에 이용하지 마세요.')
    weights_show = st.checkbox('가중치 직접 바꾸기', value=False)
    if weights_show:
        w_tech = st.slider('기술적 가치', 0, 100, 60, 5)
        w_symbol = st.slider('상징적 가치', 0, 100 - w_tech, min(20,100-w_tech), 5)
    else:
        w_tech, w_symbol = 60, 20
    w_exp = 100-w_tech-w_symbol
    st.caption(f'기술 {w_tech}% + 상징 {w_symbol}% + 경험 {w_exp}%')
    st.subheader('③ 데이터 갱신')
    refresh = st.selectbox('자동 새로고침', ['끄기', '5분 간격', '1분 간격'], index=1)
    st.caption('자동 새로고침 = 주기적 재조회. 거래소 실시간 체결가를 뜻하지 않습니다.')
    if st.button('↻ 지금 다시 조회', use_container_width=True):
        market_daily.clear()
        market_intraday.clear()
        market_benchmark.clear()
        st.rerun()
    with st.expander('📁 가치 평가 데이터 업로드', expanded=False):
        uploaded = st.file_uploader('평가 결과 CSV', type=['csv'], help='고객 만족도·브랜드 관련 검증된 조사 자료를 입력하려면 CSV를 올리세요.')
        st.download_button('CSV 입력 양식 받기', (ROOT/'data'/'ratings_template.csv').read_bytes(), 'ratings_template.csv', mime='text/csv', use_container_width=True)
        st.caption('온라인 모드에서는 가치 데이터를 자동으로 만들어내지 않습니다.')

st.title('주식 추천 알고리즘')
st.caption('가격과 거래량으로 주식 후보를 찾고, 추천 이유·위험 요인과 브랜드·고객경험 가치를 함께 확인하세요.')
st.markdown('<div class="kstep">① 종목을 고르고　→　② 아래 추천 순위를 확인하고　→　③ 관심 종목의 그래프와 위험 신호를 살펴보세요.</div>', unsafe_allow_html=True)
st.write('')

is_demo = mode == '체험용 데모'
try:
    uploaded_ratings, rating_source = import_ratings(uploaded)
except Exception as exc:
    uploaded_ratings = pd.DataFrame()
    rating_source = '파일 형식 오류'
    st.error(f'업로드한 평가 파일에 문제가 있습니다: {exc}')
ratings = merged_ratings(uploaded_ratings, is_demo)

if is_demo:
    st.warning('🧪 **체험용 데모**: 현재 가격과 가치 평가점수는 모두 가상 데이터입니다. 실제 종목의 상태나 수익률을 나타내지 않습니다.')
else:
    st.info('📡 **온라인 가격**: Yahoo Finance 제공 일봉 데이터이며 지연·누락될 수 있습니다. 앱의 1분 갱신은 거래소 실시간 체결가 제공이 아닙니다. 가치점수는 실제 조사자료를 올리거나 직접 입력해야 합니다.')


@st.fragment(run_every={'끄기': None, '5분 간격':'5m', '1분 간격':'60s'}[refresh])
def main_dashboard():
    if not symbols:
        st.warning('왼쪽에서 분석할 종목을 하나 이상 선택해 주세요.')
        return
    if len(symbols) > 16:
        st.warning('종목은 한 번에 최대 16개까지 분석할 수 있습니다. 일부 종목을 해제해 주세요.')
        return
    seen = datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S KST')
    rows, frames, detail_points, errors = [], {}, {}, []
    with st.spinner(f'{len(symbols)}개 종목의 가격·거래량을 분석하고 있어요...'):
        for symbol in symbols:
            try:
                source = demo_daily(symbol) if is_demo else market_daily(symbol)
                df = indicators(source)
                tech, points = technical_score(df)
                code = symbol.split('.')[0]
                user_entered = symbol in st.session_state.my_ratings
                if code in ratings.index:
                    rr = ratings.loc[code]
                    sv, ev = rating_score(rr, SYMBOLIC), rating_score(rr, EXPERIENCE)
                else:
                    sv, ev = np.nan, np.nan
                integrated = combined_score(tech, sv, ev, (w_tech,w_symbol,w_exp))
                value_origin = ('기술 100% (가치 미사용)' if w_tech == 100 else
                                '개인 직접 입력' if user_entered else
                                '데모 가상 설문' if is_demo else
                                '업로드 평가(미검증)')
                x = df.iloc[-1]
                end_date = df.index[-1]
                age_days = (datetime.now(KST).date() - end_date.date()).days
                rows.append({
                    '종목명':name_of(symbol), '코드':code, '티커':symbol, '시장':STOCKS.get(symbol,('','미확인',''))[1],
                    '기술점수':int(tech), '종합점수':integrated, '상징점수':sv, '경험점수':ev,
                    '판정':label_of(integrated if pd.notna(integrated) else tech),
                    '기준일':str(end_date.date()), '종가':float(x.Close),
                    '20일수익률(%)':float(x.MOM20), 'RSI':float(x.RSI),
                    '연율변동성(%)':float(x.VOLATILITY), '최근거래량배수':float(x.VOLUME_RATIO) if pd.notna(x.VOLUME_RATIO) else np.nan,
                    '데이터경과일':age_days, '데이터상태':'7일 초과 경과' if age_days > 7 else '정상 범위',
                    '가치데이터':value_origin if pd.notna(integrated) else '입력 필요',
                    '거래일수':len(df), **points})
                detail_points[symbol] = points
                frames[symbol] = df
            except Exception as exc:
                errors.append(f'{ticker_name(symbol)}: {str(exc)[:180]}')
    if not rows:
        st.error('주가 데이터가 조회되지 않았어요. 인터넷 연결·티커·외부 제공사의 응답 상태를 확인하거나, 왼쪽에서 체험용 데모를 선택하세요.')
        with st.expander('오류 내용 확인'):
            for msg in errors: st.code(msg)
        return
    result = pd.DataFrame(rows)
    result['정렬점수'] = result['종합점수'].fillna(result['기술점수'])
    result = result.sort_values(['정렬점수','기술점수'], ascending=False)
    complete = int(result['종합점수'].notna().sum())
    fresh_count = int((result['데이터경과일'] <= 7).sum())
    st.markdown('### 오늘의 종목 탐색 결과')
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('분석 성공', f'{len(result)}/{len(symbols)}개')
    c2.metric('종합 평가 가능', f'{complete}개')
    c3.metric('가격 데이터 정상 범위',f'{fresh_count}/{len(result)}개')
    c4.metric('가장 높은 기술점수',f'{int(result["기술점수"].max())}/100')
    st.caption(f'조회 시각: {seen} · 가격: {"가상 생성" if is_demo else "Yahoo Finance / 일봉"} · 상징·경험 점수: {"가상 설문 + 사용자 입력" if is_demo else "사용자 입력 또는 업로드"} · 각 가격 기준일은 순위표에서 확인하세요.')
    if errors:
        with st.expander(f'⚠️ 수집 실패한 {len(errors)}개 종목 확인'):
            for msg in errors: st.write('• '+msg)
    if fresh_count < len(result) and not is_demo:
        st.warning('최근 7일 동안 갱신되지 않은 종목이 있어요. 해당 종목의 자료는 투자 판단 전 반드시 확인하세요.')

    page1, page2, page3, page4, page5 = st.tabs([
        '✨ 추천 종목', '📈 종목 자세히', '⭐ 브랜드·경험 평가', '🧪 과거 성과', '📚 데이터·이론'
    ])
    with page1:
        rank_mode = st.radio('추천 순위 기준', ['기술 기반 순위 (누구나 바로 사용)', '3가지 가치 종합 순위'], horizontal=True)
        if rank_mode.startswith('3가지'):
            visible = result.loc[result['종합점수'].notna()].sort_values('종합점수', ascending=False).copy()
            used = '종합점수'
            if visible.empty:
                st.warning('통합 점수를 계산하려면 상징적·경험적 가치 자료가 필요합니다. 「⭐ 브랜드·경험 평가」 탭에서 10개 항목을 직접 평가하거나 CSV를 업로드하세요.')
        else:
            visible = result.sort_values('기술점수', ascending=False).copy()
            used = '기술점수'
        if not visible.empty:
            columns = ['종목명','시장','종가','20일수익률(%)',used,'판정','기준일']
            if used == '종합점수':
                columns = ['종목명','시장','종가','기술점수','상징점수','경험점수','종합점수','판정','기준일']
            st.dataframe(visible[columns], hide_index=True, use_container_width=True,
                column_config={
                    '종가':st.column_config.NumberColumn('최근 종가 (원)', format='%,.0f'),
                    '20일수익률(%)':st.column_config.NumberColumn('20일 변화',format='%.1f%%'),
                    '기술점수':st.column_config.ProgressColumn('기술점수', min_value=0,max_value=100,format='%d'),
                    '종합점수':st.column_config.ProgressColumn('종합점수',min_value=0,max_value=100,format='%.1f'),
                    '상징점수':st.column_config.NumberColumn('상징',format='%.1f'),
                    '경험점수':st.column_config.NumberColumn('경험',format='%.1f'),
                })
            st.download_button('⬇️ 종목 추천 결과 CSV 저장', visible.drop(columns=['정렬점수']).to_csv(index=False).encode('utf-8-sig'),
                '주식_추천_알고리즘_종목순위.csv', mime='text/csv', use_container_width=True)
            color_bar = '#087e88'
            fig = px.bar(visible.sort_values(used),x=used,y='종목명',orientation='h',text=used,
                        range_x=[0,105],title='비교한 종목의 점수')
            fig.update_traces(marker_color=color_bar,textposition='outside')
            fig.update_layout(height=max(310,45*len(visible)),margin=dict(t=50,l=5,r=15,b=10),
                              xaxis_title='점수 / 100',yaxis_title=None)
            st.plotly_chart(fig,use_container_width=True)
            best = visible.iloc[0]
            st.success(f'**최상위 탐색 후보: {best["종목명"]}** · {used} {best[used]:.1f}/100 · {best["판정"]}. 이는 선정된 종목 간 상대적인 규칙 기반 점수이며 상승 확률을 뜻하지 않습니다.')
        st.caption('점수는 매수 신호나 수익률 예측이 아닙니다. 종합 평가가 없는 종목을 기술점수로 자동 대체해 종합 순위에 섞지 않습니다.')

    with page2:
        st.subheader('한 종목을 자세히 확인해요')
        selected = st.selectbox('종목 찾기', options=list(frames), format_func=ticker_name, key='detail_stock')
        df = frames[selected]
        info = result.loc[result['티커']==selected].iloc[0]
        pts = detail_points[selected]
        a,b,c,dcol = st.columns(4)
        a.metric('최근 종가',fmt_num(info['종가'],0,'원'))
        b.metric('20거래일 변화',fmt_num(info['20일수익률(%)'],1,'%'))
        c.metric('기술 평가',f'{info["기술점수"]}/100')
        dcol.metric('최근 변동성 (연율)',fmt_num(info['연율변동성(%)'],1,'%'))
        st.caption(f'이 가격의 기준일: {info["기준일"]} · 데이터 제공사 시점 기준으로 최신일 수도, 지연된 자료일 수도 있습니다.')
        last_year = df.Close.tail(252)
        with st.expander('지난 1년 가격 위치와 거래량 확인'):
            lo_price, hi_price = float(last_year.min()), float(last_year.max())
            pos_price = ((float(info['종가'])-lo_price) / (hi_price-lo_price)) if hi_price > lo_price else 0.5
            st.progress(float(np.clip(pos_price,0,1)))
            st.caption(f'지난 최대 252거래일 **종가** 기준 낮은 가격 {lo_price:,.0f}원 ↔ 높은 가격 {hi_price:,.0f}원 · 현재 위치 {pos_price*100:.0f}%')
            vfig=px.bar(df.tail(90).rename_axis('날짜').reset_index(),x='날짜',y='Volume',title='최근 90거래일 거래량')
            vfig.update_traces(marker_color='#7a9eac')
            vfig.update_layout(height=230,margin=dict(t=45,l=10,r=10,b=10))
            st.plotly_chart(vfig,use_container_width=True)
        fig = go.Figure()
        for field,title,c in [('Close','종가','#1a748c'),('MA20','20일 평균','#e2a11c'),('MA60','60일 평균','#8d65bb')]:
            fig.add_trace(go.Scatter(x=df.index,y=df[field],name=title,line=dict(color=c,width=2 if field=='Close' else 1.5)))
        fig.update_layout(title=f'{name_of(selected)} 가격 흐름과 이동평균',height=390,margin=dict(t=50,l=10,b=20,r=15),hovermode='x unified')
        st.plotly_chart(fig,use_container_width=True)
        st.markdown('#### 왜 이런 점수를 받았나요?')
        pos, neg = signal_explanation(pts,df.iloc[-1])
        left,right = st.columns(2)
        with left:
            st.markdown('**✅ 확인된 긍정 신호**')
            for x in pos: st.write('• '+x)
            if not pos: st.caption('현재 확인된 긍정 신호가 제한적입니다.')
        with right:
            st.markdown('**⚠️ 주의할 신호**')
            for x in neg: st.write('• '+x)
            if not neg: st.caption('규칙상 추가 경고 신호는 없습니다. 위험이 없다는 뜻은 아닙니다.')
        st.markdown('**기술적 점수 구성**')
        names = {'추세':30,'모멘텀':25,'거래량':20,'MACD':15,'위험':10}
        parts_df = pd.DataFrame({'평가요소':list(names),'받은 점수':[pts[k] for k in names],
                                 '최대 점수':list(names.values())})
        fig2=px.bar(parts_df,x='평가요소',y='받은 점수',text='받은 점수',range_y=[0,35],
                    title='항목별 평가 (각 항목의 최대점수는 서로 다릅니다)')
        fig2.update_traces(marker_color='#3699a3')
        st.plotly_chart(fig2,use_container_width=True)
        x1,x2=st.columns(2)
        with x1:
            f=px.line(df.rename_axis('날짜').reset_index(),x='날짜',y=['MACD','SIGNAL'],title='MACD — 추세 변화')
            f.update_layout(height=310,margin=dict(t=50,l=10,r=10,b=15))
            st.plotly_chart(f,use_container_width=True)
        with x2:
            f=px.line(df.rename_axis('날짜').reset_index(),x='날짜',y='RSI',title='RSI — 과열 확인')
            f.add_hline(y=70,line_dash='dash',line_color='#e09247')
            f.add_hline(y=30,line_dash='dash',line_color='#6e8abc')
            f.update_layout(height=310,margin=dict(t=50,l=10,r=10,b=15))
            st.plotly_chart(f,use_container_width=True)
        with st.expander('📡 장중 1분봉 확인 (온라인 전용 · 제공사 지연 가능)',expanded=False):
            if is_demo:
                st.info('가상 데이터 모드에서는 장중 그래프가 제공되지 않습니다.')
            elif st.checkbox('선택 종목 장중 데이터 불러오기',key='intraday_enabled'):
                try:
                    intra=market_intraday(selected)
                    if intra.empty:
                        st.warning('이 종목의 장중 1분 데이터가 현재 제공되지 않습니다. 휴장일 또는 제공사 제한일 수 있습니다.')
                    else:
                        intra_plot=intra.copy()
                        st.line_chart(intra_plot['Close'],height=260)
                        st.caption(f'마지막 1분봉 시각: {intra_plot.index[-1]} · 공급자 지연/누락 가능. 실시간 호가·체결 보장 아님.')
                except Exception as exc:
                    st.warning(f'장중 데이터 조회 실패: {exc}')

    with page3:
        st.subheader('상징적 가치와 실제 고객경험을 점수에 반영')
        st.write('브랜드 인지도·신뢰도와 고객 만족도·재이용률을 함께 평가할 수 있습니다. **실제 조사 결과가 아닌 개인 평가는 주관적인 입력임을 명확히 표시합니다.**')
        st.caption('기본 가중치: 기술 60% · 상징 20% · 경험 20%. 온라인 모드에서 이용자가 입력하지 않은 가치점수는 계산하지 않습니다.')
        ev_sym=st.selectbox('평가할 기업',list(frames),format_func=ticker_name,key='eval_stock')
        base_values=st.session_state.my_ratings.get(ev_sym,{})
        s_col,e_col=st.columns(2)
        with st.form('ratings_entry'):
            s_col,e_col=st.columns(2)
            inputs={}
            with s_col:
                st.markdown('**상징적 가치 — 100점 중 20% 반영**')
                for key in SYMBOLIC:
                    inputs[key]=st.slider(key,1,5,int(base_values.get(key,3)),key='rating_'+key,
                                            help='1 매우 낮음 · 3 보통 · 5 매우 높음')
            with e_col:
                st.markdown('**경험적 가치 — 100점 중 20% 반영**')
                for key in EXPERIENCE:
                    inputs[key]=st.slider(key,1,5,int(base_values.get(key,3)),key='rating_'+key,
                                            help='1 매우 낮음 · 3 보통 · 5 매우 높음')
            saved=st.form_submit_button('✓ 이 종목 평가 반영하기',use_container_width=True,type='primary')
        if saved:
            st.session_state.my_ratings[ev_sym]=inputs
            st.rerun()
        code=ev_sym.split('.')[0]
        if code in ratings.index:
            rr=ratings.loc[code]
            s_score=rating_score(rr,SYMBOLIC)
            e_score=rating_score(rr,EXPERIENCE)
            first,second,third=st.columns(3)
            first.metric('상징적 가치',fmt_num(s_score,1,'/100'))
            second.metric('경험적 가치',fmt_num(e_score,1,'/100'))
            th=int(result.loc[result['티커']==ev_sym,'기술점수'].iloc[0])
            third.metric('3가지 통합점수',fmt_num(combined_score(th,s_score,e_score,(w_tech,w_symbol,w_exp)),1,'/100'))
            kind = '나의 주관적 평가' if ev_sym in st.session_state.my_ratings else '데모 가상 점수' if is_demo else '업로드된 외부 평점 (검증 안 됨)'
            st.caption(f'현재 가치점수 근거: {kind}. 이 점수는 투자 성과가 검증된 지표가 아닙니다.')
        else:
            st.info('아직 이 종목의 상징·경험 데이터가 없어요. 위 항목을 평가하고 버튼을 누르면 종합점수가 나옵니다.')
        with st.expander('평가 항목 배점·CSV 업로드 자세히 보기'):
            st.dataframe(pd.DataFrame({'상징적 측정요소':list(SYMBOLIC),'배점':list(SYMBOLIC.values())}),hide_index=True,use_container_width=True)
            st.dataframe(pd.DataFrame({'경험적 측정요소':list(EXPERIENCE),'배점':list(EXPERIENCE.values())}),hide_index=True,use_container_width=True)
            st.caption('1~5점을 0~100점으로 환산합니다. 실제 조사 수치를 쓰려면 조사기관, 표본 수, 조사일과 산업별 비교가능성 확인이 필요합니다.')
            st.caption('CSV 양식은 왼쪽 사이드바의 「가치 평가 데이터 업로드」에서 받을 수 있습니다.')
        if st.session_state.my_ratings:
            personal=pd.DataFrame([{'종목코드':sym.split('.')[0],**entry} for sym,entry in st.session_state.my_ratings.items()])
            st.download_button('⬇️ 내가 입력한 평가 CSV 저장', personal.to_csv(index=False).encode('utf-8-sig'),
                               'my_value_ratings.csv',mime='text/csv')
            st.caption('개인 평가는 현재 브라우저 세션 동안 유지되며, 파일로 저장하지 않으면 종료 후 사라질 수 있습니다.')

    with page4:
        st.subheader('과거에도 이런 규칙이 통했을까요?')
        st.caption('이 진단은 **기술적 점수만** 검증합니다. 상징·경험 가치의 과거 시점별 조사 자료가 없어 통합 알고리즘 백테스트로 해석하면 안 됩니다.')
        hold = st.select_slider('한 번 선정 후 보유 기간', options=[10,20,40],value=20,format_func=lambda n:f'{n}거래일')
        top_n = st.slider('매 구간 최대 선정 종목 수',1,min(5,len(frames)),min(3,len(frames)))
        threshold=st.slider('추천 진입 기준 점수',45,90,65,5)
        cost=st.slider('편도 거래비용 추정 (bp)',0,50,15,5,help='예: 편도 15bp는 0.15%이며 매수+매도 비용은 약 0.30%로 계산합니다.')
        bm_label = st.selectbox('비교 지수',['KOSPI','KOSDAQ'])
        benchmark=None
        if not is_demo:
            try:
                benchmark=market_benchmark('^KS11' if bm_label=='KOSPI' else '^KQ11')
            except Exception as exc:
                st.warning(f'지수 수집 실패: {exc}. 전략의 자체 수익만 표시합니다.')
        hist=technical_universe_backtest(frames,benchmark=benchmark,lookback=100,
                                         holding=hold,top_n=top_n,threshold=threshold,cost_bps=cost)
        if hist.empty:
            st.warning('분석 종목의 공통 거래일 데이터가 부족해 결과를 계산할 수 없습니다.')
        else:
            summary=strategy_summary(hist)
            m1,m2,m3=st.columns(3)
            m1.metric('과거 누적 전략 수익',fmt_num(summary['총수익률(%)'],1,'%'))
            m2.metric('과거 최대 낙폭',fmt_num(summary['최대낙폭(%)'],1,'%'))
            m3.metric('수익 구간 비율',fmt_num(summary['수익구간비율(%)'],1,'%'))
            plot=hist.rename(columns={'종료일':'기간 끝','전략누적(%)':'전략(%)','지수누적(%)':'비교지수(%)'})
            y=['전략(%)']
            if benchmark is not None and plot['비교지수(%)'].notna().any(): y.append('비교지수(%)')
            fig=px.line(plot,x='기간 끝',y=y,title='기간별 가상 누적 수익 비교')
            fig.add_hline(y=0,line_dash='dot')
            st.plotly_chart(fig,use_container_width=True)
            st.dataframe(hist,hide_index=True,use_container_width=True)
            st.download_button('⬇️ 검증 결과 CSV 저장',hist.to_csv(index=False).encode('utf-8-sig'),
                               '주식_추천_알고리즘_과거진단.csv',mime='text/csv')
        st.warning('**검증 한계**: 오늘 선택한 종목만 과거에도 존재했다고 가정하는 생존·선택편향이 있습니다. 당일 종가에 진입하는 낙관적 체결 가정, 세금·슬리피지의 불완전한 반영, 산업 편중, 통합 가치점수 미검증 등의 한계가 있습니다. 성과를 보장하는 결과가 아닙니다.')

    with page5:
        st.subheader('어떤 이론을 사용했고, 무엇이 아직 검증되지 않았을까요?')
        st.markdown('''
**① 기술적 분석·모멘텀** — 이동평균 추세, Wilder 방식 RSI(14), MACD, 거래량 비율, 20일 수익률, 변동성. 과거 가격을 이용해 현재 상태를 탐색합니다.

**② 상징적 가치** — Keller(1993)의 고객 기반 브랜드 자산, Aaker(1991)의 브랜드 자산 개념, 브랜드 신뢰·혁신 정체성·문화적 상징성. 브랜드를 기업 재무가치와 동일시하지 않습니다.

**③ 경험적 가치** — Pine & Gilmore(1998)의 경험경제, Schmitt(1999)의 경험적 마케팅, Lemon & Verhoef(2016)의 고객경험 여정. 만족도와 재이용을 측정하려면 외부 조사 데이터가 필요합니다.

**④ 행동재무학** — 투자자 관심, 과신, 유행과 같은 비가격 정보를 고려하되, 브랜드 인기가 투자수익률을 보장하지 않는다는 점을 강조합니다.

**⑤ 실무적 검토** — 종목별 동일 가중 기술적 선별의 과거 비교, 불완전한 비용 추정, 공개 시점 자료 기준의 미래정보 누출 통제, 결측치 표시, 데이터 제공자 오류 대비가 필요합니다.
''')
        st.markdown('#### 100점 점수 산식')
        st.latex(r'S=\frac{w_T T+w_S V_S+w_E V_E}{100},\quad w_T+w_S+w_E=100')
        st.write(f'현재 설정: 기술 {w_tech}% / 상징 {w_symbol}% / 경험 {w_exp}%')
        st.caption('기술점수 100점 내 배점: 추세 30점·모멘텀 25점·거래량 20점·MACD 15점·변동성 10점. 상징 5개·경험 5개, 총 10개 항목의 별도 평점으로 계산합니다.')
        st.markdown('#### 데이터 신뢰성 확인')
        quality=pd.DataFrame([
            {'항목':'시세 데이터','상태':'가상 시세' if is_demo else '외부 실제 일봉','출처/제약':'앱 생성' if is_demo else 'Yahoo Finance, 5분 캐시, 시차 가능'},
            {'항목':'상징·경험 가치','상태':f'{complete}/{len(result)}종목 입력','출처/제약':rating_source if rating_source!='미업로드' else '업로드 없음; 사용자 직접 평가 가능'},
            {'항목':'분석 데이터 누락','상태':f'{len(errors)}건','출처/제약':'네트워크·상장상태·일봉 부족 시 제외'},
            {'항목':'최신성','상태':f'{fresh_count}/{len(result)}종목 7일 이내','출처/제약':'휴장·공급자 지연은 별도 확인 필요'},
            {'항목':'백테스트','상태':'기술점수 기반 모의 진단','출처/제약':'후보군 편향, 낙관적 당일 종가, 비용 추정 한계'},
        ])
        st.dataframe(quality,hide_index=True,use_container_width=True)
        st.caption('실제 가치점수는 조사일·조사기관·표본 수와 출처를 첨부해 관리하는 것이 바람직합니다. 임의 입력 평가나 예시 데이터는 확인된 사실로 표시하지 않습니다.')
        with st.expander('연구·검증 출처'):
            st.markdown('''
- Aaker (1991), *Managing Brand Equity*.
- Keller (1993), *Conceptualizing, Measuring, and Managing Customer-Based Brand Equity*, Journal of Marketing.
- Pine & Gilmore (1998), *Welcome to the Experience Economy*, Harvard Business Review.
- Schmitt (1999), *Experiential Marketing*.
- Lemon & Verhoef (2016), *Understanding Customer Experience Throughout the Customer Journey*, Journal of Marketing.
- Jegadeesh & Titman (1993), *Returns to Buying Winners and Selling Losers*, Journal of Finance.
- Wilder (1978), *New Concepts in Technical Trading Systems*.
- Thaler (1999), *The End of Behavioral Finance*, Financial Analysts Journal.
- 기술 사용: [Streamlit 문서](https://docs.streamlit.io/) · [yfinance 문서](https://ranaroussi.github.io/yfinance/)
''')

main_dashboard()
st.divider()
st.caption('주식 추천 알고리즘 · 교육·탐색 목적의 주식 분석 도구. 종목 순위는 규칙에 따른 결과이지 특정 매수·매도 권유나 미래 수익률 예측이 아닙니다. 시세·고객평가 데이터 및 백테스트의 제한을 반드시 확인하세요.')
