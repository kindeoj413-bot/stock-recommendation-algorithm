"""Pure analysis utilities; no API calls or Streamlit dependencies."""
from __future__ import annotations
import numpy as np
import pandas as pd

SYMBOLIC = {'브랜드인지도':25,'사회적신뢰':25,'혁신상징성':20,'문화영향력':15,'브랜드충성도':15}
EXPERIENCE = {'고객만족도':30,'재구매율':25,'제품품질':20,'사용자경험':15,'경험개선':10}
ALL_FIELDS = list(SYMBOLIC) + list(EXPERIENCE)

def validate_ohlcv(df):
    if df is None or df.empty: raise ValueError('가격 데이터가 없습니다.')
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.rename(columns=str.title)
    needed = ['Open','High','Low','Close','Volume']
    missing = set(needed) - set(d.columns)
    if missing: raise ValueError(f'가격 필드 누락: {missing}')
    d = d[needed].apply(pd.to_numeric, errors='coerce')
    d.index = pd.to_datetime(d.index).tz_localize(None) if getattr(d.index,'tz',None) else pd.to_datetime(d.index)
    d = d.loc[~d.index.duplicated(keep='last')].sort_index()
    d = d.dropna(subset=['Close','Volume'])
    d = d[(d.Close > 0)&(d.Volume >= 0)]
    if len(d)<80: raise ValueError(f'80거래일 미만: {len(d)}일')
    return d

def indicators(raw):
    d=validate_ohlcv(raw).copy()
    c=d.Close
    d['MA20']=c.rolling(20).mean(); d['MA60']=c.rolling(60).mean()
    delta=c.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    rs=gain/loss.replace(0,np.nan)
    d['RSI']=100-100/(1+rs)
    d.loc[(loss==0)&(gain>0),'RSI']=100
    d.loc[(loss==0)&(gain==0),'RSI']=50
    d['MACD']=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    d['SIGNAL']=d.MACD.ewm(span=9,adjust=False).mean()
    d['VOLUME_RATIO']=d.Volume/d.Volume.shift(1).rolling(20).mean().replace(0,np.nan)
    d['MOM20']=c.pct_change(20)*100
    d['VOLATILITY']=c.pct_change().rolling(20).std()*np.sqrt(252)*100
    return d

def technical_score(d, position=-1):
    i=len(d)+position if position<0 else position
    if i<60 or i>=len(d): raise ValueError('기술점수 계산에 필요한 기간 부족')
    x=d.iloc[i]; prev=d.iloc[i-1]
    necessary=['Close','MA20','MA60','RSI','MACD','SIGNAL','MOM20','VOLATILITY']
    if x[necessary].isna().any(): raise ValueError('필수 기술지표 누락')
    points={
        '추세':int(x.Close>x.MA20)*10+int(x.MA20>x.MA60)*10+int(x.Close>x.MA60)*10,
        '모멘텀':(15 if 45<=x.RSI<=65 else 8 if 35<=x.RSI<45 else 0)+(10 if 3<=x.MOM20<=15 else 5 if 0<x.MOM20<3 else 0),
        '거래량':20 if 1.5<=x.VOLUME_RATIO<=3 else 10 if 1.2<=x.VOLUME_RATIO<1.5 else 5 if 3<x.VOLUME_RATIO<=5 else 0,
        'MACD':int(x.MACD>x.SIGNAL)*10+int(prev.MACD<=prev.SIGNAL and x.MACD>x.SIGNAL)*5,
        '위험':10 if x.VOLATILITY<30 else 5 if x.VOLATILITY<50 else 0,
    }
    return int(sum(points.values())),points

def rating_score(row,weights):
    values=pd.to_numeric(pd.Series({k:row.get(k,np.nan) for k in weights}),errors='coerce')
    if values.isna().any() or not values.between(1,5).all(): return np.nan
    return round(sum((values[k]-1)/4*w for k,w in weights.items()),2)

def combined_score(technical,symbolic,experience,weights=(60,20,20)):
    if sum(weights)!=100: raise ValueError('가중치 합계는 100이어야 합니다.')
    values = [technical, symbolic, experience]
    # Missing dimensions with zero user weight must not block a pure technical score.
    if any(not np.isfinite(value) for value, weight in zip(values, weights) if weight > 0):
        return np.nan
    return round(sum(float(value) * weight for value, weight in zip(values, weights) if weight > 0) / 100, 2)

def grade(score):
    if pd.isna(score): return '평가 불가'
    return '매수 검토 후보' if score>=80 else '관심 종목' if score>=65 else '관망' if score>=50 else '제외'

def evaluate_forward(d,step=20):
    """Descriptive historical diagnostics: score at t and realized t+step returns, not a trading strategy."""
    rows=[]
    for i in range(80,len(d)-step,step):
        try: t,_=technical_score(d,i)
        except ValueError: continue
        p0=float(d.Close.iloc[i]);p1=float(d.Close.iloc[i+step])
        rows.append({'기준일':d.index[i], '기술점수':t,'이후수익률(%)':(p1/p0-1)*100,'20일기준점수상위':t>=65})
    return pd.DataFrame(rows)


def signal_explanation(parts, row):
    """Readable explanation grounded exclusively in measured price indicators."""
    positives, cautions = [], []
    if parts['추세'] >= 20:
        positives.append('단기·중기 상승 추세가 확인됩니다')
    else:
        cautions.append('이동평균 추세가 충분히 강하지 않습니다')
    if parts['모멘텀'] >= 15:
        positives.append('RSI와 최근 수익률의 균형이 양호합니다')
    else:
        cautions.append('모멘텀 지표의 개선을 확인할 필요가 있습니다')
    if pd.isna(row['VOLUME_RATIO']):
        cautions.append('거래량 비교에 필요한 데이터가 부족합니다')
    elif parts['거래량'] >= 10:
        positives.append('최근 거래량이 평소보다 늘었습니다')
    else:
        cautions.append('거래량 증가 신호가 제한적입니다')
    if parts['MACD'] >= 10:
        positives.append('MACD가 시그널선을 웃돌고 있습니다')
    else:
        cautions.append('MACD 상승 신호가 확인되지 않습니다')
    if pd.notna(row['RSI']) and row['RSI'] >= 70:
        cautions.append('RSI 70 이상으로 단기 과열 가능성이 있습니다')
    if pd.notna(row['VOLATILITY']) and row['VOLATILITY'] >= 50:
        cautions.append('최근 변동성이 높은 편입니다')
    return positives, cautions


def normalized_rating_csv(data):
    """Read a CSV with stringent format checks; never fabricate missing survey inputs."""
    from io import BytesIO
    if isinstance(data, bytes):
        d = pd.read_csv(BytesIO(data), dtype={'종목코드': str}, encoding='utf-8-sig')
    else:
        d = pd.read_csv(data, dtype={'종목코드': str}, encoding='utf-8-sig')
    required = ['종목코드'] + ALL_FIELDS
    missing = [col for col in required if col not in d.columns]
    if missing:
        raise ValueError('필수 열이 없습니다: ' + ', '.join(missing))
    d['종목코드'] = d['종목코드'].astype(str).str.strip().str.zfill(6)
    if not d['종목코드'].str.fullmatch(r'\d{6}').all():
        raise ValueError('종목코드는 6자리 숫자여야 합니다.')
    if d['종목코드'].duplicated().any():
        raise ValueError('중복 종목코드가 있습니다. 각 종목은 1행만 입력하세요.')
    for field in ALL_FIELDS:
        d[field] = pd.to_numeric(d[field], errors='coerce')
        illegal = d[field].notna() & ~d[field].between(1, 5)
        if illegal.any():
            raise ValueError(f'{field}: 평점은 1~5 범위여야 합니다.')
    if '조사일' in d.columns:
        d['조사일'] = pd.to_datetime(d['조사일'], errors='coerce')
    return d.set_index('종목코드')


def rating_is_complete(row):
    if row is None:
        return False
    s = pd.to_numeric(pd.Series({field: row.get(field, np.nan) for field in ALL_FIELDS}), errors='coerce')
    return bool(s.notna().all() and s.between(1, 5).all())


def technical_universe_backtest(frames, benchmark=None, lookback=100, holding=20,
                                top_n=3, threshold=65, cost_bps=15):
    """Point-in-time TECHNICAL ONLY screening diagnostic.

    Uses trailing close-derived indicators available at decision date. Invests on the
    same decision-day close (optimistic execution convention, explicitly disclosed).
    Charges fixed round-trip cost each invested period; no intraday price lookahead.
    Current handpicked universe entails survival/selection bias.
    """
    candidates = {k: v for k, v in frames.items() if len(v) >= lookback + holding + 1}
    if not candidates:
        return pd.DataFrame()
    dates = None
    for df in candidates.values():
        i = pd.DatetimeIndex(df.index)
        dates = i if dates is None else dates.intersection(i)
    dates = dates.sort_values()
    if len(dates) < lookback + holding + 1:
        return pd.DataFrame()
    if benchmark is not None and not benchmark.empty:
        benchmark = benchmark.sort_index()['Close'].astype(float)
    output = []
    nav, bench_nav = 1.0, 1.0
    for pos in range(lookback - 1, len(dates) - holding, holding):
        t0 = dates[pos]
        t1 = dates[pos + holding]
        scored = []
        for symbol, d in candidates.items():
            ix = int(d.index.searchsorted(t0, side='right') - 1)
            if ix < 60 or d.index[ix] != t0:
                continue
            try:
                score, _ = technical_score(d, ix)
            except (ValueError, IndexError):
                continue
            if score >= threshold:
                scored.append((symbol, score))
        selected = [sym for sym, score in sorted(scored, key=lambda x: (-x[1], x[0]))[:top_n]]
        if selected:
            rets = [float(candidates[s].at[t1, 'Close']) / float(candidates[s].at[t0, 'Close']) - 1
                    for s in selected]
            gross = float(np.mean(rets))
            net = gross - 2 * cost_bps / 10000
        else:
            net = 0.0  # zero-interest cash proxy
        nav *= 1 + net
        bench_ret = np.nan
        if benchmark is not None and t0 in benchmark.index and t1 in benchmark.index:
            bench_ret = float(benchmark.at[t1] / benchmark.at[t0] - 1)
            bench_nav *= 1 + bench_ret
        output.append({'시작일': t0, '종료일': t1, '선정종목': ', '.join(selected) if selected else '현금',
                       '선정수':len(selected), '전략수익률(%)':round(net*100,3),
                       '지수수익률(%)':round(bench_ret*100,3) if pd.notna(bench_ret) else np.nan,
                       '전략누적(%)':round((nav-1)*100,3),
                       '지수누적(%)':round((bench_nav-1)*100,3) if benchmark is not None else np.nan})
    return pd.DataFrame(output)


def strategy_summary(result):
    if result.empty:
        return {'구간수':0,'총수익률(%)':np.nan,'최대낙폭(%)':np.nan,'수익구간비율(%)':np.nan}
    r = result['전략수익률(%)'].to_numpy(dtype=float)/100
    equity = np.r_[1., np.cumprod(1+r)]
    peak = np.maximum.accumulate(equity)
    return {'구간수':len(r), '총수익률(%)':round((equity[-1]-1)*100,2),
            '최대낙폭(%)':round(float(np.min(equity/peak-1)*100),2),
            '수익구간비율(%)':round(float(np.mean(r>0)*100),1)}
