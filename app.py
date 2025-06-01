# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# 日本語フォントを使いたい場合（Windows 例：Meiryo）
plt.rcParams['font.family'] = 'Meiryo'

# --------------------------------------------
# 1) データ取得をキャッシュ化
# --------------------------------------------
# @st.cache_data(show_spinner=False)
def load_price_data(ticker: str, start: str, end: str) -> pd.Series:
    """
    yfinance を使って日次終値を取得し、DatetimeIndex を normalize して返す。
    """
    df = yf.download(ticker, start=start, end=end, interval="1d")[["Close"]].dropna()
    s = df["Close"]
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s

# アプリ起動時に取得するデータ
st.sidebar.header("データ設定")
with st.sidebar.expander("銘柄と期間の設定"):
    ticker = st.sidebar.selectbox(
        "対象インデックス・ETF", 
        ("^N225", "^GSPC", "SPY"), 
        index=0,
        help="日経 (^N225)、S&P500 (^GSPC)、SPY のいずれかを選択"
    )
    start_date = st.sidebar.date_input("開始日", value=pd.to_datetime("2014-01-01"))
    end_date   = st.sidebar.date_input("終了日", value=pd.to_datetime("2024-12-31"))

# 日経平均などの価格 Series を取得
price = load_price_data(ticker, start_date.isoformat(), end_date.isoformat())

# --------------------------------------------
# 2) 月次積立日算出関数
# --------------------------------------------
@st.cache_data(show_spinner=False)
def get_monthly_dates(price_index: pd.DatetimeIndex, buy_day: int = 7) -> list[pd.Timestamp]:
    """
    毎月 buy_day 日（例：7日）が非営業日の場合、直近の営業日に前倒しした
    '月次積立日' のリストを返す。normalize 済みの price_index を想定。
    """
    monthly_dates = []
    years = range(price_index[0].year, price_index[-1].year + 1)
    for y in years:
        for m in range(1, 13):
            # まず  y-m-buy_day のタイムスタンプを作成し normalize
            t = pd.Timestamp(y, m, buy_day).normalize()
            # 月初より前まで調整しながら、その日が price_index に存在するか探す
            while t not in price_index and t > pd.Timestamp(y, m, 1):
                t -= pd.Timedelta(days=1)
                t = t.normalize()
            if t in price_index:
                monthly_dates.append(t)
    # 範囲外（開始以前または終了以降）の日付は除外
    monthly_dates = [d for d in monthly_dates if d >= price_index[0] and d <= price_index[-1]]
    return monthly_dates

# デフォルトでは7日を積立日にする
monthly_dates = get_monthly_dates(price.index, buy_day=7)

# --------------------------------------------
# 3) シミュレーション関数
# --------------------------------------------
def simulate_strategy(
    price: pd.Series,
    monthly_dates: list[pd.Timestamp],
    monthly_amount: float,
    x: float,   # 現金保持率
    y: float,   # 暴落判定閾値
    z: float    # 買い増し率
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    - price: normalize 済みの日次終値 Series
    - monthly_dates: "積立日" として使う日付のリスト
    - monthly_amount: 月次積立額（円）
    - x: 現金保持率 (0～1)
    - y: 暴落判定閾値 (0～1)
    - z: 買い増し率 (0～1)

    戦略を日次でシミュレートし、
    - daily_dates: シミュレーション対象の日付配列
    - daily_values: 日次の「資産評価額」（現金＋口数×価格）配列
    - final_value: 最終時点の評価額 （float）
    - total_invested: 実際に投入した累計金額（float）
    """
    cash = 0.0
    units = 0.0
    peak = -float("inf")
    total_invested = 0.0

    daily_dates = []
    daily_values = []

    for today in price.index:
        P = float(price.loc[today])

        # 1: 月次積立がある日
        if today in monthly_dates:
            invest = monthly_amount * (1 - x)
            cash += monthly_amount * x
            units += invest / P
            total_invested += monthly_amount

        # 2: 最高値更新
        peak = max(peak, P)

        # 3: 暴落判定・押し目買い
        if P <= (1 - y) * peak and cash > 0:
            buy_cash = cash * z
            units += buy_cash / P
            cash -= buy_cash

        # 日次の総資産評価額 (現金 + 株数×価格)
        current_value = units * P + cash
        daily_dates.append(today)
        daily_values.append(current_value)

    final_value = float(daily_values[-1])
    return np.array(daily_dates), np.array(daily_values), final_value, total_invested

# --------------------------------------------
# 4) サイドバー：ユーザー入力ウィジェット
# --------------------------------------------
st.sidebar.header("シミュレーション設定")

# 4-1) 月次積立額
monthly_amount = st.sidebar.slider(
    "月次積立額（円）", min_value=10_000, max_value=200_000, step=10_000, value=100_000
)

# 4-2) 現金保持率 x
x = st.sidebar.slider(
    "現金保持率 x (0～1)", min_value=0.0, max_value=1.0, step=0.01, value=0.4
)

# 4-3) 暴落閾値 y
y = st.sidebar.slider(
    "暴落閾値 y (0～1)", min_value=0.0, max_value=0.5, step=0.01, value=0.20
)

# 4-4) 買い増し率 z
z = st.sidebar.slider(
    "買い増し率 z (0～1)", min_value=0.0, max_value=1.0, step=0.01, value=1.0
)

# 4-5) 積立日（デフォルトは7日。ユーザーが変更したい場合）
buy_day = st.sidebar.number_input(
    "積立日 (1～28)", min_value=1, max_value=28, value=7, step=1
)

# 4-6) 「再計算」ボタン（押したときだけシミュレーションする仕様にしたい場合）
run_button = st.sidebar.button("シミュレーション実行")

# --------------------------------------------
# 5) ユーザー入力を反映して必要なデータを更新
# --------------------------------------------
# (A) monthly_dates を再計算
monthly_dates = get_monthly_dates(price.index, buy_day=buy_day)

# (B) ボタン任意ではなく常に再計算したい場合は
#     run_button のチェックを省いて直接実行してもOK

if run_button or True:
    # 日次シミュレーションを実行
    daily_dates, daily_values, final_value, total_invested = simulate_strategy(
        price=price,
        monthly_dates=monthly_dates,
        monthly_amount=monthly_amount,
        x=x,
        y=y,
        z=z
    )

    # 5-1) メトリクス表示
    st.subheader("シミュレーション結果")
    col1, col2, col3 = st.columns(3)
    col1.metric("最終評価額（円）", f"{int(final_value):,}")
    col2.metric("総投資額（円）", f"{int(total_invested):,}")
    col3.metric("最終リターン倍率", f"{final_value/total_invested:.2f}×")

    # 5-2) 資産推移グラフ
    st.subheader("資産推移グラフ")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily_dates, daily_values, label=f"x={x}, y={y}, z={z}")
    ax.set_title(f"{ticker} 定額＋押し目買いシミュレーション")
    ax.set_xlabel("年月日")
    ax.set_ylabel("資産評価額（円）")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# --------------------------------------------
# 6) （オプション）パラメータ感度ヒートマップなど
# --------------------------------------------
with st.expander("パラメータ感度分析"):
    st.write("ここに細かいグリッドサーチ結果のヒートマップや上位結果を表示できます。")
    # 例：st.dataframe(df_grid_results), st.pyplot(fig_heatmap) など

# --------------------------------------------
# 7) （オプション）Optuna 最適化ボタン
# --------------------------------------------
with st.expander("ベイズ最適化 (Optuna)"):
    if st.button("最適化を実行"):
        import optuna

        def objective(trial):
            xx = trial.suggest_uniform("x", 0.0, 0.6)
            yy = trial.suggest_uniform("y", 0.0, 0.3)
            zz = trial.suggest_uniform("z", 0.0, 1.0)
            fv, ti = simulate_strategy(price, monthly_dates, monthly_amount, xx, yy, zz)[2:]
            if ti <= 0:
                return float("inf")
            return -(fv / ti)

        study = optuna.create_study()
        with st.spinner("最適化中…しばらくお待ちください"):
            study.optimize(objective, n_trials=100)
        best_params = study.best_params
        best_ret = -study.best_value
        fv_best, ti_best = simulate_strategy(
            price, monthly_dates, monthly_amount,
            best_params["x"], best_params["y"], best_params["z"]
        )[2:]
        st.success(f"最適リターン: {best_ret:.4f}（x={best_params['x']:.3f}, y={best_params['y']:.3f}, z={best_params['z']:.3f}）")
        st.write(f"評価額: ¥{int(fv_best):,}、投資額: ¥{int(ti_best):,}")

