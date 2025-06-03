import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns  # ヒートマップ用
import numpy_financial as nf

# 日本語フォントを使いたい場合（Windows 例：Meiryo）
plt.rcParams['font.family'] = 'Meiryo'

# -----------------------------------------------------------------------------
# 1) データ取得関数
# -----------------------------------------------------------------------------
def load_price_data(ticker: str, start: str, end: str) -> pd.Series:
    """
    yfinance を使って日次終値を取得し、DatetimeIndex を normalize して返す。
    """
    df = yf.download(ticker, start=start, end=end, interval="1d")[["Close"]].dropna()
    s = df["Close"]
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s

# -----------------------------------------------------------------------------
# 2) 月次積立日算出関数
# -----------------------------------------------------------------------------
def get_monthly_dates(price_index: pd.DatetimeIndex, buy_day: int = 7) -> list[pd.Timestamp]:
    """
    毎月 buy_day 日（例：7日）が非営業日の場合、直近の営業日に前倒しした
    '月次積立日' のリストを返す。normalize 済みの price_index を想定。
    """
    monthly_dates = []
    years = range(price_index[0].year, price_index[-1].year + 1)
    for y in years:
        for m in range(1, 13):
            t = pd.Timestamp(y, m, buy_day).normalize()
            while t not in price_index and t > pd.Timestamp(y, m, 1):
                t -= pd.Timedelta(days=1)
                t = t.normalize()
            if t in price_index:
                monthly_dates.append(t)
    # 範囲外（開始以前または終了以降）の日付は除外
    monthly_dates = [d for d in monthly_dates if d >= price_index[0] and d <= price_index[-1]]
    return monthly_dates

# -----------------------------------------------------------------------------
# 3) シミュレーション関数
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 4) 年率IRR計算関数
# -----------------------------------------------------------------------------
def calc_annual_irr(cashflow: list[float], dates: list[pd.Timestamp]) -> float:
    """
    cashflow: float の list, 負の値は投資(支出)、正の値は回収(収入) を表す
    dates:   datetime の list, cashflow[i] に対応する日付
    戻り値:  年率リターン (小数)
    """
    start_date = dates[0]
    # 各日付が start_date から何か月後になるかを計算
    months = []
    for d in dates:
        ym_diff = (d.year - start_date.year) * 12 + (d.month - start_date.month)
        months.append(ym_diff)

    n_months = months[-1] + 1
    cash_series = np.zeros(n_months)
    for cf, m in zip(cashflow, months):
        cash_series[m] += cf

    irr_monthly = nf.irr(cash_series)
    if irr_monthly is None or np.isnan(irr_monthly):
        return np.nan

    annual_irr = (1 + irr_monthly) ** 12 - 1
    return annual_irr

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.title("定額積立＋押し目買いシミュレーション")

# 1) サイドバー：データ設定
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

# 2) サイドバー：シミュレーション設定
st.sidebar.header("シミュレーション設定")
monthly_amount = st.sidebar.slider(
    "月次積立額（円）", min_value=10_000, max_value=200_000, step=10_000, value=100_000
)
x = st.sidebar.slider(
    "現金保持率 x (0～1)", min_value=0.0, max_value=1.0, step=0.01, value=0.4
)
y = st.sidebar.slider(
    "暴落閾値 y (0～1)", min_value=0.0, max_value=0.5, step=0.01, value=0.20
)
z = st.sidebar.slider(
    "買い増し率 z (0～1)", min_value=0.0, max_value=1.0, step=0.01, value=1.0
)
buy_day = st.sidebar.number_input(
    "積立日 (1～28)", min_value=1, max_value=28, value=7, step=1
)
run_button = st.sidebar.button("シミュレーション実行")

# 3) データ取得と下準備
price = load_price_data(ticker, start_date.isoformat(), end_date.isoformat())
monthly_dates = get_monthly_dates(price.index, buy_day=buy_day)

# 4) シミュレーション実行
if run_button or True:
    daily_dates, daily_values, final_value, total_invested = simulate_strategy(
        price=price,
        monthly_dates=monthly_dates,
        monthly_amount=monthly_amount,
        x=x,
        y=y,
        z=z
    )

    # --- 年率IRR計算用キャッシュフロー作成 ---
    cf_dates = []
    cf_values = []
    # 各月の投資日で -monthly_amount
    for d in monthly_dates:
        cf_dates.append(d)
        cf_values.append(-monthly_amount)
    # 最終日に +final_value
    last_date = price.index[-1]
    cf_dates.append(last_date)
    cf_values.append(+final_value)

    annual_irr = calc_annual_irr(cf_values, cf_dates)
    if np.isnan(annual_irr):
        annual_display = "計算不可"
    else:
        annual_display = f"{annual_irr*100:.2f}%"

    # 5-1) メトリクス表示（4列に拡張）
    st.subheader("シミュレーション結果")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最終評価額（円）", f"{int(final_value):,}")
    col2.metric("総投資額（円）", f"{int(total_invested):,}")
    col3.metric("最終リターン倍率", f"{(final_value/total_invested):.2f}×")
    col4.metric("実質年率リターン", annual_display)

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

# -----------------------------------------------------------------------------
# 6) （オプション）パラメータ感度ヒートマップ
# -----------------------------------------------------------------------------
with st.expander("パラメータ感度分析"):
    st.write("x, y を固定幅でグリッドサーチし、リターンをヒートマップで表示します。")
    
    # --- 1) グリッドを定義 ---
    grid_x = np.linspace(0.0, 0.6, 11)
    grid_y = np.linspace(0.0, 0.3, 11)
    fixed_z = 1.0  # z は固定値としておく（例）
    
    results_matrix = pd.DataFrame(
        data=np.zeros((len(grid_y), len(grid_x))),
        index=[f"{yy:.2f}" for yy in grid_y],
        columns=[f"{xx:.2f}" for xx in grid_x]
    )
    
    # --- 2) グリッドサーチ実行 ---
    for i, y_val in enumerate(grid_y):
        for j, x_val in enumerate(grid_x):
            # 戻り値は (daily_dates, daily_values, final_value, total_invested)
            _, _, fv, ti = simulate_strategy(
                price=price,
                monthly_dates=monthly_dates,
                monthly_amount=monthly_amount,
                x=x_val,
                y=y_val,
                z=fixed_z
            )
            if ti > 0:
                ret = float(fv / ti)
            else:
                ret = np.nan
            results_matrix.iloc[i, j] = ret
    
    # --- 3) ヒートマップで可視化 ---
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        results_matrix.astype(float),
        annot=False,
        cmap="RdYlBu_r",
        ax=ax2,
        cbar_kws={"label": "最終リターン倍率"}
    )
    ax2.set_xlabel("現金保持率 x")
    ax2.set_ylabel("暴落閾値 y")
    ax2.set_title(f"z={fixed_z:.2f} のときのリターン感度マップ")
    st.pyplot(fig2)
    
    # --- 4) 上位 5 パターンをテーブル表示 ---
    df_long = results_matrix.stack().reset_index()
    df_long.columns = ["y（暴落閾値）", "x（現金保持率）", "return"]
    df_top5 = df_long.sort_values("return", ascending=False).head(5)
    st.write("### 感度分析：上位 5 パターン（z 固定）")
    st.dataframe(df_top5.style.format({"return": "{:.3f}"}))

# -----------------------------------------------------------------------------
# 7) （オプション）ベイズ最適化 (Optuna)
# -----------------------------------------------------------------------------
with st.expander("ベイズ最適化 (Optuna)"):
    if st.button("最適化を実行"):
        import optuna

        def objective(trial):
            xx = trial.suggest_uniform("x", 0.0, 0.6)
            yy = trial.suggest_uniform("y", 0.0, 0.3)
            zz = trial.suggest_uniform("z", 0.0, 1.0)
            # 戻り値は 4 要素なので、最後の 2 つを取り出す
            _, _, fv_opt, ti_opt = simulate_strategy(
                price=price,
                monthly_dates=monthly_dates,
                monthly_amount=monthly_amount,
                x=xx,
                y=yy,
                z=zz
            )
            if ti_opt <= 0:
                return float("inf")
            return -(fv_opt / ti_opt)

        study = optuna.create_study()
        with st.spinner("最適化中…しばらくお待ちください"):
            study.optimize(objective, n_trials=100)
        best_params = study.best_params
        best_ret = -study.best_value
        _, _, fv_best, ti_best = simulate_strategy(
            price=price,
            monthly_dates=monthly_dates,
            monthly_amount=monthly_amount,
            x=best_params["x"],
            y=best_params["y"],
            z=best_params["z"]
        )
        st.success(
            f"最適リターン: {best_ret:.4f}（x={best_params['x']:.3f}, "
            f"y={best_params['y']:.3f}, z={best_params['z']:.3f}）"
        )
        st.write(f"評価額: ¥{int(fv_best):,}、投資額: ¥{int(ti_best):,}")
