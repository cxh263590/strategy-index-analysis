import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import psycopg2
import logging
import os
from datetime import datetime, timedelta

LOG_DIR = os.environ.get("LOG_DIR", "./logs")
if os.path.exists(LOG_DIR) or os.access(os.path.dirname(LOG_DIR) or ".", os.W_OK):
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(LOG_DIR, "app.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        encoding='utf-8'
    )

def log_page_access():
    logging.info("=" * 50)
    logging.info(f"Page accessed - Time: {datetime.now()}")
    logging.info("=" * 50)

def log_function_call(func_name, **kwargs):
    params = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logging.info(f"Function called: {func_name}({params})")

log_page_access()

def get_db_connection():
    return psycopg2.connect(
        host=st.secrets["postgres"]["host"],
        port=int(st.secrets["postgres"]["port"]),
        user=st.secrets["postgres"]["user"],
        password=st.secrets["postgres"]["password"],
        database=st.secrets["postgres"]["database"]
    )

@st.cache_data
def load_data(strategy_name):
    log_function_call("load_data", strategy_name=strategy_name)
    conn = get_db_connection()
    query = "SELECT date, net_value FROM strategy_data WHERE strategy_name = %s ORDER BY date"
    df = pd.read_sql(query, conn, params=(strategy_name,))
    conn.close()
    return df

@st.cache_data
def load_all_strategies():
    log_function_call("load_all_strategies")
    conn = get_db_connection()
    query = "SELECT DISTINCT strategy_name FROM strategy_data ORDER BY strategy_name"
    df = pd.read_sql(query, conn)
    conn.close()
    return df['strategy_name'].tolist()

@st.cache_data
def load_all_data():
    log_function_call("load_all_data")
    conn = get_db_connection()
    query = "SELECT date, strategy_name, net_value FROM strategy_data ORDER BY date, strategy_name"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def calc_stats(data):
    log_function_call("calc_stats", data_rows=len(data))
    series = data['net_value'].dropna()
    if len(series) < 2:
        return None
    
    start_val = series.iloc[0]
    end_val = series.iloc[-1]
    returns = series.pct_change().dropna()
    
    days = len(returns)
    annual_return = (end_val / start_val) ** (252 / max(days, 1)) - 1
    
    cumulative = series / start_val
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    start_date = data['date'].dropna().iloc[0]
    
    return {
        'start_date': start_date,
        'annual_return': annual_return,
        'max_drawdown': max_dd
    }

def calc_period_return(df, strategy, days):
    strategy_data = df[df['strategy_name'] == strategy].copy()
    strategy_data['date'] = pd.to_datetime(strategy_data['date'])
    strategy_data = strategy_data.sort_values('date')
    
    if len(strategy_data) < 2:
        return None
    
    cutoff_date = strategy_data['date'].max() - timedelta(days=days)
    recent_data = strategy_data[strategy_data['date'] >= cutoff_date]
    
    if len(recent_data) < 2:
        return None
    
    start_val = recent_data['net_value'].iloc[0]
    end_val = recent_data['net_value'].iloc[-1]
    return (end_val / start_val - 1) * 100

st.set_page_config(page_title="策略指数分析平台", layout="wide")

page = st.sidebar.radio("导航", ["策略走势", "策略统计"])

if page == "策略走势":
    st.title("📊 策略指数分析平台")
    
    strategies = load_all_strategies()
    
    st.sidebar.header("选择策略")
    selected_strategy = st.sidebar.selectbox("策略", strategies)
    log_function_call("sidebar_strategy_selection", selected_strategy=selected_strategy)
    
    st.subheader(f"{selected_strategy} 走势")
    
    data = load_data(selected_strategy)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date')
    
    dates = data['date'].dt.strftime('%Y-%m-%d').tolist()
    
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        start_idx = 0
        start_date_sel = st.selectbox("起始日期", range(len(dates)), index=start_idx, 
                                       format_func=lambda x: dates[x], key="start_date")
    with col_date2:
        end_idx = len(dates) - 1
        end_date_sel = st.selectbox("结束日期", range(len(dates)), index=end_idx,
                                     format_func=lambda x: dates[x], key="end_date")
    
    log_function_call("user_selection", selected_strategy=selected_strategy, 
                      start_date=dates[start_date_sel], end_date=dates[end_date_sel])
    
    if start_date_sel > end_date_sel:
        st.error("起始日期不能晚于结束日期")
        st.stop()
    
    filtered_data = data.iloc[start_date_sel:end_date_sel+1]
    
    stats = calc_stats(filtered_data)
    
    if stats:
        col1, col2, col3 = st.columns(3)
        col1.metric("年化收益率", f"{stats['annual_return']:.2%}")
        col2.metric("最大回撤", f"{stats['max_drawdown']:.2%}")
        col3.metric("起始日期", str(stats['start_date'])[:10])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_data['date'],
        y=filtered_data['net_value'],
        mode='lines',
        name=selected_strategy,
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        xaxis_title="日期",
        yaxis_title="净值",
        hovermode="x unified",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("数据明细")
    st.dataframe(filtered_data, use_container_width=True)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("目录")
    for s in strategies:
        st.sidebar.markdown(f"- {s}")

elif page == "策略统计":
    st.title("📊 策略收益统计")
    
    with st.spinner("加载数据中..."):
        df = load_all_data()
        df['date'] = pd.to_datetime(df['date'])
        
        strategies = load_all_strategies()
        
        results = []
        for s in strategies:
            week_return = calc_period_return(df, s, 7)
            year_return = calc_period_return(df, s, 365)
            two_year_return = calc_period_return(df, s, 730)
            
            results.append({
                '策略': s,
                '近1周(%)': f"{week_return:.2f}" if week_return is not None else "N/A",
                '近1年(%)': f"{year_return:.2f}" if year_return is not None else "N/A",
                '近2年(%)': f"{two_year_return:.2f}" if two_year_return is not None else "N/A"
            })
        
        result_df = pd.DataFrame(results)
        
        st.subheader("各策略收益对比")
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        
        st.markdown("### 导出数据")
        csv = result_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="下载 CSV",
            data=csv,
            file_name="strategy_returns.csv",
            mime="text/csv"
        )
