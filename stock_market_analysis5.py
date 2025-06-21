import openai
import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer 
from sklearn.preprocessing import normalize 
import faiss
import streamlit as st 
import datetime
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- API Keys ---
# openai.api_key = "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"
# twelvedata_api_key = "8b12c89c35be4fd0b13bcacbfba4700a" 
# news_api_key = "0d1e0cc62cad47b4aa7623adfb2d4684" 
# #alpha_vantage_api_key = "KXOVYT1RRA7HT88W" 
# alpha_vantage_api_key="UTRNX3Y6VG4ZXL24"

import streamlit as st
import openai

openai.api_key = st.secrets["api_keys"]["openai"]
alpha_vantage_api_key = st.secrets["api_keys"]["alpha_vantage_api_key"]
twelvedata_api_key = st.secrets["api_keys"]["twelvedata_api_key"]
news_api_key =  st.secrets["api_keys"]["news_api_key"]
# --- Load Embedding Model ---
import requests
# --- PostgreSQL Config ----

SUPABASE_URL = "https://rtxwmpbwggrlxqeajgvm.supabase.co"
SUPABASE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ0eHdtcGJ3Z2dybHhxZWFqZ3ZtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTA0Nzk1ODcsImV4cCI6MjA2NjA1NTU4N30.k1hRAnUdodcsnnc1x2SI9mIP3Y6WRW6nDMJpPffNnY8"  # 🔐 Replace with your actual key

headers = {
    "apikey": SUPABASE_API_KEY,
    "Authorization": f"Bearer {SUPABASE_API_KEY}"
}

# --- Fetch Company Symbol & Exchange from PostgreSQL ---
def fetch_ticker_from_db(company_name):
    url = f"{SUPABASE_URL}/rest/v1/stock_symbol"
    params = {
        "company_name": f"eq.{company_name}",
        "select": "symbol,exchange"
    }
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    
    if data:
        return data[0]["symbol"], data[0]["exchange"]
    return None

# --- Match Query with Company in DB ---
def match_company_in_db(user_query):
    url = f"{SUPABASE_URL}/rest/v1/stock_symbol"
    params = {"select": "company_name"}
    response = requests.get(url, headers=headers, params=params)
    companies = [row["company_name"] for row in response.json()]
    
    for name in companies:
        if name.lower() in user_query.lower():
            return name
    return None

# --- Fetch News ---
def fetch_live_news(company):
    url = f"https://newsapi.org/v2/everything?q={company}&sortBy=publishedAt&language=en&apiKey={news_api_key}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return [f"{a['title']}. {a.get('description', '')}" for a in articles if a.get('title')]

# --- Build FAISS Index ---
def build_faiss_index(corpus):
    embeddings = embedding_model.encode(corpus)
    embeddings = normalize(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, corpus

# --- Fetch Historical Data from Alpha Vantage for Indian Companies ---
def get_alpha_vantage_data(symbol):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": alpha_vantage_api_key,
        "outputsize": "full"
    }
    r = requests.get(url, params=params)
    data = r.json()

    if "Time Series (Daily)" not in data:
        raise ValueError(f"Alpha Vantage error: {data}")

    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
    df = df.rename(columns={
        '1. open': 'open', '2. high': 'high', '3. low': 'low',
        '4. close': 'close', '5. volume': 'volume'
    })
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df.last("6M")[['close']].astype(float)

# --- Fetch Historical Data from Twelve Data for US Companies ---
def get_twelve_data(symbol, exchange):
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=180)
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1day",
        "start_date": start_date.date(),
        "end_date": end_date.date(),
        "outputsize": 500,
        "apikey": twelvedata_api_key
    }
    if exchange:
        params["exchange"] = exchange

    response = requests.get(url, params=params)
    data = response.json()

    if "values" not in data:
        raise ValueError(f"Twelve Data error: {data}")

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df.sort_index()
    return df[["close"]].astype(float)

# --- Generate Insight ---
def get_stock_insight(query, corpus, index, symbol, hist_df):
    query_embedding = embedding_model.encode([query])
    query_embedding = normalize(query_embedding)
    D, I = index.search(query_embedding, k=3)
    context = "\n".join([corpus[i] for i in I[0]])

    latest_price = hist_df["close"].iloc[-1]
    past_price = hist_df["close"].iloc[0]
    change_pct = ((latest_price - past_price) / past_price) * 100
    trend_context = (
        f"The current stock price of {symbol} is {latest_price:.2f}. "
        f"It changed from {past_price:.2f} over the past 6 months, a {change_pct:.2f}% move."
    )

    prompt = f"""
    You are a financial advisor. Based on the stock trend and the recent news, provide an investment insight and recommendation:

    Stock Trend:
    {trend_context}

    News:
    {context}

    Question:
    {query}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful financial advisor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )
    return response['choices'][0]['message']['content'] 

# --- Streamlit UI ---
st.set_page_config(page_title="Stock Market Consultant", layout="centered") 
st.title("📈 STAT-TECH-AI-Powered Stock Market Consultant") 


query = st.text_input("🔍 Ask a question about a company (e.g., Infosys 6-month trend)")

if query:
    with st.spinner("Fetching insights..."):
        company_name = match_company_in_db(query)
        if company_name:
            result = fetch_ticker_from_db(company_name)
            if result:
                symbol, exchange = result
                st.write(f"📌 Company: **{company_name.title()}**, Symbol: **{symbol}**, Exchange: **{exchange}**")

                news = fetch_live_news(company_name)
                if news:
                    index, corpus = build_faiss_index(news)
                    try:
                        if exchange in ["NSE", "BSE"]:
                            hist_df = get_alpha_vantage_data(symbol)
                        else:
                            hist_df = get_twelve_data(symbol, exchange)

                        st.line_chart(hist_df, use_container_width=True)
                        insight = get_stock_insight(query, corpus, index, symbol, hist_df)
                        st.success("💡 Investment Insight:")
                        st.write(insight)
                    except Exception as e:
                        st.error(f"❌ Historical data error: {e}")
                else:
                    st.warning("⚠️ No recent news found for this company.")
            else:
                st.error("❌ Company not found in database. Please check the name or add it to your PostgreSQL table.")
        else:
            st.error("❌ Company not found in database. Please check the name or add it to your PostgreSQL table.")
