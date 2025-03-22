import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import urllib.parse
from dotenv import load_dotenv
import os


load_dotenv()

username = os.getenv('MYSQL_USERNAME')
password = os.getenv('MYSQL_PASSWORD')
host = os.getenv('MYSQL_HOST')
port = os.getenv('MYSQL_PORT')
database = os.getenv('MYSQL_DB')

encoded_password = urllib.parse.quote_plus(password)
engine = create_engine(f'mysql+mysqlconnector://{username}:{encoded_password}@{host}:{port}/{database}')

# --- Sidebar for Schema Viewer ---
st.sidebar.title("📂 Database Explorer")
st.sidebar.markdown("View tables and schemas in your database.")

with engine.connect() as conn:
    tables_query = "SHOW TABLES"
    tables = conn.execute(text(tables_query)).fetchall()
    table_names = [table[0] for table in tables]

selected_table = st.sidebar.selectbox("Select a table to view schema:", table_names)

if selected_table:
    with engine.connect() as conn:
        schema_query = f"DESCRIBE {selected_table}"
        schema_df = pd.read_sql(text(schema_query), conn)
    st.sidebar.write("📑 **Schema:**")
    st.sidebar.dataframe(schema_df)

# --- Main SQL Runner UI ---
st.markdown("""
    <style>
        h1 { color: #4a90e2; text-align: center; }
        .stTextArea > label { font-weight: bold; }
        .stButton > button {
            background-color: #4a90e2;
            color: white;
            border: None;
            padding: 0.5em 1em;
            border-radius: 8px;
        }
        .stButton > button:hover { background-color: #357ABD; }
        .result-container {
            background-color: white;
            padding: 1em;
            border-radius: 10px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💻 MySQL Query Runner")
st.write("👉 Enter any **SQL query** and see results or execution status.")

query = st.text_area("📝 Enter your SQL query here:", height=150)

if st.button("🚀 Run Query"):
    try:
        with engine.begin() as conn:
            if query.strip().lower().startswith('select'):
                result_df = pd.read_sql(text(query), conn)
                st.success("✅ Query executed successfully!")
                with st.container():
                    st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                    st.dataframe(result_df)
                    st.markdown("</div>", unsafe_allow_html=True)
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Result as CSV",
                    data=csv,
                    file_name='query_results.csv',
                    mime='text/csv'
                )
            else:
                conn.execute(text(query))
                st.success("✅ Query executed and committed successfully!")
    except Exception as e:
        st.error(f"❌ Error occurred: {e}")

st.markdown("---")
st.markdown("<p style='text-align:center; color: gray;'>Built for SQL exploration with ❤️</p>", unsafe_allow_html=True)
