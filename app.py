# Aplikasi Streamlit pembungkus web 3D AAS Shimadzu AA-7000 (index.html)
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="AAS AA-7000 – Penetapan Cu",
    page_icon="⚗️",
    layout="wide",
)

# Rapikan margin bawaan Streamlit agar tampilan web memenuhi layar
st.markdown(
    """
    <style>
      .block-container {padding-top: 0.5rem; padding-bottom: 0; max-width: 100%;}
      header[data-testid="stHeader"] {height: 0; visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

html_path = Path(__file__).parent / "index.html"

if not html_path.exists():
    st.error("File index.html tidak ditemukan. Pastikan index.html berada di folder yang sama dengan app.py.")
    st.stop()

html = html_path.read_text(encoding="utf-8")

# Tinggi area tampilan dapat diatur dari sidebar
with st.sidebar:
    st.header("Pengaturan tampilan")
    tinggi = st.slider("Tinggi halaman (piksel)", 700, 2400, 1300, step=50)
    st.caption("Perbesar jika bagian bawah halaman terpotong.")

components.html(html, height=tinggi, scrolling=True)
