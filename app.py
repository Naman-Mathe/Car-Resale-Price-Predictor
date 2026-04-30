import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Car Resale Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  CUSTOM CSS  (dark automotive theme)
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

/* ── Root ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d0f14;
    color: #e0e4ef;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13161f 0%, #0d0f14 100%);
    border-right: 1px solid #1e2330;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }

/* ── Main container ── */
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1400px; }

/* ── Hero title ── */
.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    background: linear-gradient(90deg, #f0a500 0%, #e06020 60%, #c0392b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    line-height: 1.1;
}
.hero-sub {
    font-size: 0.85rem;
    color: #6b7492;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #161b27 0%, #1a2035 100%);
    border: 1px solid #252d42;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    text-align: center;
}
.metric-card .val {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #f0a500;
}
.metric-card .lbl {
    font-size: 0.72rem;
    color: #6b7492;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.1rem;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: #c8cedf;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-left: 3px solid #f0a500;
    padding-left: 0.7rem;
    margin: 2rem 0 1rem;
}

/* ── Result banner ── */
.result-banner {
    background: linear-gradient(135deg, #1a2035 0%, #1e1a10 100%);
    border: 1px solid #f0a500;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
    box-shadow: 0 0 40px rgba(240,165,0,0.12);
}
.result-banner .price {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    color: #f0a500;
    letter-spacing: 0.03em;
}
.result-banner .tagline {
    font-size: 0.8rem;
    color: #6b7492;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-top: 0.4rem;
}

/* ── Plotly chart cards ── */
.chart-card {
    background: #13161f;
    border: 1px solid #1e2330;
    border-radius: 12px;
    padding: 0.5rem;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stSelectbox"] > div,
div[data-testid="stNumberInput"] > div {
    background: #161b27 !important;
    border-color: #252d42 !important;
    border-radius: 8px !important;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #f0a500, #e06020);
    color: #0d0f14;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    margin-top: 1rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* hide default streamlit chrome */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  LOAD ASSETS
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model   = pickle.load(open("model.pkl", "rb"))
    columns = pickle.load(open("columns.pkl", "rb"))
    return model, columns

@st.cache_data
def load_data():
    df = pd.read_csv("data/car_data.csv")
    df.columns = [c.lower() for c in df.columns]
    return df

model, columns = load_model()
df_raw = load_data()

# ─────────────────────────────────────────
#  DERIVED DATA FOR CHARTS
# ─────────────────────────────────────────
df_raw['brand'] = df_raw['name'].apply(lambda x: x.split()[0])
df_raw['car_age'] = 2026 - df_raw['year']

brand_counts = df_raw['brand'].value_counts().head(10).reset_index()
brand_counts.columns = ['Brand', 'Count']

fuel_counts = df_raw['fuel'].value_counts().reset_index()
fuel_counts.columns = ['Fuel', 'Count']

# ─────────────────────────────────────────
#  PLOTLY THEME HELPER
# ─────────────────────────────────────────
LAYOUT = dict(
    paper_bgcolor="#13161f",
    plot_bgcolor="#13161f",
    font=dict(family="Inter", color="#a0a8c0", size=12),
    margin=dict(l=16, r=16, t=36, b=16),
    colorway=["#f0a500", "#e06020", "#c0392b", "#2980b9", "#27ae60"],
)

def themed(fig, title=""):
    fig.update_layout(**LAYOUT, title=dict(text=title, font=dict(family="Rajdhani", size=16, color="#c8cedf")))
    fig.update_xaxes(gridcolor="#1e2330", linecolor="#252d42", zerolinecolor="#1e2330")
    fig.update_yaxes(gridcolor="#1e2330", linecolor="#252d42", zerolinecolor="#1e2330")
    return fig

# ─────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="hero-title">🚗 Car Resale Price Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">AI-powered valuation · Indian used car market</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
#  QUICK METRICS
# ─────────────────────────────────────────
st.markdown("")
m1, m2, m3, m4 = st.columns(4)
avg_price  = int(df_raw['selling_price'].mean())
max_price  = int(df_raw['selling_price'].max())
total_cars = len(df_raw)
avg_age    = round(df_raw['car_age'].mean(), 1)

for col, val, label in zip(
    [m1, m2, m3, m4],
    [f"₹{avg_price:,}", f"₹{max_price:,}", f"{total_cars:,}", f"{avg_age} yrs"],
    ["Avg. Resale Price", "Highest Price", "Cars in Dataset", "Avg. Car Age"],
):
    col.markdown(
        f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{label}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ─────────────────────────────────────────
#  DATA INSIGHTS (interactive charts)
# ─────────────────────────────────────────
st.markdown('<div class="section-header">Data Insights</div>', unsafe_allow_html=True)

row1_l, row1_r = st.columns(2)

# ── Price distribution
with row1_l:
    fig1 = px.histogram(
        df_raw, x="selling_price", nbins=60,
        labels={"selling_price": "Selling Price (₹)"},
        color_discrete_sequence=["#f0a500"],
    )
    fig1 = themed(fig1, "Selling Price Distribution")
    fig1.update_traces(opacity=0.85)
    st.plotly_chart(fig1, use_container_width=True)

# ── Fuel split donut
with row1_r:
    fig2 = px.pie(
        fuel_counts, names="Fuel", values="Count",
        hole=0.55,
        color_discrete_sequence=["#f0a500", "#e06020", "#c0392b", "#2980b9", "#27ae60"],
    )
    fig2 = themed(fig2, "Fuel Type Split")
    fig2.update_traces(textfont_color="#c8cedf")
    st.plotly_chart(fig2, use_container_width=True)

row2_l, row2_r = st.columns(2)

# ── Top brands bar
with row2_l:
    fig3 = px.bar(
        brand_counts.sort_values("Count"), x="Count", y="Brand",
        orientation="h",
        labels={"Count": "Number of Cars", "Brand": ""},
        color="Count",
        color_continuous_scale=[[0, "#1a2035"], [1, "#f0a500"]],
    )
    fig3 = themed(fig3, "Top 10 Brands by Listings")
    fig3.update_coloraxes(showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

# ── Age vs Price scatter
with row2_r:
    fig4 = px.scatter(
        df_raw.sample(min(1000, len(df_raw)), random_state=42),
        x="car_age", y="selling_price",
        color="fuel",
        labels={"car_age": "Car Age (years)", "selling_price": "Price (₹)", "fuel": "Fuel"},
        opacity=0.65,
        color_discrete_sequence=["#f0a500", "#e06020", "#2980b9", "#27ae60", "#c0392b"],
    )
    fig4 = themed(fig4, "Car Age vs Selling Price")
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────
#  SIDEBAR — INPUT FORM
# ─────────────────────────────────────────
st.sidebar.markdown(
    '<div style="font-family:Rajdhani,sans-serif;font-size:1.4rem;font-weight:700;'
    'color:#f0a500;letter-spacing:0.08em;margin-bottom:1.2rem;">🔧 Car Details</div>',
    unsafe_allow_html=True,
)

# Car name — searchable from dataset
all_car_names = sorted(df_raw['name'].unique().tolist())
car_name = st.sidebar.selectbox("Car Model", options=[""] + all_car_names, index=0)

# Auto-derive brand
brand = car_name.split()[0].lower() if car_name else ""
if brand:
    st.sidebar.markdown(
        f'<div style="font-size:0.78rem;color:#6b7492;margin-top:-0.6rem;margin-bottom:0.8rem;">'
        f'Brand detected: <span style="color:#f0a500;">{brand.title()}</span></div>',
        unsafe_allow_html=True,
    )

km_driven    = st.sidebar.number_input("Kilometers Driven", min_value=0, max_value=900000, value=50000, step=1000)
year         = st.sidebar.slider("Year of Purchase", 1992, 2020, 2016)
fuel         = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
seller_type  = st.sidebar.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
owner        = st.sidebar.selectbox(
    "Ownership",
    ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"],
)

predict_btn = st.sidebar.button("⚡  Predict Resale Price")

# ─────────────────────────────────────────
#  PREDICTION SECTION
# ─────────────────────────────────────────
st.markdown('<div class="section-header">Price Prediction</div>', unsafe_allow_html=True)

if predict_btn:
    if not car_name:
        st.warning("⚠️  Please select a car model from the dropdown.")
    else:
        car_age = 2026 - year

        # Build input row (all zeros, then fill known columns)
        input_data = pd.DataFrame(columns=columns)
        input_data.loc[0] = 0

        # Numeric
        for col, val in [("km_driven", km_driven), ("car_age", car_age)]:
            if col in input_data.columns:
                input_data.loc[0, col] = val

        # One-hot encoded flags
        mapping = {
            f"fuel_{fuel}":                   1,
            f"seller_type_{seller_type}":     1,
            f"transmission_{transmission}":   1,
            f"owner_{owner}":                 1,
            f"brand_{brand}":                 1,
        }
        for col, val in mapping.items():
            if col in input_data.columns:
                input_data.loc[0, col] = val

        predicted_price = model.predict(input_data)[0]
        predicted_price = max(0, predicted_price)   # clamp negatives

        # ── Banner
        st.markdown(
            f"""
            <div class="result-banner">
                <div class="tagline">Estimated Resale Value for {car_name}</div>
                <div class="price">₹ {predicted_price:,.0f}</div>
                <div class="tagline">{year} · {fuel} · {transmission} · {owner}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Gauge chart
        st.markdown("")
        g_col, b_col = st.columns([1, 1])

        with g_col:
            max_ref = df_raw['selling_price'].quantile(0.99)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_price,
                number={"prefix": "₹ ", "valueformat": ",.0f",
                        "font": {"family": "Rajdhani", "size": 28, "color": "#f0a500"}},
                gauge={
                    "axis": {"range": [0, max_ref], "tickformat": ",.0s",
                             "tickfont": {"color": "#6b7492", "size": 10}},
                    "bar": {"color": "#f0a500"},
                    "bgcolor": "#13161f",
                    "bordercolor": "#252d42",
                    "steps": [
                        {"range": [0,               max_ref * 0.33], "color": "#1a2035"},
                        {"range": [max_ref * 0.33,  max_ref * 0.66], "color": "#1e2535"},
                        {"range": [max_ref * 0.66,  max_ref],        "color": "#222b40"},
                    ],
                    "threshold": {
                        "line": {"color": "#e06020", "width": 3},
                        "thickness": 0.75,
                        "value": predicted_price,
                    },
                },
                title={"text": "Predicted Price Gauge",
                       "font": {"family": "Rajdhani", "size": 15, "color": "#c8cedf"}},
            ))
            fig_gauge.update_layout(**LAYOUT)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with b_col:
            # Comparison bar: this car vs dataset percentiles
            p25 = df_raw['selling_price'].quantile(0.25)
            p50 = df_raw['selling_price'].quantile(0.50)
            p75 = df_raw['selling_price'].quantile(0.75)

            comp_df = pd.DataFrame({
                "Segment":  ["Budget (P25)", "Median (P50)", "Upper (P75)", "Your Car"],
                "Price":    [p25, p50, p75, predicted_price],
                "Highlight": [False, False, False, True],
            })

            fig_cmp = px.bar(
                comp_df, x="Segment", y="Price",
                color="Highlight",
                color_discrete_map={True: "#f0a500", False: "#1e2535"},
                labels={"Price": "Price (₹)"},
            )
            fig_cmp = themed(fig_cmp, "Your Car vs Market Percentiles")
            fig_cmp.update_traces(marker_line_width=0)
            fig_cmp.update_layout(showlegend=False)
            st.plotly_chart(fig_cmp, use_container_width=True)

        # ── Similar cars from dataset
        st.markdown('<div class="section-header">Similar Listings in Dataset</div>', unsafe_allow_html=True)
        similar = df_raw[
            (df_raw['brand'] == brand.title()) &
            (df_raw['fuel']  == fuel)
        ][['name', 'year', 'km_driven', 'fuel', 'transmission', 'seller_type', 'owner', 'selling_price']]\
            .sort_values('selling_price').head(8).reset_index(drop=True)

        if similar.empty:
            similar = df_raw[df_raw['brand'] == brand.title()]\
                [['name', 'year', 'km_driven', 'fuel', 'transmission', 'seller_type', 'owner', 'selling_price']]\
                .sort_values('selling_price').head(8).reset_index(drop=True)

        if not similar.empty:
            similar['selling_price'] = similar['selling_price'].apply(lambda x: f"₹ {x:,.0f}")
            similar.columns = ['Name', 'Year', 'KM Driven', 'Fuel', 'Transmission', 'Seller', 'Owner', 'Price']
            st.dataframe(similar, use_container_width=True, hide_index=True)
        else:
            st.info("No similar listings found in the dataset for this brand.")

else:
    # Placeholder
    st.markdown(
        """
        <div style="background:#13161f;border:1px dashed #252d42;border-radius:16px;
                    padding:3rem;text-align:center;color:#3a4260;">
            <div style="font-family:Rajdhani,sans-serif;font-size:1.5rem;margin-bottom:0.5rem;">
                Select a car model and hit ⚡ Predict
            </div>
            <div style="font-size:0.85rem;">
                Fill in the details in the left panel to get an AI-powered price estimate.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )