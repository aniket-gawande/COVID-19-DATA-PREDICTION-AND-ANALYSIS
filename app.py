import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Set up page and theme
st.set_page_config(
    page_title="COVID-19 India Pro Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for final polish
st.markdown("""
    <style>
        .main, .block-container {padding-top:1.6rem;padding-bottom:2.2rem;background:#151623;}
        .css-1d391kg {background:#151623!important;}
        [data-testid="stSidebar"] {background: #181b2a!important; color: #f6fafb;}
        .stDataFrame {background-color: #22243b!important;}
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown, h3, h2, h1 {
            color: #fff!important;
        }
        .stMetric {padding-bottom:18px;}
        .metric-label {color: #36D399;}
        .metric-value {font-weight:700;}
        hr {border-top: 1.7px dashed #36d399; margin-top:38px; margin-bottom:20px; width:95%;}
        @media only screen and (max-width: 600px) { #credit-div {font-size:12px; padding:6px 10px 6px 8px;} }
    </style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def load_csv(filename, parse_dates=None):
    return pd.read_csv(os.path.join(DATA_DIR, filename), parse_dates=parse_dates)

# Load data
cases = load_csv('case_time_series.csv', ['Date_YMD'])
states = load_csv('states.csv', ['Date'])
districts = load_csv('districts.csv')
vax = load_csv('vaccine_doses_statewise_v2.csv')  # Vaccination

# Vaccination data safe columns
vax_cols = [col.lower() for col in vax.columns]
state_col = next((c for c in vax.columns if 'state' in c.lower()), vax.columns[0])
dose_col_candidates = [c for c in vax.columns if 'total' in c.lower() and 'dose' in c.lower()]
dose_col = dose_col_candidates[0] if dose_col_candidates else vax.columns[-1]

st.sidebar.title("COVID-19 Pro Dashboard")
section = st.sidebar.radio("Menu", [
    "India Trends", "State Trends", "District Trends", "Vaccination"
])

with st.sidebar:
    st.markdown("<div style='margin-top:18px; color:#36d399;'>Data: covid19india.org, MOHFW, CoWIN</div>", unsafe_allow_html=True)

# --- MAIN ---
if section == "India Trends":
    st.title("🇮🇳 India COVID-19 – National Trends")
    metric = st.sidebar.selectbox("Metric", ["Daily Confirmed", "Daily Recovered", "Daily Deceased"])
    min_date, max_date = cases.Date_YMD.min(), cases.Date_YMD.max()
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
    filtered = cases[(cases["Date_YMD"] >= pd.Timestamp(date_range[0])) & (cases["Date_YMD"] <= pd.Timestamp(date_range[1]))].copy()
    st.markdown("<br/>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Confirmed Cases", f"{int(filtered['Daily Confirmed'].sum()):,}")
    c2.metric("Total Recovered", f"{int(filtered['Daily Recovered'].sum()):,}")
    c3.metric("Total Deceased", f"{int(filtered['Daily Deceased'].sum()):,}")
    chart_type = st.sidebar.radio("Chart Type", ["Line", "Area", "Bar"])
    if chart_type == "Line":
        fig = px.line(filtered, x="Date_YMD", y=metric, markers=True,
                      title=f"<b>{metric}</b> (India)", labels={"Date_YMD": "Date", metric: metric}, 
                      color_discrete_sequence=["#36d399"])
    elif chart_type == "Bar":
        fig = px.bar(filtered, x="Date_YMD", y=metric, title=f"<b>{metric}</b> (India)", color_discrete_sequence=["#36d399"])
    else:
        fig = px.area(filtered, x="Date_YMD", y=metric, title=f"<b>{metric}</b> (India)", color_discrete_sequence=["#36d399"])
    fig.update_layout(
        paper_bgcolor="#151623", plot_bgcolor="#232652", font_color="#efefef",
        margin=dict(t=64, r=22, b=36, l=7),
        title=dict(font=dict(size=26, color='#36d399')),
        hoverlabel=dict(bgcolor="#22243b", font_size=14, font_color="#fff")
    )
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Data Table")
    st.dataframe(filtered, use_container_width=True)
    st.download_button("Download as CSV", filtered.to_csv(index=False), file_name="India_trends.csv")
    st.caption("🔷 Powered by official MOHFW and crowd-sourced sources")

elif section == "State Trends":
    st.title("📈 State-Level Trends")
    state = st.sidebar.selectbox("State", sorted(states.State.unique()))
    metric = st.sidebar.radio("Metric", ["Confirmed", "Recovered", "Deceased"])
    min_date, max_date = states.Date.min(), states.Date.max()
    date_range = st.sidebar.date_input("Date Range (States)", [min_date, max_date], min_value=min_date, max_value=max_date)
    sdf = states[states.State == state]
    sdf = sdf[(sdf["Date"] >= pd.Timestamp(date_range[0])) & (sdf["Date"] <= pd.Timestamp(date_range[1]))]
    fig = px.line(sdf, x="Date", y=metric, markers=True, title=f"{state} — {metric} over Time",
                  color_discrete_sequence=["#f6b035"])
    fig.update_layout(
        paper_bgcolor="#151623", plot_bgcolor="#232652", font_color="#efefef",
        margin=dict(t=48, r=22, b=28, l=6),
        title=dict(font=dict(size=22, color='#36d399')))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(sdf, use_container_width=True)
    st.download_button(f"Download {state} CSV", sdf.to_csv(index=False), file_name=f"{state}_trends.csv")

elif section == "District Trends":
    st.title("🏙️ District Trends")
    # Only keep rows where State is string
    districts = districts[districts['State'].notnull()]
    string_states = sorted([str(s) for s in districts['State'].unique() if isinstance(s, str)])
    chosen_state = st.selectbox("Select State", string_states)
    dist_state_df = districts[districts['State'] == chosen_state]

    available_districts = sorted([str(d) for d in dist_state_df['District'].unique() if isinstance(d, str)])
    chosen_district = st.selectbox("Select District", available_districts)
    metric = st.selectbox("Metric", [c for c in ["Confirmed", "Recovered", "Deceased"] if c in dist_state_df.columns])
    # Simulate a date axis if available, else just show bar
    if "Date" in dist_state_df.columns:
        dtdf = dist_state_df[dist_state_df['District'] == chosen_district]
        fig = px.line(dtdf, x="Date", y=metric, markers=True, title=f"{chosen_district} — {metric} over Time ({chosen_state})",
                      color_discrete_sequence=["#f27c4f"])
    else:
        dtdf = dist_state_df[dist_state_df['District'] == chosen_district]
        fig = px.bar(dtdf, x="District", y=metric, title=f"{chosen_district} — {metric} ({chosen_state})",
                     color_discrete_sequence=["#f27c4f"])
    fig.update_layout(
        paper_bgcolor="#151623", plot_bgcolor="#232652", font_color="#efefef",
        margin=dict(t=38, r=18, b=26, l=8),
        title=dict(font=dict(size=21, color='#36d399')))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(dtdf, use_container_width=True)
    st.download_button("Download District Data", dtdf.to_csv(index=False), file_name=f"{chosen_state}_{chosen_district}_trend.csv")
    st.caption("Tip: For full analytics, combine with geocodes if available.")

elif section == "Vaccination":
    st.title("💉 Statewise Vaccination Progress (Time Series)")
    # Try to reshape for one record per state per date
    time_col_candidates = [c for c in vax.columns if "date" in c.lower() or "updated" in c.lower()]
    time_col = time_col_candidates[0] if time_col_candidates else None

    chosen_state = st.selectbox("Select State for Vaccination Trend", sorted(vax[state_col].unique()))
    v2 = vax[vax[state_col] == chosen_state]
    if time_col:
        v2[time_col] = pd.to_datetime(v2[time_col], errors='coerce')
        v2 = v2.sort_values(by=time_col)
        fig = px.line(v2, x=time_col, y=dose_col,
                      title=f"Vaccine Doses Over Time: {chosen_state}",
                      markers=True, color_discrete_sequence=["#36d399"],
                      labels={time_col: "Date", dose_col: "Total Doses"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No time series found. Showing latest data for all states.")
        gdf = vax.sort_values(dose_col, ascending=False).head(15)
        fig = px.bar(gdf, y=state_col, x=dose_col,
                     orientation='h', color=dose_col,
                     title="Current: Top 15 States by Vaccine Doses",
                     color_continuous_scale="viridis")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(v2 if time_col else gdf, use_container_width=True)
    st.download_button("Download Vaccination Data", v2.to_csv(index=False), file_name=f"vax_{chosen_state}.csv")
    st.caption("Source: MOHFW/CoWIN")


st.markdown("<hr style='border-top: 1.7px dashed #36d399; margin-top:40px; margin-bottom:24px; width:96%;' />", unsafe_allow_html=True)

custom_footer = """
<style>
#credit-div {position: fixed; right: 32px; bottom: 22px; background: rgba(20,23,39,0.93); color: #36D399; border-radius: 14px; padding: 10px 20px 10px 16px;
font-size: 16px; z-index:9999; box-shadow:0 8px 30px 0 rgba(44,230,221,.12);}
#credit-div strong {color:#7dfa81;}
</style>
<div id='credit-div'>
Made with❤️ by <strong>Atharv Shinde </strong>, <strong>Aniket Gawande </strong>, <strong>Ashutosh More </strong>
</div>
<script>document.body.appendChild(Object.assign(document.createElement("div"),{innerHTML:`${document.getElementById('credit-div').outerHTML}`}));document.getElementById('credit-div').remove()</script>
"""

st.markdown(custom_footer, unsafe_allow_html=True)




