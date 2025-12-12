# app.py
"""
Heart Survey — Stable Feature Visual Explorer (minimal interaction).
- No predictive models / no diagnostics.
Run: python app.py
"""
from pathlib import Path
import pandas as pd
import numpy as np

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
DATA_PATH = Path("data/heart_data.csv")
APP_TITLE = "Heart Survey — Feature Visual Explorer"
SAMPLE_MAX = 2500   # sampling ceiling for heavy plots
# ----------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}.")

# Load & robust-clean
df = pd.read_csv(DATA_PATH)

# ensure predictable dtypes and fill NaNs
obj_cols = df.select_dtypes(include="object").columns.tolist()
num_cols = df.select_dtypes(include=["float64","int64"]).columns.tolist()
df[obj_cols] = df[obj_cols].fillna("Unknown")
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Ensure expected columns exist (best-effort)
expected = ["AgeCategory","GeneralHealth","SmokerStatus","BMI","SleepHours",
            "PhysicalHealthDays","MentalHealthDays","WeightInKilograms","HeightInMeters","State","Sex","RemovedTeeth"]
for col in expected:
    if col not in df.columns:
        if col in num_cols:
            df[col] = np.nan
        else:
            df[col] = "Unknown"

# Map states to abbreviations (best-effort)
STATE_MAP = {
 "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO",
 "Connecticut":"CT","Delaware":"DE","Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID",
 "Illinois":"IL","Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA",
 "Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN","Mississippi":"MS",
 "Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH","New Jersey":"NJ",
 "New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND","Ohio":"OH","Oklahoma":"OK",
 "Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD",
 "Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA",
 "West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY"
}
df["state_abbr"] = df["State"].map(STATE_MAP).fillna("")

# ---------------- Figure builders (robust) ----------------

def fig_correlation_heatmap(dataframe):
    numeric = ["BMI","SleepHours","PhysicalHealthDays","MentalHealthDays","WeightInKilograms","HeightInMeters"]
    present = [c for c in numeric if c in dataframe.columns]
    if not present:
        fig = go.Figure()
        fig.update_layout(title="No numeric features available")
        return fig
    corr = dataframe[present].corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", text_auto=".2f", aspect="auto", template="plotly_dark")
    fig.update_layout(title="Correlation heatmap (numeric features)", margin=dict(l=10,r=10,t=30,b=10))
    return fig

def fig_sunburst_age_health_smoke(dataframe):
    cols = ["AgeCategory","GeneralHealth","SmokerStatus"]
    for c in cols:
        if c not in dataframe.columns:
            dataframe[c] = "Unknown"
    fig = px.sunburst(dataframe, path=cols, maxdepth=3, template="plotly_dark")
    fig.update_layout(title="Sunburst: Age → GeneralHealth → SmokerStatus", margin=dict(l=10,r=10,t=30,b=10))
    return fig

def fig_violin_sleep_by_health(dataframe):
    if "SleepHours" not in dataframe.columns or "GeneralHealth" not in dataframe.columns:
        return go.Figure()
    fig = px.violin(dataframe, x="GeneralHealth", y="SleepHours", color="GeneralHealth", box=True, points="outliers", template="plotly_dark")
    fig.update_layout(title="Sleep hours distribution by General Health", showlegend=False, margin=dict(l=10,r=10,t=30,b=10))
    return fig

def fig_density_bmi_sleep(dataframe):
    # Use go.Histogram2dContour for compatibility (works well across Plotly 5+)
    if "BMI" not in dataframe.columns or "SleepHours" not in dataframe.columns:
        return go.Figure()
    s = dataframe[["BMI","SleepHours"]].dropna().sample(min(len(dataframe), SAMPLE_MAX), random_state=1)
    fig = go.Figure()
    fig.add_trace(go.Histogram2dContour(
        x=s["BMI"],
        y=s["SleepHours"],
        colorscale=[[0, 'rgba(155,124,255,0.05)'], [0.5, 'rgba(0,255,195,0.1)'], [1, 'rgba(155,124,255,0.25)']],
        contours_coloring='fill',
        reversescale=False,
        showscale=False,
        nbinsx=30,
        nbinsy=30,
        line=dict(width=0)
    ))
    # overlay scatter (light) for points density sense
    fig.add_trace(go.Scatter(
        x=s["BMI"],
        y=s["SleepHours"],
        mode="markers",
        marker=dict(size=4, opacity=0.35, color="rgba(255,255,255,0.06)"),
        hoverinfo="skip",
        showlegend=False
    ))
    fig.update_layout(title="BMI vs Sleep (density contour)", xaxis_title="BMI", yaxis_title="Sleep Hours",
                      template="plotly_dark", margin=dict(l=10,r=10,t=30,b=10))
    return fig

def fig_box_bmi_by_age(dataframe):
    if "BMI" not in dataframe.columns or "AgeCategory" not in dataframe.columns:
        return go.Figure()
    order = sorted(dataframe["AgeCategory"].unique(), key=lambda x: str(x))
    fig = px.box(dataframe, x="AgeCategory", y="BMI", category_orders={"AgeCategory":order}, points="outliers", template="plotly_dark")
    fig.update_layout(title="BMI distribution across Age categories", margin=dict(l=10,r=10,t=30,b=10))
    return fig

def fig_parallel_coords_numeric(dataframe):
    numeric = ["BMI","SleepHours","PhysicalHealthDays","MentalHealthDays","WeightInKilograms"]
    present = [c for c in numeric if c in dataframe.columns]
    if not present:
        return go.Figure()
    s = dataframe[present].dropna().sample(min(len(dataframe), 800), random_state=2)
    fig = px.parallel_coordinates(s, dimensions=present, color=s["BMI"] if "BMI" in s.columns else None, template="plotly_dark")
    fig.update_layout(title="Parallel coordinates (sampled)", margin=dict(l=10,r=10,t=30,b=10))
    return fig

def fig_scatter_matrix_sample(dataframe):
    numeric = ["BMI","SleepHours","PhysicalHealthDays","MentalHealthDays"]
    present = [c for c in numeric if c in dataframe.columns]
    if len(present) < 2:
        return go.Figure()
    s = dataframe[present].dropna().sample(min(len(dataframe), 1200), random_state=3)
    fig = px.scatter_matrix(s, dimensions=present, template="plotly_dark")
    fig.update_layout(title="Scatter matrix (sampled)", margin=dict(l=10,r=10,t=40,b=10))
    return fig

def fig_removed_teeth_pie(dataframe):
    if "RemovedTeeth" not in dataframe.columns:
        return go.Figure()
    fig = px.pie(dataframe, names="RemovedTeeth", hole=0.5, template="plotly_dark")
    fig.update_layout(title="Removed teeth breakdown", margin=dict(l=10,r=10,t=30,b=10))
    return fig

# Build figures once (static) to reduce runtime callbacks
FIG_CORR = fig_correlation_heatmap(df)
FIG_SUNBURST = fig_sunburst_age_health_smoke(df)
FIG_VIOLIN = fig_violin_sleep_by_health(df)
FIG_DENSITY = fig_density_bmi_sleep(df)
FIG_BOX_BMI = fig_box_bmi_by_age(df)
FIG_PARALLEL = fig_parallel_coords_numeric(df)
FIG_MATRIX = fig_scatter_matrix_sample(df)
FIG_TEETH = fig_removed_teeth_pie(df)

# ---------------- Dash app ----------------
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

header = dbc.Row([
    dbc.Col(html.Img(src="/assets/avatar.png", className="avatar"), width="auto"),
    dbc.Col(html.Div([html.H2(APP_TITLE, className="app-title"),
                      html.Div("Feature-focused visual explorer — descriptive only", className="muted")]), style={"paddingLeft":"12px"}),
    dbc.Col(dbc.Button("Replay Intro", id="replay-intro", className="ghost-btn"), width="auto")
], align="center", className="header-row")

info_card = html.Div([
    html.H5("Dataset summary", className="panel-title"),
    html.Div(f"Rows: {len(df):,}", className="muted"),
    html.Div(f"Numeric cols: {len(num_cols)} • Categorical cols: {len(obj_cols)}", className="muted"),
    html.Hr(className="soft-hr"),
    html.Div("Design: focus on features, stable UX, no predictive outputs.", className="muted tiny")
], className="sidebar glass-card")

tabs = dcc.Tabs(id="tabs", value="overview", children=[
    dcc.Tab(label="Overview", value="overview", className="tab", selected_className="tab--selected"),
    dcc.Tab(label="Feature Visuals", value="features", className="tab", selected_className="tab--selected"),
    dcc.Tab(label="Deep Views", value="deep", className="tab", selected_className="tab--selected"),
], className="tabs")

app.layout = dbc.Container([
    html.Div(id="intro-wrap", className="intro-wrap play"),
    header,
    dbc.Row([
        dbc.Col(info_card, width=3),
        dbc.Col([tabs, html.Div(id="tab-content", className="tab-content")], width=9)
    ], className="main-row"),
    html.Footer("Built with ❤️ — Descriptive-only visual explorer", className="footer")
], fluid=True, className="app-container")

# Minimal tab rendering callback
@app.callback(Output("tab-content","children"), Input("tabs","value"))
def render_tab(tab):
    if tab == "overview":
        return html.Div([
            dbc.Row([
                dbc.Col(dbc.Card([html.Div("Total Responses", className="kpi-label"), html.Div(f"{len(df):,}", className="kpi-value")], className="glass-card kpi-card"), width=3),
                dbc.Col(dbc.Card([html.Div("Median BMI", className="kpi-label"), html.Div(f"{df['BMI'].median():.1f}", className="kpi-value")], className="glass-card kpi-card"), width=3),
                dbc.Col(dbc.Card([html.Div("Avg Sleep (hrs)", className="kpi-label"), html.Div(f"{df['SleepHours'].mean():.1f}", className="kpi-value")], className="glass-card kpi-card"), width=3),
                dbc.Col(dbc.Card([html.Div("Distinct Age groups", className="kpi-label"), html.Div(f"{df['AgeCategory'].nunique()}", className="kpi-value")], className="glass-card kpi-card"), width=3),
            ], className="kpi-row"),
            dbc.Row([
                dbc.Col(dbc.Card(dcc.Graph(figure=FIG_SUNBURST)), width=6),
                dbc.Col(dbc.Card(dcc.Graph(figure=FIG_CORR)), width=6),
            ]),
            dbc.Row([
                dbc.Col(dbc.Card(dcc.Graph(figure=FIG_VIOLIN)), width=6),
                dbc.Col(dbc.Card(dcc.Graph(figure=FIG_BOX_BMI)), width=6),
            ])
        ], className="tab-panel")

    elif tab == "features":
        return html.Div([
            dbc.Row([
                dbc.Col(dbc.Card(dcc.Graph(figure=FIG_DENSITY)), width=6),
                dbc.Col(dbc.Card(dcc.Graph(figure=FIG_TEETH)), width=6),
            ]),
            dbc.Row([
                dbc.Col(dbc.Card(dcc.Graph(figure=FIG_MATRIX)), width=12)
            ])
        ], className="tab-panel")

    else:  # deep
        return html.Div([
            dbc.Row([
                dbc.Col(dbc.Card(dcc.Graph(figure=FIG_PARALLEL)), width=12)
            ]),
            dbc.Row([
                dbc.Col(dbc.Card(html.Div([
                    html.H6("Notes", className="muted"),
                    html.P("این نماها برای نمایش الگوها طراحی شده‌اند؛ هیچ پیش‌بینی یا تشخیصی تولید نمی‌شود.", className="muted tiny")
                ]), className="glass-card"), width=12)
            ])
        ], className="tab-panel")

# simple client-side replay intro (no heavy logic)
app.clientside_callback(
    """
    function(n_clicks) {
        const el = document.getElementById('intro-wrap');
        if(!el) return '';
        el.classList.remove('play');
        void el.offsetWidth;
        el.classList.add('play');
        return '';
    }
    """,
    Output('intro-wrap','children'),
    Input('replay-intro','n_clicks')
)

if __name__ == "__main__":
    app.run_server(debug=True)
