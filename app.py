import dash
import pandas as pd
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, r2_score

# Load and prepare dataset from CSV
def load_and_prepare_data(file_path="inquiry_data.csv"):
    """Load and prepare inquiry data from CSV."""
    try:
        df = pd.read_csv(file_path)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        return df
    except FileNotFoundError:
        print("inquiry_data.csv not found. Generating fallback data.")
        from data import generate_inquiry_data
        return generate_inquiry_data(2000)

df = load_and_prepare_data()

# Data validation
def validate_dataset(df):
    """Validate the dataset for missing or invalid values."""
    if df.empty:
        print("Dataset is empty after loading and preprocessing.")
    if df["Inquiries"].isnull().any():
        print("Some inquiry values are missing or invalid.")
        df["Inquiries"] = df["Inquiries"].fillna(0)
    if df["PreviousInquiries"].isnull().any():
        print("Some previous inquiries values are missing or invalid.")
        df["PreviousInquiries"] = df["PreviousInquiries"].fillna(0)
    if df["TargetInquiries"].isnull().any():
        print("Some target inquiries values are missing or invalid.")
        df["TargetInquiries"] = df["TargetInquiries"].fillna(0)
    if df["StatusCode"].isnull().any():
        print("Some StatusCode values are missing or invalid.")
        df["StatusCode"] = df["StatusCode"].fillna(200)
    if df["HttpMethod"].isnull().any():
        print("Some HttpMethod values are missing or invalid.")
        df["HttpMethod"] = df["HttpMethod"].fillna("GET")
    if df["Url"].isnull().any():
        print("Some Url values are missing or invalid.")
        df["Url"] = df["Url"].fillna("/api/default/0")
    if df["Sales"].isnull().any():
        print("Some sales values are missing or invalid.")
        df["Sales"] = df["Sales"].fillna(0)
    if df["ForecastedSales"].isnull().any():
        print("Some forecasted sales values are missing or invalid.")
        df["ForecastedSales"] = df["ForecastedSales"].fillna(0)
    if (df["Sales"] < 0).any() or (df["ForecastedSales"] < 0).any():
        print("Invalid negative sales values detected.")
        df["Sales"] = df["Sales"].clip(lower=0)
        df["ForecastedSales"] = df["ForecastedSales"].clip(lower=0)
    if df["Timestamp"].isnull().any():
        print("Some Timestamp values are invalid (NaT). Dropping affected rows.")
        df = df.dropna(subset=["Timestamp"])
    if "Hour" not in df.columns:
        print("Hour column missing. Adding Hour column by extracting from Timestamp.")
        df["Hour"] = df["Timestamp"].dt.hour
    if df["Hour"].isnull().any():
        print("Some Hour values are invalid (NaN). Dropping affected rows.")
        df = df.dropna(subset=["Hour"])
    if (df["Hour"] < 0).any() or (df["Hour"] > 23).any():
        print("Invalid Hour values detected (outside 0-23 range). Dropping affected rows.")
        df = df[(df["Hour"] >= 0) & (df["Hour"] <= 23)]
    df["Hour"] = df["Hour"].astype(int)
    return df.dropna()

df = validate_dataset(df)

# Initialize app
app = dash.Dash(__name__)

# Chart creation functions with improved visuals and labeling
def create_peak_inquiry_hours(df_filtered):
    """Create a bar chart showing the hourly distribution of inquiries, highlighting the peak hour."""
    total_days = (df_filtered["Timestamp"].dt.date.max() - df_filtered["Timestamp"].dt.date.min()).days + 1 if not df_filtered.empty else 1

    hourly_df = df_filtered.groupby("Hour").agg({"Inquiries": "sum", "TargetInquiries": "sum"}).reset_index()
    hourly_df = hourly_df.set_index("Hour").reindex(range(24), fill_value=0).reset_index()
    
    hourly_df["Inquiries"] = hourly_df["Inquiries"].clip(upper=500)
    hourly_df["TargetInquiries"] = hourly_df["TargetInquiries"].clip(upper=500)
    
    peak_hour = hourly_df.loc[hourly_df["Inquiries"].idxmax()]
    peak_hour_value = peak_hour["Hour"]
    peak_inquiries = peak_hour["Inquiries"]
    
    colors = ["#1f77b4"] * len(hourly_df)
    peak_idx = hourly_df[hourly_df["Hour"] == peak_hour_value].index[0]
    colors[peak_idx] = "#ff4d4d"
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly_df["Hour"],
        y=hourly_df["Inquiries"],
        name="Actual Inquiries",
        marker_color=colors,
        opacity=0.7,
        width=0.5,
        hovertemplate="Hour: %{x}:00<br>Inquiries: %{y}<extra></extra>"
    ))
    
    fig.add_trace(go.Scatter(
        x=hourly_df["Hour"],
        y=hourly_df["TargetInquiries"],
        name="Target Inquiries",
        mode="lines+markers",
        line=dict(color="#2ca02c", width=2, dash="dash"),
        marker=dict(size=6),
        hovertemplate="Hour: %{x}:00<br>Target: %{y}<extra></extra>"
    ))
    
    total_inquiries = hourly_df["Inquiries"].sum()
    total_target = hourly_df["TargetInquiries"].sum()
    performance = (total_inquiries / total_target * 100) if total_target > 0 else 0
    perf_color = "red" if performance < 70 else "yellow" if performance < 90 else "green"
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        title="Hourly Inquiry Distribution",
        title_font=dict(size=14, color="#333"),
        font=dict(size=12, color="#666"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        legend=dict(x=1.05, y=0.5, font=dict(size=10)),
        xaxis=dict(
            title="Hour of Day",
            title_font=dict(size=12),
            tickfont=dict(size=10),
            gridcolor="#eee",
            tickvals=list(range(24)),
            ticktext=[f"{h}:00" for h in range(24)]
        ),
        yaxis=dict(
            title="Number of Inquiries",
            title_font=dict(size=12),
            tickfont=dict(size=10),
            gridcolor="#ddd",
            range=[0, max(hourly_df["Inquiries"].max(), hourly_df["TargetInquiries"].max()) * 1.2]
        )
    )
    
    fig.add_annotation(
        text=f"Perf: {int(performance)}%",
        xref="paper",
        yref="paper",
        x=1.05,
        y=0.95,
        showarrow=False,
        font=dict(size=12, color=perf_color, weight="bold"),
        bgcolor="white",
        bordercolor=perf_color,
        borderwidth=2
    )
    
    fig.add_annotation(
        text=f"Over {total_days} Days",
        xref="paper",
        yref="paper",
        x=1.05,
        y=0.85,
        showarrow=False,
        font=dict(size=10, color="#666"),
        bgcolor="white",
        bordercolor="#666",
        borderwidth=1
    )
    
    fig.add_annotation(
        text=f"Peak: {int(peak_hour_value)}:00 ({int(peak_inquiries)})",
        x=peak_hour_value,
        y=peak_inquiries,
        xref="x",
        yref="y",
        showarrow=True,
        arrowhead=2,
        ax=30 if peak_hour_value < 12 else -30,
        ay=-40,
        font=dict(size=10, color="#ff4d4d", weight="bold"),
        bgcolor="white",
        bordercolor="#ff4d4d",
        borderwidth=1
    )
    
    return fig

def create_sales_vs_inquiries_scatter(df_filtered):
    """Create a scatter plot with positive trend lines for Sales vs Inquiries by Salesperson."""
    fig = px.scatter(
        df_filtered,
        x="Inquiries",
        y="Sales",
        color="Salesperson",
        title="Sales vs Inquiries by Salesperson",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hover_data={"Inquiries": True, "Sales": True, "Salesperson": True, "Product": True, "ReferralSource": True}
    )

    for salesperson in df_filtered["Salesperson"].unique():
        df_sp = df_filtered[df_filtered["Salesperson"] == salesperson]
        if len(df_sp) > 1:
            X = df_sp["Inquiries"].values.reshape(-1, 1)
            y = df_sp["Sales"].values
            model = LinearRegression().fit(X, y)
            x_range = np.array([[df_filtered["Inquiries"].min()], [df_filtered["Inquiries"].max()]])
            y_pred = model.predict(x_range)
            slope = model.coef_[0]
            fig.add_trace(
                go.Scatter(
                    x=x_range.flatten(),
                    y=y_pred,
                    mode="lines",
                    name=f"{salesperson} Trend (Slope: {slope:.2f})",
                    line=dict(dash="dash", width=2),
                    showlegend=True
                )
            )

    fig.update_traces(marker=dict(size=10, opacity=0.6))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        title_font=dict(size=14, color="#333"),
        font=dict(size=12, color="#666"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        legend=dict(x=1.05, y=0.5, font=dict(size=10)),
        xaxis=dict(
            title="Number of Inquiries",
            title_font=dict(size=12),
            tickfont=dict(size=10),
            gridcolor="#ddd",
            range=[0, df_filtered["Inquiries"].max() * 1.1]
        ),
        yaxis=dict(
            title="Sales (Units)",
            title_font=dict(size=12),
            tickfont=dict(size=10),
            gridcolor="#ddd",
            range=[0, df_filtered["Sales"].max() * 1.1]
        )
    )
    return fig

def create_sales_forecast(df_filtered):
    """Create a line chart for actual vs forecasted sales with accuracy metrics and daily resampling."""
    trend_df = df_filtered.resample("D", on="Timestamp").agg({"Sales": "sum", "ForecastedSales": "sum"}).reset_index()
    
    trend_df["Sales"] = trend_df["Sales"].rolling(window=7, min_periods=1).mean()
    trend_df["ForecastedSales"] = trend_df["ForecastedSales"].rolling(window=7, min_periods=1).mean()

    if len(trend_df) > 1:
        y_true = trend_df["Sales"].values
        y_pred = trend_df["ForecastedSales"].values
        mae = mean_absolute_error(y_true[~np.isnan(y_true)], y_pred[~np.isnan(y_pred)])
        r2 = r2_score(y_true[~np.isnan(y_true)], y_pred[~np.isnan(y_pred)])
    else:
        mae = 0
        r2 = 0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend_df["Timestamp"], y=trend_df["Sales"], name="Actual Sales",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Actual: %{y}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=trend_df["Timestamp"], y=trend_df["ForecastedSales"], name="Forecasted Sales",
        line=dict(color="#2ca02c", width=2, dash="dash"),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Forecast: %{y}<extra></extra>"
    ))
    fig.update_layout(
        height=200, margin=dict(l=10, r=10, t=30, b=10),
        title="Actual vs Forecasted Sales", title_font=dict(size=14, color="#333"),
        font=dict(size=12, color="#666"), plot_bgcolor="white",
        paper_bgcolor="white", showlegend=True, legend=dict(x=1.05, y=0.5, font=dict(size=10)),
        xaxis=dict(title="Date", title_font=dict(size=12), tickfont=dict(size=10), gridcolor="#eee"),
        yaxis=dict(title="Sales (Units)", title_font=dict(size=12), tickfont=dict(size=10), gridcolor="#eee")
    )

    return fig, mae, r2

def create_individual_gauge(df_filtered):
    """Create a gauge chart for individual or team performance, fully labeled."""
    if "salesperson-filter" in df_filtered.columns:
        salesperson = df_filtered["salesperson-filter"].iloc[0]
        if salesperson == "All":
            total_inquiries = df_filtered["Inquiries"].sum()
            target = df_filtered["TargetInquiries"].sum()
            title_text = "Team Performance vs Target (%)"
        else:
            salesperson_df = df_filtered[df_filtered["Salesperson"] == salesperson]
            total_inquiries = salesperson_df["Inquiries"].sum()
            target = salesperson_df["TargetInquiries"].sum()
            title_text = f"{salesperson} Performance vs Target (%)"
        
        value = (total_inquiries / target * 100) if target > 0 else 0
        color = "red" if value < 70 else "yellow" if value < 90 else "green"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title_text, 'font': {'size': 14, 'color': "#333"}},
            number={'font': {'size': 20, 'color': color}, 'suffix': "%"},
            gauge={
                'axis': {'range': [0, 150], 'tickfont': {'size': 12}, 'tickcolor': "#666", 'tickmode': 'linear', 'dtick': 30},
                'bar': {'color': color, 'thickness': 0.3},
                'steps': [
                    {'range': [0, 70], 'color': "red"},
                    {'range': [70, 90], 'color': "yellow"},
                    {'range': [90, 150], 'color': "green"}
                ],
                'threshold': {'line': {'color': "black", 'width': 2}, 'thickness': 0.75, 'value': 100}
            }
        ))
        fig.update_layout(
            height=200, margin=dict(l=10, r=10, t=30, b=10),
            font=dict(size=12, color="#666"), plot_bgcolor="white", paper_bgcolor="white",
            annotations=[
                dict(
                    text=f"Total: {int(total_inquiries)} | Target: {int(target)}",
                    xref="paper", yref="paper", x=0.5, y=-0.1, showarrow=False,
                    font=dict(size=10, color="#666")
                )
            ]
        )
        return fig
    return go.Figure().update_layout(
        height=200, margin=dict(l=10, r=10, t=30, b=10),
        title="Select a Salesperson", title_font=dict(size=14, color="#333"),
        font=dict(size=12, color="#666"), plot_bgcolor="white", paper_bgcolor="white"
    )

def create_demographics_engagement(df_filtered):
    """Create a bar chart for inquiries by demographics without target, fully labeled."""
    demo_df = df_filtered.groupby(["AgeGroup", "Gender"]).agg({"Inquiries": "sum"}).reset_index()
    fig = px.bar(
        demo_df, x="AgeGroup", y="Inquiries", color="Gender",
        title="Engagement by Demographics",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hover_data={"Inquiries": True}
    )
    fig.update_layout(
        height=200, margin=dict(l=10, r=10, t=30, b=10),
        title_font=dict(size=14, color="#333"), font=dict(size=12, color="#666"),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=True,
        legend=dict(x=1.05, y=0.5, font=dict(size=10)),
        xaxis=dict(title="Age Group", title_font=dict(size=12), tickfont=dict(size=10), gridcolor="#eee"),
        yaxis=dict(title="Number of Inquiries", title_font=dict(size=12), tickfont=dict(size=10), gridcolor="#eee")
    )
    return fig

def create_products_by_region(df_filtered):
    """Create a bar chart for products by region with only inquiries, fully labeled."""
    product_region_df = df_filtered.groupby(["Continent", "Product"]).agg({"Inquiries": "sum"}).reset_index()
    fig = px.bar(
        product_region_df, x="Continent", y="Inquiries", color="Product",
        title="Products by Region",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hover_data={"Inquiries": True}
    )
    fig.update_layout(
        height=200, margin=dict(l=10, r=10, t=30, b=10),
        title_font=dict(size=14, color="#333"), font=dict(size=12, color="#666"),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=True,
        legend=dict(x=1.05, y=0.5, font=dict(size=10)),
        xaxis=dict(title="Continent", title_font=dict(size=12), tickfont=dict(size=10), tickangle=45, gridcolor="#eee"),
        yaxis=dict(title="Number of Inquiries", title_font=dict(size=12), tickfont=dict(size=10), gridcolor="#eee")
    )
    return fig

def create_referral_sources(df_filtered):
    """Create a funnel chart for inquiries by referral source with salesperson attribution, fully labeled."""
    referral_df = df_filtered.groupby(["ReferralSource", "Salesperson"]).agg({"Inquiries": "sum", "TargetInquiries": "sum"}).reset_index()
    fig = px.funnel(
        referral_df, x="Inquiries", y="ReferralSource", color="Salesperson",
        title="Inquiries by Referral Source",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        hover_data={"Inquiries": True, "TargetInquiries": True, "Salesperson": True}
    )
    performance = referral_df["Inquiries"].sum() / referral_df["TargetInquiries"].sum() * 100 if referral_df["TargetInquiries"].sum() > 0 else 0
    color = "red" if performance < 70 else "yellow" if performance < 90 else "green"
    fig.update_layout(
        height=200, margin=dict(l=10, r=10, t=30, b=10),
        title_font=dict(size=14, color="#333"), font=dict(size=12, color="#666"),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=True,
        legend=dict(x=1.05, y=0.5, font=dict(size=10)),
        xaxis=dict(title="Number of Inquiries", title_font=dict(size=12), tickfont=dict(size=10), gridcolor="#eee"),
        yaxis=dict(title="Referral Source", title_font=dict(size=12), tickfont=dict(size=10))
    )
    fig.add_annotation(
        text=f"Perf: {int(performance)}%", xref="paper", yref="paper",
        x=1.05, y=1, showarrow=False, font=dict(size=12, color=color, weight="bold")
    )
    return fig

def create_product_interest_by_country(df_filtered):
    """Create a choropleth map for inquiries per country with target comparison, fully labeled."""
    product_country_df = df_filtered.groupby("Country").agg({"Inquiries": "sum", "TargetInquiries": "sum"}).reset_index()
    fig = px.choropleth(
        product_country_df, locations="Country", locationmode="country names",
        color="Inquiries", hover_name="Country", hover_data={"Inquiries": True, "TargetInquiries": True},
        title="Inquiries by Country",
        color_continuous_scale=px.colors.diverging.RdYlGn_r
    )
    performance = product_country_df["Inquiries"].sum() / product_country_df["TargetInquiries"].sum() * 100 if product_country_df["TargetInquiries"].sum() > 0 else 0
    color = "red" if performance < 70 else "yellow" if performance < 90 else "green"
    fig.update_layout(
        height=200, margin=dict(l=10, r=10, t=30, b=10),
        title_font=dict(size=14, color="#333"), font=dict(size=12, color="#666"),
        geo=dict(bgcolor="white", showframe=False, projection_scale=1),
        coloraxis_colorbar=dict(
            title="Number of Inquiries", title_font=dict(size=12), tickfont=dict(size=10)
        )
    )
    fig.add_annotation(
        text=f"Perf: {int(performance)}%", xref="paper", yref="paper",
        x=1.05, y=1, showarrow=False, font=dict(size=12, color=color, weight="bold")
    )
    return fig

def create_salesperson_performance(df_filtered):
    """Create a bar chart for salesperson performance with target and previous period, fully labeled."""
    sales_df = df_filtered.groupby("Salesperson").agg({"Inquiries": "sum", "PreviousInquiries": "sum", "TargetInquiries": "sum"}).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sales_df["Salesperson"], y=sales_df["Inquiries"], name="Actual Inquiries",
        marker_color="#1f77b4", opacity=0.8, width=0.4,
        hovertemplate="Salesperson: %{x}<br>Inquiries: %{y}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=sales_df["Salesperson"], y=sales_df["PreviousInquiries"], name="Last Month Inquiries",
        marker_color="#ff7f0e", opacity=0.6, width=0.4,
        hovertemplate="Salesperson: %{x}<br>Last Month: %{y}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=sales_df["Salesperson"], y=sales_df["TargetInquiries"], name="Target Inquiries",
        mode="lines+markers", line=dict(color="#2ca02c", width=2, dash="dash"), marker=dict(size=8),
        hovertemplate="Salesperson: %{x}<br>Target: %{y}<extra></extra>"
    ))
    performance = sales_df["Inquiries"].sum() / sales_df["TargetInquiries"].sum() * 100 if sales_df["TargetInquiries"].sum() > 0 else 0
    color = "red" if performance < 70 else "yellow" if performance < 90 else "green"
    fig.update_layout(
        height=200, margin=dict(l=10, r=10, t=30, b=10),
        title="Salesperson Performance", title_font=dict(size=14, color="#333"),
        font=dict(size=12, color="#666"), plot_bgcolor="white",
        paper_bgcolor="white", showlegend=True, legend=dict(x=1.05, y=0.5, font=dict(size=10)),
        xaxis=dict(title="Salesperson", title_font=dict(size=12), tickfont=dict(size=10), gridcolor="#eee"),
        yaxis=dict(title="Number of Inquiries", title_font=dict(size=12), tickfont=dict(size=10), gridcolor="#eee"),
        barmode="group"
    )
    fig.add_annotation(
        text=f"Perf: {int(performance)}%", xref="paper", yref="paper",
        x=1.05, y=1, showarrow=False, font=dict(size=12, color=color, weight="bold")
    )
    return fig

# Layout with refined internal styles and KPI section
app.layout = html.Div([
    html.H1("AI-Solutions Analytics Dashboard", style={
        "backgroundColor": "#2c3e50", "color": "white", "padding": "20px", "textAlign": "center",
        "borderRadius": "8px 8px 0 0", "margin": "0", "fontSize": "26px", "fontWeight": "600",
        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
    }),
    # Filters with labeled headings
    html.Div([
        html.Div([
            html.Label("Select Salesperson", style={
                "fontSize": "14px", "color": "#333", "fontWeight": "bold", "marginBottom": "5px"
            }),
            dcc.Dropdown(
                id='salesperson-filter',
                options=[{'label': 'All', 'value': 'All'}] + [{'label': sp, 'value': sp} for sp in df["Salesperson"].unique()],
                value='All',
                placeholder="Choose a salesperson",
                style={"width": "100%", "fontSize": "14px", "borderRadius": "5px", "border": "1px solid #d1d5db"}
            ),
        ], style={"width": "25%", "display": "inline-block", "padding": "5px"}),
        html.Div([
            html.Label("Select Continent", style={
                "fontSize": "14px", "color": "#333", "fontWeight": "bold", "marginBottom": "5px"
            }),
            dcc.Dropdown(
                id='continent-filter',
                options=[{'label': 'All', 'value': 'All'}] + [{'label': c, 'value': c} for c in df["Continent"].unique()],
                value='All',
                placeholder="Choose a continent",
                style={"width": "100%", "fontSize": "14px", "borderRadius": "5px", "border": "1px solid #d1d5db"}
            ),
        ], style={"width": "25%", "display": "inline-block", "padding": "5px"}),
        html.Div([
            html.Label("Select Product", style={
                "fontSize": "14px", "color": "#333", "fontWeight": "bold", "marginBottom": "5px"
            }),
            dcc.Dropdown(
                id='product-filter',
                options=[{'label': 'All', 'value': 'All'}] + [{'label': sp, 'value': sp} for sp in df["Product"].unique()],
                value='All',
                placeholder="Choose a product",
                style={"width": "100%", "fontSize": "14px", "borderRadius": "5px", "border": "1px solid #d1d5db"}
            ),
        ], style={"width": "25%", "display": "inline-block", "padding": "5px"}),
        html.Div([
            html.Label("Select Referral Source", style={
                "fontSize": "14px", "color": "#333", "fontWeight": "bold", "marginBottom": "5px"
            }),
            dcc.Dropdown(
                id='referral-filter',
                options=[{'label': 'All', 'value': 'All'}] + [{'label': r, 'value': r} for r in df["ReferralSource"].unique()],
                value='All',
                placeholder="Choose a referral source",
                style={"width": "100%", "fontSize": "14px", "borderRadius": "5px", "border": "1px solid #d1d5db"}
            ),
        ], style={"width": "25%", "display": "inline-block", "padding": "5px"}),
    ], style={
        "backgroundColor": "white", "padding": "15px", "borderRadius": "8px",
        "boxShadow": "0 2px 4px rgba(0,0,0,0.05)", "marginBottom": "20px", "marginTop": "5px"
    }),
    # KPI Section
    html.Div(id="kpi-section", style={
        "display": "flex", "justifyContent": "space-between", "marginBottom": "20px",
        "backgroundColor": "white", "padding": "15px", "borderRadius": "8px",
        "boxShadow": "0 2px 4px rgba(0,0,0,0.05)"
    }),
    # 3x3 Grid with 9 visualizations
    html.Div([
        html.Div([
            html.Div(dcc.Graph(id="peak-inquiry-hours"), style={
                "width": "33%", "display": "inline-block", "padding": "5px",
                "border": "1px solid #e0e0e0", "borderRadius": "5px", "backgroundColor": "#fafafa"
            }),
            html.Div(dcc.Graph(id="sales-vs-inquiries-scatter"), style={
                "width": "33%", "display": "inline-block", "padding": "5px",
                "border": "1px solid #e0e0e0", "borderRadius": "5px", "backgroundColor": "#fafafa"
            }),
            html.Div([
                dcc.Graph(id="sales-forecast"),
                html.Div(id="forecast-accuracy", style={
                    "textAlign": "center", "marginTop": "10px", "fontSize": "14px", "color": "#333"
                })
            ], style={
                "width": "33%", "display": "inline-block", "padding": "5px",
                "border": "1px solid #e0e0e0", "borderRadius": "5px", "backgroundColor": "#fafafa"
            }),
        ], style={"display": "flex", "justifyContent": "space-between"}),
        html.Div([
            html.Div(dcc.Graph(id="demographics-engagement"), style={
                "width": "33%", "display": "inline-block", "padding": "5px",
                "border": "1px solid #e0e0e0", "borderRadius": "5px", "backgroundColor": "#fafafa"
            }),
            html.Div(dcc.Graph(id="products-by-region"), style={
                "width": "33%", "display": "inline-block", "padding": "5px",
                "border": "1px solid #e0e0e0", "borderRadius": "5px", "backgroundColor": "#fafafa"
            }),
            html.Div(dcc.Graph(id="referral-sources"), style={
                "width": "33%", "display": "inline-block", "padding": "5px",
                "border": "1px solid #e0e0e0", "borderRadius": "5px", "backgroundColor": "#fafafa"
            }),
        ], style={"display": "flex", "justifyContent": "space-between"}),
        html.Div([
            html.Div(dcc.Graph(id="inquiries-by-country"), style={
                "width": "33%", "display": "inline-block", "padding": "5px",
                "border": "1px solid #e0e0e0", "borderRadius": "5px", "backgroundColor": "#fafafa"
            }),
            html.Div(dcc.Graph(id="salesperson-performance"), style={
                "width": "33%", "display": "inline-block", "padding": "5px",
                "border": "1px solid #e0e0e0", "borderRadius": "5px", "backgroundColor": "#fafafa"
            }),
            html.Div(dcc.Graph(id="individual-gauge"), style={
                "width": "33%", "display": "inline-block", "padding": "5px",
                "border": "1px solid #e0e0e0", "borderRadius": "5px", "backgroundColor": "#fafafa"
            }),
        ], style={"display": "flex", "justifyContent": "space-between"}),
    ], style={
        "backgroundColor": "white", "borderRadius": "8px", "padding": "15px",
        "boxShadow": "0 2px 4px rgba(0,0,0,0.05)"
    })
], style={
    "backgroundColor": "#f4f7fa", "fontFamily": "Segoe UI, Tahoma, Geneva, Verdana, sans-serif",
    "margin": "0", "padding": "10px"
})

# Callback to update charts and KPIs
@app.callback(
    [Output("peak-inquiry-hours", "figure"),
     Output("sales-vs-inquiries-scatter", "figure"),
     Output("sales-forecast", "figure"),
     Output("forecast-accuracy", "children"),
     Output("demographics-engagement", "figure"),
     Output("products-by-region", "figure"),
     Output("referral-sources", "figure"),
     Output("inquiries-by-country", "figure"),
     Output("salesperson-performance", "figure"),
     Output("individual-gauge", "figure"),
     Output("kpi-section", "children")],
    [Input("salesperson-filter", "value"),
     Input("continent-filter", "value"),
     Input("product-filter", "value"),
     Input("referral-filter", "value")]
)
def update_charts(salesperson, continent, product, referral):
    """Update all charts and KPIs based on filter selections."""
    filtered_df = df.copy()
    if salesperson != "All":
        filtered_df = filtered_df[filtered_df["Salesperson"] == salesperson]
    if continent != "All":
        filtered_df = filtered_df[filtered_df["Continent"] == continent]
    if product != "All":
        filtered_df = filtered_df[filtered_df["Product"] == product]
    if referral != "All":
        filtered_df = filtered_df[filtered_df["ReferralSource"] == referral]
    filtered_df["salesperson-filter"] = salesperson  # For individual gauge
    
    # Compute KPIs
    total_inquiries = int(filtered_df["Inquiries"].sum())
    total_sales = int(filtered_df["Sales"].sum())
    total_target = filtered_df["TargetInquiries"].sum()
    performance = (total_inquiries / total_target * 100) if total_target > 0 else 0
    perf_color = "red" if performance < 70 else "yellow" if performance < 90 else "green"
    num_salespersons = filtered_df["Salesperson"].nunique() if salesperson == "All" else 1
    avg_inquiries_per_salesperson = total_inquiries / num_salespersons if num_salespersons > 0 else 0

    # KPI Display
    kpi_section = [
        html.Div([
            html.H4("Total Inquiries", style={"fontSize": "14px", "color": "#333", "margin": "0"}),
            html.P(f"{total_inquiries}", style={"fontSize": "20px", "fontWeight": "bold", "color": "#1f77b4", "margin": "5px 0"})
        ], style={"textAlign": "center", "width": "22%", "padding": "10px", "borderRadius": "5px", "backgroundColor": "#f9f9f9"}),
        html.Div([
            html.H4("Total Sales", style={"fontSize": "14px", "color": "#333", "margin": "0"}),
            html.P(f"{total_sales}", style={"fontSize": "20px", "fontWeight": "bold", "color": "#1f77b4", "margin": "5px 0"})
        ], style={"textAlign": "center", "width": "22%", "padding": "10px", "borderRadius": "5px", "backgroundColor": "#f9f9f9"}),
        html.Div([
            html.H4("Performance vs Target", style={"fontSize": "14px", "color": "#333", "margin": "0"}),
            html.P(f"{int(performance)}%", style={"fontSize": "20px", "fontWeight": "bold", "color": perf_color, "margin": "5px 0"})
        ], style={"textAlign": "center", "width": "22%", "padding": "10px", "borderRadius": "5px", "backgroundColor": "#f9f9f9"}),
        html.Div([
            html.H4("Avg Inquiries per Salesperson", style={"fontSize": "14px", "color": "#333", "margin": "0"}),
            html.P(f"{int(avg_inquiries_per_salesperson)}", style={"fontSize": "20px", "fontWeight": "bold", "color": "#1f77b4", "margin": "5px 0"})
        ], style={"textAlign": "center", "width": "22%", "padding": "10px", "borderRadius": "5px", "backgroundColor": "#f9f9f9"}),
    ]

    # Update charts
    fig, mae, r2 = create_sales_forecast(filtered_df)
    r2_color = "green" if r2 >= 0.7 else "yellow" if r2 >= 0.5 else "red"
    mae_color = "green" if mae <= 1.5 else "yellow" if mae <= 3 else "red"
    accuracy_text = [
        html.H5("Model Accuracy Metrics", style={"fontWeight": "bold", "color": "#333"}),
        html.P([
            "MAE: ",
            html.Span(f"{mae:.2f}", style={"color": mae_color, "fontWeight": "bold"}),
            " (Average prediction error in sales units)"
        ]),
        html.P([
            "R²: ",
            html.Span(f"{r2:.2f}", style={"color": r2_color, "fontWeight": "bold"}),
            " (Proportion of variance explained, 0 to 1)"
        ])
    ]
    
    return (
        create_peak_inquiry_hours(filtered_df),
        create_sales_vs_inquiries_scatter(filtered_df),
        fig,
        accuracy_text,
        create_demographics_engagement(filtered_df),
        create_products_by_region(filtered_df),
        create_referral_sources(filtered_df),
        create_product_interest_by_country(filtered_df),
        create_salesperson_performance(filtered_df),
        create_individual_gauge(filtered_df),
        kpi_section
    )

if __name__ == '__main__':
    app.run(debug=True)