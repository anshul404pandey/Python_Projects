# interactive_app.py

import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.graph_objs as go
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# -----------------------------
# APP SETUP
# -----------------------------
app = dash.Dash(__name__)
app.title = "Dynamic Portfolio Dashboard"

# Default settings
TICKERS = ['VOO', 'VTI', 'QQQ', 'AAPL', 'MSFT', 'TSLA']
DEFAULT_WEIGHTS = [0.25, 0.25, 0.20, 0.10, 0.10, 0.10]

# ✅ FIX: Remove time component using `.date()`
DEFAULT_START = (datetime.today() - timedelta(days=5 * 365)).date()
DEFAULT_END = datetime.today().date()

# -----------------------------
# LAYOUT
# -----------------------------
app.layout = html.Div([
    html.H1("📊 Interactive Portfolio Dashboard", style={'textAlign': 'center'}),

    html.Div([
        html.Label("📅 Select Date Range:"),
        dcc.DatePickerRange(
            id='date-range',
            start_date=DEFAULT_START,
            end_date=DEFAULT_END,
            display_format='YYYY-MM-DD',
        ),
    ], style={'marginBottom': '20px'}),

    html.Div([
        html.H3("⚖️ Adjust Asset Weights (must sum to 1.0)"),
        html.Div([
            html.Div([
                html.Label(f"{ticker}:"),
                dcc.Input(
                    id=f'weight-{ticker}',
                    type='number',
                    min=0,
                    max=1,
                    step=0.01,
                    value=DEFAULT_WEIGHTS[i],
                    debounce=True
                )
            ], style={'padding': '5px'}) for i, ticker in enumerate(TICKERS)
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px'})
    ]),

    html.Button("Update Dashboard", id='update-button', n_clicks=0, style={'marginTop': '20px'}),

    html.H2("📈 Portfolio Cumulative Return"),
    dcc.Graph(id='cumulative-returns'),

    html.H2("📋 Portfolio Metrics"),
    dash_table.DataTable(id='portfolio-metrics', style_cell={'textAlign': 'center'}),

    html.H2("📊 Individual Asset Metrics"),
    dash_table.DataTable(id='asset-metrics', style_cell={'textAlign': 'center'}),

    html.Footer("Data Source: Yahoo Finance", style={'textAlign': 'center', 'marginTop': 40})
])

# -----------------------------
# CALLBACKS
# -----------------------------
@app.callback(
    Output('cumulative-returns', 'figure'),
    Output('portfolio-metrics', 'data'),
    Output('portfolio-metrics', 'columns'),
    Output('asset-metrics', 'data'),
    Output('asset-metrics', 'columns'),
    Input('update-button', 'n_clicks'),
    State('date-range', 'start_date'),
    State('date-range', 'end_date'),
    *[State(f'weight-{ticker}', 'value') for ticker in TICKERS]
)
def update_dashboard(n_clicks, start_date, end_date, *weights):
    weights = np.array(weights, dtype=np.float64)
    if not np.isclose(np.sum(weights), 1.0):
        return go.Figure(), [{'Warning': 'Weights must sum to 1.0'}], [{'name': 'Warning', 'id': 'Warning'}], [], []

    # Download data
    raw_data = yf.download(
        TICKERS,
        start=start_date,
        end=end_date,
        group_by='ticker',
        auto_adjust=True,
        progress=False
    )

    price_data = raw_data.xs('Close', axis=1, level=1)

    # ✅ FIX: Avoid SettingWithCopyWarning
    price_data = price_data.dropna()
    daily_returns = price_data.pct_change().dropna()

    portfolio_returns = daily_returns.dot(weights)
    cumulative_portfolio = (1 + portfolio_returns).cumprod()

    def calculate_performance(returns):
        ann_return = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol
        total_ret = (1 + returns).prod() - 1
        return {
            'Total Return (%)': round(total_ret * 100, 2),
            'Annualized Return (%)': round(ann_return * 100, 2),
            'Annualized Volatility (%)': round(ann_vol * 100, 2),
            'Sharpe Ratio': round(sharpe, 2)
        }

    def max_drawdown(cumulative_returns):
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()

    # Portfolio metrics
    portfolio_metrics = calculate_performance(portfolio_returns)
    portfolio_df = pd.DataFrame(portfolio_metrics, index=['Portfolio'])

    # Asset metrics
    individual_metrics = []
    for ticker in daily_returns.columns:
        ret = daily_returns[ticker]
        cum_ret = (1 + ret).cumprod()
        metrics = calculate_performance(ret)
        metrics['Max Drawdown (%)'] = round(max_drawdown(cum_ret) * 100, 2)
        metrics['Asset'] = ticker
        individual_metrics.append(metrics)

    assets_df = pd.DataFrame(individual_metrics).set_index('Asset')

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative_portfolio.index,
        y=cumulative_portfolio,
        mode='lines',
        name='Portfolio'
    ))
    fig.update_layout(
        title="Cumulative Portfolio Return",
        xaxis_title="Date",
        yaxis_title="Cumulative Return"
    )

    return (
        fig,
        portfolio_df.reset_index().to_dict('records'),
        [{'name': col, 'id': col} for col in portfolio_df.reset_index().columns],
        assets_df.reset_index().to_dict('records'),
        [{'name': col, 'id': col} for col in assets_df.reset_index().columns],
    )

# -----------------------------
# RUN
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
