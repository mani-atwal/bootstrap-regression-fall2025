from dash import Dash, html, dcc, Input, Output, State, dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.express as px
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from src.simulate import generate_linear_data
from src.bootstrap_methods import (
    fit_ols,
    bootstrap_parametric_normal,
    bootstrap_pairs,
    bootstrap_residuals,
    bootstrap_wild,
    bootstrap_summary,
)
from src.evaluate import (
    plot_bootstrap_lines,
    plot_coef_histogram,
)

def render_graph(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    return f'data:image/png;base64,{fig_data}'

def create_slider(label, id, min_val=-5, max_val=5, step=0.1, value=0):
    return html.Div([
        html.Label(label, style={'fontWeight': 'bold', 'textAlign':'center', 'display': 'block'}),
        dcc.Slider(
            min=min_val,
            max=max_val,
            step=step,
            value=value,
            marks=None,
            tooltip={"placement": "bottom", "always_visible": True},
            id=id
        )
    ])

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([

    html.H1("Bootstrap Dashboard", className='mb-2',
        style={
            'textAlign': 'center',
            'paddingTop': '16px',
            'paddingBottom': '16px'
        }
    ),

    dbc.Row([
        dbc.Col([
            dbc.Card(
                [
                    dbc.CardHeader(html.Label("Bootstrap Model", style={'fontWeight': 'bold', 'textAlign':'center', 'display': 'block'})),
                    dbc.CardBody(
                        dcc.RadioItems(
                            id='bootstrap-models',
                            options=[
                                {'label':'Parametric Normal', 'value':'parametric_normal'},
                                {'label':'Pairs','value':'pairs'},
                                {'label':'Residuals','value':'residuals'},
                                {'label':'Wild', 'value':'wild'}
                            ],
                            value='parametric_normal',
                            inline=False,
                            labelStyle={'display': 'block', 'margin': '8px 0', 'fontSize': '18px'},
                            inputStyle={'marginRight': '12px', 'width': '18px', 'height': '18px'}
                        )
                    )
                ],
                style={'boxShadow': '0 2px 6px rgba(0,0,0,0.1)', 'borderRadius': '10px', 'padding': '10px'}
            )
        ], width=3),
        dbc.Col([
            dbc.Row([
                dbc.Col([create_slider("True Intercept (β₀)", 'beta-0', -5, 5, 0.1, 0)], width = 4),
                dbc.Col([create_slider("True Slope (β₁)", 'beta-1', -5, 5, 0.1, 1)], width = 4),
                dbc.Col([create_slider("True Standard Deviation (σ)", 'sigma', 0.1, 5, 0.1, 2)], width = 4)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([create_slider("Number of Data Points (n)", 'n-points', 0, 200, 1, 50)], width = 4),
                dbc.Col([create_slider("Heteroskedasticity Stregth", 'hetero-strength', 0, 5, 0.1, 1)], width = 4),
                dbc.Col([create_slider("Number of Bootstap Models", 'n-boot', 10, 10000, 1, 2000)], width = 4)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    html.Label("Heteroskedastic?", style={'fontWeight': 'bold', 'textAlign':'center', 'display': 'block'}),
                    daq.BooleanSwitch(
                        on=True,
                        id='hetero'
                    )
                ], width = 6),
                dbc.Col([
                    html.Label("Fat Tails?", style={'fontWeight': 'bold', 'textAlign':'center', 'display': 'block'}),
                    daq.BooleanSwitch(
                        on=False,
                        id='fat-tails'
                    )
                ], width = 6)
            ])
        ], className='mb-4'),
        dbc.Row([
            dbc.Col([
                html.Button("Run Simulation", id="run-btn", n_clicks=0, className="btn btn-primary")
            ], width=12, style={'textAlign': 'center', 'display':'block'})
        ])
    ], className='mb-4'),


    dbc.Row([
        dcc.Loading(
            id='loading-sim',
            type="circle",
            color="#0d6efd",
            children=dbc.Row([
                dbc.Col([
                    html.Img(id='parametric-normal-bootstrap-scatter', style={'width': '100%'})
                ], width=6),
                dbc.Col([
                    html.Img(id='parametric-normal-bootstrap-histogram', style={'width': '100%'})
                ], width=6)
            ])
        )
    ]),

    dbc.Row([
        dcc.Loading(
            id='loading-table',
            type="circle",
            color="#0d6efd",
            children=html.Div(
                dash_table.DataTable(
                    columns=[
                        {"name": i, "id": i} for i in ["Coefficient", "Mean", "Std. Dev.", "2.5%", "97.5%"]
                    ],
                    data=[],  # will be filled by callback
                    style_table={'overflowX': 'auto', 'width': '820px', 'margin': '0 auto'},
                    style_cell={'textAlign': 'center', 'padding': '5px', 'width': '160px', 'minWidth': '160px', 'maxWidth': '160px'},
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    },
                    id='bootstrap-summary-table'
                ),
                id='table-container',
                style={'display': 'none'}
            )
        )
    ], className='mb-4')

])

@app.callback(
    Output('parametric-normal-bootstrap-scatter', 'src'),
    Output('parametric-normal-bootstrap-histogram', 'src'),
    Output('bootstrap-summary-table', 'data'),
    Output('table-container', 'style'),
    Input('run-btn', 'n_clicks'),
    State('bootstrap-models', 'value'),
    State('beta-0', 'value'),
    State('beta-1', 'value'),
    State('sigma', 'value'),
    State('n-points', 'value'),
    State('hetero-strength', 'value'),
    State('n-boot', 'value'),
    State('hetero', 'on'),
    State('fat-tails', 'on')
)
def run_simulation(n_clicks, model, beta0, beta1, sigma, n, hetero_strength, n_boot, hetero, fat_tails):
    if n_clicks == 0:
        raise PreventUpdate

    random_seed = np.random.default_rng()

    # Simulate data
    df, true_params = generate_linear_data(
        n=n,
        beta0=beta0,
        beta1=beta1,
        sigma=sigma,
        heteroskedastic=hetero,
        hetero_strength=hetero_strength,
        heavy_tails=fat_tails,
        seed=random_seed,
    )

    X = df["x"].values
    y = df["y"].values

    model_dict = {
        'parametric_normal':bootstrap_parametric_normal,
        'pairs':bootstrap_pairs,
        'residuals':bootstrap_residuals,
        'wild':bootstrap_wild
    }

    boot_param = model_dict[model](X, y, n_boot=n_boot, seed=1).rename(columns={'const': 'β̂₀', 0: "β̂₁"})
    ols_fit = fit_ols(X, y)

    graph = render_graph(plot_bootstrap_lines(X, y, boot_param, n_lines=50))
    hist = render_graph(plot_coef_histogram(boot_param, coef="β̂₁", alpha=0.05, ols_res=ols_fit))

    summary_df = bootstrap_summary(boot_param)
    summary_df = summary_df.reset_index().rename(columns={'index':'Coefficient','mean':'Mean','std':'Std. Dev.'})
    summary_df[summary_df.columns[1:]] = summary_df[summary_df.columns[1:]].map(lambda x: round(x, 4))

    table_data = summary_df.to_dict('records')

    table_style = {'display': 'block'} if table_data else {'display': 'none'}

    return graph, hist, table_data, table_style

if __name__ == '__main__':
    app.run(debug=True)
