from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import base64
from io import BytesIO
from src.simulate import generate_linear_data, save_df, plot_data
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
    compute_bias_var,
)

slider_style = {'fontWeight': 'bold', 'textAlign':'center', 'display': 'block'}

def render_graph(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    return f'data:image/png;base64,{fig_data}'

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Bootstrap Dashboard", className='mb-2', style={'textAlign':'center'}),

    # Row 1
    dbc.Row([
        dbc.Col([
            html.Label('Bootstrap Models', style={'fontWeight': 'bold'}),
            dcc.RadioItems(
                id='bootstrap-models',
                options=[{'label':'Parametric Normal', 'value':'parametric_normal'},
                        {'label':'Pairs','value':'pairs'},
                        {'label':'Residuals','value':'residuals'},
                        {'label':'Wild', 'value':'wild'}
                ],
                value='parametric_normal',
                inline=True
            )
        ])
    ]),
    
    # Row 2: n point, beta0, beta1
    dbc.Row([
        dbc.Col([
            html.Label("Number of Data Points (n)", style=slider_style),
            dcc.Slider(min=0, max=200, step=1, value=50, marks=None,
                tooltip={"placement": "bottom", "always_visible": True},
                id='n-points'
            )
        ], width = 4),
        dbc.Col([
            html.Label("True Intercept (β₀)", style=slider_style),
            dcc.Slider(min=-5, max=5, step=.1, value=0, marks=None,
                tooltip={"placement": "bottom", "always_visible": True},
                id='beta-0'
            )
        ], width = 4),
        dbc.Col([
            html.Label("True Slope (β₁)", style=slider_style),
            dcc.Slider(min=-5, max=5, step=.1, value=0, marks=None,
                tooltip={"placement": "bottom", "always_visible": True},
                id='beta-1'
            )
        ], width = 4)
    ], className="mb-4"),

    # Row 3: sigma^2, hetero_strength, something else
    dbc.Row([
        dbc.Col([
            html.Label("Variance (σ²)", style=slider_style),
            dcc.Slider(min=0, max=10, step=1, value=2, marks=None,
                tooltip={"placement": "bottom", "always_visible": True},
                id='sigma-2'
            )
        ], width = 4),
        dbc.Col([
            html.Label("Heteroskedacity Stregth", style=slider_style),
            dcc.Slider(min=0, max=10, step=1, value=1, marks=None,
                tooltip={"placement": "bottom", "always_visible": True},
                id='hetero-strength'
            )
        ], width = 4),
        dbc.Col([
            html.Label("True Slope (β₁)", style=slider_style),
            dcc.Slider(min=-5, max=5, step=.1, value=0, marks=None,
                tooltip={"placement": "bottom", "always_visible": True},
                id='filler'
            )
        ], width = 4)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.Img(id='parametric-normal-bootstrap-scatter')
        ], width=6),
        dbc.Col([
            html.Img(id='parametric-normal-bootstrap-histogram')
        ], width=6)
    ])
])

@app.callback(
    Output('parametric-normal-bootstrap-scatter', 'src'),
    Output('parametric-normal-bootstrap-histogram', 'src'),
    Input('bootstrap-models', 'value'),
    Input('n-points', 'value'),
    Input('beta-0', 'value'),
    Input('beta-1', 'value'),
    Input('sigma-2', 'value'),
    Input('hetero-strength', 'value')
)
def test_callback(model, n, beta0, beta1, sigma2, hetero_strength):
    random_seed = 2025
    n_boot = 10000   # lower for quicker runtime

    # Simulate messy data (heteroskedastic + heavy-tailed)
    df, true_params = generate_linear_data(
        n=n,
        beta0=beta0,
        beta1=beta1,
        sigma=sigma2,
        heteroskedastic=True,
        hetero_strength=hetero_strength,
        heavy_tails=False,
        seed=random_seed,
    )

    X = df["x"].values
    y = df["y"].values

    boot_param = bootstrap_parametric_normal(X, y, n_boot=n_boot, seed=1)
    boot_param = boot_param.rename(columns={0: "x1"})  # nicer column names

    ols_fit = fit_ols(X, y)
    
    parametric_normal_bootstrap = render_graph(plot_bootstrap_lines(X, y, boot_param, n_lines=50))
    parametric_normal_bootstrap_hist = render_graph(plot_coef_histogram(boot_param, coef="x1", alpha=0.05, ols_res=ols_fit))

    return (
        parametric_normal_bootstrap,
        parametric_normal_bootstrap_hist
    )

if __name__ == '__main__':
    app.run(debug=True)
