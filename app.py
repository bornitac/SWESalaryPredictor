import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('ds_salaries.csv')

exp_map = {'EN': 'Entry Level', 'MI': 'Mid Level', 'SE': 'Senior', 'EX': 'Executive'}
size_map = {'S': 'Small', 'M': 'Medium', 'L': 'Large'}
remote_map = {0: 'On-Site', 50: 'Hybrid', 100: 'Remote'}
emp_map = {'FT': 'Full-Time', 'PT': 'Part-Time', 'CT': 'Contract', 'FL': 'Freelance'}

df['experience_label'] = df['experience_level'].map(exp_map)
df['size_label'] = df['company_size'].map(size_map)
df['remote_label'] = df['remote_ratio'].map(remote_map)
df['employment_label'] = df['employment_type'].map(emp_map)

exp_score_map = {'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4}
size_score_map = {'S': 1, 'M': 2, 'L': 3}

with open('swe_salary_rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('swe_salary_lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)
with open('swe_salary_dt_model.pkl', 'rb') as f:
    dt_model = pickle.load(f)
with open('model_metrics.json', 'r') as f:
    metrics = json.load(f)

lr_r2 = metrics['lr']['r2']
lr_mae = metrics['lr']['mae']
lr_rmse = metrics['lr']['rmse']
dt_r2 = metrics['dt']['r2']
dt_mae = metrics['dt']['mae']
dt_rmse = metrics['dt']['rmse']
rf_r2 = metrics['rf']['r2']
rf_mae = metrics['rf']['mae']
rf_rmse = metrics['rf']['rmse']

job_titles = sorted(df['job_title'].unique())
locations = sorted(df['company_location'].unique())
exp_options = [{'label': v, 'value': k} for k, v in exp_map.items()]
size_options = [{'label': v, 'value': k} for k, v in size_map.items()]
remote_options = [{'label': v, 'value': k} for k, v in remote_map.items()]
emp_options = [{'label': v, 'value': k} for k, v in emp_map.items()]

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

BLUE = '#6495ED'
BG = '#eef2fb'
WHITE = '#ffffff'
DARK = '#1e293b'
GRAY = '#64748b'

card = {
    'backgroundColor': WHITE,
    'border': '1px solid #dde3f0',
    'padding': '20px',
    'marginBottom': '16px'
}

app.layout = html.Div(style={'backgroundColor': BG, 'minHeight': '100vh', 'fontFamily': 'Arial, sans-serif'}, children=[

    html.Div(style={'backgroundColor': BLUE, 'padding': '20px 32px', 'marginBottom': '24px'}, children=[
        html.H1('Tech & Data Science Salary Predictor',
                style={'color': 'white', 'margin': 0, 'fontSize': '22px'}),
        html.P(f'Trained on {len(df):,} records | Random Forest | R\u00b2 = {rf_r2:.2f}',
               style={'color': 'rgba(255,255,255,0.8)', 'margin': '4px 0 0', 'fontSize': '13px'})
    ]),

    html.Div(style={'maxWidth': '1100px', 'margin': '0 auto', 'padding': '0 20px 40px'}, children=[

        dcc.Tabs(id='tabs', value='predict', style={'marginBottom': '20px'}, children=[
            dcc.Tab(label='Salary Predictor', value='predict'),
            dcc.Tab(label='Visualizations', value='viz'),
            dcc.Tab(label='Model Performance', value='models'),
        ]),

        html.Div(id='tab-content')
    ]),

    html.Div(style={'backgroundColor': DARK, 'padding': '14px', 'textAlign': 'center', 'marginTop': '40px'}, children=[
        html.P('Built by Bornita Chowdhury | Data source: Kaggle (AI/ML/Data Science Salary Dataset)',
               style={'color': 'rgba(255,255,255,0.7)', 'margin': 0, 'fontSize': '12px'})
    ])
])

predict_layout = html.Div([
    html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '16px'}, children=[

        html.Div(style=card, children=[
            html.H3('Enter Your Profile', style={'marginTop': 0, 'color': DARK}),

            html.Label('Job Title', style={'color': DARK}),
            dcc.Dropdown(id='job-title', options=[{'label': j, 'value': j} for j in job_titles],
                         value='Data Analyst', clearable=False, style={'marginBottom': '14px'}),

            html.Label('Experience Level', style={'color': DARK}),
            dcc.Dropdown(id='exp-level', options=exp_options, value='MI',
                         clearable=False, style={'marginBottom': '14px'}),

            html.Label('Employment Type', style={'color': DARK}),
            dcc.Dropdown(id='emp-type', options=emp_options, value='FT',
                         clearable=False, style={'marginBottom': '14px'}),

            html.Label('Company Size', style={'color': DARK}),
            dcc.Dropdown(id='company-size', options=size_options, value='M',
                         clearable=False, style={'marginBottom': '14px'}),

            html.Label('Work Setting', style={'color': DARK}),
            dcc.Dropdown(id='remote-ratio', options=remote_options, value=100,
                         clearable=False, style={'marginBottom': '14px'}),

            html.Label('Company Location', style={'color': DARK}),
            dcc.Dropdown(id='company-loc', options=[{'label': l, 'value': l} for l in locations],
                         value='US', clearable=False, style={'marginBottom': '14px'}),

            html.Label('Work Year', style={'color': DARK}),
            dcc.Slider(id='work-year', min=2020, max=2023, step=1, value=2023,
                       marks={y: str(y) for y in range(2020, 2024)},
                       tooltip={'placement': 'bottom'}),

            html.Br(),
            html.Button('Predict Salary', id='predict-btn', n_clicks=0,
                        style={'backgroundColor': BLUE, 'color': 'white', 'border': 'none',
                               'padding': '10px 24px', 'fontSize': '14px',
                               'cursor': 'pointer', 'width': '100%', 'marginTop': '10px'})
        ]),

        html.Div(style=card, children=[
            html.H3('Prediction Result', style={'marginTop': 0, 'color': DARK}),
            html.Div(id='prediction-output', style={'textAlign': 'center', 'padding': '28px 0'}),
            html.Hr(),
            html.H4('How does this compare to the dataset?', style={'color': DARK}),
            dcc.Graph(id='comparison-chart', config={'displayModeBar': False})
        ])
    ])
])

viz_layout = html.Div([
    html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '16px'}, children=[

        html.Div(style=card, children=[
            html.H4('Median Salary by Experience Level', style={'marginTop': 0, 'color': DARK}),
            dcc.Graph(id='exp-chart', config={'displayModeBar': False})
        ]),

        html.Div(style=card, children=[
            html.H4('Top 10 Highest Paying Roles', style={'marginTop': 0, 'color': DARK}),
            dcc.Graph(id='job-chart', config={'displayModeBar': False})
        ]),

        html.Div(style=card, children=[
            html.H4('Salary Growth Over Time', style={'marginTop': 0, 'color': DARK}),
            dcc.Graph(id='year-chart', config={'displayModeBar': False})
        ]),

        html.Div(style=card, children=[
            html.H4('Remote vs Hybrid vs On-Site', style={'marginTop': 0, 'color': DARK}),
            dcc.Graph(id='remote-chart', config={'displayModeBar': False})
        ]),

        html.Div(style={**card, 'gridColumn': 'span 2'}, children=[
            html.H4('Salary by Company Size', style={'marginTop': 0, 'color': DARK}),
            dcc.Graph(id='size-chart', config={'displayModeBar': False})
        ]),
    ])
])

models_layout = html.Div([
    html.P('Three models were trained and compared. Random Forest performed best overall.',
           style={'color': GRAY, 'marginBottom': '16px'}),

    html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr', 'gap': '16px',
                    'marginBottom': '16px'}, children=[

        html.Div(style=card, children=[
            html.H4('Linear Regression', style={'color': GRAY, 'marginTop': 0}),
            html.H2(f'{lr_r2:.2%}', style={'color': DARK, 'margin': '6px 0'}),
            html.P('R\u00b2 Score', style={'color': GRAY, 'margin': 0}),
            html.Hr(),
            html.P(f'MAE: ${lr_mae:,.0f}', style={'color': GRAY, 'fontSize': '13px'}),
            html.P(f'RMSE: ${lr_rmse:,.0f}', style={'color': GRAY, 'fontSize': '13px'})
        ]),

        html.Div(style=card, children=[
            html.H4('Decision Tree', style={'color': GRAY, 'marginTop': 0}),
            html.H2(f'{dt_r2:.2%}', style={'color': DARK, 'margin': '6px 0'}),
            html.P('R\u00b2 Score', style={'color': GRAY, 'margin': 0}),
            html.Hr(),
            html.P(f'MAE: ${dt_mae:,.0f}', style={'color': GRAY, 'fontSize': '13px'}),
            html.P(f'RMSE: ${dt_rmse:,.0f}', style={'color': GRAY, 'fontSize': '13px'})
        ]),

        html.Div(style={**card, 'border': f'2px solid {BLUE}'}, children=[
            html.H4('Random Forest', style={'color': BLUE, 'marginTop': 0}),
            html.H2(f'{rf_r2:.2%}', style={'color': BLUE, 'margin': '6px 0'}),
            html.P('R\u00b2 Score', style={'color': GRAY, 'margin': 0}),
            html.Hr(),
            html.P(f'MAE: ${rf_mae:,.0f}', style={'color': GRAY, 'fontSize': '13px'}),
            html.P(f'RMSE: ${rf_rmse:,.0f}', style={'color': GRAY, 'fontSize': '13px'})
        ]),
    ]),

    html.Div(style=card, children=[
        html.H4('R\u00b2 Score Comparison', style={'marginTop': 0, 'color': DARK}),
        dcc.Graph(id='model-comparison-chart', config={'displayModeBar': False})
    ])
])


@app.callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render_tab(tab):
    if tab == 'predict':
        return predict_layout
    elif tab == 'viz':
        return viz_layout
    return models_layout


@app.callback(
    Output('prediction-output', 'children'),
    Output('comparison-chart', 'figure'),
    Input('predict-btn', 'n_clicks'),
    State('job-title', 'value'),
    State('exp-level', 'value'),
    State('emp-type', 'value'),
    State('company-size', 'value'),
    State('remote-ratio', 'value'),
    State('company-loc', 'value'),
    State('work-year', 'value'),
    prevent_initial_call=False
)
def predict_salary(n_clicks, job_title, exp_level, emp_type, company_size, remote_ratio, company_loc, work_year):
    input_df = pd.DataFrame([{
        'experience_level': exp_level,
        'employment_type': emp_type,
        'job_title': job_title,
        'company_location': company_loc,
        'company_size': company_size,
        'remote_ratio': remote_ratio,
        'work_year': work_year,
        'experience_score': exp_score_map.get(exp_level, 2),
        'is_remote': 1 if remote_ratio == 100 else 0,
        'size_score': size_score_map.get(company_size, 2),
        'is_us': 1 if company_loc == 'US' else 0
    }])

    predicted = rf_model.predict(input_df)[0]

    avg_by_exp = df.groupby('experience_label')['salary_in_usd'].median().reset_index()
    avg_by_exp.columns = ['Experience', 'Median Salary']

    fig = px.bar(avg_by_exp, x='Experience', y='Median Salary',
                 color='Experience', color_discrete_sequence=px.colors.qualitative.Set2)
    fig.add_hline(y=predicted, line_dash='dash', line_color=BLUE,
                  annotation_text=f'Predicted: ${predicted:,.0f}',
                  annotation_position='top right')
    fig.update_layout(showlegend=False, plot_bgcolor='white',
                      paper_bgcolor='white', margin=dict(t=20, b=20))

    result = html.Div([
        html.P('Estimated Annual Salary (USD)', style={'color': GRAY, 'fontSize': '13px', 'margin': 0}),
        html.H1(f'${predicted:,.0f}', style={'color': BLUE, 'fontSize': '46px', 'margin': '8px 0'}),
        html.P(f'{exp_map.get(exp_level, "")} — {job_title}', style={'color': GRAY, 'fontSize': '13px'}),
        html.Br(),
        html.P(f'Dataset median: ${df["salary_in_usd"].median():,.0f}',
               style={'color': GRAY, 'fontSize': '12px', 'margin': 0}),
        html.P(f'Model R\u00b2: {rf_r2:.2%}', style={'color': GRAY, 'fontSize': '12px', 'margin': 0}),
    ])
    return result, fig


@app.callback(
    Output('exp-chart', 'figure'),
    Output('job-chart', 'figure'),
    Output('year-chart', 'figure'),
    Output('remote-chart', 'figure'),
    Output('size-chart', 'figure'),
    Input('tabs', 'value')
)
def update_viz(tab):
    exp_df = df.groupby('experience_label')['salary_in_usd'].median().reset_index()
    exp_order = ['Entry Level', 'Mid Level', 'Senior', 'Executive']
    exp_df['experience_label'] = pd.Categorical(exp_df['experience_label'], categories=exp_order, ordered=True)
    exp_df = exp_df.sort_values('experience_label')
    fig_exp = px.bar(exp_df, x='experience_label', y='salary_in_usd',
                     color='experience_label', color_discrete_sequence=px.colors.qualitative.Set2,
                     labels={'experience_label': 'Experience', 'salary_in_usd': 'Median Salary (USD)'})
    fig_exp.update_layout(showlegend=False, plot_bgcolor='white', paper_bgcolor='white', margin=dict(t=10, b=10))

    top_jobs = df.groupby('job_title')['salary_in_usd'].median().nlargest(10).reset_index()
    fig_job = px.bar(top_jobs, x='salary_in_usd', y='job_title', orientation='h',
                     color='salary_in_usd', color_continuous_scale='Blues',
                     labels={'salary_in_usd': 'Median Salary (USD)', 'job_title': ''})
    fig_job.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                          coloraxis_showscale=False, margin=dict(t=10, b=10))
    fig_job.update_yaxes(autorange='reversed')

    year_df = df.groupby('work_year')['salary_in_usd'].median().reset_index()
    fig_year = px.line(year_df, x='work_year', y='salary_in_usd', markers=True,
                       labels={'work_year': 'Year', 'salary_in_usd': 'Median Salary (USD)'},
                       color_discrete_sequence=[BLUE])
    fig_year.update_layout(plot_bgcolor='white', paper_bgcolor='white', margin=dict(t=10, b=10))

    remote_df = df.groupby('remote_label')['salary_in_usd'].median().reset_index()
    fig_remote = px.bar(remote_df, x='remote_label', y='salary_in_usd',
                        color='remote_label', color_discrete_sequence=px.colors.qualitative.Pastel,
                        labels={'remote_label': 'Work Setting', 'salary_in_usd': 'Median Salary (USD)'})
    fig_remote.update_layout(showlegend=False, plot_bgcolor='white', paper_bgcolor='white', margin=dict(t=10, b=10))

    size_df = df.groupby('size_label')['salary_in_usd'].median().reset_index()
    fig_size = px.bar(size_df, x='size_label', y='salary_in_usd',
                      color='size_label', color_discrete_sequence=px.colors.qualitative.Set3,
                      labels={'size_label': 'Company Size', 'salary_in_usd': 'Median Salary (USD)'})
    fig_size.update_layout(showlegend=False, plot_bgcolor='white', paper_bgcolor='white', margin=dict(t=10, b=10))

    return fig_exp, fig_job, fig_year, fig_remote, fig_size


@app.callback(
    Output('model-comparison-chart', 'figure'),
    Input('tabs', 'value')
)
def update_model_chart(tab):
    model_df = pd.DataFrame({
        'Model': ['Linear Regression', 'Decision Tree', 'Random Forest'],
        'R2': [lr_r2, dt_r2, rf_r2]
    })
    fig = px.bar(model_df, x='Model', y='R2',
                 color='Model',
                 color_discrete_sequence=['#94a3b8', '#64748b', BLUE],
                 text=model_df['R2'].apply(lambda x: f'{x:.2%}'))
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, plot_bgcolor='white',
                      paper_bgcolor='white', yaxis_range=[0, 1],
                      yaxis_title='R\u00b2 Score',
                      margin=dict(t=20, b=20))
    return fig


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)