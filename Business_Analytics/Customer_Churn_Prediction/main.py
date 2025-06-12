import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import io
import base64
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import plotly.graph_objs as go

# ---------- Helper Functions ----------

def preprocess_data(df):
    try:
        df = df.copy()
        if 'customerID' in df.columns:
            df.drop('customerID', axis=1, inplace=True)

        df['TotalCharges'] = pd.to_numeric(df.get('TotalCharges'), errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

        if 'Churn' not in df.columns:
            raise ValueError("Missing required column: 'Churn'")

        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}) if df['Churn'].dtype == 'object' else df['Churn']

        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].nunique() == 2:
                df[col] = LabelEncoder().fit_transform(df[col])
            else:
                df = pd.get_dummies(df, columns=[col], drop_first=True)

        return df

    except Exception as e:
        raise ValueError(f"Error in preprocessing: {str(e)}")

def train_model(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    return roc_auc, cm, report, importance

# ---------- Dash App Setup ----------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Customer Churn Dashboard"

app.layout = dbc.Container([
    html.H2("Customer Churn Prediction Dashboard", className='text-center my-4'),

    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    '📤 Drag and Drop or ',
                    html.A('Select a CSV File')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'marginBottom': '10px'
                },
                multiple=False
            ),
            dbc.Button("Train Model", id='train-button', color='primary', className='mb-2', n_clicks=0),
            html.Div(id='error-message', style={'color': 'red'}),
        ], width=4),
        dbc.Col([
            html.Div(id='metrics-output')
        ], width=8),
    ]),

    dcc.Graph(id='churn-pie'),
    dcc.Graph(id='feature-importance'),

    dcc.Store(id='stored-data')
], fluid=True)

# ---------- Callbacks ----------

@app.callback(
    Output('stored-data', 'data'),
    Output('error-message', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def parse_upload(contents, filename):
    if contents is None:
        # Use default dataset
        try:
            df = pd.read_csv('https://raw.githubusercontent.com/blastchar/telco-churn/master/Telco-Customer-Churn.csv')
            #df = pd.read_csv('https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv')

            df = preprocess_data(df)
            return df.to_json(date_format='iso', orient='split'), ""
        except Exception as e:
            return None, f"Failed to load default data: {e}"

    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        df = preprocess_data(df)
        return df.to_json(date_format='iso', orient='split'), ""
    except Exception as e:
        return None, f"Error uploading file: {str(e)}"

@app.callback(
    [Output('metrics-output', 'children'),
     Output('churn-pie', 'figure'),
     Output('feature-importance', 'figure')],
    Input('train-button', 'n_clicks'),
    State('stored-data', 'data')
)
def update_dashboard(n_clicks, data):
    if not n_clicks or data is None:
        return "", go.Figure(), go.Figure()

    df = pd.read_json(data, orient='split')
    try:
        roc_auc, cm, report, importance = train_model(df)

        # Churn pie chart
        pie_fig = px.pie(
            df, names='Churn', title="Churn Distribution",
            color='Churn', color_discrete_map={0: 'green', 1: 'red'}
        )

        # Feature importance
        feat_fig = px.bar(
            importance.head(15),
            x=importance.head(15).values,
            y=importance.head(15).index,
            orientation='h',
            title='Top 15 Features'
        )

        # Metrics
        metrics = f"""
        **Model Performance**
        - Precision (Churn): {report['1']['precision']:.2f}
        - Recall (Churn): {report['1']['recall']:.2f}
        - F1 Score (Churn): {report['1']['f1-score']:.2f}
        - ROC AUC Score: {roc_auc:.2f}
        - Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}
        """
        metrics_card = dbc.Card([dbc.CardBody([dcc.Markdown(metrics)])], color="light", className="mb-4")

        return metrics_card, pie_fig, feat_fig
    except Exception as e:
        return f"Error in training model: {e}", go.Figure(), go.Figure()

# ---------- Run App ----------
if __name__ == "__main__":
    app.run(debug=True)

