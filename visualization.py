import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Load the processed data from the CSV file
file_path = 'processed_event_log.csv'

# Debug: Verify if the file exists
import os
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

# Debug: Load and check the data
try:
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully with {len(df)} records")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Debug: Ensure data types are correct
print("Data types before conversion:")
print(df.dtypes)

# Convert data types if necessary
df['relative_time'] = pd.to_numeric(df['relative_time'], errors='coerce')
df.dropna(subset=['relative_time'], inplace=True)

print("Data types after conversion:")
print(df.dtypes)

# Limit the data for initial rendering (e.g., 10,000 records)
sample_df = df.sample(n=10000, random_state=1)

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1('Interactive Violin Chart for Process Mining'),
    dcc.Dropdown(
        id='event-type-dropdown',
        options=[{'label': event, 'value': event} for event in df['event_name'].unique()],
        value=df['event_name'].unique().tolist(),
        multi=True
    ),
    dcc.Slider(
        id='quartile-slider',
        min=0,
        max=100,
        step=25,
        marks={0: '0%', 25: '25%', 50: '50%', 75: '75%', 100: '100%'},
        value=50
    ),
    dcc.Graph(id='violin-plot'),
    html.Div(id='quartile-output'),
    html.Button('Load More Data', id='load-more-data', n_clicks=0),
])

@app.callback(
    Output('violin-plot', 'figure'),
    [Input('event-type-dropdown', 'value'),
     Input('quartile-slider', 'value'),
     Input('load-more-data', 'n_clicks')]
)
def update_violin(selected_event_types, selected_quartile, n_clicks):
    filtered_df = sample_df if n_clicks == 0 else df

    if isinstance(selected_event_types, str):
        selected_event_types = [selected_event_types]

    filtered_df = filtered_df[filtered_df['event_name'].isin(selected_event_types)]

    if filtered_df.empty:
        return {}

    # Determine the threshold for the selected quartile
    thresholds = filtered_df.groupby('event_name')['relative_time'].quantile(selected_quartile / 100.0).to_dict()

    # Add a new column to highlight cases within the selected quartile
    filtered_df['highlight'] = filtered_df.apply(
        lambda row: 'Highlighted' if row['relative_time'] <= thresholds.get(row['event_name'], float('inf')) else 'Normal', axis=1
    )

    # Create the violin plot
    fig = px.violin(filtered_df, x='event_name', y='relative_time', color='highlight', box=True, points='all', violinmode='overlay')
    fig.update_layout(title='Distribution of Event Types Over Time', xaxis_title='Event Type', yaxis_title='Time Since Start (seconds)')

    return fig

@app.callback(
    Output('quartile-output', 'children'),
    [Input('event-type-dropdown', 'value'),
     Input('quartile-slider', 'value')]
)
def update_quartile_output(selected_event_types, selected_quartile):
    filtered_df = sample_df if n_clicks == 0 else df

    if isinstance(selected_event_types, str):
        selected_event_types = [selected_event_types]

    filtered_df = filtered_df[filtered_df['event_name'].isin(selected_event_types)]

    # Calculate quartiles
    quartiles = filtered_df.groupby('event_name')['relative_time'].quantile([0.25, 0.5, 0.75]).unstack(level=-1)
    quartile_output = html.Div([
        html.H4('Quartile Information'),
        html.Table([
            html.Tr([html.Th('Event Type'), html.Th('Q1'), html.Th('Median'), html.Th('Q3')])] +
            [html.Tr([html.Td(event), html.Td(quartiles.loc[event, 0.25]), html.Td(quartiles.loc[event, 0.5]), html.Td(quartiles.loc[event, 0.75])]) for event in quartiles.index]
        )
    ])

    return quartile_output

if __name__ == '__main__':
    app.run_server(debug=True, port=8059)
