import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.util import interval_lifecycle
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dask.dataframe as dd

# Function to process a single batch of traces
def process_batch(traces):
    data = []
    for trace in traces:
        trace_id = trace.attributes['concept:name']
        # Find the start timestamp of the trace
        start_timestamp = None
        for event in trace:
            if 'time:timestamp' in event:
                start_timestamp = event['time:timestamp']
                break
        if start_timestamp is None:
            continue

        # Calculate the relative time for each event
        for event in trace:
            event_name = event['concept:name']
            if 'time:timestamp' in event:
                timestamp = event['time:timestamp']
                relative_time = (timestamp - start_timestamp).total_seconds() / 3600.0  # relative time in hours
                data.append([trace_id, event_name, relative_time])
    return data

# Load the event log in batches
file_path = '/Users/korin.hoxha/Library/CloudStorage/OneDrive-RTLGroup/Korin Hoxha - Private/Personale/HU/SoSe 24/ProMi and VA/BPI Challenge 2017_1_all/BPI Challenge 2017.xes'
event_log = xes_importer.apply(file_path)

# Split the event log into batches
batch_size = 1000  # Adjust batch size as needed
batches = [event_log[i:i + batch_size] for i in range(0, len(event_log), batch_size)]

# Process each batch and collect the data
all_data = []
for batch in batches:
    batch_data = process_batch(batch)
    all_data.extend(batch_data)

# Convert the data into a pandas DataFrame
df = pd.DataFrame(all_data, columns=['trace_id', 'event_name', 'relative_time'])

# Save the DataFrame to a file for later use if needed
df.to_csv('processed_event_log.csv', index=False)

# Create a violin plot for the relative times of each event type
plt.figure(figsize=(15, 10))
sns.violinplot(x='event_name', y='relative_time', data=df)
plt.xticks(rotation=90)
plt.xlabel('Event Name')
plt.ylabel('Relative Time (hours)')
plt.title('Distribution of Event Types Relative to Start Event')
plt.show()

# Load the preprocessed data
df = pd.read_csv('processed_event_log.csv')

# Create a Dash application
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='violin-plot'),
    dcc.Dropdown(
        id='event-dropdown',
        options=[{'label': event, 'value': event} for event in df['event_name'].unique()],
        value=df['event_name'].unique()[0]
    ),
    dcc.RangeSlider(
        id='quartile-slider',
        min=0,
        max=100,
        step=1,
        value=[25, 75]
    )
])

@app.callback(
    Output('violin-plot', 'figure'),
    [Input('event-dropdown', 'value'),
     Input('quartile-slider', 'value')]
)
def update_violin(selected_event, selected_quartiles):
    filtered_df = df[df['event_name'] == selected_event]
    lower_quantile = filtered_df['relative_time'].quantile(selected_quartiles[0] / 100.0)
    upper_quantile = filtered_df['relative_time'].quantile(selected_quartiles[1] / 100.0)
    highlighted_df = filtered_df[(filtered_df['relative_time'] >= lower_quantile) &
                                 (filtered_df['relative_time'] <= upper_quantile)]

    fig = px.violin(filtered_df, x='event_name', y='relative_time', box=True, points="all",
                    hover_data=filtered_df.columns)
    fig.add_trace(px.scatter(highlighted_df, x='event_name', y='relative_time').data[0])

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
