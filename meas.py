import os
import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.dependencies import ALL
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import dash_player as dp

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server= app.server

dir_path = ".\human"
directory_names = [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]

dropdown_items = []
for name in directory_names:
    image_filename = f"{name}.jpg"
    image_path = f"/assets/{image_filename}"
    image = html.Img(src=image_path, height="30px", style={"margin-right": "10px"})
    item = dbc.DropdownMenuItem(children=[image, name], id=name)
    dropdown_items.append(item)

dropdown = dbc.DropdownMenu(
    id="dropdown",
    label="People",
    menu_variant="dark",
    children=dropdown_items,
    style={"margin-top": "-38px"}
)

def split_dataframe(df, ranges_str):
    ranges = ranges_str.split(',')
    dataframes = []
    for r in ranges:
        start, end = map(float, r.split('-'))
        df_range = df[(df.index >= start) & (df.index <= end)]
        dataframes.append(df_range)
    return dataframes

def create_accordion(selected_person):
    if selected_person is None:
        return dbc.Card(
            [
                dbc.CardImg(src="/assets/logo.jpg", top=True),
                dbc.CardBody(
                    [
                        html.H1("Welcome!", className="display-3"),
                        html.P(
                            "Please choose a character from the top right menu to start analysis...",
                            className="lead",
                        ),
                    ]
                ),
            ],
            className="text-center",
        )

    accordion_items = []
    csv_dir = f'{dir_path}//{selected_person}/data'
    csv_files = {}
    for f in os.listdir(csv_dir):
        if f.endswith('.csv') and f not in ['fades.csv', 'text.csv']:
            file_key = f[:-4].replace('_', ' ')
            csv_files[file_key] = pd.read_csv(os.path.join(csv_dir, f), nrows=0).columns.tolist()

    for filename, cols in csv_files.items():
        checkboxes = []
        for col in cols:
            checkbox_id = f"{filename.replace('.', '_')}-{col}-checkbox"
            checkboxes.append(
                dbc.Checklist(
                    options=[{'label': col, 'value': f"{filename}:{col}"} for col in [col]],
                    value=[],
                    id={'type': 'dynamic-checklist', 'index': checkbox_id},
                    inputClassName="checkbox-input",
                    style={"font-size": "12px"}
                )

            )
        accordion_item = dbc.AccordionItem(
            [html.Div(checkboxes)],
            title=filename,
            id=f"{filename.replace('.', '_')}-accordion-item"
        )
        accordion_items.append(accordion_item)
    accordion = dbc.Accordion(accordion_items)
    return accordion

update_button = dbc.Button("Update Graph", id="update-button", color="primary", className="mt-3")
clear_button =  dbc.Button('Clear', id='clear-button', color="primary", className="mt-3", n_clicks=0)
toggle_button = dbc.Button("Frame and notes", id="toggle-button", className="mt-3", active=False)

# noinspection PyInterpreter
app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("ΜΕΑΣ - Multimodal Emotion Analysis Sandbox",
                        style={'color': 'white', 'margin': '20px 0 0 0', 'text-align': 'right', 'padding': '0 60px 0 0',
                               'font-size': '34px'}),
                dbc.Row([
                    dbc.Col(dropdown, width=2),  # adjust width as necessary
                    dbc.Col(html.Div([
                        # scroll to movie
                        html.A(dbc.Button('Scroll to video', color='success', id='video-btn'),
                               href='#video-player-large'),
                    ], style={'margin-top': '-37px', 'margin-left': '120px'}), width=1),  # adjust width as necessary
                    dbc.Col(html.Div([
                        # switch
                        dbc.Checklist(
                            options=[
                                {"label": "", "value": 1},
                            ],
                            id="checklist-switch",
                            switch=True,
                        ),
                    ], style={'margin-top': '-26px',  'margin-left': '1090px',  'position':'absolute', 'z-index': '1900'}), width=2)  # adjust width as necessary
                ]),

                dcc.Store(id='selected-value', data=None),
                dcc.Store(id='toggle-value', data=False),  # Store for the toggle status
            ], width=12, style={"background-color": "#333", "padding": "5px 0", "height": "80px"}),
        ]),
        dbc.Row([
            dbc.Col([html.Div(id="accordion-container", children=create_accordion(selected_person=None)),
                     update_button,
                     clear_button],
                    width=2, style={"background-color": "white", "padding": "5px"}),

            dbc.Col([
                # photo
                html.Div([
                    html.Div(id='person-image',
                             style={'font-size': '24px', 'color': '#ggg', 'margin-top': '90px', 'margin-left': '115px'}),
                ], className='name-and-image-container', style={'position': 'absolute', 'z-index': '2', 'top': '0'}),

                # name
                html.Div([
                    html.Div(id='person-name', style={'color': '#ddd','margin-top': '20px'}),
                ], className='name-and-image-container', style={'position': 'absolute', 'z-index': '3', 'top': '0'}),


                # graph & video
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='line-graph', style={'position': 'relative', 'height': '500px'}),

                        # Normalize
                        dbc.Checklist(
                            options=[
                                {"label": "Normalize data", "value": "normalize"},
                            ],
                            value=[],
                            id="normalize-check",
                        ),

                        html.P('Smoothing :', style={
                            'font-size': '15',
                            'font-weight': '600',
                            'line-height': '38px',
                        }),

                        # smoothing slider
                        dcc.Slider(
                            id='window-slider',
                            min=1,
                            max=500,
                            step=1,
                            value=10,
                            marks={i: str(i) for i in range(0, 500, 50)}
                        ),
                    ], id='graph-column', width=8),
                    dbc.Col([
                        html.Div([
                            html.Div(id='video-small', children=[
                                dp.DashPlayer(
                                    id='video-player-small',
                                    url='',
                                    controls=False,
                                    width='100%',
                                    height='385px'
                                )
                            ], style={'display': 'flex', 'align-items': 'flex-start', 'justify-content': 'center',
                                      'height': '390px', 'overflow': 'auto', 'margin-top': '-60px'}),
                        ]),
                        html.Div([
                            # Adjust margin-top here
                            html.P('Notes :', style={
                                'font-size': '15',
                                'font-weight': '600',
                                'line-height': '38px',
                                'margin-top': '-5px'
                            }),
                            dcc.Textarea(
                                id='textarea',
                                placeholder='Enter your text here...',
                                style={'width': '100%', 'height': 'auto', 'overflow': 'auto', 'margin-top': '0px'}
                            ),
                            html.Div([
                                dbc.Button(" Modalities ", color='warning', id="modalities-button", size="sm",  className="me-3", style={"margin-right": "10px"}),
                                dbc.Button("  Time  ",  color='warning', id="time-button", size="sm",  className="me-3", style={"margin-right": "10px"}),
                                dbc.Button("  Save Notes  ",  color='success', id="save-note",  className="me-3"),
                            ], style={'justify-content': 'right', 'margin-top': '10px'}),

                            # CNN
                            html.P('Compute a model :', style={
                                'font-size': '15',
                                'font-weight': '600',
                                'line-height': '38px',
                            }),
                            dcc.Input(id='etiquette', type='text', placeholder='Name (eg. anger)', style={'width': '70%','margin-top': '5px'}),
                            dbc.Button("Compute",  color='success', id="save-button",  className="me-3",style={"margin-left": "10px"}),
                            dcc.Input(id='ranges-input', type='text', placeholder='Enter ranges separated by commas', style={'width': '100%','margin-top': '10px'}),

                        ], style={'height': '100%', 'overflow': 'auto', 'margin-top': '-50px'}),
                    ], id='right-column', width=4),
                ]),

            ], className='lead', width=10),


            ], justify="start", align="start"),
    ], fluid=True),
    dbc.Row([
        dbc.Col([
            dbc.Col(html.Div(id='video-large', children=[
                dp.DashPlayer(
                    id='video-player-large',
                    url='',
                    controls=True,
                    width='100%',
                    height='1080px'
                )
            ]), style={"background-color": "#333", "padding": "5px 0px", "height": "1400px"}),
            html.Div(id="checked-info", style={"padding": "20px", "margin-top": "20px", "visibility": "hidden"}),
        ], style={"background-color": "#333", "padding": "5px 0px", "height": "1400px"}),
    ])
])




@app.callback(
    [
        Output("selected-value", "data"),
        Output("accordion-container", "children"),
        Output("person-image", "children"),
        Output("person-name", "children"),
        Output('video-player-small', 'url'),  # Update 'video-player-small' url here
        Output('video-player-large', 'url'),  # Update 'video-player-large' url here
    ],
    [Input(name, "n_clicks") for name in directory_names],
    [State("selected-value", "data")],
    prevent_initial_call=True,
)
def combined_callback(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return None, create_accordion(selected_person=None), None, None, None
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if not trigger_id:
        return None, create_accordion(selected_person=None), None, None, None
    selected_value = trigger_id

    accordion = create_accordion(selected_person=selected_value)

    image_filename = f"{selected_value}.jpg"
    image_path = f"/assets/{image_filename}"
    image = html.Img(src=image_path, style={"margin-right": "10px", "border": "2px solid white",'border-radius': '7px'})

    video_filename = f"{selected_value}.mp4"
    video_path = f"assets/videos/{video_filename}"

    return selected_value, accordion, selected_value, image, video_path, video_path  # return video_path

@app.callback(
    Output('accordion-container', 'children', allow_duplicate=True),
    Input("clear-button", "n_clicks"),
    State("selected-value", "data"),
    prevent_initial_call=True
)
def clear_output(n_clicks_clear, selected_person):
    if n_clicks_clear:
        return create_accordion(selected_person)
    else:
        raise PreventUpdate



@app.callback(
    [Output("checked-info", "children"),
     Output('line-graph', 'figure')],
    [Input('window-slider', 'value'),
     Input("update-button", "n_clicks"),
     Input("normalize-check", "value"),
     Input("clear-button", "n_clicks")],
    [State({'type': 'dynamic-checklist', 'index': ALL}, 'value'),
     State("selected-value", "data")],
    prevent_initial_call=True
)
def update_output(window_size, n_clicks, normalize_checked, n_clicks_clear, checkbox_values, selected_person):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'clear-button':
        return "No checkboxes checked.", go.Figure()
    elif button_id == 'update-button' and n_clicks:
        normalize = "normalize" in normalize_checked

        # Filter out unchecked (empty) checkboxes
        checked_checkboxes = [val[0].split(":") for val in checkbox_values if val]

        if checked_checkboxes:
            checked_info = html.Ul([html.Li(f"File: {file}, Column: {column}") for file, column in checked_checkboxes])
        else:
            checked_info = "No checkboxes checked."

        if selected_person is None:
            return checked_info, go.Figure()

        # Load each selected CSV file and select the chosen columns
        dataframes = []
        for file, column in checked_checkboxes:
            # Convert spaces in file name back to underscores
            file = file.replace(' ', '_')
            csv_dir = f'{dir_path}/{selected_person}/data/{file}.csv'
            df = pd.read_csv(csv_dir)
            df = df[[column]]
            dataframes.append(df)

        # Concatenate all dataframes (assuming they have the same length)
        final_df = pd.concat(dataframes, axis=1)
        final_df = final_df.rolling(window=window_size).mean()

        # Normalize the data if the checkbox is checked
        if normalize:
            final_df = (final_df - final_df.min()) / (final_df.max() - final_df.min())

        text_file = f'{dir_path}//{selected_person}/data/text.csv'
        text_data = pd.read_csv(text_file)['Sous-titres']

        # Create line graph figure with Plotly
        fig = go.Figure()
        for column in final_df.columns:
            time_values = final_df.index / 25
            # Check if the column contains boolean values
            if final_df[column].dropna().isin([0,1]).all():
                # Add a bar chart for boolean values
                fig.add_trace(go.Bar(
                    x=time_values,
                    y=final_df[column],
                    name=column,
                    hovertemplate="%{text}<extra></extra>",
                    text=[f"{t // 60:.0f}:{t % 60:02.0f}<br>{a}" for t, a in zip(time_values, text_data)]
                ))
            else:
                # Existing scatter plot code
                fig.add_trace(go.Scatter(
                    x=time_values,
                    y=final_df[column],
                    mode='lines',
                    name=column,
                    line=dict(width=1),
                    hovertemplate="%{text}<extra></extra>",
                    text=[f"{t // 60:.0f}:{t % 60:02.0f}<br>{a}" for t, a in zip(time_values, text_data)]
                ))

        # Calculate the x tick locations and labels
        max_seconds = int(max(time_values))
        tickvals = list(range(0, max_seconds, 15))
        ticktext = [f"{t // 60:.0f}:{t % 60:02.0f}" for t in tickvals]

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation="h"),
            xaxis=dict(
                tickmode='array',
                tickvals=tickvals,
                ticktext=ticktext
            )
        )

        return checked_info, fig
    else:
        raise PreventUpdate


@app.callback(
    Output('video-btn', 'style'),
    Input('selected-value', 'data')
)
def show_hide_button(data):
    if data is None:
        # Hide the button
        return {'display': 'none'}
    else:
        # Show the button
        return {}

@app.callback(
    [
        Output('video-player-small', 'seekTo'),
        Output('video-player-large', 'seekTo')
    ],
    [Input('line-graph', 'clickData')],
)
def update_video_frame(clickData):
    if clickData is not None:
        new_time = clickData['points'][0]['x']  # Assuming 'x' is time in seconds
        return new_time, new_time
    else:
        return dash.no_update, dash.no_update



@app.callback(
    Output('video', 'style'),
    [Input('selected-value', 'data')],
)
def toggle_container(data):
    if data is None:
        return {'display': 'none'}
    else:
        return {'display': 'block'}

@app.callback(
    Output('toggle-value', 'data'),
    Input('checklist-switch', 'value'),
    State('toggle-value', 'data'),
    prevent_initial_call=True,
)
def toggle_visibility(n, visibility):
    if n is None:
        return True
    else:
        return not visibility

@app.callback(
    [Output('right-column', 'style'), Output('graph-column', 'width')],
    Input('toggle-value', 'data'),
)
def update_style(visibility):
    if visibility:
        return {}, 8  # or whatever width your graph column originally had
    else:
        return {'display': 'none'}, 12

@app.callback(
    Output('left-side', 'style'),
    Output('right-side', 'style'),
    [Input('checklist-switch', 'value')]
)
def toggle_column(n):
    if n:
        return dict(display='block', width='50%'), dict(display='block', width='50%')
    else:
        return dict(display='none', width='0%'), dict(display='block', width='100%')

# ------ CNN -----------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def train_cnn(df, person_name, label):
    data = df.values.reshape((df.shape[0], df.shape[1], 1))
    model = Sequential()
    model.add(Conv1D(32, 3, activation='relu', input_shape=data.shape[1:]))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    encoder = LabelEncoder()
    y = encoder.fit_transform([label]*df.shape[0])
    y = to_categorical(y)

    model.fit(data, y, epochs=10, batch_size=32)

    # Sauvegarder le modèle
    model.save(f'{dir_path}/{person_name}/data/model.h5')

    return model


@app.callback(
    Output('model-output', 'children'),
    Input('save-button', 'n_clicks'),
    State('ranges-input', 'value'),
    State('data-table', 'data'),
    State('selected-value', 'data'),
    prevent_initial_call=True
)
def train_model(n_clicks, ranges_str, data, selected_person):
    if n_clicks is None:
        raise PreventUpdate

    df = pd.DataFrame(data)
    dataframes = split_dataframe(df, ranges_str)

    for df in dataframes:
        model = train_cnn(df, selected_person)
    return "Model trained successfully!"


if __name__ == '__main__':
    app.run_server(debug=True)
