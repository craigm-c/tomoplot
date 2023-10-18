
from dash import Dash, dcc, html, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from io import StringIO
import scipy.io
from scipy.spatial import cKDTree
import shutil
import webbrowser
from tomo_functions import *


# Below is the layout for the app

fig = go.Figure()

app = Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])
server = app.server

app.layout = html.Div([
    dbc.Container([
    	dbc.Row([
    		dbc.Col(
    			html.H1('Tomoplot'),
                        style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "none",
                        "borderRadius": "5px",
                        "textAlign": "center",}
            ),
    	]),

    	dbc.Row([
            dbc.Col(
                    dcc.Upload(id='uploading-file', children=html.Div('Click to choose file',  style={'color': 'red'}), multiple=False,
                        ),
                    style={
                        "height": "60px",
                        "lineHeight": "60px",
                        "font-size":"30px",
                        "borderWidth": "1px",
                        "borderStyle": "solid",
                        "borderRadius": "5px",
                        "textAlign": "center",},
                        width=3,
                        className="ml-auto"
                    
                ),
    			dbc.Col([
                    html.Div(id='selected-file', children='test_data.mat'),
                    html.Div(id='data-store', style={'display': 'none'}),
                    html.Div(id='clean-data-store', children=None, style={'display': 'none'}),
                    html.Div(id='orientation-data-store', children=None, style={'display': 'none'}),
                    html.Div(id='index-store', children=0, style={'display': 'none'}),
                    dcc.Store(id='clean-indexes-store', data=[], storage_type='memory')],
                    style={
                    "height": "60px",
                    "lineHeight":"60px",
                    "fontSize":"30px",
                    "borderWidth": "1px",
                    "borderStyle": "solid",
                    "borderRadius": "5px",
                    "textAlign": "center",},
                    width={"size":9}
                ),
    	]),
    dbc.Row([
            dbc.Col([
                dbc.Button(id='load-data', children='Load data',
                    style={
                        "height": "50px",
                        "width": "200px",
                        "lineHeight": "30px",
                        "font-size": "30px",
                        "borderWidth": "1px",
                        "borderStyle": "none",
                        "borderRadius": "5px",
                        "textAlign": "center",
                    },
                )],
                style={
                    "height": "60px",
                    "lineHeight": "60px",
                    "font-size": "30px",
                    "borderWidth": "1px",
                    "borderStyle": "none",
                    "borderRadius": "5px",
                    "textAlign": "center",
                },
                width=3,
                className="ml-auto"
            ),
    		dbc.Col(
                html.Div(id='image-number', children=html.Div('No. of tomograms')),
                style={
                    "height": "60px",
                    "lineHeight": "60px",
                    "font-size":"30px",
                    "borderWidth": "1px",
                    "borderStyle": "solid",
                    "borderRadius": "5px",
                    "textAlign": "center",},
                    width=3,
                    className="ml-auto"
                    
                ),
            dbc.Col([
                    html.Div(dcc.Slider(id='image-slider', min=1, max=8, step=1, value=8))            
                ],
                style={
                    "height": "60px",
                    "lineHeight": "60px",
                    "font-size": "30px",
                    "borderWidth": "1px",
                    "borderStyle": "solid",
                    "borderRadius": "5px",
                    "textAlign": "center",
                },
                width=3,
                className="ml-auto"
            ),
    		dbc.Col(
                html.Div(id='load-confirmation', children=''),
                style={
                    "height": "60px",
                    "lineHeight": "60px",
                    "font-size":"20px",
                    "borderWidth": "1px",
                    "borderStyle": "solid",
                    "borderRadius": "5px",
                    "textAlign": "center",},
                    width=3,
                    className="ml-auto"
                    
                ),
        ]),
        dbc.Row([
            dbc.Col(
                html.Div(id='distance-label', children=html.Div('Distance | ±')),
                style={
                    "height": "60px",
                    "lineHeight": "60px",
                    "font-size":"30px",
                    "borderWidth": "1px",
                    "borderStyle": "none",
                    "borderRadius": "5px",
                    "textAlign": "center",},
                    width=2,
                    className="ml-auto"
                
            ),
            dbc.Col(
                html.Div(id='angle-label', children=html.Div('Angle | ±')),
                style={
                    "height": "60px",
                    "lineHeight": "60px",
                    "font-size":"30px",
                    "borderWidth": "1px",
                    "borderStyle": "none",
                    "borderRadius": "5px",
                    "textAlign": "center",},
                    width=2,
                    className="ml-auto"
                
            ),
            dbc.Col(
                html.Div(id='neighbours-label', children=html.Div('Neighbours')),
                style={
                    "height": "60px",
                    "lineHeight": "60px",
                    "font-size":"30px",
                    "borderWidth": "1px",
                    "borderStyle": "none",
                    "borderRadius": "5px",
                    "textAlign": "center",},
                    width=2,
                    className="ml-auto"
                
            ),
            dbc.Col(
                html.Div(id='fidelity-label', children='Orientation fidelity'),
                style={
                    "height": "60px",
                    "lineHeight": "60px",
                    "font-size":"30px",
                    "borderWidth": "1px",
                    "borderStyle": "none",
                    "borderRadius": "5px",
                    "textAlign": "center",},
                    width=3,
                    className="ml-auto"                
            ),
            dbc.Col(
                html.Div(id='new-filename-label', children=html.Div('New filename')),
                style={
                    "height": "60px",
                    "lineHeight": "60px",
                    "font-size":"30px",
                    "borderWidth": "1px",
                    "borderStyle": "none",
                    "borderRadius": "5px",
                    "textAlign": "center",},
                    width=3,
                    className="ml-auto"
                
            ),
        ]),
    	dbc.Row([
    			dbc.Col(
                    dcc.Input(id='distance-value', type='number', value=0, style={'width': '80%'}),
                        className="mx-auto my-auto"  
                ),
    			dbc.Col(
                    dcc.Input(id='distance-error', type='number', value=0,  style={'width': '80%'}),
                        width=1,
                        className="mx-auto my-auto"  
                ),
    			dbc.Col(
                    dcc.Input(id='angle-value', type='number', value=0,  style={'width': '80%'}),
                        width=1,
                        className="mx-auto my-auto"
                ),
    			dbc.Col(
                    dcc.Input(id='angle-error', type='number', value=0,  style={'width': '80%'}),
                        width=1,
                        className="mx-auto my-auto"
                ),
    			dbc.Col(
                    dcc.Input(id='neighbour-value', type='number', value=0),
                        width=2,
                        className="mx-auto my-auto" 
                ),
    			dbc.Col(
                    dcc.Input(id='fidelity-value', type='number', value=0, max=0, style={'float': 'right'}),
                        width=3,
                        className="mx-auto my-auto"
                ),   
    			dbc.Col(
                    dcc.Input(id='new-filename-value', type='text', value='cleaned.mat', style={'float': 'right'}),
                        width=3,
                        className="mx-auto my-auto"
                ),   
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Button(id='plot-data', children='Plot data',
                    style={
                        "height": "50px",
                        "width": "200px",
                        "lineHeight": "30px",
                        "font-size": "30px",
                        "borderWidth": "1px",
                        "borderStyle": "none",
                        "borderRadius": "5px",
                        "textAlign": "center",
                    },
                )],
                style={
                    "height": "60px",
                    "lineHeight": "60px",
                    "font-size": "30px",
                    "borderWidth": "1px",
                    "borderStyle": "none",
                    "borderRadius": "5px",
                    "textAlign": "center",
                },
                width=3,
                className="ml-auto"
            ),
            dbc.Col([
                dbc.Button(id='clean-data', children='Clean data',
                    style={
                        "height": "50px",
                        "width": "200px",
                        "lineHeight": "30px",
                        "font-size": "30px",
                        "borderWidth": "1px",
                        "borderStyle": "none",
                        "borderRadius": "5px",
                        "textAlign": "center",
                    },
                )],
                style={
                    "height": "60px",
                    "lineHeight": "60px",
                    "font-size": "30px",
                    "borderWidth": "1px",
                    "borderStyle": "none",
                    "borderRadius": "5px",
                    "textAlign": "center",
                },
                width=6,
                className="ml-auto"
            ),
            dbc.Col([
                dbc.Button(id='save-data', children='Save data',
                    style={
                        "height": "50px",
                        "width": "200px",
                        "lineHeight": "30px",
                        "font-size": "30px",
                        "borderWidth": "1px",
                        "borderStyle": "none",
                        "borderRadius": "5px",
                        "textAlign": "center",
                    },
                )],
                style={
                    "height": "60px",
                    "lineHeight": "60px",
                    "font-size": "30px",
                    "borderWidth": "1px",
                    "borderStyle": "",
                    "borderRadius": "5px",
                    "textAlign": "center",
                },
                width=3,
                className="ml-auto"
            )      
        ]),
        dbc.Row([
        	dbc.Col(
    			html.H1(''),
                        style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "none",
                        "borderRadius": "5px",
                        "textAlign": "center",})
    	]),
        dbc.Row([
            dbc.Col([
                dbc.Button(id='previous-tomogram', children='Previous',
                    style={
                        "height": "50px",
                        "width": "200px",
                        "lineHeight": "30px",
                        "font-size": "30px",
                        "borderWidth": "1px",
                        "borderStyle": "none",
                        "borderRadius": "5px",
                        "textAlign": "center",
                    },
                ), 
                dbc.Button(id='next-tomogram', children='Next',
                    style={
                        "height": "50px",
                        "width": "200px",
                        "lineHeight": "30px",
                        "font-size": "30px",
                        "borderWidth": "1px",
                        "borderStyle": "none",
                        "borderRadius": "5px",
                        "textAlign": "center",
                    },
                )],
                style={
                    "height": "60px",
                    "lineHeight": "60px",
                    "font-size": "30px",
                    "borderWidth": "1px",
                    "borderStyle": "none",
                    "borderRadius": "5px",
                    "textAlign": "center",
                },
            ),  
        ]),
        dbc.Row([
    			dbc.Col([
                    html.Div(id='measure-label', children='Measure: Click 3 points'),
                    html.Div(id='vector-data-store', children=[], style={'display': 'none'})],
                    style={
                        "height": "60px",
                        "lineHeight": "60px",
                        "font-size":"30px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",},
                        width=4,
                        className="ml-auto"
                ),
    			dbc.Col(
                    html.Div(id='measure-distance-label', children=html.Div('Distance:')),
                    style={
                        "height": "60px",
                        "lineHeight": "60px",
                        "font-size":"30px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",},
                        width=2,
                        className="ml-auto"
                    
                ),
    			dbc.Col(
                    html.Div(id='measure-distance-value', children=''),
                    style={
                        "height": "60px",
                        "lineHeight": "60px",
                        "font-size":"30px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",},
                        width=2,
                        className="ml-auto"
                    
                ),
    			dbc.Col(
                    html.Div(id='measure-angle', children=html.Div('Angle:')),
                    style={
                        "height": "60px",
                        "lineHeight": "60px",
                        "font-size":"30px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",},
                        width=2,
                        className="ml-auto"
                    
                ),
    			dbc.Col(
                    html.Div(id='measure-angle-value', children=''),
                    style={
                        "height": "60px",
                        "lineHeight": "60px",
                        "font-size":"30px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",},
                        width=2,
                        className="ml-auto"
                    
                ),
    	]),
    	dbc.Row(
    			dbc.Col(
                    html.Div(
                        dcc.Graph(id='3d-scatter-plot', figure=fig, style={'height': '900px'}),
                        style={'border': '20px solid #002b36'}
                    )
                ),       
        ),

    ])
])


# Selects the filename and updates the slider wwith the number of tomograms in the .mat file to allow the user to 
# select how many they want to work on

@app.callback(
    Output("selected-file", "children"),
    Output('image-slider', 'max'),
    Output('load-confirmation', 'children', allow_duplicate=True),
    Input("uploading-file", "filename"),
    prevent_initial_call=True
)

def select_data(filename):
    if filename is not None:
        image_slider_number = len(scipy.io.loadmat(filename, simplify_cells=True)['subTomoMeta']['cycle000']['geometry'].keys())
        return filename, image_slider_number, None
    else:
        return "Select .mat file", None, None



# load the x,y,z and orientation vector data from the number of tomograms selected by the user. Also resets the cleaned data to be 
# empty and sets the index of the tomogram being veiwed to 0

@app.callback(
    
    Output('data-store', 'children'),
    Output('clean-data-store', 'children'),
    Output('orientation-data-store', 'children'),
    Output('index-store', 'children', allow_duplicate=True),
    Output('load-confirmation', 'children', allow_duplicate=True),
    Input("load-data", "n_clicks"),
    State("selected-file", "children"),
    State("image-slider", "value"),
    prevent_initial_call=True  
)

def load_data(_, filename, slider_number):
    if filename is not None:
        json_data_list = []
        orientation_json_data_list= []
        mat_data = scipy.io.loadmat(filename, simplify_cells=True)['subTomoMeta']['cycle000']['geometry']
        for x in range(slider_number):
            tomogram = list(mat_data.keys())[x]
            
            mat_data_prelim = mat_data[tomogram].T[10:13]
            mat_data_processed = mat_data_prelim.T              
            data_df = pd.DataFrame(mat_data_processed, columns=['x','y','z'])
            json_data = data_df.to_json(orient='columns')
            json_data_list.append(json_data)

            orientation_data_prelim = mat_data[tomogram].T[22:25]
            orientation_data_processed = orientation_data_prelim.T              
            orientation_data_df = pd.DataFrame(orientation_data_processed, columns=['a','b','c'])
            orientation_json_data = orientation_data_df.to_json(orient='columns')
            orientation_json_data_list.append(orientation_json_data)

        return json_data_list, None, orientation_json_data_list, 0, 'Data loaded, click plot data'
    else:
        return None, None, None, 0,'data not loaded'



# Plots the data on the graph

@app.callback(
    Output(component_id='3d-scatter-plot', component_property='figure', allow_duplicate=True),
    Input(component_id='plot-data', component_property='n_clicks'),
    State(component_id='data-store', component_property='children'),

    prevent_initial_call=True  
)


def update_graph(n_clicks, json_data):
    # Parse the JSON data
    json_io = StringIO(json_data[0])
    # Convert the JSON data back to a Pandas DataFrame
    df_data = pd.read_json(json_io, orient='columns')
    # Create a new scatter plot with updated data
    trace = go.Scatter3d(
        x=df_data['x'],
        y=df_data['y'],
        z=-df_data['z'],
        mode='markers',
        marker=dict(
            size=3,
            color='blue',
            colorscale='Viridis',
            opacity=0.8
        ),
        hoverinfo='none'
    )

    fig = go.Figure(data=[trace])
    return fig



# Runs the cleaning operation on the data and plots the result of the tomogram that is currently being views to the graph


@app.callback(
    Output(component_id='3d-scatter-plot', component_property='figure', allow_duplicate=True),
    Output(component_id='clean-data-store', component_property='children', allow_duplicate=True),
    Output(component_id='clean-indexes-store', component_property='data'),
    Input(component_id='clean-data', component_property='n_clicks'),
    State(component_id='data-store', component_property='children'),
    State(component_id='orientation-data-store', component_property='children'),
    State(component_id='distance-value', component_property='value'),
    State(component_id='distance-error', component_property='value'),
    State(component_id='angle-value', component_property='value'),
    State(component_id='angle-error', component_property='value'),
    State(component_id='neighbour-value', component_property='value'),
    State(component_id='index-store', component_property='children'),
    State(component_id='fidelity-value', component_property='value'),
    
    prevent_initial_call=True
)


def clean_and_plot(n_clicks, json_data_list, orientation_json_data_list, dist_val, dist_error, angle_val, angle_error, neighbour, index, fidelity_number):
    clean_indexes = []
    clean_json_data_list = []
    for list_number in range(len(json_data_list)):


        json_io= StringIO(json_data_list[list_number])
        orientation_json_io = StringIO(orientation_json_data_list[list_number])

        df_data = pd.read_json(json_io, orient='columns')
        orientation_df_data = pd.read_json( orientation_json_io, orient='columns')

        np_data = df_data.to_numpy()
        orientation_np_data = orientation_df_data.to_numpy()

        kd_tree = cKDTree(np_data)
        
        initial_indexes = get_close_points_indexes(np_data, kd_tree, dist_val, dist_error, neighbour)

        orientation_indexes = get_good_orientation_points_indexes(orientation_np_data, initial_indexes, fidelity_number)

        confirmed_indexes = get_lattice_points_indexes(orientation_indexes, np_data, angle_val, angle_error)
        clean_indexes.append(confirmed_indexes)
        cleaned_array = np_data[confirmed_indexes]
        cleaned_df = pd.DataFrame(cleaned_array, columns=['x', 'y', 'z'])
        clean_json_data = cleaned_df.to_json(orient='columns')
        clean_json_data_list.append(clean_json_data)
    
    cleaned_json_plot = StringIO(clean_json_data_list[index])
    cleaned_df_plot = pd.read_json(cleaned_json_plot, orient='columns')


    trace = go.Scatter3d(
        x=cleaned_df_plot['x'],
        y=cleaned_df_plot['y'],
        z=-cleaned_df_plot['z'],
        mode='markers',
        marker=dict(
            size=3,
            color='blue',
            colorscale='Viridis',
            opacity=0.8
        ),
        hoverinfo='none'
    )

    fig = go.Figure(data=[trace])
    return fig, clean_json_data_list, clean_indexes




# Plots the data from the next tomogram from from the input file, will display the cleaned data is the tomograms have been cleaned
# other plot the orignal data


@app.callback(
    Output(component_id='3d-scatter-plot', component_property='figure', allow_duplicate=True),
    Output(component_id='index-store', component_property='children',  allow_duplicate=True),
    Input(component_id='next-tomogram', component_property='n_clicks'),
    State(component_id='clean-data-store', component_property='children'),
    State(component_id='data-store', component_property='children'),
    State(component_id='index-store', component_property='children'),
    prevent_initial_call=True  
)


def next_button(n_clicks, clean_json_data_list, json_data_list, index):
    if index < len(json_data_list)-1:
        index = index + 1
    if clean_json_data_list is not None:
        json_io = StringIO(clean_json_data_list[index])
        df_data = pd.read_json(json_io, orient='columns')
    else:
        json_io = StringIO(json_data_list[index])
        df_data = pd.read_json(json_io, orient='columns')

    trace = go.Scatter3d(
        x=df_data['x'],
        y=df_data['y'],
        z=-df_data['z'],
        mode='markers',
        marker=dict(
            size=3,
            color='blue',
            colorscale='Viridis',
            opacity=0.8
        ),
        hoverinfo='none'
    )
    fig = go.Figure(data=[trace])
    return fig, index






# Plots the data from the previous tomogram from from the input file, will display the cleaned data is the tomograms have been cleaned
# other plot the orignal data

@app.callback(
    Output(component_id='3d-scatter-plot', component_property='figure', allow_duplicate=True),
    Output(component_id='index-store', component_property='children',  allow_duplicate=True),
    Input(component_id='previous-tomogram', component_property='n_clicks'),
    State(component_id='clean-data-store', component_property='children'),
    State(component_id='data-store', component_property='children'),
    State(component_id='index-store', component_property='children'),
    prevent_initial_call=True  
)


def previous_button(n_clicks, clean_json_data_list, json_data_list, index):
    if index > 0:
        index = index - 1
    if clean_json_data_list is not None:
        json_io = StringIO(clean_json_data_list[index])
        df_data = pd.read_json(json_io, orient='columns')
    else:
        json_io = StringIO(json_data_list[index])
        df_data = pd.read_json(json_io, orient='columns')

    trace = go.Scatter3d(
        x=df_data['x'],
        y=df_data['y'],
        z=-df_data['z'],
        mode='markers',
        marker=dict(
            size=3,
            color='blue',
            colorscale='Viridis',
            opacity=0.8
        ),
        hoverinfo='none'
    )

    fig = go.Figure(data=[trace])
    return fig, index




# Allowd the user to click three points on the graph and outputs the distance and angle between these points


@app.callback(
    Output(component_id='measure-label', component_property='children'),
    Output(component_id='measure-distance-value', component_property='children'),
    Output(component_id='measure-angle-value', component_property='children'),
    Output(component_id='vector-data-store', component_property='children'),
    Input(component_id='3d-scatter-plot', component_property='clickData'),
    State(component_id='vector-data-store', component_property='children'),
    prevent_initial_call=True 
)


def select_points(clickData, data_store):
    clicked_vector = [clickData["points"][0]['x'], clickData["points"][0]['y'], clickData["points"][0]['z']]
    data_store.extend(clicked_vector)
    if len(data_store) == 9:
        np_data_store = np.array(data_store).reshape(3, 3)
        v1 = np_data_store[0]
        v2 = np_data_store[1]
        v3 = np_data_store[2]
        distance, angle = calculate_parameters(v1,v2,v3)
        return 'Measure: Click 3 points', round(distance, 2), round(angle, 2), []
    if len(data_store) == 6:
        return 'Measure: Click 1 point', '', '', data_store
    if len(data_store) == 3:
        return 'Measure: Click 2 points', '', '', data_store


# Makes a new file containing all of the data from the indexes selected by the cleaning proceedure

@app.callback(
    Output(component_id='load-confirmation', component_property='children'),
    Input(component_id='save-data', component_property='n_clicks'),
    State(component_id='selected-file', component_property='children'),
    State(component_id='new-filename-value', component_property='value'),
    State(component_id='clean-indexes-store', component_property='data'),
    State("image-slider", "value"),
    prevent_initial_call=True  
)


def save_data(n_clicks, original_filename, new_filename, indexes_list, tomogram_number):
    shutil.copyfile(original_filename, new_filename)
    mat_data = scipy.io.loadmat(original_filename, simplify_cells=True)
    output_data = scipy.io.loadmat(new_filename, simplify_cells=True)
    for index, tomogram in enumerate(list(mat_data['subTomoMeta']['cycle000']['geometry'].keys())[:tomogram_number]):
        output_data['subTomoMeta']['cycle000']['geometry'][tomogram] = mat_data['subTomoMeta']['cycle000']['geometry'][tomogram][indexes_list[index]]
    scipy.io.savemat(new_filename, output_data)
    return 'Data saved'


# sets the max vlaue for the orientation fidelity value which is the number of nearest neighbours valued squared

@app.callback(
    Output(component_id='fidelity-value', component_property='max'),
    Input(component_id='neighbour-value', component_property='value'),
    prevent_initial_call=True  
)


def set_orientation_fidelity_max(num_neighbours):
    num_neighbours_squared = num_neighbours**2
    return num_neighbours_squared 


if __name__ == '__main__':
    webbrowser.open("http://localhost:8050/")
    app.run_server(debug=True, use_reloader=False)
