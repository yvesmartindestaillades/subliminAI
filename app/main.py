import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, ClientsideFunction

import numpy as np
import pandas as pd
import datetime
from datetime import datetime as dt
import pathlib

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Clinical Analytics Dashboard"

server = app.server
app.config.suppress_callback_exceptions = True

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()

# Read data

clinic_list ='clinic_list'
admit_list = ['admit_list']

day_list = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def description_card():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("Clinical Analytics"),
            html.H3("Welcome to the Clinical Analytics Dashboard"),
            html.Div(
                id="intro",
                children="Explore clinic patient volume by time of day, waiting time, and care score. Click on the heatmap to visualize patient experience at different time points.",
            ),
        ],
    )


def generate_control_card():
    """

    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.P("Select Clinic"),
            dcc.Dropdown(
                id="clinic-select",
                options=[{"label": i, "value": i} for i in clinic_list],
                value=clinic_list[0],
            ),
            html.Br(),
            html.P("Select Check-In Time"),
            dcc.DatePickerRange(
                id="date-picker-select",
                start_date=dt(2014, 1, 1),
                end_date=dt(2014, 1, 15),
                min_date_allowed=dt(2014, 1, 1),
                max_date_allowed=dt(2014, 12, 31),
                initial_visible_month=dt(2014, 1, 1),
            ),
            html.Br(),
            html.Br(),
            html.P("Select Admit Source"),
            dcc.Dropdown(
                id="admit-select",
                options=[{"label": i, "value": i} for i in admit_list],
                value=admit_list[:],
                multi=True,
            ),
            html.Br(),
            html.Div(
                id="reset-btn-outer",
                children=html.Button(id="reset-btn", children="Reset", n_clicks=0),
            ),
        ],
    )



app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[html.Img(src=app.get_asset_url("plotly_logo.png"))],
        ),
        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card(), generate_control_card()]
            + [
                html.Div(
                    ["initial child"], id="output-clientside", style={"display": "none"}
                )
            ],
        ),
        # Right column
        html.Div(
            id="right-column",
            
        ),
    ],
)

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)