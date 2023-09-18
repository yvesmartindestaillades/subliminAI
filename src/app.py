import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, ClientsideFunction, State
import base64
import numpy as np
import pandas as pd
import os
from datetime import datetime as dt
import pathlib
from generate_animation import GenerateAnimation
import dash_mantine_components as dmc
import flask

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Clinical Analytics Dashboard"

server = app.server
app.config.suppress_callback_exceptions = True
generate_animation = GenerateAnimation()

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
from os.path import dirname, abspath, join

out_path = 'output.gif'

# Read data

app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[
                html.Img(src=app.get_asset_url("plotly_logo.png")),
                html.H2("SubliminAI"),
            ],
        ),
        # Left column
        html.Div(
            id="left-column",
            # max width = 30% of the page
            style={
                "width": "49%",
                "display": "inline-block",
                "vertical-align": "middle",
                "margin-right": "1%",
                "horizontal-align": "left",
            },
            className="four columns",
            children=[
                # hidden text prompt
                html.Div(
                    id="hidden-text-prompt",
                    children=[
                        html.P(
                            id="hidden-text-prompt-text",
                            children="Hidden text prompt",
                        ),
                        dcc.Input(
                            id="hidden-text",
                            value="You're the best!",
                            style={"width": "100%", "height": 100},
                        ),
                    ],
                ),
                # prompt
                html.Div(
                    id="prompt",
                    children=[
                        html.P(
                            id="prompt-text",
                            children="Theme prompt",
                        ),
                        dcc.Input(
                            id="theme-text",
                            value="An italian restaurant with a cozy atmosphere.",
                            # expandable
                            style={"height": 100, "width": "100%"},
                        ),
                    ],
                ),
                html.Div(
                    id="control-strength-and-number-of-frames",
                    children=[
                        # float for control strength
                        html.Div(
                            id="control-strength",
                            style={
                                "margin-top": "40px",
                                "width": "90%",
                                "margin-bottom": "20px",
                                "margin-left": "20px",
                            },
                            children=[
                                html.P(
                                    id="control-strength-text",
                                    children="Control strength",
                                ),
                                dcc.Slider(
                                    id="control-strength-slider",
                                    min=0.1,
                                    max=2,
                                    step=0.1,
                                    value=1.7,
                                    marks={i / 10: str(i / 10) for i in range(1, 21)},
                                ),
                            ],
                        ),
                        # number of frames
                        html.Div(
                            id="number-of-frames",
                            style={
                                "margin-top": "40px",
                                "width": "90%",
                                "margin-bottom": "20px",
                                "margin-left": "20px",
                            },
                            children=[
                                html.P(
                                    id="number-of-frames-text",
                                    children="Number of frames",
                                ),
                                dcc.Slider(
                                    id="number-of-frames-slider",
                                    min=1,
                                    max=10,
                                    step=1,
                                    value=3,
                                    marks={i: str(i) for i in range(1, 11)},
                                ),
                            ],
                        ),
                        # checkbox for morphing
                        html.Div(
                            id="morphing-checkbox",
                            style={
                                "margin-top": "40px",
                                "width": "46%",
                                "margin-bottom": "20px",
                                "margin-left": "20px",
                            },
                            children=[
                                dcc.Checklist(
                                    id="morphing-checkbox__input",
                                    options=[
                                        {
                                            "label": "Use morphing",
                                            "value": "use_morphing",
                                        }
                                    ],
                                    value=[],
                                )
                            ],
                        ),
                    ],
                ),
                html.Button(
                    id="submit-button",
                    children="Generate",
                    n_clicks=0,
                    style={"width": "100%"},
                ),
                        # progress bar
                
                html.Div(
                    id="progress-bar",
                    style={
                        "margin-top": "40px",
                        "width": "90%",
                        "margin-bottom": "20px",
                        "margin-left": "20px",
                    },
                    children=[
                        dcc.Interval(
                            id="progress-interval",
                            n_intervals=0,
                            interval=500,
                            max_intervals=-1,
                        ),
                        dmc.Progress(
                            id="loading",
                            value=0,
                            color="green"),        
                        html.P(
                            id="progress-bar-text",
                            children="Progress",
                        ),
                    ],
                ),
            ],
        ),
        # Right column
        html.Div(
            id="right-column",
            style={
                "width": "45%",
                "display": "inline-block",
                "vertical-align": "middle",
                "horizontal-align": "right",
            },
            # embed image
            children=[
                html.Img(
                    id="image",
                    src=app.get_asset_url("app/assets/placeholder.png"),
                    style={
                        "width": "80%",
                        "display": "inline-block",
                        "vertical-align": "middle",
                        "horizontal-align": "right",
                    },
                ),
                # download button
                # html.Div(
                #     id="download-button",
                #     children=[
                #         html.Button("Download", id="btn_download"), 
                #         dcc.Download(id="download"),
                        
                #     ],
                # ),
            ],
        ),
    ],
)

@app.callback(
    Output("download-text", "data"),
    [Input("btn_download", "n_clicks")],
    prevent_initial_call=True,
)
def create_download_file(n_clicks):
    filename = out_path
    # Alternatively:
    # filename = f"{uuid.uuid1()}.txt"
    print(filename)
    return flask.send_file(filename)

@app.callback([Output("loading", "value"), Output("progress-bar-text","children"), Output("loading", "color")], [Input("progress-interval", "n_intervals")])
def progress_bar_update(n):
    progress_bar_val = (
        generate_animation.progress_bar_val / generate_animation.progress_bar_val_max
    )
    return (progress_bar_val*100, generate_animation.state, 'blue' if progress_bar_val < 1 else 'green')


# callback for submit-button
@app.callback(
    Output("image", "src"),
    [Input("submit-button", "n_clicks")],
    [
        State("hidden-text", "value"),
        State("theme-text", "value"),
        State("control-strength-slider", "value"),
        State("number-of-frames-slider", "value"),
        State("morphing-checkbox__input", "value"),
    ],
    prevent_initial_call=True)

def update_output(
    n_clicks: int,
    hidden_text: str,
    theme_text: str,
    control_strength: float,
    number_of_frames: int,
    use_morphing: bool,
) -> str:
    generate_animation.state = "Generating alternative prompts..."
    generate_animation.progress_bar_val = 0
    generate_animation.progress_bar_val_max = number_of_frames
    try:
        # generate animation
        generate_animation.run(
            text=hidden_text,
            prompt=theme_text,
            control_strength=control_strength,
            number_of_frames=number_of_frames,
            use_morphing="use_morphing" in use_morphing,
            output_path=out_path,
        )
        # return the image
        encoded_image = base64.b64encode(open(out_path, "rb").read())
    except:
        encoded_image = base64.b64encode(open("assets/placeholder.png", "rb").read())
    return f"data:image/png;base64,{encoded_image.decode()}"


# Run the server
if __name__ == "__main__":
    app.run_server()  # debug=True)
