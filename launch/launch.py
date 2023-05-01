# def launch_app():
#     from shiny import App, render, ui, run_app
#     app_ui = ui.page_fluid(
#         ui.input_slider("n", "N", 0, 100, 20),
#         ui.output_text_verbatim("txt"),
#     )

#     def server(input, output, session):
#         @output
#         @render.text
#         def txt():
#             return f"n*2 is {input.n() * 2}"

#     ret = run_app(App(app_ui, server))
#     return ret

# launch_app()


# # run_app("launch::launch_app")

# from shiny import ui, App, run_app
# app_ui = ui.page_fluid(
#     ui.input_slider("n", "N", 0, 100, 20),
#     ui.output_text_verbatim("txt"),
# )


# def server(input, output, session):
#     @output
#     @render.text
#     def txt():
#         return f"n*2 is {input.n() * 2}"


# run_app(App(app_ui, server), port=8001)


# from IPython.display import Markdown
# from shiny import ui, App, run_app

# Markdown.tagify = lambda x: ui.markdown(x.data)

# app_ui = ui.page_fluid(
#     Markdown("## Hello world")
# )

# run_app(App(app_ui, server=None))

from shiny import *
from shinywidgets import output_widget, register_widget, reactive_read
import ipyleaflet as L

app_ui = ui.page_fluid(
    ui.input_slider("zoom", "Map zoom level", value=4, min=1, max=10),
    output_widget("map"),
    ui.output_text("map_bounds"),
)


def server(input, output, session):

    # Initialize and display when the session starts (1)
    map = L.Map(center=(52, 360), zoom=4)
    register_widget("map", map)

    # When the slider changes, update the map's zoom attribute (2)
    @reactive.Effect
    def _():
        map.zoom = input.zoom()

    # When zooming directly on the map, update the slider's value (2 and 3)
    @reactive.Effect
    def _():
        ui.update_slider("zoom", value=reactive_read(map, "zoom"))

    # Everytime the map's bounds change, update the output message (3)
    @output
    @render.text
    def map_bounds():
        b = reactive_read(map, "bounds")
        lat = [b[0][0], b[0][1]]
        lon = [b[1][0], b[1][1]]
        return f"The current latitude is {lat} and longitude is {lon}"


run_app(App(app_ui, server), port=8001)
