from pathlib import Path
import numpy as np
import pandas as pd
import pyrsm as rsm
import io
from contextlib import redirect_stdout
import sys
from shiny import App, render, ui, reactive, Inputs, Outputs, Session
from shiny.types import FileInfo

app_ui = ui.page_fluid(
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.panel_well(
                ui.input_file(
                    id="logistic_data",
                    label="Upload a Pickle file",
                    accept=[".pkl"],
                    multiple=False,
                ),
            ),
            ui.panel_conditional(
                "input.tabs_logistic == 'Summary'",
                ui.panel_well(
                    ui.input_action_button(
                        "run", "Estimate model", class_="btn-success", width="100%"
                    ),
                ),
                ui.panel_well(
                    ui.output_ui("ui_rvar"),
                    ui.output_ui("ui_evar"),
                    width=3,
                ),
            ),
            ui.panel_conditional(
                "input.tabs_logistic == 'Plot'",
                ui.panel_well(
                    ui.input_action_button(
                        "plot", "Create plot", class_="btn-success", width="100%"
                    ),
                ),
                ui.panel_well(
                    ui.input_select(
                        id="logistic_plots",
                        label="Plots",
                        selected=None,
                        choices={
                            "or": "OR plot",
                            "pred": "Prediction plot",
                            "vimp": "Permutation importance",
                        },
                    ),
                    width=3,
                ),
            ),
        ),
        ui.panel_main(
            ui.navset_tab_card(
                ui.nav("Data", ui.output_ui("show_data")),
                ui.nav("Summary", ui.output_text_verbatim("logistic_summary")),
                ui.nav("Plot", ui.output_plot("logistic_plot")),
                id="tabs_logistic",
            )
        ),
    ),
)


def server(input: Inputs, output: Outputs, session: Session):
    @reactive.Calc
    def load_data():
        if input.logistic_data() is None:
            return "Please upload a Pickle file"
        f: list[FileInfo] = input.logistic_data()
        return pd.read_pickle(f[0]["datapath"])

    @output(id="show_data")
    @render.ui
    def show_data():
        if input.logistic_data() is None:
            return "Please upload a Pickle file"
        else:
            return ui.HTML(
                load_data()
                .head()
                .to_html(classes="table table-striped data_preview", index=False)
            )

    @output(id="ui_rvar")
    @render.ui
    def ui_rvar():
        if isinstance(load_data(), str):
            df_cols = []
        else:
            df_cols = list(load_data().columns)
        return ui.input_select(
            id="rvar",
            label="Response Variable",
            selected=None,
            choices=df_cols,
        )

    @output(id="ui_evar")
    @render.ui
    def ui_evar():
        if isinstance(load_data(), str):
            df_cols = []
        else:
            df_cols = list(load_data().columns)
            if (input.rvar() is not None) and (input.rvar() in df_cols):
                df_cols.remove(input.rvar())
        return ui.input_select(
            id="evar",
            label="Explanatory Variables",
            selected=None,
            choices=df_cols,
            multiple=True,
            selectize=False,
        )

    @reactive.Calc
    def logistic_regression():
        return rsm.logistic(
            dataset=load_data(), rvar=input.rvar(), evar=list(input.evar())
        )

    @output(id="logistic_summary")
    @render.text
    @reactive.event(input.run, ignore_none=True)
    def logistic_summary():
        out = io.StringIO()
        with redirect_stdout(out):
            logistic_regression().summary()
        return out.getvalue()

    @output(id="logistic_plot")
    @render.plot()
    @reactive.event(input.plot, ignore_none=True)
    def logistic_plot():
        return logistic_regression().plot(plots=input.logistic_plots())

    @output(id="code_snippet")
    @render.text
    @reactive.event(input.copy_to_clipboard, ignore_none=False)
    def code_snippet():
        """Generate Python code for logistic regression using the 'pyrsm' package."""
        rvar = input.rvar
        evar = input.evar
        cmd = f"""import pyrsm as rsm
lr = rsm.logistic_(dataset='{df_name}', rvar='{rvar}', evars={evar})
lr.summary()
        """

        try:
            return exec(cmd)
        except Exception as e:
            return f"Error: {str(e)}"

        # pyperclip.copy(cmd)
        return cmd


app = App(app_ui, server)
