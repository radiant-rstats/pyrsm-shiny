from pathlib import Path

import numpy as np
import pandas as pd
import pyrsm as rsm
import pyperclip

import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import logit

from shiny import App, render, ui, reactive

df_name = "dvd"
infile = Path(__file__).parent / f"data/{df_name}.csv"
df = pd.read_csv(infile)


def create_ui(data: pd.DataFrame):
    df_cols = list(data.columns)
    app_ui = ui.page_fluid(
        ui.row(
            ui.column(
                2,
                offset=1,
                *[
                    ui.input_selectize(
                        id="resp_var",
                        label="Response Variable",
                        selected=None,
                        choices=df_cols,
                    )
                ],
            ),
            ui.column(1),
            ui.column(6, ui.output_text("resp_var_select")),
        ),
        ui.row(
            ui.column(
                2,
                offset=1,
                *[
                    ui.input_selectize(
                        id="expl_var",
                        label="Explanatory Variables",
                        selected=None,
                        choices=df_cols,
                        multiple=True,
                    )
                ],
            ),
            ui.column(1),
            ui.column(6, ui.output_text("expl_var_select")),
        ),
        ui.row(ui.column(6, ui.output_text("logit_ouput"))),
        ui.row(
            ui.column(
                6,
                ui.input_action_button(
                    "copy_to_clipboard", "Generate code", class_="btn-success"
                ),
            )
        ),
        ui.panel_conditional(
            "input.copy_to_clipboard > 0",
        ui.row(
            ui.column(
                6,
                ui.output_text_verbatim("code_snippet"))
            ),
        ),
    )
    return app_ui


frontend = create_ui(df)

# wrapper function for the server, allows the data to be passed in
def create_server(data):
    def f(input, output, session):
        @output(
            id="resp_var_select"
        )  # decorator to link this function to the "resp_var_select" id in the UI
        @render.text
        def resp_var_select():
            if input.resp_var() != None:
                return f'Selected resp_var is: "{input.resp_var()}"'
            else:
                return f"Please select a response variable"

        # expl_var_list = list(list(input.expl_var()))
        @output(
            id="expl_var_select"
        )  # decorator to link this function to the "expl_var_select" id in the UI
        @render.text
        def expl_var_select():
            if input.expl_var() != None:
                return f'Selected expl_var are: "{list(input.expl_var())}"'
            else:
                return f"Please select at least one explanatory variable"

        @output(
            id="logit_ouput"
        )  # decorator to link this function to the "expl_var_select" id in the UI
        @render.text
        def logit_ouput():
            resp_var = input.resp_var()
            expl_var_list = list(input.expl_var())
            # df["resp_var_yes"] = rsm.ifelse(df[resp_var] == "yes", 1, 0)
            lr_expl_var = " + ".join(expl_var_list)
            form = input.resp_var() + " ~ " + " + ".join(list(input.expl_var()))
            lr = smf.glm(
                formula=form,
                family=Binomial(link=logit()),
                data=df,
            ).fit()
            # return lr.summary()
            return rsm.or_ci(lr)
            # return f'resp_var column is:\n{df[resp_var]}'

        @output(id="code_snippet")
        @render.text
        @reactive.event(input.copy_to_clipboard, ignore_none=False)
        def code_snippet():
            """Generate Python code for logistic regression using the 'pyrsm' package."""
            resp_var = input.resp_var
            expl_var = input.expl_var
        
            code_template = """import pyrsm as rsm

lr = rsm.logistic_regression(dataset='{df_name}', rvar='{resp_var}', evars={expl_var})

lr.regress()
            """
            code_string = code_template.format(
                df_name  = df_name,
                resp_var=resp_var(),
                expl_var=expl_var()
                
            )
            
            try:
               exec(code_string) 
               return f'eval("""{code_string}""")'
            except Exception as e:
                return f"Error: {str(e)}"
        

            pyperclip.copy(code_string)

            return code_string

    return f


server = create_server(df)

# Connect everything
app = App(frontend, server)
