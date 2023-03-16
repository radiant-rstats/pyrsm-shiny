from pathlib import Path

import numpy as np
import pandas as pd
# import pyrsm as rsm

import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import logit

from shiny import App, render, ui, reactive

infile = Path(__file__).parent / "data/dvd.csv"
df = pd.read_csv(infile)

def create_ui(data: pd.DataFrame):
    df_cols = list(data.columns)
    app_ui = ui.page_fluid(
        ui.row(
            ui.column(2, offset=1, *[
                ui.input_selectize(
                    id = "resp_var",
                    label = "Response Variable",
                    selected=None,
                    choices = df_cols
                )
            ]),
            ui.column(1),
            ui.column(6,
                     ui.output_text("resp_var_select"))
        ),
        ui.row(
            ui.column(2, offset=1, *[
                ui.input_selectize(
                    id = "expl_var",
                    label = "Explanatory Variables",
                    selected=None,
                    choices = df_cols,
                    multiple=True
                )
            ]),
            ui.column(1),
            ui.column(6,
                     ui.output_text("expl_var_select"))
        ),
        ui.row(
            ui.column(6,
                     ui.output_text("logit_ouput"))
        )
    )
    return app_ui

frontend = create_ui(df)

# wrapper function for the server, allows the data to be passed in
def create_server(data):
  def f(input, output, session):
    
    @output(id="resp_var_select") # decorator to link this function to the "resp_var_select" id in the UI
    @render.text
    def resp_var_select():
        if (input.resp_var() != None):
            return f'Selected resp_var is: "{input.resp_var()}"'
        else:
            return f'Please select a response variable'
    
    # expl_var_list = list(list(input.expl_var()))
    @output(id="expl_var_select") # decorator to link this function to the "expl_var_select" id in the UI
    @render.text
    def expl_var_select():
        if (input.expl_var() != None):
            return f'Selected expl_var are: "{list(input.expl_var())}"'
        else:
            return f'Please select at least one explanatory variable'
    
    @output(id="logit_ouput") # decorator to link this function to the "expl_var_select" id in the UI
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
        return lr.summary()
        # return rsm.or_ci(lr)
        # return f'resp_var column is:\n{df[resp_var]}'

  return f

server = create_server(df)

# Connect everything
app = App(frontend, server)