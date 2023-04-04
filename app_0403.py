from pathlib import Path

import numpy as np
import pandas as pd
import pyrsm as rsm

import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import logit

import seaborn as sns

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
                     ui.output_table("logit_or_ci"))
        ),
        ui.row(
            ui.column(6,
                     ui.output_plot("logit_or_plot"))
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
    
    def logit_func():
        form = input.resp_var() + " ~ " + " + ".join(list(input.expl_var()))
        lr = smf.glm(
            formula=form,
            family=Binomial(link=logit()),
            data=df,
            ).fit()
        return lr
    
    @output(id="logit_or_ci") # decorator to link this function to the "expl_var_select" id in the UI
    @render.table
    def logit_or_ci():
        lr = logit_func()
        # or_ci_df = rsm.or_ci(lr)
        return (
                # or_ci_df.style.set_table_attributes(
                rsm.or_ci(lr).style.set_table_attributes(
                    'class="dataframe shiny-table table w-auto"'
                )
                # .hide(axis="index")
                # .set_table_styles(
                #     [
                #         dict(selector="th", props=[("text-align", "right")]),
                #         dict(
                #             selector="tr>td",
                #             props=[
                #                 ("padding-top", "0.1rem"),
                #                 ("padding-bottom", "0.1rem"),
                #             ],
                #         ),
                #     ]
                # )
            )
        # return or_ci_df
    
    @output(id="logit_or_plot") # decorator to link this function to the "expl_var_select" id in the UI
    @render.plot()
    def logit_or_plot():
        lr = logit_func()
        sns.set_style('whitegrid')
        fig = rsm.or_plot(lr)
        fig.set(title = "Odds-Ratios of Model 'lr'", xlabel = "Odds-Ratios", ylabel = "Explanatory Variables")
        return fig

  return f

server = create_server(df)

# Connect everything
app = App(frontend, server)