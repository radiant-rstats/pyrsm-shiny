from pathlib import Path

import numpy as np
import pandas as pd
import pyrsm as rsm
import pyperclip

import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import logit

import seaborn as sns

from shiny import App, render, ui, reactive

df_name = "dvd"
infile = Path(__file__).parent / f"data/{df_name}.csv"
df = pd.read_csv(infile)

def create_ui(data: pd.DataFrame):
    df_cols = list(data.columns)
    app_ui = ui.page_fluid(
        ui.layout_sidebar(
            ui.panel_sidebar(ui.input_select(id = "resp_var",
                                                label = "Response Variable",
                                                selected=None,
                                                choices = df_cols),
                            ui.input_selectize(id = "expl_var",
                                               label = "Explanatory Variables",
                                               selected=None,
                                               choices = df_cols,
                                               multiple=True),
                            width=2
                                                ),
            ui.panel_main(ui.output_text("resp_var_select"),
                          ui.row(),
                          ui.output_text("expl_var_select")),
        ),
        # ui.row(
        #     ui.column(2),
        #     ui.column(6, ui.output_table("logit_or_ci"))
        # ),
        # ui.row(
        #     ui.column(2),
        #     ui.column(6, ui.output_table("logit_model_fit"))
        # ),
        # ui.row(
        #     ui.column(2),
        #     ui.column(6, ui.output_plot("logit_or_plot"))
        # ),,
        # ui.row(
        #     ui.column(2),
        #     ui.column(6, ui.output_plot("logit_perm_imp_plot"))
        # ),
        ui.row(
            ui.column(2),
            ui.column(9, ui.navset_tab_card(
                            ui.nav("Summary", ui.output_table("logit_or_ci"), ui.output_table("logit_model_fit")),
                            ui.nav("Plot", ui.output_plot("logit_or_plot"), ui.output_plot("logit_perm_imp_plot")),
                        )
            )
        ),
        ui.row(
            ui.column(6,
                      ui.input_action_button("copy_to_clipboard",
                                             "Generate code",
                                             class_="btn-success"
                                             ),
            )
        ),
        ui.panel_conditional(
            "input.copy_to_clipboard > 0",
            ui.row(
                ui.column(6, ui.output_text_verbatim("code_snippet"))
            ),
        ),
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
            return f'Selected response variable: "{input.resp_var()}"'
        else:
            return f'Please select a response variable'
    
    # expl_var_list = list(list(input.expl_var()))
    @output(id="expl_var_select") # decorator to link this function to the "expl_var_select" id in the UI
    @render.text
    def expl_var_select():
        if (input.expl_var() != None):
            return f'Selected explanatory variables: "{list(input.expl_var())}"'
        else:
            return f'Please select at least one explanatory variable'
    
    def logit_func():
        form = input.resp_var() + " ~ " + " + ".join(list(input.expl_var()))
        # lr = rsm.logistic_regression(dataset=df,
        #                              rvar=input.resp_var(),
        #                              evars=list(input.expl_var())
        #                              ).regress()
        lr = smf.glm(
            formula=form,
            family=Binomial(link=logit()),
            data=df,
            ).fit()
        return lr
    
    @output(id="logit_or_ci") # decorator to link this function to the "logit_or_ci" id in the UI
    @render.table
    def logit_or_ci():
        lr = logit_func()
        # or_ci_df = rsm.or_ci(lr)
        return (
                # or_ci_df.style.set_table_attributes(
                rsm.or_ci(lr).style.set_table_attributes(
                    'class="dataframe shiny-table table w-auto"'
                )
                .hide(axis="index")
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
    
    @output(id="logit_model_fit") # decorator to link this function to the "logit_model_fit" id in the UI
    @render.table
    def logit_model_fit():
        lr = logit_func()
        mfit = rsm.model_fit(lr, prn=False)
        mfit_dict = [{'Measure': "Pseudo R-squared (McFadden):",
                    'Value': "{:.3f}".format(mfit.pseudo_rsq_mcf.values[0])},
                {'Measure': "Pseudo R-squared (McFadden adjusted):",
                    'Value': "{:.3f}".format(mfit.pseudo_rsq_mcf_adj.values[0])},
                {'Measure': "Area under the RO Curve (AUC):",
                    'Value': "{:.3f}".format(mfit.AUC.values[0])},
                {'Measure': "Log-likelihood:",
                    'Value': "{:.3f}".format(mfit.log_likelihood.values[0])},
                {'Measure': "AIC:",
                    'Value': "{:.3f}".format(mfit.AIC.values[0])},
                {'Measure': "BIC:",
                    'Value': "{:.3f}".format(mfit.BIC.values[0])},
                {'Measure': "Chi-squared:",
                    'Value': "{:.3f}".format(mfit.chisq.values[0])},
                {'Measure': "Chi-squared df:",
                    'Value': mfit.chisq_df.values[0]},
                {'Measure': "p.value",
                    'Value': np.where(mfit.chisq_pval.values[0] < .001, "< 0.001", mfit.chisq_pval.values[0].round(3))},
                {'Measure': "Nr obs:",
                    'Value': "{:.0f}".format(mfit.nobs.values[0])}]
        
        mfit_df = pd.DataFrame(mfit_dict)

        mfit_text = f"""
Pseudo R-squared (McFadden): {mfit.pseudo_rsq_mcf.values[0].round(3)}
Pseudo R-squared (McFadden adjusted): {mfit.pseudo_rsq_mcf_adj.values[0].round(3)}
Area under the RO Curve (AUC): {mfit.AUC.values[0].round(3)}
Log-likelihood: {mfit.log_likelihood.values[0].round(3)}, AIC: {mfit.AIC.values[0].round(3)}, BIC: {mfit.BIC.values[0].round(3)}
Chi-squared: {mfit.chisq.values[0].round(3)} df({mfit.chisq_df.values[0]}), p.value {np.where(mfit.chisq_pval.values[0] < .001, "< 0.001", mfit.chisq_pval.values[0].round(3))} 
Nr obs: {mfit.nobs.values[0]:,.0f}
"""
        # return mfit_text
        return (
                mfit_df.style.set_table_attributes(
                    'class="dataframe shiny-table table w-auto"'
                )
                .hide(axis="index")
            )
    
    @output(id="logit_or_plot") # decorator to link this function to the "logit_or_plot" id in the UI
    @render.plot()
    def logit_or_plot():
        lr = logit_func()
        sns.set_style('whitegrid')
        fig = rsm.or_plot(lr)
        fig.set(title = "Odds-Ratios of Model 'lr'", xlabel = "Odds-Ratios", ylabel = "Explanatory Variables")
        return fig
    
    @output(id="logit_perm_imp_plot") # decorator to link this function to the "logit_perm_imp_plot" id in the UI
    @render.plot()
    def logit_perm_imp_plot():
        lr = logit_func()
        sns.set_style('whitegrid')
        fig = rsm.vimp_plot_sm(lr, df)
        # fig.set(title = "Odds-Ratios of Model 'lr'", xlabel = "Odds-Ratios", ylabel = "Explanatory Variables")
        return fig
    
    @output(id="code_snippet")
    @render.text
    @reactive.event(input.copy_to_clipboard, ignore_none=False)
    def code_snippet():
        """Generate Python code for logistic regression using the 'pyrsm' package."""
        resp_var = input.resp_var
        expl_var = input.expl_var
        code_template = """import pyrsm as rsm
lr = rsm.logistic_regression(dataset='{df_name}', rvar='{resp_var}', evars={expl_var})
lr.summary()
            """
        code_string = code_template.format(
            df_name  = df_name,
            resp_var=resp_var(),
            expl_var=expl_var()
            )
          
        try:
            return exec(code_string)
        #    return f'eval("""{code_string}""")'
        except Exception as e:
            return f"Error: {str(e)}"
        
        pyperclip.copy(code_string)
        return code_string

  return f

server = create_server(df)

# Connect everything
app = App(frontend, server)