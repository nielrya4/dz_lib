import pandas as pd
def dataframe_to_html(df: pd.DataFrame):
    return (
        df.to_html(
            classes="table table-bordered table-striped",
            justify="center"
        ).replace(
            '<th>',
            '<th style="background-color: White;">'
        ).replace(
            '<td>',
            '<td style="background-color: White;">'
        )
    )
