import pandas as pd
import numpy as np
from datetime import datetime


def generate_disaster_matrix(input_csv, output_csv,
                              start_date='2012-01-01', end_date='2024-12-31',
                              county_col='County', code_col='countyCode',
                              date_col='Incident Begin Date'):

    # read data
    df = pd.read_csv(input_csv, parse_dates=[date_col])
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # date range
    all_dates = pd.date_range(start=start_date, end=end_date)

    # Get basic information for all counties
    unique_counties = df[[county_col, code_col]].drop_duplicates().reset_index(drop=True)
    date_matrix = pd.DataFrame(0, index=np.arange(len(unique_counties)), columns=all_dates)
    result_df = pd.concat([unique_counties, date_matrix], axis=1)

    # Mark disaster occurrence
    for idx, row in df.iterrows():
        county = row[county_col]
        code = row[code_col]
        date = row[date_col].date()
        match = (result_df[county_col] == county) & (result_df[code_col] == code)
        col = pd.Timestamp(date)
        if col in result_df.columns:
            result_df.loc[match, col] = 1

    # Sort and save the result
    result_df = result_df[[county_col, code_col] + sorted(all_dates)]
    result_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    generate_disaster_matrix(
        input_csv=r"E:\OneDrive - National University of Singapore\研二下\Flooding\reg_disaster.csv",
        output_csv=r"E:\OneDrive - National University of Singapore\研二下\Flooding\county_disaster_daily_matrix.csv",
        start_date='2012-01-01',
        end_date='2024-12-31',
        county_col='County',
        code_col='countyCode',
        date_col='Incident Begin Date'
    )