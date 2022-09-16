import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(file_name):
    df = pd.read_csv(f"./data/raw/{file_name}")
    df.date = df.date.astype("datetime64[ns]")
    return df.set_index(df.columns[0])


def rescale_weekly_data(df_monthly, df_weekly):
    years_to_consider = [2018, 2019, 2020, 2021, 2022]
    weekly_dates = df_weekly.date
    list_rescaled_weekly_values = []
    for yr in years_to_consider:
        weekly_for_current_year = df_weekly.loc[weekly_dates.dt.year == yr]
        # Since monthly data are only on the first of each month
        relevant_timestamps = weekly_for_current_year.loc[
            weekly_dates.dt.day == 1
        ].index
        relevant_monthly_values = df_monthly.loc[
            relevant_timestamps, "value_month"
        ].values
        relevant_weekly_values = df_weekly.loc[
            relevant_timestamps, "value_week"
        ].values
        ave_scale_factor = np.mean(
            relevant_monthly_values/relevant_weekly_values
        )
        list_rescaled_weekly_values.append(
            ave_scale_factor * weekly_for_current_year.value_week
        )
    return pd.concat(list_rescaled_weekly_values)


def rescale_hourly_data(df_rescaled_weekly, df_hourly):
    hourly_times = df_hourly.date
    relevant_hourly_times = hourly_times.loc[df_rescaled_weekly.index]
    years = relevant_hourly_times.dt.isocalendar().year
    weeks = relevant_hourly_times.dt.isocalendar().week
    scale_factors = (
            df_rescaled_weekly /
            df_hourly.loc[df_rescaled_weekly.index, "value_hour"]
    )
    list_rescaled_hourly_values = []
    for yr, wk, scale in zip(years, weeks, scale_factors):
        hourly_for_current_week = hourly_times.loc[
            (hourly_times.dt.isocalendar().year == yr) &
            (hourly_times.dt.isocalendar().week == wk)
        ].index
        list_rescaled_hourly_values.append(
            scale * df_hourly.loc[hourly_for_current_week, "value_hour"]
        )
    return pd.concat(list_rescaled_hourly_values)


def include_datetime_index(df_rescaled, datetimes):
    df = pd.concat([
        df_rescaled, datetimes.loc[df_rescaled.index]
    ], axis=1).set_index("date")
    return df


def plot_time_series(df_monthly, df_weekly, df_hourly, fname, ylim=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))

    df_hourly.plot(ax=ax, style='b-', alpha=0.1)
    df_weekly.plot(ax=ax, style='go-', alpha=0.4)
    df_monthly.plot(ax=ax, style='ro')

    if ylim:
        ax.set_ylim(0, ylim)

    ax.set_title(fname)
    plt.tight_layout()
    fig.savefig(f"./data/result/{fname}.png")


if __name__ == "__main__":
    df_monthly_entire_range = load_data('monthly_data.csv')
    df_weekly_each_year = load_data('weekly_data.csv')
    df_hourly_each_week = load_data('hourly_data.csv')

    # Rescale
    rescaled_weekly_values = rescale_weekly_data(
        df_monthly_entire_range, df_weekly_each_year
    )
    rescaled_hourly_values = rescale_hourly_data(
        rescaled_weekly_values, df_hourly_each_week
    )

    # Plotting and saving results
    rescaled_hourly = include_datetime_index(
        rescaled_hourly_values, df_hourly_each_week["date"]
    )
    rescaled_weekly = include_datetime_index(
        rescaled_weekly_values, df_weekly_each_year["date"]
    )

    rescaled_hourly.to_csv("./data/result/rescaled_hourly_data.csv")

    raw_monthly = df_monthly_entire_range.set_index('date').loc['2018':]
    raw_weekly = df_weekly_each_year.set_index('date').loc['2018':]
    raw_hourly = df_hourly_each_week.set_index('date').loc['2018':]

    plot_time_series(
        raw_monthly, raw_weekly, raw_hourly, "Raw_Data", ylim=150
    )

    plot_time_series(
        raw_monthly, rescaled_weekly, rescaled_hourly,
        "Rescaled_Data_Zoomed", ylim=150
    )

    plot_time_series(
        raw_monthly, rescaled_weekly, rescaled_hourly,
        "Rescaled_Data", ylim=None
    )
