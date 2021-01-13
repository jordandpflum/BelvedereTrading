from Data_Master.read_data import *
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import itertools
import subprocess
import os
from shutil import copyfile
import time
from pathlib import Path
from isoweek import Week

def generate_polar_vis(df, futures, feature="correlations", time_interval="weekly", connect_first_and_last=False, save_cat=None):
    """
    :param df: relevant data frame to use (x_df, y_df, or merged_df)
    :param futures: list of futures of interest. Only supply one (for single futures analysis) or two (for pairs).
    :param feature: desired feature to explore
    :param time_interval: time period of interest (options: weekly, biweekly, monthly, seasonal)
    :param connect_first_and_last: connect the first point in the polar plot to the last point (not necessarily valid data between these points, but appearance is cleaner)
    :param save_cat: directory name (category) to save the visualization to in ../Visualizations
    :return: displays or saves a polar plotly plot for desired feature
    """
    years = [2015, 2016, 2017]
    years_data = {2015: {}, 2016: {}, 2017: {}}

    pairs = []
    for future_name1 in futures:
        for future_name2 in futures:
            if (future_name1 + '-' + future_name2 not in pairs and future_name2 + '-' + future_name1 not in pairs and future_name1 != future_name2):
                pairs.append(future_name1 + '-' + future_name2)

    # used for individual future exploration
    future_data = {}
    for future in futures:
        future_data[future] = {}
        for year in years:
            future_data[future][year] = {}

    # used for pair explorations
    pairs_data = {}
    for pair in pairs:
        pairs_data[pair] = {}
        for year in years:
            pairs_data[pair][year] = {}

    # we need to get all the start and end dates for the time interval of interest
    if(time_interval=='weekly'):
        weeks = []
        for filename in os.listdir('../Belvedere_Spr20/Data_Master/correlations/cor_res_weekly'):
            if (filename != '.DS_Store'):
                a, b, c, d, e = filename.split('_')
                week_number, f = e.split('.')
                weeks.append(week_number)
        sorted_weeks = sorted(weeks)

        for week_number in sorted_weeks:
            year,week = week_number.split('-')
            # d = year + '-W' + week
            # start_date = datetime.datetime.strptime(d + '-1', "%Y-W%W-%w")
            w = Week(int(year), int(week))
            start_date = w.monday()
            end_date = start_date + datetime.timedelta(days=4)
            if(int(year) in years):
                years_data[int(year)]["W" + week.lstrip("0")] = {'start_date': start_date, 'end_date': end_date}
    elif(time_interval == 'biweekly'):
        weeks = []

        for filename in os.listdir('../Belvedere_Spr20/Data_Master/correlations/cor_res_biweek/'):
            if filename != '.DS_Store':
                a, b, c, d, e = filename.split('_')
                week_number, f = e.split('.')
                weeks.append(week_number)
        sorted_weeks = sorted(weeks)

        for week_number in sorted_weeks:
            first_week_number = week_number[0:7]
            second_week_number = week_number[7:14]
            first_week_number_year, first_week_number_week, = first_week_number.split('-')
            second_week_number_year, second_week_number_week = second_week_number.split('-')
            if(first_week_number_week.lstrip("0") in [str(i) for i in range(1, 53, 2)]):
                year, week = first_week_number_year, first_week_number_week
            else:
                year, week = second_week_number_year, second_week_number_week

            if (int(year) in years):
                # d = year + '-W' + week
                # start_date = datetime.datetime.strptime(d + '-1', "%Y-W%W-%w")
                w = Week(int(year), int(week))
                start_date = w.monday()
                end_date = start_date + datetime.timedelta(days=11)
                years_data[int(year)]["W" + week.lstrip("0")] = {'start_date': start_date, 'end_date': end_date}

    elif(time_interval == 'monthly'):
        # get all the month files and sort the dates
        months = []

        for filename in os.listdir('../Belvedere_Spr20/Data_Master/correlations/cor_res_month/'):
            if filename != '.DS_Store':
                a, b, c, d, e = filename.split('_')
                month, f = e.split('.')
                months.append(month)
        sorted_months = sorted(months)

        row_count = 0
        for year_month in sorted_months:
            year, month = year_month.split('-')
            year = int(year)
            month = int(month.lstrip("0"))
            if (int(year) in [2015, 2016, 2017]):
                a, num_days = calendar.monthrange(year, month)
                start_date = datetime.date(year, month, 1)
                end_date = datetime.date(year, month, num_days)
                years_data[int(year)]["M" + str(month)] = {'start_date': start_date, 'end_date': end_date}

    elif (time_interval == 'seasonal'):
        # get all the seasonal files and sort the dates
        quarters = []

        for filename in os.listdir('../Belvedere_Spr20/Data_Master/correlations/cor_res_season/'):
            if filename != '.DS_Store':
                a, b, c, d, e = filename.split('_')
                quarter, f = e.split('.')
                quarters.append(quarter)
        sorted_quarters = sorted(quarters)

        for quarter in sorted_quarters:
            first_year_month = quarter[0:7]
            year, month = first_year_month.split('-')
            year = int(year)
            month = int(month.lstrip("0"))
            second_month,third_month = month + 1, month + 2
            if (int(year) in years):
                a, num_days_first = calendar.monthrange(year, month)
                a, num_days_second = calendar.monthrange(year, second_month)
                a, num_days_third = calendar.monthrange(year, third_month)
                total_num_days = num_days_first + num_days_second + num_days_third
                start_date = datetime.date(year, month, 1)
                end_date = datetime.date(year, third_month, num_days_third)
                if(month == 1):
                    quarter = "Q1"
                elif(month == 4):
                    quarter = "Q2"
                elif(month == 7):
                    quarter = "Q3"
                elif(month == 10):
                    quarter = "Q4"
                years_data[int(year)][quarter] = {'start_date': start_date, 'end_date': end_date}

    # for each period in each year, we get the first entry in that period and grab the feature value
    for year,vals in years_data.items():
        for interval,dates in vals.items():
            mask = (df['Timestamp'] >= pd.Timestamp(dates['start_date'])) & (df['Timestamp'] <= pd.Timestamp(dates['end_date']))
            sub = df.loc[mask].iloc[0]
            if (feature == 'correlations'):
                index = '_corr_'
            elif (feature == 'correlation_differences'):
                index = '_corr_diff_'

            for pair in pairs:
                pairs_data[pair][year][interval] = sub[time_interval + index + pair]

            # elif(feature=='open_price'):
            #     for future in futures:
            #         future_data[future][year][interval] = sub[future + '_OPEN']

    # create the visualization (single futures)
    if(feature == 'open_price'):
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'polar'}] * 1] * 1)
        for future in futures:
            for year in years:
                weeks = []
                prices = []
                for week,price in future_data[future][year].items():
                    prices.append(price)
                    weeks.append(week)

                fig.add_trace(go.Scatterpolar(
                    name = 'Open Price ' + future + ' ' + str(year),
                    r=prices,
                    theta=weeks,
                ), 1, 1)

            plt_title = 'Open Prices ' + future_to_name(future)

    # create the visualization (futures pairs)
    if(feature in ['correlations', 'correlation_differences']):
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'polar'}] * 1] * 1)
        for pair in pairs:
            for year in years:
                intervals = []
                corrs = []
                for interval,corr in pairs_data[pair][year].items():
                    intervals.append(interval)
                    corrs.append(corr)
                if(connect_first_and_last):
                    first_key = list(pairs_data[pair][year].keys())[0]
                    intervals.append(first_key)
                    corrs.append(pairs_data[pair][year][first_key])

                if(feature == 'correlations'):
                    fig.add_trace(go.Scatterpolar(
                        name = pair + ' ' + str(year) + ' Correlation',
                        r=corrs,
                        theta=intervals,
                    ))

                    plt_title = ' Correlations: ' + pair_to_names(pair)
                elif(feature == 'correlation_differences'):
                    fig.add_trace(go.Scatterpolar(
                        name=pair + ' ' + str(year) + ' Correlation Difference',
                        r=corrs,
                        theta=intervals,
                    ))

                    plt_title = ' Correlation Differences: ' + pair_to_names(pair)

    # update the layout depending on time interval
    if(time_interval=='weekly'):
        fig.update_layout(
            title='Weekly' + plt_title,
            polar=dict(
                angularaxis_categoryarray=["W" + str(i) for i in range(1,53)]
            )
        )
    elif(time_interval == 'biweekly'):
        fig.update_layout(
            title='Biweekly' + plt_title,
            polar=dict(
                angularaxis_categoryarray=["W" + str(i) for i in range(1, 53, 2)]
            )
        )
    elif(time_interval == 'monthly'):
        fig.update_layout(
            title='Monthly' + plt_title,
            polar=dict(
                angularaxis_categoryarray=["M" + str(i) for i in range(1, 13)]
            )
        )
    elif (time_interval == 'seasonal'):
        fig.update_layout(
            title='Seasonal' + plt_title,
            polar=dict(
                angularaxis_categoryarray=["Q" + str(i) for i in range(1, 5)]
            )
        )
    # if save option supplied, save to correct directory name, and otherwise just display it
    if(save_cat == None):
        fig.show()
    else:
        if(feature == 'correlations'):
            name = pair
            short = 'corrs'
        elif(feature == 'correlation_differences'):
            name = pair
            short = 'corr-diffs'
        img_name = name + '-' + time_interval + '-' + short
        dload = os.path.expanduser('~/Downloads')
        save_dir = '../Belvedere_Spr20/Visualizations/' + save_cat + '/' + pair + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plotly.offline.plot(fig, image_filename=img_name, image='png')

        saved = False
        while(saved == False):
            try:
                copyfile('{}/{}.png'.format(dload, img_name), '{}/{}.png'.format(save_dir, img_name))
                saved = True
            except:
                print('here')
                time.sleep(1)
