# -*- coding: UTF-8 -*-
__author__ = 'lixiaobo'

import pandas as pd
import numpy as np
import json
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math

# 绘图配置
row_size = 2
col_size = 3
cell_size = row_size * col_size

def count(dataFrame, columns, dropna=False, format_width=40):
    '标称属性，给出每个可能取值的频数'
    format_text = '{{:<{0}}}{{:<{0}}}'.format(format_width)
    for col in columns:
        print('标称属性 <{}> 频数统计'.format(col))
        print(format_text.format('value', 'count'))
        print('- ' * format_width)

        counts = pd.value_counts(dataFrame[col].values, dropna=False)
        for i, index in enumerate(counts.index):
            if pd.isnull(index): # NaN?
                print(format_text.format('-NaN-', counts.values[i]))
            else:
                print(format_text.format(index, counts[index]))
        print('--' * format_width)
        print()

def describe(dataFrame, columns):
    '数值属性，给出最大、最小、均值、中位数、四分位数及缺失值的个数'
    desc = dataFrame[columns].describe()
    statistic = DataFrame()
    statistic['max'] = desc.loc['max']   #最大值
    statistic['min'] = desc.loc['min']   #最小值
    statistic['mean'] = desc.loc['mean'] #均值
    statistic['50%'] = desc.loc['50%']    #中位数
    statistic['25%'] = desc.loc['25%']    #四分位数
    statistic['75%'] = desc.loc['75%']
    statistic['NaN'] = dataFrame[columns].isnull().sum()#缺失值
    print(statistic)

def histogram(dataFrame, columns):
    '直方图'
    for i, col in enumerate(columns):
        if i % cell_size == 0:
            fig = plt.figure()
        ax = fig.add_subplot(col_size, row_size, (i % cell_size) + 1)
        dataFrame[col].hist(ax=ax, grid=False, figsize=(15, 15), bins=50)
        plt.title(col)
        if (i + 1) % cell_size == 0 or i + 1 == len(columns):
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            plt.show()


def qqplot(dataFrame, columns):
    'qq图'
    for i, col in enumerate(columns):
        if i % cell_size == 0:
            fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(col_size, row_size, (i % cell_size) + 1)
        sm.qqplot(dataFrame[col], ax=ax)
        ax.set_title(col)
        if (i + 1) % cell_size == 0 or i + 1 == len(columns):
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            plt.show()


def boxplot(dataFrame, columns):
    '盒图'
    for i, col in enumerate(columns):
        if i % cell_size == 0:
            fig = plt.figure()
        ax = fig.add_subplot(col_size, row_size, (i % cell_size) + 1)
        dataFrame[col].plot.box(ax=ax, figsize=(15, 15))
        if (i + 1) % cell_size == 0 or i + 1 == len(columns):
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            plt.show()

def compare(df1, df2, columns, bins=50):
    '直方图比较'
    for col in columns:
        mean1 = df1[col].mean()
        mean2 = df2[col].mean()

        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        df1[col].hist(ax=ax1, grid=False, figsize=(15, 5), bins=bins)
        ax1.axvline(mean1, color='r')
        plt.title('origin\n{}\nmean={}'.format(col, str(mean1)))
        ax2 = fig.add_subplot(122)
        df2[col].hist(ax=ax2, grid=False, figsize=(15, 5), bins=bins)
        ax2.axvline(mean2, color='b')
        plt.title('filled\n{}\nmean={}'.format(col, str(mean2)))
        plt.subplots_adjust(wspace = 0.3, hspace = 10)
        plt.show()


if __name__=="__main__":
    #打开excel获取数据
    file=('NFL Play by Play 2009-2017 (v4).csv')
    df=pd.read_csv(file,low_memory=False)
    #标称属性
    name_category = ['Drive', 'qtr', 'down', 'SideofField', 'ydstogo', 'GoalToGo', 'FirstDown', 'posteam',
                 'DefensiveTeam', 'PlayAttempted', 'sp', 'Touchdown', 'ExPointResult', 'TwoPointConv',
                 'DefTwoPoint', 'Safety', 'Onsidekick', 'PuntResult', 'PlayType', 'Passer', 'Passer_ID',
                 'PassAttempt', 'PassOutcome', 'PassLength', 'QBHit', 'PassLocation', 'InterceptionThrown',
                 'Interceptor', 'Rusher', 'Rusher_ID', 'RushAttempt', 'RunLocation', 'RunGap', 'Receiver',
                 'Receiver_ID', 'Reception', 'ReturnResult', 'Returner', 'BlockingPlayer', 'Tackler1', 'Tackler2',
                 'FieldGoalResult', 'Fumble', 'RecFumbTeam', 'RecFumbPlayer', 'Sack', 'Challenge.Replay',
                 'ChalReplayResult', 'Accepted.Penalty', 'PenalizedTeam', 'PenaltyType', 'PenalizedPlayer',
                 'HomeTeam', 'AwayTeam', 'Timeout_Indicator', 'Timeout_Team', 'Season']
    # 数值属性
    name_value =  ['TimeUnder', 'TimeSecs', 'PlayTimeDiff', 'yrdln', 'yrdline100', 'ydsnet', 'Yards.Gained',
              'AirYards', 'YardsAfterCatch', 'FieldGoalDistance', 'Penalty.Yards', 'PosTeamScore', 'DefTeamScore',
              'ScoreDiff', 'AbsScoreDiff', 'posteam_timeouts_pre', 'HomeTimeouts_Remaining_Pre', 'AwayTimeouts_Remaining_Pre',
              'HomeTimeouts_Remaining_Post', 'AwayTimeouts_Remaining_Post', 'No_Score_Prob', 'Opp_Field_Goal_Prob',
              'Opp_Safety_Prob', 'Opp_Touchdown_Prob', 'Field_Goal_Prob', 'Safety_Prob', 'Touchdown_Prob', 'ExPoint_Prob',
              'TwoPoint_Prob', 'ExpPts', 'EPA', 'airEPA', 'yacEPA', 'Home_WP_pre', 'Away_WP_pre', 'Home_WP_post',
              'Away_WP_post', 'Win_Prob', 'WPA', 'airWPA', 'yacWPA']

   #标称属性统计频数
count(df,name_category)

#数值属性统计最小、最大、均值、中位数、四分位数及缺失值个数
describe(df,name_value)

#绘制直方图
histogram(df,name_value)

#绘制qq图
qqplot(df,name_value)

#绘制盒图
boxplot(df,name_value)

#可填充属性
cols = ['No_Score_Prob', 'Opp_Field_Goal_Prob', 'Opp_Safety_Prob', 'Opp_Touchdown_Prob', 'Field_Goal_Prob', 
        'Safety_Prob', 'Touchdown_Prob', 'ExpPts', 'EPA', 'airEPA', 'yacEPA', 'Home_WP_pre', 'Away_WP_pre', 
        'Home_WP_post', 'Away_WP_post', 'Win_Prob', 'WPA', 'airWPA', 'yacWPA']
#将缺失值剔除并进行直方图比较
index = df[cols].isnull().sum(axis=1) == 0
df_fillna = df[index]

compare(df,df_fillna,cols)

#用最高频率值来填补缺失值并进行直方图比较
df_filled = df.copy()
for col in cols:
    # 计算最高频率的值
    most_frequent_value = df_filled[col].value_counts().idxmax()
    # 替换缺失值\n",
    df_filled[col].fillna(value=most_frequent_value, inplace=True)

compare(df,df_filled,cols)

# 通过属性的相关关系来填补缺失值并进行直方图比较
df_filled_inter = df.copy()
# 对每一列数据，分别进行处理
for col in cols:
    df_filled_inter[col].interpolate(inplace=True)

    compare(df,df_filled_inter,cols)

