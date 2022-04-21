import random
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from datetime import datetime


def import_ex_plates_schedule(filepath):
    df_schedule = pd.read_csv(filepath, encoding='euc-kr')
    plates = []
    for i, row in df_schedule.iterrows():
        plate = Plate(row['plate_id'], row['inbound_date'], row['outbound_date'])
        plates.append(plate)
    return plates


def import_plates_schedule(filepath, graph=False):
    df_schedule = pd.read_csv(filepath, encoding='euc-kr')
    df_schedule.dropna(subset=['자재번호', '최근입고일', '블록S/C일자'], inplace=True)
    df_schedule['최근입고일'] = pd.to_datetime(df_schedule['최근입고일'], format='%Y.%m.%d')
    df_schedule['블록S/C일자'] = pd.to_datetime(df_schedule['블록S/C일자'], format='%Y.%m.%d')
    df_schedule = df_schedule[df_schedule['최근입고일'] >= datetime(2019, 1, 1)]
    df_schedule = df_schedule[df_schedule['최근입고일'] <= df_schedule['블록S/C일자']]
    initial_date = df_schedule['최근입고일'].min()
    df_schedule['최근입고일'] = (df_schedule['최근입고일'] - initial_date).dt.days
    df_schedule['블록S/C일자'] = (df_schedule['블록S/C일자'] - initial_date).dt.days
    df_schedule.sort_values(by=['최근입고일'], inplace=True)
    df_schedule.reset_index(drop=True, inplace=True)

    if graph:
        inter_arrival_time = (df_schedule['최근입고일'].diff()).dropna()
        stock_time = (df_schedule['블록S/C일자'] - df_schedule['최근입고일'])[df_schedule['블록S/C일자'] >= df_schedule['최근입고일']]
        #loc_expon, scale_expon = stats.expon.fit(inter_arrival_time)
        #a, b, loc_beta, scale_beta = stats.beta.fit(stock_time)
        fig, ax = plt.subplots(2, 1, squeeze=False)
        ax[0][0].set_title('Inter Arrival Time'); ax[0][0].set_xlabel('time'); ax[0][0].set_ylabel('normalized frequency of occurrence')
        ax[1][0].set_title('Stock Time'); ax[1][0].set_xlabel('time'); ax[1][0].set_ylabel('normalized frequency of occurrence')
        ax[0][0].hist(list(inter_arrival_time), bins=100)
        ax[1][0].hist(list(stock_time), bins=100)
        plt.tight_layout()
        plt.show()

    plates = [[]]
    for i, row in df_schedule.iterrows():
        plate = Plate(row['자재번호'], row['최근입고일'], row['블록S/C일자'])
        plates[0].append(plate)
    return plates


def generate_schedule(num_plate=100, graph=False):
    inter_arrival_time = np.floor(stats.expon.rvs(loc=0.0, scale=0.273, size=num_plate))
    stock_time = np.floor(stats.beta.rvs(1.85, 32783.4, loc=2.52, scale=738938.8, size=num_plate))
    #inter_arrival_time = [0 for _ in range(num_plate)]
    #stock_time = np.floor(stats.beta.rvs(4.39, 0.227, loc=0.608, scale=6.39, size=num_plate)) #week
    #stock_time = np.sort(stats.uniform.rvs(loc=0.0, scale=100.0, size=num_plate))[::-1]
    current_date = 0
    plates = [[]]
    for i in range(num_plate):
        plate_id = 'plate' + str(i)
        inbound_date = current_date + inter_arrival_time[i]
        outbound_date = inbound_date if stock_time[i] < 0.0 else inbound_date + stock_time[i]
        current_date = inbound_date
        plate = Plate(plate_id, inbound_date, outbound_date)
        plates[0].append(plate)

    if graph:
        fig, ax = plt.subplots(2, 1, squeeze=False)
        ax[0][0].set_title('Inter Arrival Time'); ax[0][0].set_xlabel('time[day]'); ax[0][0].set_ylabel('occurrence')
        ax[1][0].set_title('Stock Time'); ax[1][0].set_xlabel('time[day]'); ax[1][0].set_ylabel('occurrence')
        ax[0][0].hist(list(inter_arrival_time), bins=100)
        ax[1][0].hist(list(stock_time), bins=100)

        plt.tight_layout()
        plt.show()

    return plates


def import_plates_schedule_by_week(filepath, graph=False):
    df_schedule = pd.read_csv(filepath, encoding='euc-kr')
    df_schedule.dropna(subset=['자재번호', '최근입고일', '블록S/C일자'], inplace=True)
    df_schedule['최근입고일'] = pd.to_datetime(df_schedule['최근입고일'], format='%Y.%m.%d')
    df_schedule['블록S/C일자'] = pd.to_datetime(df_schedule['블록S/C일자'], format='%Y.%m.%d')
    df_schedule = df_schedule[df_schedule['최근입고일'] >= datetime(2019, 1, 1)]
    df_schedule = df_schedule[df_schedule['최근입고일'] <= df_schedule['블록S/C일자']]
    initial_date = df_schedule['최근입고일'].min()
    df_schedule['최근입고일'] = (df_schedule['최근입고일'] - initial_date).dt.days
    df_schedule['블록S/C일자'] = (df_schedule['블록S/C일자'] - initial_date).dt.days
    df_schedule.sort_values(by=['블록S/C일자'], inplace=True)
    df_schedule.reset_index(drop=True, inplace=True)

    plates = []
    day = df_schedule['블록S/C일자'].min()
    while len(df_schedule) != 0:
        plates_by_week = []
        temp = df_schedule[df_schedule['블록S/C일자'] <= day]
        temp.sort_values(by=['최근입고일'], inplace=True)
        temp.reset_index(drop=True, inplace=True)
        steel_num = len(temp)

        if steel_num > 0:
            for i, row in temp.iterrows():
                # plate = Plate(row['자재번호'], row['최근입고일'], row['블록S/C일자'])
                plate = Plate(row['자재번호'], day - 7, row['블록S/C일자'])  # 주간 물량의 입고 기준일
                plates_by_week.append(plate)
            plates.append(plates_by_week)
            df_schedule.drop([_ for _ in range(steel_num)], inplace=True)
            df_schedule.reset_index(drop=True, inplace=True)
        random.shuffle(plates_by_week)
        day += 7

        if graph:
            stock_time = [plate.outbound - plate.inbound for plate in plates_by_week]
            #a, b, loc_beta, scale_beta = stats.beta.fit(stock_time)
            fig, ax = plt.subplots(1, 1, squeeze=False)
            ax[0][0].set_title('Stock Time')
            ax[0][0].set_xlabel('time')
            ax[0][0].set_ylabel('normalized frequency of occurrence')
            ax[0][0].hist(list(stock_time), bins=100)
            plt.tight_layout()
            plt.show()

    return plates


def import_plates_schedule_by_day(filepath):
    df_schedule = pd.read_excel(filepath, header=[0, 1], encoding='euc-kr')
    columns = map(lambda x:x[0].replace('\n','') if 'Unnamed' in x[1] else x[0]+'_'+x[1], df_schedule.columns)
    df_schedule.columns = columns
    df_schedule.dropna(subset=['자재번호'], inplace=True)
    df_schedule['불출요구일'] = pd.to_datetime(df_schedule['불출요구일'], format='%Y.%m.%d')
    initial_date = df_schedule['불출요구일'].min()
    df_schedule['불출요구일'] = (df_schedule['불출요구일'] - initial_date).dt.days
    df_schedule.reset_index(drop=True, inplace=True)

    plates = []
    for (date, yard), group in df_schedule.groupby(['불출요구일', '적치장']):
        group.reset_index(drop=True, inplace=True)
        plates_by_day = []

        priority = 1
        while len(group) != 0:
            temp = group[group['절단장비'] == group.iloc[0]['절단장비']]
            steel_num = len(temp)
            for i, row in temp.iterrows():
                plate = Plate(row['자재번호'], date, date + priority)
                plates_by_day.append(plate)
            group.drop([_ for _ in range(steel_num)], inplace=True)
            group.reset_index(drop=True, inplace=True)
            priority += 1

        plates.append(plates_by_day)

    return plates


# 강재 정보 클래스 id, 입출고일정 포함
class Plate(object):
    def __init__(self, plate_id=None, inbound=0, outbound=1):
        self.id = str(plate_id)
        self.inbound = inbound
        self.outbound = outbound
        if outbound == -1:  # 강재 데이터가 없으면 임의로 출고일 생성
            self.outbound = random.randint(1, 5)


if __name__ == "__main__":
    import os
    np.random.seed(42)

    num_plate = [100, 120, 140, 160, 180, 200]
    num_instance = 1

    data_path = '../benchmark/data/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    for num_p in num_plate:
        for num_i in range(num_instance):
            plates = generate_schedule(num_plate=num_p)
            data = pd.DataFrame(columns=["plate_id", "inbound", "outbound"])
            for i, plate in enumerate(plates[0]):
                data.loc[i] = [plate.id, plate.inbound, plate.outbound]
            data.to_csv(data_path + 'data_plate{0}_{1}.csv'.format(num_p, num_i), index=False)