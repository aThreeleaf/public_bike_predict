# 定义函数，用于获取并处理数据
import pandas as pd


def get_data(path):
    # 1.获取数据
    df = pd.read_csv(path)
    # 2.数据的预处理+特征工程
    # 筛选出特征列和标签列
    x = df.iloc[:, 1:6]
    y = df.iloc[:, -1]
    # 对weather列one-hot热编码
    x = pd.get_dummies(x, columns=['weather'])
    # 1代表晴天，2表示多云，3表示小雨，4表示大雨
    x.rename(columns={'weather_1': 'sunny', 'weather_2': 'cloudy', 'weather_3': 'rainy', 'weather_4': 'heavy_rainy'},
             inplace=True)
    # 对hour列热编码
    x = pd.get_dummies(x, columns=['hour'])
    return x, y
