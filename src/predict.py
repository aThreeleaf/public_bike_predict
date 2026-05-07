import joblib
from utils.common import get_data

if __name__ == '__main__':
    # 获取数据
    x_test, y_test = get_data('../data/test.csv')
    # 加载模型
    estimator = joblib.load('../model/xgb.pkl')
    # 预测
    y_pre = estimator.predict(x_test)
    print(f'预测结果为:{y_pre}')
