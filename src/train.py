# 导包
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error
from utils.common import get_data


# 定义函数，用于模型调优并保存模型
def model_train(x, y):
    # 切分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)
    # 模型训练
    # 创建XGBoost模型对象
    estimator = XGBRegressor()
    # 创建参数列表
    param_dict = {'max_depth': [3, 5, 6, 7], 'n_estimators': [50, 100, 150, 200],
                  'learning_rate': [0.01, 0.1, 0.2, 0.3]}
    # 网格搜索+交叉验证
    gs = GridSearchCV(estimator=estimator, param_grid=param_dict, cv=4)
    # 训练
    gs.fit(x_train, y_train)
    # 预测
    y_pre = gs.best_estimator_.predict(x_test)
    # 查看训练结果
    print(f'均方误差为:{mean_squared_error(y_test, y_pre)}')
    print(f'均方根误差为:{root_mean_squared_error(y_test, y_pre)}')
    print(f'平均绝对误差为:{mean_absolute_error(y_test, y_pre)}')
    # 模型保存
    joblib.dump(gs.best_estimator_, '../model/xgb.pkl')


if __name__ == '__main__':
    # 获取数据
    x, y = get_data('../data/train.csv')
    # 模型训练并调优
    model_train(x, y)
