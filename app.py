import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pickle
from imblearn.over_sampling import SMOTE
import os
import random
import logging
from flask import Flask, request, jsonify, render_template

# 配置日志记录
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

current_dir = os.getcwd()
logger.info(f"当前工作目录: {current_dir}")
new_dir = "/Users/wangning/Desktop/app dployment"
os.chdir(new_dir)
logger.info(f"当前工作目录: {os.getcwd()}")

# 读取数据
try:
    data = pd.read_csv('data.csv')
except FileNotFoundError:
    logger.error("错误: 未找到指定的 CSV 文件，请检查文件路径和文件名。")
    exit()

# 分离特征和目标变量
X = data.drop('Class Attribute', axis=1)
y = data['Class Attribute']

# 处理数据不平衡问题
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 创建 LightGBM 数据集
dtrain = lgb.Dataset(X_train, label=y_train)
dtest = lgb.Dataset(X_test, label=y_test)

# 定义模型参数范围
param_grid = {
    'num_leaves': [20, 31, 40],
    'learning_rate': [0.01, 0.05, 0.1],
    'lambda_l1': [0, 1, 5],
    'lambda_l2': [0, 1, 5]
}

# 创建模型
model = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', boosting_type='gbdt', verbose=-1)

# 超参数调优
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# 获取最优模型
best_model = grid_search.best_estimator_

# 预测概率（置信度）
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# 计算 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 打印患病的置信度及对应是否患病
logger.info("患病的置信度（前 10 个样本）、是否患病及对应样本的所有特征值：")
for i in range(min(10, len(y_pred_proba))):
    sample_features = X_test.iloc[i].to_dict()
    feature_str = ", ".join([f"{key}: {value}" for key, value in sample_features.items()])
    logger.info(
        f"样本 {i + 1}: 置信度 {y_pred_proba[i]:.6f}, 是否患病: {'是' if y_test.iloc[i] == 1 else '否'}, 特征值: {feature_str}")

# 保存模型
try:
    with open('lgb_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    logger.info("模型已成功保存为 lgb_model.pkl")
except Exception as e:
    logger.error(f"保存模型时出现错误: {e}")

# 加载模型
try:
    with open('lgb_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    logger.info("模型加载成功！")
except Exception as e:
    logger.error(f"加载模型时出现错误: {e}")


# 定义预测函数
def get_model(data, model, training_features):
    try:
        # 检查输入数据的特征是否完整
        for feature in training_features:
            if feature not in data:
                logger.error(f"输入数据缺少特征: {feature}")
                return "Unknown", None

        # 将传入的数据转换为 DataFrame
        input_df = pd.DataFrame([data])

        # 尝试将所有列转换为数值类型，无法转换的值设为 NaN
        for col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        # 检查是否有缺失值
        if input_df.isnull().any().any():
            logger.warning("输入数据包含无法转换为数值的值，已将其设为 NaN")
            # 这里可以根据需求对缺失值进行处理，比如填充均值、中位数等
            # 示例：使用均值填充缺失值
            input_df = input_df.fillna(input_df.mean())

        # 进行预测
        prediction = model.predict(input_df)
        # 获取预测的类别标签
        prediction_label = prediction[0]

        # 获取预测的置信度（概率）
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_df)
            confidence = probabilities[0][1]  # 取最大概率作为置信度
        else:
            confidence = None  # 如果模型不支持 predict_proba 方法，则置信度设为 None

        return prediction_label, confidence

    except Exception as e:
        logger.error(f"预测时出现错误: {e}", exc_info=True)
        return "Unknown", None


# 对整体数据中的随机样本进行预测
random_index = random.randint(0, len(data) - 1)
random_sample = data.iloc[random_index].drop('Class Attribute').to_dict()
true_label = data.iloc[random_index]['Class Attribute']

assert isinstance(loaded_model, object)
training_features = X.columns.tolist()
prediction_label, confidence = get_model(random_sample, loaded_model, training_features)

feature_str = ", ".join([f"{key}: {value}" for key, value in random_sample.items()])

logger.info(f"\n随机样本预测结果：")
logger.info(f"样本所有特征的值: {feature_str}")
logger.info(f"真实疾病类型: {'是' if true_label == 1 else '否'}")
logger.info(f"预测疾病类型: {'是' if prediction_label == 1 else '否'}")
logger.info(f"预测置信度: {confidence:.6f}")


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# 定义 Flask 路由
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "未提供有效的 JSON 数据"}), 400
        training_features = X.columns.tolist()
        logger.debug(f"接收到的数据: {data}")
        prediction_label, confidence = get_model(data, loaded_model, training_features)
        # 将 prediction_label 转换为 Python 内置的 int 类型
        if isinstance(prediction_label, np.int64):
            prediction_label = int(prediction_label)
        return jsonify({
            "disease_risk": prediction_label,
            "confidence": confidence
        })
    except Exception as e:
        logger.error(f"处理 /predict 请求时出现错误: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/result')
def result():
    disease_risk = request.args.get('disease_risk')
    confidence = request.args.get('confidence')
    return render_template('result.html', disease_risk=disease_risk, confidence=confidence)


if __name__ == '__main__':
    app.run(debug=True)
