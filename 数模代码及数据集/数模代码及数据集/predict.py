import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.font_manager as fm

# 读取数据
data = pd.read_csv('广州市交通指数.txt', parse_dates=['统计时间'], date_format='%Y年%m月')
data.set_index('统计时间', inplace=True)

# 分离数值列和非数值列
numeric_data = data[['交通指数']]
non_numeric_data = data[['拥堵级别']]

# 填补缺失值（使用线性插值）
numeric_data = numeric_data.resample('ME').mean().interpolate(method='linear')

# 如果有必要，填补非数值列的缺失值
non_numeric_data = non_numeric_data.resample('ME').ffill()

# 将数值列和非数值列重新合并
data = pd.concat([numeric_data, non_numeric_data], axis=1)

# 训练SARIMA模型
model = SARIMAX(data['交通指数'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# 预测2024年全年
forecast = results.get_forecast(steps=12)
forecast_index = pd.date_range(start='2024-01-01', periods=12, freq='M')
forecast_values = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# 设置matplotlib显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 绘制图像
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['交通指数'], label='历史数据')
plt.plot(forecast_index, forecast_values, label='预测数据', color='red')
plt.fill_between(forecast_index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='red', alpha=0.3)
plt.xlabel('时间')
plt.ylabel('交通指数')
plt.title('2024年交通指数预测')
plt.legend()
plt.grid(True)
plt.show()
