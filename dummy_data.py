import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 样本数据
sample_data = {
    'datetime': ['2024-01-09', '2024-01-09', '2024-01-09', '2024-01-09',
                 '2024-01-10', '2024-01-10', '2024-01-10', '2024-01-10',
                 '2024-01-11', '2024-01-11', '2024-01-11', '2024-01-11'],
    'instrument': ['AAPL', 'AMZN', 'GOOGL', 'MSFT',
                   'AAPL', 'AMZN', 'GOOGL', 'MSFT',
                   'AAPL', 'AMZN', 'GOOGL', 'MSFT'],
    'Open': [183.92, 148.33, 138.5, 372.01,
             184.35, 152.06, 141, 376.37,
             186.54, 155.04, 143.49, 386],
    'High': [185.15, 151.71, 141.49, 375.99,
             186.4, 154.42, 143, 384.17,
             187.05, 157.17, 145.22, 390.68],
    'Low': [182.73, 148.21, 138.15, 371.19,
            183.92, 151.88, 140.91, 376.32,
            183.62, 153.12, 140.64, 380.38],
    'Close': [185.14, 151.37, 140.95, 375.79,
              186.19, 153.73, 142.28, 382.77,
              185.59, 155.18, 142.08, 384.63],
    'Volume': [42841800, 43812600, 24759600, 20830000,
               46792900, 44421800, 21320200, 25514200,
               49128400, 49072700, 24008700, 27850800],
    'change': [-0.00226341, 0.0152246, 0.0151974, 0.00293578,
               0.0056714, 0.0155909, 0.00943598, 0.0185742,
               -0.00322255, 0.0094321, -0.00140566, 0.00485936],
    'label': [0.0056714, 0.0155909, 0.00943598, 0.0185742,
              -0.00322255, 0.0094321, -0.00140566, 0.00485936,
              0.00177812, -0.0036087, 0.00401177, 0.00998361]
}

# 转换为DataFrame
df = pd.DataFrame(sample_data)

# 扩展数据到2023年之前
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 1, 11)
delta = timedelta(days=1)

# 生成新的日期列表
date_list = []
while start_date <= end_date:
    date_list.append(start_date.strftime('%Y-%m-%d'))
    start_date += delta

# 生成新的数据
expanded_data = []
for date in date_list:
    for instrument in df['instrument'].unique():
        row = df[(df['datetime'] == '2024-01-09') & (df['instrument'] == instrument)].copy()
        row['datetime'] = date

        # 添加随机扰动
        row['Open'] += np.random.normal(0, 1)
        row['High'] += np.random.normal(0, 1)
        row['Low'] += np.random.normal(0, 1)
        row['Close'] += np.random.normal(0, 1)
        row['Volume'] += np.random.randint(-1000000, 1000000)
        row['change'] += np.random.normal(0, 0.001)
        row['label'] += np.random.normal(0, 0.001)

        expanded_data.append(row)

# 合并生成的数据
expanded_df = pd.concat(expanded_data, ignore_index=True)

# 保存为pkl文件
data_dir = './data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

expanded_df.to_pickle(os.path.join(data_dir, "usdataset.pkl"))

print("usdataset.pkl 文件已生成！")
