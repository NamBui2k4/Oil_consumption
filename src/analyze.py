import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df0 = pd.read_csv('data/Oil Consumption by Country 1965 to 2023.csv')

# normalize
df0.index = df0.iloc[:,0]
df = df0.T[1:]
# with open('observation/data.txt', 'w', encoding='utf-8') as f:
#     f.write(df.to_string())

# # statistic
# stat = df.describe().T
# with open('observation/stastistic.txt', 'w', encoding='utf-8') as f:
#     f.write(stat.to_string())

# stat.iloc[0:5].plot(kind='bar',rot=True,title='Oil consumption by sampled countries'),
# plt.savefig('observation/Visualize.png')

# # procssing missing data
# check_nan = df.isnull()
# ax = plt.figure(figsize=(12,5))
# ax = sns.heatmap(check_nan.T)
# ax.set_label('NaN observations')
# plt.savefig('observation/check_nan.png')


# add time series
df['time'] = pd.to_datetime(df.index)
cols = ['time'] + [col for col in df.columns if col != 'time']
df = df[cols]

# analyze the Iraq consumption
target = 'Iraq'
Iraq_df = df['Iraq'].fillna(0)
time = df['time']

fig, ax = plt.subplots()
ax.plot(time,Iraq_df)
ax.set_xlabel('time')
ax.set_ylabel('oil consumption')
ax.set_title('Trending of oil consumption in Iraq')
plt.savefig('observation/Africa_trend.png')


