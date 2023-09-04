import pandas as pd

df1 = pd.read_csv("./data/train.csv", sep="\t", encoding="utf-8")
df2 = pd.read_csv("./data/train1.csv", sep="\t", encoding="utf-8")
df3 = pd.read_csv("./data/train2.csv", sep="\t", encoding="utf-8")
# print(df1)
# print(df3)
print(len(df1))
print(len(df2))
print(len(df3))
# lst = df3["label"].tolist()
# print(lst)
# for i in range(len(lst)):
#     if lst[i] not in [0.0, 1.0, 2.0]:
#         print(lst[i])
#         print(i)

df = pd.concat([df1, df2, df3])
print(len(df))
df.to_csv("./data/three_sentiment_train.csv", sep="\t", encoding="utf-8", index=False)

import matplotlib.pyplot as plt
fig = plt.figure()
fig.add_subplot(111)

plt.figure(figsize=(12, 6))

import numpy as np
import seaborn as sns
import pandas as pd

data = pd.DataFrame()
np.random.normal()
data.hist()
plt.subplots()
data.plot(kind='box')
