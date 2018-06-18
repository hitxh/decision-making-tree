
import pandas as pd

def read_dataset(fname):
    # 指定第一列作为行索引
    data = pd.read_csv(fname, index_col=0)
    # 丢弃无用的数据
    data.drop(['Name', 'Ticket',  'Cabin'], axis=1, inplace=True)
    # 处理性别数据
    data['Sex'] = (data['Sex'] == 'male').astype('int')
    # 处理登船港口数据
    labels = data['Embarked'].unique().tolist()
    data['Embarked'] =  data['Embarked'].apply(lambda n: labels.index(n))
    # 处理缺失数据
    data = data.fillna(0)
    return data

train = read_dataset(r'.\kaggle-titanic-master\input\train.csv')
# print(train)

from sklearn.model_selection import train_test_split

y = train['Survived'].values
X = train.drop(['Survived'], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print('train dataset:{0}; testdataset:{1}'.format(X_train.shape, X_test.shape))

# 使用决策树对其进行拟合
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print('train score:{0};test score: {1}'.format(train_score, test_score))
