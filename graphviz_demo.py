from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd

wine = load_wine()
df1 = pd.concat([pd.DataFrame(wine.data), pd.DataFrame(wine.target)], axis=1)
print(df1)
#查看特征
print(wine.feature_names)
#查看标签
print(wine.target_names)
#分割数据集
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)
print(x_train.shape)
print(wine.data.shape)
# 实例化classifier
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
print(score)

feature_name = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','花青素','颜 色强度','色调','od280/od315稀释葡萄酒','脯氨酸']
import graphviz 
dot_data = tree.export_graphviz(clf,out_file = None,feature_names= feature_name,class_names=["琴酒","雪莉","贝尔摩德"],filled=True,rounded=True) 
graph = graphviz.Source(dot_data)
graph
