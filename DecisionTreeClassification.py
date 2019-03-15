#常用的三個參考
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#讀取csv
dataset = pd.read_csv('Social_Network_Ads.csv')
#性別id不影響
X = dataset.iloc[:, 2:4].values#:-1 所有去除最後
Y = dataset.iloc[:, 4].values

#數據集 化分測試集 跟 訓練集
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.25,random_state =0)

#不用數據特徵縮放  因為算法裡面沒用到歐氏距離 但是為了圖解還是縮放一下
from sklearn.preprocessing import StandardScaler
sc_X =StandardScaler()
X_train =sc_X.fit_transform(X_train)
X_test =sc_X.transform(X_test)

#決策樹有不同的生成算法 cart(GINI)  id3=>c4.5、c5.0(資訊獲利entropy)   針對的資料也不近相同
#cart 用於離散與連續性資料

#回歸樹   連續的數據  年齡 溫度
#分類樹  分類離散的數據 性別 品種


#決策樹
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state =0)
#entropy 中文是 熵(ㄉ一) 念作 商 ，
#物理學:  熱力學的一種概念，不能做功的能量總數
#生態學:  薛丁格 :生物就是負熵的過程
#資訊理論的熵是指 資訊的平均量  可以想成 非秩序量 混亂量 不確定性之度量
#越乾淨的資料 資訊量越少 熵越小
#太陽從東邊出來 熵=0
#投擲硬幣  熵=1                      -0.5*(-1) - 0.5*(-1)
#Entropy = -p * log2 p – q * log2 q
#p：成功的機率（或true的機率） q：失敗的機率（或false的機率）

classifier = DecisionTreeClassifier(criterion='gini',random_state =0)
#criterion='gini' 吉尼係數 越平均 gini越低 最經典是配合收入曲線 用於顯示貧富差距
# 1-p^2 -q^2
#台灣的吉尼係數為 0.336   0.4以上就是過大
#gini速度較快 但就模型性能來說兩者區別不大

classifier.fit(X_train,Y_train)

#預測
Y_pred = classifier.predict(X_test)
#Making the Confusion Matrix 混郩矩陣 可以看正確率
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)

#視覺化
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('orange', 'blue'))(i), label = j)
plt.title('Decision Tree (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
#過度擬合




#visualising the test set result
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('orange', 'blue'))(i), label = j)
plt.title('Decision Tree (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#視覺化決策樹
from sklearn import tree
import graphviz
dot_data = tree.export_graphviz(classifier, out_file = None)
graph = graphviz.Source(dot_data) 
graph
#減少過度擬合 預先減枝
#每個結點(node)至少五個樣本 94%
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, min_samples_leaf = 5)
#深度最多4
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, max_depth = 4)
#知道怎麼種一顆樹 才能開始造森林