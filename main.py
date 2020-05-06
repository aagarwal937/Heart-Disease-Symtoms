import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn import metrics
from sklearn.decomposition import PCA
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/root/Desktop/heart.csv')
print(df.head())
print(df.tail())
print(df.describe())
print(df.isnull())
print(df.target.value_counts())
print(df.columns)

print(sns.distplot(df.age)


pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
#plt.savefig('heartDiseaseAndAges.png')
print(plt.show())


plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")
plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
print(plt.show())


plt.figure(figsize=(10,6))
print(sns.countplot(df.cp))


plt.figure(figsize=(10,6))
print(sns.distplot(df.chol))


corr = df.corr()
plt.figure(figsize=(15,6))
print(sns.heatmap(corr))


X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = df['target']

sc = StandardScaler()
Xscaled = sc.fit_transform(X)

scaled_df = pd.DataFrame(Xscaled,columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal'])
print(scaled_df.head())


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


from sklearn.ensemble import AdaBoostClassifier
LR = LogisticRegression()
NB = GaussianNB()
LR_boost = AdaBoostClassifier(base_estimator = LR,n_estimators = 170)
NB_boost = AdaBoostClassifier(base_estimator = NB,n_estimators = 350)
DT_boost = AdaBoostClassifier(n_estimators = 100)

NB_boost.fit(X_train,y_train)
ypred3= NB_boost.predict(X_test)

LR_boost.fit(X_train,y_train)
ypred2= LR_boost.predict(X_test)

DT_boost.fit(X_train,y_train)
ypred4= DT_boost.predict(X_test)

from sklearn.metrics import accuracy_score,auc,confusion_matrix,f1_score,mean_squared_error,roc_curve,precision_score,recall_score

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, ypred2)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, ypred2)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, ypred2)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, ypred2)
print('F1 score: %f' % f1)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, ypred3)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, ypred3)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, ypred3)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, ypred3)
print('F1 score: %f' % f1)

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, ypred4)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, ypred4)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, ypred4)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, ypred4)
print('F1 score: %f' % f1)
