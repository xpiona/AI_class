import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn import tree
import sys
import io


sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')


# wine = pd.read_csv('wine.csv')
wine = pd.read_csv('winequality-red.csv')

# 품질이 좋고 나쁜 것을 나누는 기준 설정
# 6.5를 기준으로 bad(0) good(1)으로 나눈다 (임의로 나눈 것임)
# my_bins = (2.5, 6.5, 8.5)
# groups = [0, 1]
# wine['qual'] = pd.cut(wine['quality'], bins = my_bins, labels = groups)
wine['qual'] = (wine['quality'] > 7) *1
print('qual의 value count : ', wine['qual'].value_counts())

X = wine.drop(['quality', 'qual'], axis = 1)
y = wine['qual']

q = (wine["quality"] > 7)*1

sc = StandardScaler()
X = sc.fit_transform(X)

np.random.seed(11)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

sgd = SGDClassifier()
sgd.fit(X_train, y_train)
print('sgd스코어 : ', sgd.score(X_test,y_test))

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
print('트리스코어 : ', clf.score(X_test,y_test))

rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train, y_train)
print('랜덤포레스트스코어 : ', rfc.score(X_test,y_test))

svc = SVC()
svc.fit(X_train, y_train)
print('svc스코어 : ', svc.score(X_test,y_test))

log = LogisticRegression()
log.fit(X_train, y_train)
print('lr스코어 : ', log.score(X_test,y_test))

# estimator = 모델, cv는 분할 테스트 숫자
rfc_eval = cross_val_score(rfc, X = X, y = y, cv = 5)
rfc_eval  # 5번의 교차 검증 결과를 보여준다

y_pred = sgd.predict(X_test)

print('confushion :', confusion_matrix(y_test, y_pred))
print('report : ', classification_report(y_test, y_pred))

y_score = sgd.decision_function(X_test)

result = pd.DataFrame(list(zip(y_score, y_pred, y_test)),
                      columns=['score', 'predict', 'real'])
result['correct'] = (result.predict == result.real)
print(result.head(5))

fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()
