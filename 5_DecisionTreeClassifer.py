import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree

Data=pd.read_csv(r"C:\Users\Admin\Downloads\diabetes - diabetes (2).csv")
Data.describe()
X=pd.DataFrame(Data,columns=['Glucose','BloodPressure','SkinThinkness','Insulin','BMP','DiabtesPedigreeFunction','Age'])

y=Data.Outcome.values.reshape(-1,1)
x_train,x_test,y_test,y_train=train_test_split(x,y,test_size=.3,random_state=1)
clf=DecisionTreeClassifier(max_depth=3)
clf=clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("Accuracy ",metrics.accuracy_score(y_test,y_pred))
feature_names=['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabatesIgreeFunction','Age']
target_names=['0','1']
fig=plt.figure(figsize=(23,20))
Plot=tree.plot_tree(clf.feature_names,class_names=target_names,filled=True)