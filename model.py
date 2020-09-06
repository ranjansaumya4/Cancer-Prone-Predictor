import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle



dt = pd.read_csv('Data.csv')
X = dt.iloc[:, 1:-1].values
y = dt.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X, y , test_size=0.25, random_state=0)


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train= sc.fit_transform(X_train)
X_test= sc.transform(X_test)


from sklearn.tree import DecisionTreeClassifier
reg=DecisionTreeClassifier(criterion = 'entropy' , random_state=0)
reg.fit(X_train,y_train)

y_pred=reg.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)

accuracy_score(y_test,y_pred)

from sklearn.externals.joblib import dump, load

dump(sc, 'std_scaler.bin', compress=True)

if reg.predict(sc.transform([[4	,1	,1	,3	,2	,1	,3	,1	,1]])) == [[4]]:
    print('prone')
else:
    print('non_prone')
    
pickle.dump(reg, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

if model.predict(sc.transform([[4	,1	,1	,3	,2	,1	,3	,1	,1]])) == [[4]]:
    print('Prone')
else:
    print('Non Prone')