import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv('train.csv')
df['Age']=df['Age'].fillna(df['Age'].mean())
y=df['Survived']
'''df=df.drop('Survived',axis=1,)
df=df.drop('SibSp',axis=1)
df=df.drop('Parch',axis=1)
df=df.drop('Name',axis=1)
df=df.drop('Ticket',axis=1)
df=df.drop('Fare',axis=1)
df=df.drop('Cabin',axis=1)'''
X=df[['PassengerId','Pclass','Sex','Age']]
X['Sex']=X['Sex'].map({'male':1,'female':0})
regresor=RandomForestClassifier(n_estimators=100)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33)
regresor.fit(X_train,y_train)
y_pred=regresor.predict(X_test)
print(regresor.predict([[124,2,0,32]]))