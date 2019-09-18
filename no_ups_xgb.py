import numpy as np
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.drop('Username',axis=1,inplace=True)
test.drop('Username',axis=1,inplace=True)

test_labels = test.iloc[:,0].values
test.drop('ID',axis=1,inplace=True)
train.drop('ID',axis=1,inplace=True)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
sc.fit(train.iloc[:,1:4])
train.iloc[:,1:4] = sc.transform(train.iloc[:,1:4])
test.iloc[:,1:4] = sc.transform(test.iloc[:,1:4])

from sklearn.preprocessing import LabelEncoder , OneHotEncoder

le = LabelEncoder()
train['Tag'] = le.fit_transform(train['Tag'])
test['Tag'] = le.transform(test['Tag'])

oe = OneHotEncoder(categorical_features = [0])
train = oe.fit_transform(train).toarray()
test = oe.transform(test).toarray()
train = train[:,1:]
test = test[:,1:]

X = train[:,:-1]
y = train[:,-1]

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
X_train ,  X_val , y_train ,y_val = train_test_split(X , y,test_size=0.2,random_state=0)

'''import xgboost as xgb

xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=0)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_val)
y_pred2 = xgb_model.predict(test)
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred))

submission = pd.DataFrame({'ID' : test_labels , 'Upvotes' : y_pred2 })'''

from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(X_train,y_train)
y_pred3 = gbr.predict(X_val)
y_pred4 = gbr.predict(test)
rmse_gbr = np.sqrt(mean_squared_error(y_val, y_pred3))

submission = pd.DataFrame({'ID' : test_labels , 'Upvotes' : y_pred4 })
submission.to_csv('submission.csv',index=False)
