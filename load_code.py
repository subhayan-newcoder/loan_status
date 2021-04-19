import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv('/content/drive/MyDrive/DATA/train_indessa.csv')

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')

df_num = df.drop(['member_id', 'funded_amnt', 'funded_amnt_inv', 'term', 'batch_enrolled', 
                  'grade', 'sub_grade', 'emp_title', 'emp_length', 'home_ownership','verification_status','pymnt_plan', 'desc', 'purpose', 'title', 'zip_code',
                  'addr_state', 'initial_list_status', 'application_type', 'verification_status_joint', 'last_week_pay', 'loan_status'], axis = 1)

print(df_num.columns)
df_num_columns = df_num.columns

df_num = imputer.fit_transform(df_num)

df_num = pd.DataFrame(data = df_num, columns= df_num_columns)

df_cat = df[[ 'grade', 'sub_grade', 'emp_length', 'home_ownership','verification_status','pymnt_plan', 'purpose', 'title', 
                  'addr_state', 'initial_list_status', 'application_type']]

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()

df_cat_columns = df_cat.columns

df_cat['emp_length'] = df_cat['emp_length'].fillna('U')
df_cat['title'] = df_cat['title'].fillna('V')

df_cat_new = df_cat.drop(['title', 'pymnt_plan', 'sub_grade', 'addr_state'], axis=1)

df_cat_1hot = cat_encoder.fit_transform(df_cat_new)
df_cat_1hot = df_cat_1hot.toarray()
df_cat_1hot = pd.DataFrame(df_cat_1hot)
df_new = pd.concat([df_num, df_cat_1hot], axis=1)

X = np.array(df_new)
Y = np.array(df['loan_status'])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
std_scl = StandardScaler()
std_scl.fit(x_train)

x_train = std_scl.transform(x_train)
x_test = std_scl.transform(x_test)

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.8, gamma=4,
              learning_rate=0.02, max_delta_step=0, max_depth=15,
              min_child_weight=2, missing=None, n_estimators=600, n_jobs=1,
              nthread=1, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=True, subsample=0.8, tree_method='gpu_hist', verbosity=1)

xgb.fit(x_train,y_train)
y_pred = xgb.predict(x_test)

from sklearn.metrics import  confusion_matrix, classification_report, roc_auc_score, f1_score
print(classification_report(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))
print('roc auc score: {}'.format(roc_auc_score(y_pred,y_test)))
print('f1 score: {}'.format(f1_score(y_pred,y_test)))

df_test = pd.read_csv('/content/drive/MyDrive/DATA/test_indessa.csv')
df_test_num = df_test.drop(['member_id', 'funded_amnt', 'funded_amnt_inv', 'term', 'batch_enrolled', 
                  'grade', 'sub_grade', 'emp_title', 'emp_length', 'home_ownership','verification_status','pymnt_plan', 'desc', 'purpose', 'title', 'zip_code',
                  'addr_state', 'initial_list_status', 'application_type', 'verification_status_joint', 'last_week_pay'], axis = 1)
df_test_num = imputer.fit_transform(df_test_num)
df_test_num = pd.DataFrame(data = df_test_num)
df_test_cat = df_test[[ 'grade', 'sub_grade', 'emp_length', 'home_ownership','verification_status','pymnt_plan', 'purpose', 'title', 
                  'addr_state', 'initial_list_status', 'application_type']]
df_test_cat['emp_length'] = df_test_cat['emp_length'].fillna('U')
df_test_cat['title'] = df_test_cat['title'].fillna('V')
df_cattest_new = df_test_cat.drop(['title', 'pymnt_plan', 'sub_grade', 'addr_state'], axis=1)
cat_encoder.fit(df_cat_new)
df_cattest_new = cat_encoder.transform(df_cattest_new)
df_cattest_new = df_cattest_new.toarray()
df_cattest = pd.DataFrame(df_cattest_new)
df_test_new = pd.concat([df_test_num, df_cattest], axis=1)
x_test_new = np.array(df_test_new)
print(df_test_new.shape)
std_scl = StandardScaler()
std_scl.fit(x_train)
x_test = std_scl.fit_transform(x_test_new)
# x_test_new = np.array(df_test_new)
# x_test = std_scl.transform(x_test_new)
type(x_test)
y_pred = xgb.predict(x_test)
y_pred = pd.DataFrame(y_pred)
y_pred.to_csv('/content/drive/MyDrive/DATA/test_pred.csv', index=True)



