# Note: I didn't get the time to transfer everything from the playbook file, if you have time please take a look in it for feature selection, linreg, lasso and gridsearch
import pickle
import pandas as pd
from utils import psi_func, company_func
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool


raw_df = pd.read_csv('model/flavors_of_cacao.csv', na_values=['\xa0'])
df = raw_df.dropna(subset= ['Broad Bean\nOrigin'])
df.columns = [column.replace('\n','_') for column in df.columns]
df.columns = [column.replace('\xa0','') for column in df.columns]
df.head()
X = df[['Cocoa_Percent', 'Company_Location']]
y = df['Rating']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
mapper = DataFrameMapper([
    ('Cocoa_Percent', psi_func()),
    ('Company_Location', company_func())
], df_out = True)
Z_train = mapper.fit_transform(X_train)
Z_test = mapper.transform(X_test)


# Naive model
linreg = LinearRegression()
linreg.fit(Z_train[['Cocoa_Percent']], y_train)
linreg.score(Z_test[['Cocoa_Percent']], y_test)


model = CatBoostRegressor(iterations = 50, learning_rate= 0.1)
pipe = make_pipeline(model)
pipe.fit(Z_train,y_train)
# pipe.score(Z_test, y_test)
#
# pipe.predict(pd.DataFrame({'Cocoa_Percent': 65,
#                            'Company_Location': 1}, index = [0]))

# pipe with mapper doesnt work on existing data yet, due to errors with custom transformer mixins
# Pipe does work on new data, because new data has been formatted to enter pipe in functional manner
# pipe.fit(Z_train, y_train)
# pipe.score(Z_test, y_test)


pickle.dump(pipe, open("model/pipe.pkl", "wb"))
