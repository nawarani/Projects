# imports
import pickle
import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso, LinearRegression


from sklearn_pandas import CategoricalImputer
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures, LabelBinarizer, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.pipeline import make_pipeline

# make life easier
pd.set_option('display.max_rows', None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.base import TransformerMixin

class SafeLabelBinarizer(TransformerMixin):

    def __init__(self):
        self.lb = LabelBinarizer()

    def fit(self, X):
        X = np.array(X)
        self.lb.fit(X)
        self.classes_ = self.lb.classes_

    def transform(self, X):
        K = np.append(self.classes_, ['__FAKE__'])
        X = label_binarize(X, K, pos_label=1, neg_label=0)
        X = np.delete(X, np.s_[-1], axis=1)
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

df = pd.read_csv('data/train.csv', na_values = '', keep_default_na = False)
df = df.drop(df[df['Gr Liv Area'] > 4000].index)
na_df = pd.DataFrame(df.isna().sum(), columns = ['na_values'])
nz = na_df['na_values'].to_numpy().nonzero()
na_df.iloc[nz]
df = df.dropna(subset=['Bsmt Qual','Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin SF 1', 'BsmtFin Type 2', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF','Bsmt Full Bath', 'Bsmt Half Bath', 'Garage Finish', 'Garage Cars', 'Garage Area', 'Garage Qual', 'Garage Cond'])
na_df = pd.DataFrame(df.isna().sum(), columns = ['na_values'])
nz = na_df['na_values'].to_numpy().nonzero()
na_df.iloc[nz]
target = 'SalePrice'
y = df[target]
X = df.drop([target], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

mapper = DataFrameMapper([
    ('MS SubClass', SafeLabelBinarizer()),
    ('MS Zoning', SafeLabelBinarizer()),
    (['Lot Frontage'], [SimpleImputer(strategy = 'constant', fill_value = 0), StandardScaler()]),
    (['Lot Area'], StandardScaler()),
    ('Street', SafeLabelBinarizer()),
    ('Alley', SafeLabelBinarizer()),
    (['Lot Shape'], OrdinalEncoder(categories = [['Reg', 'IR1', 'IR2', 'IR3']])),
    (['Land Contour'], OrdinalEncoder(categories = [['Lvl', 'Low', 'Bnk', 'HLS']])),
    (['Utilities'], SafeLabelBinarizer()),
    ('Lot Config', SafeLabelBinarizer()),
    (['Land Slope'], OrdinalEncoder(categories = [['Gtl', 'Mod', 'Sev']])),
    ('Neighborhood', SafeLabelBinarizer()),
    ('Condition 1', SafeLabelBinarizer()),
    ('Condition 2', SafeLabelBinarizer()),
    ('Bldg Type', SafeLabelBinarizer()),
    ('House Style', SafeLabelBinarizer()),
    (['Overall Qual'], OrdinalEncoder(categories = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])),
    (['Overall Cond'], OrdinalEncoder(categories = [[1, 2, 3, 4, 5, 6, 7, 8, 9]])),
    (['Year Built'], None),
    (['Year Remod/Add'], None),
    ('Roof Style', SafeLabelBinarizer()),
    ('Roof Matl', SafeLabelBinarizer()),
    ('Exterior 1st', SafeLabelBinarizer()),
    ('Exterior 2nd', SafeLabelBinarizer()),
    (['Mas Vnr Type'], [SimpleImputer(strategy = 'constant', fill_value = 'Not_applicable'), SafeLabelBinarizer()]),
    (['Mas Vnr Area'], [SimpleImputer(strategy = 'constant', fill_value = 0), StandardScaler()]),
    (['Exter Qual'], OrdinalEncoder(categories = [['Ex', 'Gd', 'TA', 'Fa']])),
    (['Exter Cond'], OrdinalEncoder(categories = [['Ex', 'Gd', 'TA', 'Fa', 'Po']])),
    ('Foundation', SafeLabelBinarizer()),
    (['Bsmt Qual'], SafeLabelBinarizer()),
    (['Bsmt Cond'], OrdinalEncoder(categories = [['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']])),
    (['Bsmt Exposure'], OrdinalEncoder(categories = [['NA', 'No', 'Mn', 'Av', 'Gd']])),
    (['BsmtFin Type 1'], OrdinalEncoder(categories = [['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ','GLQ']])),
    (['BsmtFin SF 1'], StandardScaler()),
    (['BsmtFin Type 2'], [SimpleImputer(strategy = 'constant', fill_value = 'Not_applicable'), OrdinalEncoder(categories = [['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ','GLQ']])]),
    (['BsmtFin SF 2'], StandardScaler()),
    (['Bsmt Unf SF'], StandardScaler()),
    (['Total Bsmt SF'], StandardScaler()),
    (['Heating'], OrdinalEncoder(categories = [['GasA', 'GasW', 'Grav', 'OthW', 'Wall']])),
    (['Heating QC'], OrdinalEncoder(categories = [['Ex', 'Gd', 'TA', 'Fa', 'Po']])),
    ('Central Air', LabelEncoder()),
    (['Electrical'], OrdinalEncoder(categories = [['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix']])),
    (['1st Flr SF'], StandardScaler()),
    (['2nd Flr SF'], StandardScaler()),
    (['Low Qual Fin SF'], StandardScaler()),
    (['Gr Liv Area'], StandardScaler()),
    ('Bsmt Full Bath', None),
    ('Bsmt Half Bath', None),
    ('Full Bath', None),
    ('Half Bath', None),
    ('Bedroom AbvGr', None),
    ('Kitchen AbvGr', None),
    (['Kitchen Qual'], OrdinalEncoder(categories = [['Fa', 'TA', 'Gd', 'Ex']])),
    ('TotRms AbvGrd', None),
    (['Functional'], SafeLabelBinarizer()),
    ('Fireplaces', None),
    (['Fireplace Qu'], OrdinalEncoder(categories = [['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex']])),
    ('Garage Type', SafeLabelBinarizer()),
    (['Garage Yr Blt'], [SimpleImputer(strategy = 'constant', fill_value = 0), StandardScaler()]),
    (['Garage Finish'], OrdinalEncoder(categories = [['NA', 'Unf', 'RFn', 'Fin']])),
    ('Garage Cars', None),
    (['Garage Area'], StandardScaler()),
    (['Garage Qual'], OrdinalEncoder(categories = [['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']])),
    (['Garage Cond'], OrdinalEncoder(categories = [['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']])),
    (['Paved Drive'], OrdinalEncoder(categories = [['Y', 'P', 'N']])),
    (['Wood Deck SF'], StandardScaler()),
    (['Open Porch SF'], StandardScaler()),
    (['Enclosed Porch'], StandardScaler()),
    (['3Ssn Porch'], StandardScaler()),
    (['Screen Porch'], StandardScaler()),
    (['Pool Area'], StandardScaler()),
    (['Pool QC'], OrdinalEncoder(categories = [['NA', 'Fa', 'TA', 'Gd', 'Ex']])),
    (['Fence'], OrdinalEncoder(categories = [['NA', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv']])),
    (['Misc Feature'], SafeLabelBinarizer()),
    (['Misc Val'], StandardScaler()),
    ('Mo Sold', None),
    ('Yr Sold', None),
    ('Sale Type', SafeLabelBinarizer())], df_out = True)


#Naive model
linreg = LinearRegression()
linreg.fit(X_train[['Bedroom AbvGr']], y_train)
linreg.score(X_test[['Bedroom AbvGr']], y_test)


Z_train = mapper.fit_transform(X_train)

select = RFE(RandomForestClassifier(n_estimators=10, random_state=42), n_features_to_select=3)
select.fit(Z_train, y_train)
Z_train_selected = select.transform(Z_train)
selection = select.get_support()
select_df = pd.DataFrame(zip(Z_train.columns.ravel(), selection), columns = ['feature', 'bool'])
select_df[select_df['bool']]
lassomodel = Lasso()
lassomodel.fit(Z_train_selected, y_train)
lassomodel.score(Z_train_selected, y_train)
new_mapper  = DataFrameMapper([
    (['Lot Area'], StandardScaler()),
    (['Gr Liv Area'], StandardScaler()),
    (['1st Flr SF'], StandardScaler())
], df_out = True)

pipe = make_pipeline(new_mapper, lassomodel)
pipe.fit(X_train, y_train)
# pipe.score(X_test, y_test)

pickle.dump(pipe, open("pipe.pkl", "wb"))
