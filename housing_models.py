import pandas
import numpy
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.model_selection import cross_val_score

housing_dataset = pandas.read_csv(r"C:\Users\Pc\Desktop\Materials\housing.csv")
housing_dataset["income"] = pandas.cut(housing_dataset["median_income"], bins=[
                           0.0, 1.5, 3.0, 4.5, 6.0, numpy.inf], labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing_dataset,housing_dataset["income"]):
    s_tr_s = housing_dataset.loc[train_index]
    s_te_s = housing_dataset.loc[test_index]
for set_ in (s_tr_s, s_te_s):
    set_.drop("income", axis=1, inplace=True)
c_m = s_tr_s.corr()
print(c_m["median_house_value"].sort_values(ascending=False))
s_tr_s = s_tr_s.drop("total_bedrooms", axis=1)
print(s_tr_s.info())
y_train = s_tr_s["median_house_value"]
y_test = s_te_s["median_house_value"]
n_p = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("s_s", StandardScaler())
])
n_a = list(s_tr_s.drop("ocean_proximity", axis=1))
c_a = ["ocean_proximity"]
f_p = ColumnTransformer([
    ("n_a", n_p, n_a),
    ("c_a", OrdinalEncoder(), c_a)
])
s_tr_s = pandas.DataFrame(f_p.fit_transform(s_tr_s), columns=s_tr_s.columns)
print(s_tr_s.info())
print(s_tr_s.head())
X_train = s_tr_s.drop("median_house_value", axis=1)
s_te_s = s_te_s.drop("total_bedrooms", axis=1)
s_te_s = pandas.DataFrame(f_p.fit_transform(s_te_s), columns=s_te_s.columns)
X_test = s_te_s.drop("median_house_value", axis=1)
def displays(s):
    print("scores: ", s, ".")
    print("mean: ", s.mean(), ".")
    print("standard deviation: ", s.std(), ".")

print("linear regression.");   
print();
print("training data.");
l_r = LinearRegression().fit(X_train,y_train);
print(numpy.sqrt(mean_squared_error(y_train,l_r.predict(X_train))));
displays(numpy.sqrt(-cross_val_score(l_r,X_train,y_train,scoring="neg_mean_squared_error",cv=10)));
print("test data.");
print(numpy.sqrt(mean_squared_error(y_test,l_r.predict(X_test))));
displays(numpy.sqrt(-cross_val_score(l_r,X_test,y_test,scoring="neg_mean_squared_error",cv=10)));
print();

print("linearsupportvectormachineregressor.");
print();
print("training data.");
l_s_v_m_r = LinearSVR().fit(X_train,y_train);
print(numpy.sqrt(mean_squared_error(y_train,l_s_v_m_r.predict(X_train))));
displays(numpy.sqrt(-cross_val_score(l_s_v_m_r,X_train,y_train,scoring="neg_mean_squared_error",cv=10)));
print();
print("test data.");
print(numpy.sqrt(mean_squared_error(y_test,l_s_v_m_r.predict(X_test))));
displays(numpy.sqrt(-cross_val_score(l_s_v_m_r,X_test,y_test,scoring="neg_mean_squared_error",cv=10)));
print();

print("randomforestregressor.");
print();
print("training data.");
r_f_r = RandomForestRegressor().fit(X_train,y_train);
print(numpy.sqrt(mean_squared_error(y_train,r_f_r.predict(X_train))));
displays(numpy.sqrt(-cross_val_score(r_f_r,X_train,y_train,scoring="neg_mean_squared_error",cv=10)));
print();
print("test data. ");
print(numpy.sqrt(mean_squared_error(y_test,r_f_r.predict(X_test))));
displays(numpy.sqrt(-cross_val_score(r_f_r,X_test,y_test,scoring="neg_mean_squared_error",cv=10)));
