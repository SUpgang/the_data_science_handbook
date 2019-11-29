from housing import fetch_housing_data, load_housing_data, split_train_test, CombinedAttributesAdder

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np

#fetch_housing_data()

housing = load_housing_data()
# print(housing.head())
# print(housing.info())
# print(housing["ocean_proximity"].value_counts())
# print(housing.describe())

# housing.hist(bins=50, figsize=(20,15))
# plt.show()

# train_set, test_set = split_train_test(housing, 0.2)
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# print(train_set.info())
# print(test_set.info())
# print(housing.info())
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0.0,1.5,3.0,4.5,6.0, np.inf],
                               labels=[1,2,3,4,5])
# print(housing.info())
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()

# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# plt.show()

# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#              s=housing["population"]/100, label="population", figsize=(10,7),
#              c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
# plt.legend()
# plt.show()

corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

# attributes = ["median_house_value", "median_income", "total_rooms",
              # "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12,8))
# plt.show()

# housing.plot(kind="scatter", x="median_income", y="median_house_value",
#              alpha = 0.1)
# plt.show()

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
# imputer.fit(housing_num)
# print(imputer.statistics_)
# print(housing_num.median().values)

# X = imputer.transform(housing_num)
# fit_transform macht das gleiche wie fit und dann transform
# hierbei schätzt fit die werte für NA und transform setzt sie ein
# Bei neuen Daten sollten dann wieder die werte von fit eingefügt werden
# und nicht noch mal neue Werte geschätzt werden

X = imputer.fit_transform(housing_num)
# print(imputer.statistics_)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

# housing[["..."]] retuniert ein DataFrame, housing["..."] eine Series
housing_cat = housing[["ocean_proximity"]]
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
# print(housing_cat_encoded[:10])
# print(ordinal_encoder.categories_)

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# print(cat_encoder.categories_)

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
    ])
housing_num_tr = num_pipeline.fit_transform(housing_num)

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)
# print(type(housing_prepared))
# print(housing_prepared[0:3,0:16])
