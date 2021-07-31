import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


# -----------------------------------------------------
# READ DATA
# -----------------------------------------------------

df = pd.read_csv("./../data/placements_data.csv")

df = df.rename(
    columns={
        "ssc_p": "10th_percentage",
        "ssc_b": "10th_board",
        "hsc_p": "12th_percentage",
        "hsc_b": "12th_board",
        "hsc_s": "stream",
        "degree_p": "undergrad_percentage",
        "degree_t": "undergrad_stream",
        "mba_p": "mba_percentage",
    }
)

df = df.drop("sl_no", axis=1)

# -----------------------------------------------------
# ENCODE INPUT FEATURES
# -----------------------------------------------------

target = "status"

features_numerical = [
    "10th_percentage",
    "12th_percentage",
    "undergrad_percentage",
    "mba_percentage",
    "etest_p",
]

features_categorical = [
    "gender",
    "10th_board",
    "12th_board",
    "stream",
    "undergrad_stream",
    "workex",
    "specialisation",
]

df_numerical = df.loc[:, features_numerical]
df_categorical = df.loc[:, features_categorical]
df_target = df.loc[:, target]
arr_target = df_target.values

# Standardise input numeric variables
ss_input_vars = StandardScaler()
ss_input_vars.fit(df_numerical)
arr_numerical = ss_input_vars.transform(df_numerical)

# Encode input categorical variables
ohe_input_vars = OneHotEncoder()
ohe_input_vars.fit(df_categorical)
arr_categorical = ohe_input_vars.transform(df_categorical)

# Merge arrays to create a single input array for the model
x_data = np.hstack((arr_categorical.A, arr_numerical))

# Encode output variable
le_target_vars = LabelEncoder()
le_target_vars.fit(arr_target)
y_data = le_target_vars.transform(arr_target)


# -----------------------------------------------------
# TRAIN MODEL
# -----------------------------------------------------

# train
model = LogisticRegression()
model.fit(X=x_data, y=y_data)


# Cross validation
acc_scores = cross_val_score(model, x_data, y_data, cv=10)
print(acc_scores)

# Save models and other utilities
pickle.dump(model, open("./model_logistic_regression.pkl", "wb"))
pickle.dump(ss_input_vars, open("./ss_input_vars.pkl", "wb"))
pickle.dump(ohe_input_vars, open("./ohe_input_vars.pkl", "wb"))
pickle.dump(le_target_vars, open("./le_target_vars.pkl", "wb"))
