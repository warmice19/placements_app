import pickle
import numpy as np

# -----------------------------------------------------
# Load Model and Utilities
# -----------------------------------------------------

# Load model and other utilities
model = pickle.load(open("./model_logistic_regression.pkl", "rb"))
ss_input_vars = pickle.load(open("./ss_input_vars.pkl", "rb"))
ohe_input_vars = pickle.load(open("./ohe_input_vars.pkl", "rb"))
le_target_vars = pickle.load(open("./le_target_vars.pkl", "rb"))

# -----------------------------------------------------
# Get User Input
# -----------------------------------------------------

# User input
user_input_dict = {
    "10th_percentage": 0.8,
    "12th_percentage": 0.7,
    "undergrad_percentage": 0.85,
    "mba_percentage": 0.91,
    "gender": "M",
    "10th_board": "Central",
    "12th_board": "Others",
    "stream": "Science",
    "undergrad_stream": "Sci&Tech",
    "workex": "No",
    "specialisation": "Mkt&Fin",
}


# Convertig user input dictionary to lists
user_input_numerical = [v for i, v in enumerate(user_input_dict.values()) if i <= 3]
user_input_categorical = [v for i, v in enumerate(user_input_dict.values()) if i > 3]

# Converting user input list to numpy array and reshape. Reshape is required because
# number of rows should be equal to one as we are running predcitions on a single
# input row
arr_user_input_numerical = np.array(user_input_numerical).reshape(1, -1)
arr_user_input_categorical = np.array(user_input_categorical).reshape(1, -1)

# One Hot Encoding categorical features array.
# Normalising numerical features list using standard scaler.
# Both one hot encoder and the standard scaler were created in the training set
arr_user_input_numerical = ss_input_vars.transform(arr_user_input_numerical)
arr_user_input_categorical = ohe_input_vars.transform(arr_user_input_categorical)

# Joining the two lists
arr_user_input = np.hstack((arr_user_input_numerical, arr_user_input_categorical.A))


# -----------------------------------------------------
# Generate Predictions
# -----------------------------------------------------
pred = model.predict_proba(arr_user_input)
print(le_target_vars.classes_)
print(pred)
