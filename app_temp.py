import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import matplotlib.pyplot as plt

# Load the data
with open('X_copy.pkl', 'rb') as f:
    X_copy = pickle.load(f)

with open('y_copy.pkl', 'rb') as f:
    y_copy = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

global globvar
globvar = 0

def set_globvar_to_one():
    global globvar    # Needed to modify global copy of globvar
    globvar += 1
    
# Function to make predictions
def make_predictions(X_copy, y_copy, ex, scaler):
    accuracy_scores = []
    classification_reports = []
    predictions = []
    dim_predictions = []
    prediction_probabilities = []

    ex = scaler.transform(ex)
    
    for i in range(16):
        X_train, X_test, y_train, y_test = train_test_split(X_copy.iloc[:69], y_copy.iloc[:69], test_size=0.2, random_state=i)

        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        xgb_classifier = XGBClassifier(random_state=4, max_depth=3, learning_rate=0.214, objective='binary:logistic')

        xgb_classifier.fit(X_train, y_train)

        y_pred = xgb_classifier.predict(X_test)
        y_new = xgb_classifier.predict(ex)
        y_prob = xgb_classifier.predict_proba(ex)[:, 1]
        
        predictions.append(y_new[0])
        prediction_probabilities.append(y_prob[0])
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        accuracy_scores.append(accuracy)
        classification_reports.append(report)

    mean_accuracy = np.mean(accuracy_scores)
    std_accuracy = np.std(accuracy_scores)

    mean_prediction = np.mean(predictions)
    mean_probability = np.mean(prediction_probabilities)
    ans = 1 if mean_prediction > 0.5 else 0
    set_globvar_to_one()
    if(ans == 1):
        dim_predictions.append(ex)
        st.write(f"Optimal Bush found, progress {globvar}")
    else:
        st.write(f"Current Bush not fine, progress {globvar}")
    return dim_predictions, mean_probability

# Function to get user inputs from the app
def get_user_inputs():
    user_input = {}

    col1, col2 = st.columns(2)
    
    with col1:
        user_input['Tons'] = st.number_input("Tons: ", min_value=0.0, step=0.1)
        
        user_input['WP'] = st.number_input("Working Pressure: ", min_value=0.0, step=0.1)
        
        user_input['Rod'] = st.number_input("Rod: ", min_value=0.0, step=0.01)
        clear_min = st.number_input("Minimum Clearance: ", min_value=0.0, step=0.01)
        
        Notches_min = st.number_input("Minimum number of Notches: ", min_value=0, step=1)
        Notch_angle_degree_min = st.number_input("Minimum Notch Angle degree: ", min_value=0.0, step=0.001)
        Notch_length_min = st.number_input("Minimum Notch length (mm): ", min_value=0.0, step=0.01)
        Bush_OD_min = st.number_input("Minimum Bush OD: ", min_value=0.0, step=0.01)
        
        user_input['Type'] = st.selectbox("Type", options=[0, 1], format_func=lambda x: "CEC" if x == 0 else "HEC")
    with col2:
        user_input['Bush Length'] = st.number_input("Bush Length: ", min_value=0.0, step=0.01)
        user_input['Bore'] = st.number_input("Bore: ", min_value=0.0, step=0.01)
        user_input['Stroke'] = st.number_input("Stroke: ", min_value=0.0, step=0.01)
        
        clear_max = st.number_input("Maximum Clearance: ", min_value=0.0, step=0.01)
        
        Notches_max = st.number_input("Maximum number of Notches: ", min_value=0, step=1)
        
        Notch_angle_degree_max = st.number_input("Maximum Notch Angle degree: ", min_value=0.0, step=0.001)
        
        Notch_length_max = st.number_input("Maximum Notch length (mm): ", min_value=0.0, step=0.01)
        Bush_OD_max = st.number_input("Maximum Bush OD: ", min_value=0.0, step=0.01)
        user_input['samples'] = st.number_input("Number of Iterations Needed:(Minimum 10) ", min_value=10, step=1)

    return user_input, Bush_OD_min, Bush_OD_max, clear_min, clear_max, Notches_min, Notches_max, Notch_angle_degree_min, Notch_angle_degree_max, Notch_length_min, Notch_length_max

def save_data(X_copy, y_copy):
    with open('X_copy.pkl', 'wb') as f:
        pickle.dump(X_copy, f)
    with open('y_copy.pkl', 'wb') as f:
        pickle.dump(y_copy, f)
    st.write("Data has been saved successfully.")


def get_new_data_points():
    st.subheader("Add New Data Points")
    new_data = {}

    with st.form(key='new_data_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            new_data['Tons'] = st.number_input("Tons: ", min_value=0.0, step=0.1)
            new_data['WP'] = st.number_input("Working Pressure (New): ", min_value=0.0, step=0.1)
            new_data['Rod'] = st.number_input("Rod (New): ", min_value=0.0, step=0.01)
            new_data['Clearance'] = st.number_input("Clearance (New): ", min_value=0.0, step=0.01)
            new_data['Notches'] = st.number_input("Notches (New): ", min_value=0, step=1)
            new_data['Bore'] = st.number_input("Bore (New): ", min_value=0.0, step=0.01)
            new_data['Stroke'] = st.number_input("Stroke (New): ", min_value=0.0, step=0.01)
            new_data['Type'] = st.selectbox("Type", options=[0, 1], format_func=lambda x: "CEC" if x == 0 else "HEC")
            new_data['Bush OD'] = st.number_input("Bush OD (New): ", min_value=0.0, step=0.01)
            new_data['Bush Length'] = st.number_input("Bush Length (New): ", min_value=0.0, step=0.01)
        
        with col2:
            new_data['Notch Angle degree'] = st.number_input("Notch Angle degree (New): ", min_value=0.0, step=0.001)
            new_data['Notch length(mm)'] = st.number_input("Notch length(mm) (New): ", min_value=0.0, step=0.01)
            Working_fine = st.number_input("Bush Working Fine?:(Enter 1 if yes, 0 if no) ", min_value=0, step=1)
        
        add_data = st.form_submit_button("Add Data Point")

        if add_data:
            new_data_point = pd.DataFrame([new_data], columns=['Tons', 'WP', 'Rod', 'Clearance', 'Notches', 'Bore', 'Stroke', 'Type', 'Bush OD', 'Bush Length', 'Notch Angle degree', 'Notch length(mm)'])
            return new_data_point, Working_fine
        else:
            return None, None
    return None,None


# Define search space for optimization
def optimize_parameters(user_input, Bush_OD_min, Bush_OD_max, clear_min, clear_max, Notches_min, Notches_max, Notch_angle_degree_min, Notch_angle_degree_max, Notch_length_min, Notch_length_max, X_copy, y_copy, scaler):
    search_space = [
        Real(Bush_OD_min, Bush_OD_max, name='Bush OD'),
        Real(clear_min, clear_max, name='Clearance'),
        Integer(Notches_min, Notches_max, name='Notches'),
        Real(Notch_angle_degree_min, Notch_angle_degree_max, name='Notch Angle degree'),
        Real(Notch_length_min, Notch_length_max, name='Notch length(mm)')
    ]

    @use_named_args(search_space)
    def objective(**params):
        sample = {
            'WP': user_input['WP'],
            'Bore': user_input['Bore'],
            'Stroke': user_input['Stroke'],
            'Clearance': params['Clearance'],
            'Bush OD': params['Bush OD'],
            'Notch Angle degree': params['Notch Angle degree'],
            'Notches': params['Notches'],
            'Notch length(mm)': params['Notch length(mm)'],
            'Rod': user_input['Rod'],
        }
        sample_df = pd.DataFrame([sample])

        sample_df['diff'] = (sample_df['Notches'] - sample_df['Notch length(mm)'])**2
        sample_df['Clearance_area'] = np.pi * (2 * sample_df['Bush OD'] * sample_df['Clearance'] + (sample_df['Clearance'])**2)
        
        sample_df_test = sample_df.drop(['Notches', 'Notch length(mm)', 'Rod', 'Bush OD'], axis=1)

        dim_predictions, mean_probability = make_predictions(X_copy, y_copy, sample_df_test, scaler)
        
        return -mean_probability
    
    samples = user_input['samples']
    X_copy_scaled = pd.DataFrame(scaler.fit_transform(X_copy), columns=X_copy.columns)

    res = gp_minimize(objective, search_space, n_calls=samples, random_state=0)

    # Best result
    st.write("Best parameters found:")
    st.write(res.x)
    

    user_input['Bush OD'] = res.x[0]
    user_input['Clearance'] = res.x[1]
    user_input['Notches'] = res.x[2]
    user_input['Notch Angle degree'] = res.x[3]
    user_input['Notch length(mm)'] = res.x[4]

    return user_input


# Streamlit app layout
st.title("Bush Optimization App")

st.write("Provide the necessary inputs to optimize the bush parameters.")

user_input, Bush_OD_min, Bush_OD_max, clear_min, clear_max, Notches_min, Notches_max, Notch_angle_degree_min, Notch_angle_degree_max, Notch_length_min, Notch_length_max = get_user_inputs()

# Initialize session state for new data point if not already present
if 'new_data_point' not in st.session_state:
    st.session_state.new_data_point = None

if 'new_working_fine' not in st.session_state:
    st.session_state.new_working_fine = None

if st.button("Add new Data points"):
    st.session_state.new_data_point, st.session_state.new_working_fine = get_new_data_points()

if st.session_state.new_data_point is not None:
    if st.button("Confirm to add the data point"):
        # Scale the new data point
        st.write(st.session_state.new_data_point)
        cols_to_drop = ['Tons']
        st.session_state.new_data_point = st.session_state.new_data_point.drop(columns=cols_to_drop, axis=1)
        st.write(st.session_state.new_data_point)
        
        st.session_state.new_data_point = None  # Clear the session state after saving
    elif st.button("Re-enter the given data-point"):
        st.session_state.new_data_point, st.session_state.new_working_fine = get_new_data_points()  # Clear the session state to re-enter data
        
st.write("*Note: If Bush OD/ID is B and tolerance is +h_B & -l_B and the counter-part has an ID/OD of C and tol is +h_C & -l_C")
st.write("Then clearance is : absolute(((B-C) + (h_B-l_B) + (h_C-h_B ))/2")

if st.button("Optimize Parameters"):
    # Initialize scaler and fit with combined data
    scaler = StandardScaler().fit(X_copy)
    optimized_params = optimize_parameters(user_input, Bush_OD_min, Bush_OD_max, clear_min, clear_max, Notches_min, Notches_max, Notch_angle_degree_min, Notch_angle_degree_max, Notch_length_min, Notch_length_max, X_copy, y_copy, scaler)
    st.write("Optimized Parameters:")
    st.write(optimized_params)
