import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

with open('X_copy_unscaled.pkl', 'rb') as f:
    X_copy_unscaled = pickle.load(f)

with open('y_copy.pkl', 'rb') as f:
    y_copy = pickle.load(f)

def upload_csv():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file:
        new_data = pd.read_csv(uploaded_file)
        return new_data
    return None

def save_data(X_copy, y_copy):
    with open('X_copy.pkl', 'wb') as f:
        pickle.dump(X_copy, f)
    with open('y_copy.pkl', 'wb') as f:
        pickle.dump(y_copy, f)
    st.write("Data has been saved successfully.")

def collect_new_data():
    col1, col2 = st.columns(2)
    with col1:
        tons = st.number_input("Tons:", min_value=0.0, step=0.1)
        wp = st.number_input("Working Pressure:", min_value=0.0, step=0.1)
        rod = st.number_input("Rod:", min_value=0.0, step=0.01)
        clearance = st.number_input("Clearance:", min_value=0.0, step=0.01)
        notches = st.number_input("Notches:", min_value=0, step=1)
        bore = st.number_input("Bore:", min_value=0.0, step=0.01)
        
        
    with col2:
        
        type_ = st.selectbox("Type:", options=[0, 1], format_func=lambda x: "CEC" if x == 0 else "HEC")
        bush_od = st.number_input("Bush OD:", min_value=0.0, step=0.01)
        bush_length = st.number_input("Bush Length:", min_value=0.0, step=0.01)
        notch_angle_degree = st.number_input("Notch Angle Degree:", min_value=0.0, step=0.001)
        notch_length_mm = st.number_input("Notch Length (mm):", min_value=0.0, step=0.01)
        stroke = st.number_input("Stroke:", min_value=0.0, step=0.01)
    working_fine = st.selectbox("Working Fine?", options=[0, 1], format_func=lambda x: "YES" if x == 0 else "NO")
    #submit_button = st.form_submit_button("Submit new data")

        #if submit_button:
            
    new_data = {
        'Tons': tons, 'WP': wp ,'Bore ': bore, 'Rod': rod, 'Stroke': stroke, 'Bush OD': bush_od, 'Clearance': clearance ,
        'Notches': notches,  'Type': type_,
        'Bush Length': bush_length,
        'Notch Angle degree': notch_angle_degree, 'Notch length(mm)': notch_length_mm
    }
   
    return pd.DataFrame([new_data] , columns=['Tons', 'WP','Bore ','Rod', 'Stroke','Bush OD', 'Clearance', 'Notches', 'Type', 'Bush Length', 'Notch Angle degree', 'Notch length(mm)']), pd.DataFrame([working_fine])

st.title("Data Adding App")

st.write("Provide the necessary inputs to add data.")

new_data, new_y = collect_new_data()
to_download = X_copy_unscaled

uploaded_data = upload_csv()
if st.button("Confirm to upload this file"):
    X_copy_temp = uploaded_data
    with open('X_copy_unscaled.pkl', 'wb') as f:
        pickle.dump(X_copy_temp, f)
    X_copy_temp['diff'] = ((X_copy_temp['Notches']-X_copy_temp['Notch length(mm)']))**2 
    X_copy_temp['Clearance_area'] = np.pi*(2*X_copy_temp['Bush OD']*X_copy_temp['Clearance'] + (X_copy_temp['Clearance'])**2)
    X_copy_temp = X_copy_temp.drop(['Notches','Notch length(mm)','Rod', 'Bush OD'] , axis= 1)
    
    save_data(X_copy_temp,y_copy)

if st.button("Confirm adding this data"):
    scaler = StandardScaler()
    cols_to_drop = ['Tons', 'Bush Length' , 'Type']
    new_data = new_data.drop(columns= cols_to_drop , axis = 1)
    
    
    y_copy = pd.concat([y_copy,new_y], axis = 0)
    X_copy_temp = pd.concat([X_copy_unscaled,new_data] , axis = 0)
    to_download = X_copy_temp
    
    with open('X_copy_unscaled.pkl', 'wb') as f:
        pickle.dump(X_copy_temp, f)
    X_copy_temp = pd.DataFrame(scaler.fit_transform(X_copy_temp) , columns = X_copy_temp.columns)

    X_copy_temp['diff'] = ((X_copy_temp['Notches']-X_copy_temp['Notch length(mm)']))**2 
    X_copy_temp['Clearance_area'] = np.pi*(2*X_copy_temp['Bush OD']*X_copy_temp['Clearance'] + (X_copy_temp['Clearance'])**2)
    X_copy_temp = X_copy_temp.drop(['Notches','Notch length(mm)','Rod', 'Bush OD'] , axis= 1)
    
    save_data(X_copy_temp,y_copy)

if st.button("Download CSV"):
    csv = to_download.to_csv(index=False)
    #csv = pd.concat([csv,
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='Latest_data.csv',
        mime='text/csv',
    )
