# Import required libraries
import streamlit as st
import numpy as np
import pickle  # For loading trained models
import warnings
warnings.filterwarnings("ignore")

# Load the trained model (replace 'model.pkl' with your model file name)
with open(r'Heart_Disease_Prediction_Model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the Streamlit app
def main():
    st.set_page_config(layout="wide", page_title="Heart Disease Prediction")
    
    # Title and Description
    st.title(":blue[Heart Disease Prediction]")
    st.markdown("""
    **Heart disease is one of the leading causes of death worldwide, affecting millions each year.**
    Early detection and timely medical intervention can significantly reduce risks.  
    Use this tool to assess your heart health by entering key medical parameters below.
    """)

    # Form Layout and Grouping
    st.markdown("### Please provide your medical details for heart disease prediction:")
    col1, col2, col3 = st.columns(3)

    # Input features
    with col1:
        age = st.text_input("Age", placeholder="Enter age (e.g., 25)")                                                                                                             
        sex = st.selectbox("Gender", options=["Select Gender", "Female (0)", "Male (1)"])
        cp = st.selectbox("Chest Pain Type", 
                          options=["Select Chest Pain Type from below options", 
                                   "Typical (0)", "Atypical (1)", 
                                   "Non-Anginal (2)", "Asymptomatic (3)"],
                          help="Select the type of chest pain experienced. Typical is the most common form.")
        
        resting_bp = st.text_input("Resting Blood Pressure (mm Hg)", 
                                   placeholder="Enter resting BP in mm Hg (e.g., 120)",
                                   help="Enter your resting blood pressure measured in mm Hg.")
        
    with col2:
        chol = st.text_input("Cholesterol (mg/dl)", placeholder="Enter cholesterol value (e.g., 200)",
                             help="Cholesterol level measured in mg/dl. Ideal range is below 200 mg/dl.")
        
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", 
                                  options=["Select Fasting Blood Sugar from below options", "No (0)", "Yes (1)"],
                                  help="Indicates whether your fasting blood sugar is greater than 120 mg/dl.")
        
        resting_ecg = st.selectbox("Resting ECG Results", 
                                   options=["Select ECG results from below options", "Normal (0)", 
                                            "ST-T wave abnormality (1)", 
                                            "Hypertrophy (2)"],
                                    help="Choose your resting ECG results. 'Normal' is typical, but abnormalities may indicate heart issues.")
        
        exercise_angina = st.selectbox("Exercise-Induced Angina", 
                                       options=["Select Exercise-Induced Angina from below options", "No (0)", "Yes (1)"], help="Indicates whether exercise causes angina.")
       
    with col3:
        max_hr = st.text_input("Maximum Heart Rate Achieved", 
                               placeholder="Enter maximum heart rate achieved (e.g., 135)",
                               help="Enter the maximum heart rate achieved during exercise. Normal ranges from 60-202.")
        
        oldpeak = st.text_input("Oldpeak (ST depression)", 
                                placeholder="Enter ST depression value (e.g., 2.6)",
                                 help="Indicates ST depression during exercise, which may indicate heart disease.")
        
        slope = st.selectbox("Slope of Peak Exercise ST Segment", 
                             options=["Select Slope of Peak Exercise from below options", "Upsloping (0)", 
                                      "Flat (1)", "Downsloping (2)"],
                             help="Select the slope of the peak exercise ST segment. It can be upsloping, flat, or downsloping.")
    st.markdown("#### Please click the button below after filling in all the details above")
    # Validation and processing
    if st.button("Predict"): #Evaluate the Heart Condition
        # Basic validation to ensure no "Select" or empty fields are submitted
        if (
            age and resting_bp and chol and max_hr and oldpeak and
            sex != "Select Gender" and cp != "Select Chest Pain Type from below options" and
            fasting_bs != "Select Fasting Blood Sugar from below options" and resting_ecg != "Select ECG results from below options" and 
            exercise_angina != "Select Exercise-Induced Angina from below options" and slope != "Select Slope of Peak Exercise from below options"
        ):
            try:
                # Extract numeric values from dropdown options
                sex_val = int(sex.split(" ")[-1][1:-1])
                cp_val = int(cp.split(" ")[-1][1:-1])
                fasting_bs_val = int(fasting_bs.split(" ")[-1][1:-1])
                resting_ecg_val = int(resting_ecg.split(" ")[-1][1:-1])
                exercise_angina_val = int(exercise_angina.split(" ")[-1][1:-1])
                slope_val = int(slope.split(" ")[-1][1:-1])

                # Prepare input data for prediction
                input_data = np.array(([[int(age), sex_val, cp_val, int(resting_bp), 
                                         int(chol), fasting_bs_val, resting_ecg_val, 
                                         int(max_hr), exercise_angina_val, 
                                         float(oldpeak), slope_val]]))
    
                
                # Make prediction
                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data)
                
                # Display the result
                if prediction[0] == 1:
                    st.error(f"### Prediction: \u2620 Positive for Heart Disease (Risk).") #\u26A0
                else:
                    st.success(f"### Prediction: Negative for Heart Disease (No Risk) ❤️ Your Heart is Safe!") #\u2764

                # Display prediction probabilities
                st.write("Prediction Probability:")
                st.write(f"No Risk (0): {prediction_proba[0][0]:.2f}")
                st.write(f"Risk (1): {prediction_proba[0][1]:.2f}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.error("Please fill in all fields correctly.")

# Run the app
if __name__ == '__main__':
    main()
