# App to predict the chances of admission using a pre-trained ML model in Streamlit

# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')


# Reading the pickle file that we created before 
model_pickle = open('dt_fetal.pickle', 'rb') 
dt_model = pickle.load(model_pickle) 
model_pickle.close()

model_pickle = open('rf_fetal.pickle', 'rb') 
rf_model = pickle.load(model_pickle) 
model_pickle.close()

model_pickle = open('ada_fetal.pickle', 'rb') 
ada_model = pickle.load(model_pickle) 
model_pickle.close()

model_pickle = open('vc_fetal.pickle', 'rb') 
vc_model = pickle.load(model_pickle) 
model_pickle.close()

# Load the default dataset
default_df = pd.read_csv('fetal_health.csv')
default_df.drop(columns=["fetal_health"], inplace=True)

# Create a sidebar for input collection
st.sidebar.header('Fetal Health Features Input')
st.sidebar.write("Upload your data")

with st.sidebar:
    st.header('Fetal Health Features Input')
    st.write("Upload your data")
    file = st.file_uploader("Drag and drop file here", type='csv', accept_multiple_files=False)
    st.warning("⚠️Ensure your data follows the format oulined below")
    st.dataframe(default_df.head())
    st.write("Choose Model for Prediction")
    model = st.radio("Choose Model for Prediction", options=["Random Forest", "Decision Tree", "Adaboost", "Soft Voting"])
    st.info(f'You selected: {model}')

# Set up the app title and image
st.title('Fetal Health Classification: A Machine Learning App')
st.image('fetal_health_image.gif', use_column_width = True)
st.write("Utilize our advanced Machine Learning application to predict fetal health classication")

def color_class(val):
    if val == "Normal": 
        return 'background-color: lime'
    elif val == "Suspect": 
        return 'background-color: yellow'
    elif val == "Pathological":
        return 'background-color: orange'
    else:
        return ''

if file:
    st.success("CSV file uploaded successfully.")
    st.header(f"Predicting Fetal Health Class Using {model} Model")

    df1 = pd.read_csv(file)

    df = pd.get_dummies(df1, columns=["histogram_tendency"])

    expected_columns = ['baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions', 'light_decelerations',
    'severe_decelerations', 'prolongued_decelerations', 'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
    'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability','histogram_width','histogram_min',
    'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean','histogram_median',
    'histogram_variance','histogram_tendency_0.0','histogram_tendency_1.0']

    
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    X = df[expected_columns]

    if model == "Random Forest":
        df["Predicted Class"] = rf_model.predict(X)
        df["Predicted Probability"] = np.max(rf_model.predict_proba(X), axis=1)
        id = "rf_"

    elif model == "Decision Tree":
        df["Predicted Class"] = dt_model.predict(X)
        df["Predicted Probability"] = np.max(dt_model.predict_proba(X), axis=1)
        id = "dt_"
    
    elif model == "Adaboost":
        df["Predicted Class"] = ada_model.predict(X)
        df["Predicted Probability"] = np.max(ada_model.predict_proba(X), axis=1)
        id = "ada_"

    elif model == "Soft Voting":
        df["Predicted Class"] = vc_model.predict(X)
        df["Predicted Probability"] = np.max(vc_model.predict_proba(X), axis=1)
        id = "vc_"

    df1["Predicted Class"] = df["Predicted Class"]
    df1["Predicted Probability"] = df["Predicted Probability"]
    df1 = df1.style.applymap(color_class, subset=["Predicted Class"])

    st.dataframe(data=df1)

    # Additional tabs for DT model performance
    st.subheader("Model Insights")
    tab1, tab2, tab3 = st.tabs(["Feature Importance", 
                                "Confusion Matrix", 
                                "Classification Report"])
    with tab1:
        st.write("### Feature Importance")
        st.image(f'{id}feature_importance.svg')
        st.caption("Relative importance of features in prediction.")
    with tab2:
        st.write("### Confusion Matrix")
        st.image(f'{id}confusion_matrix.svg')
    with tab3:
        st.write("### Classification Report")
        report_df = pd.read_csv(f'{id}class_report.csv', index_col = 0).transpose()
        st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))

else:
    st.info("Please upload data to proceed")





