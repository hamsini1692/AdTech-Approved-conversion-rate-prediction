import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from data_processing import add_conversion_flags, split_data, column_types, scale_numerical_columns, one_hot_encode, prepare_features_and_target
from model_training import fit_and_predict_ridgeclassifier
from metrics import get_metrics, plot_confusion_matrix

# Streamlit application
st.title("Ad Conversion Rate Analysis")

# Initialize session state variables
if 'columns_to_encode' not in st.session_state:
    st.session_state['columns_to_encode'] = ['xyz_campaign_id', 'gender']
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'X_train' not in st.session_state:
    st.session_state['X_train'] = None
if 'y_train' not in st.session_state:
    st.session_state['y_train'] = None
if 'X_val' not in st.session_state:
    st.session_state['X_val'] = None
if 'y_val' not in st.session_state:
    st.session_state['y_val'] = None
if 'X_test' not in st.session_state:
    st.session_state['X_test'] = None
if 'y_test' not in st.session_state:
    st.session_state['y_test'] = None

# File Upload Section
uploaded_file = st.file_uploader("Upload Ad Campaign Data (CSV)", type=["csv"])
if uploaded_file is not None:
    # Reading the uploaded CSV file into a Pandas DataFrame
    df = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data")
    st.write(df.head())

    # Adding conversion flags
    df = add_conversion_flags(df)
    # Dropping unnecessary columns
    df = df.drop(['Approved_Conversion', 'Total_Conversion', 'Clicks', 'fb_campaign_id', 'ad_id'], axis=1)

    # Display summary statistics for key columns
    st.write("Summary statistics for key columns:")
    st.write(df[['Impressions', 'Spent', 'interest']].describe())

    # Process Data Button
    if st.button('Process Data'):
        # Splitting the data
        train_df, val_df, test_df = split_data(df)
        st.write("Data split successfully.")
        st.write("Train DataFrame shape:", train_df.shape)
        st.write("Validation DataFrame shape:", val_df.shape)
        st.write("Test DataFrame shape:", test_df.shape)

        # Identifying numerical and categorical columns
        numerical_columns, categorical_columns = column_types(train_df)
        st.write("Numerical columns:", numerical_columns)
        st.write("Categorical columns:", categorical_columns)

        # Scaling numerical columns
        train_df, val_df, test_df = scale_numerical_columns(train_df, val_df, test_df, numerical_columns)
        st.write("Numerical columns scaled successfully.")

        # One-hot encoding categorical columns
        train_df_encoded, val_df_encoded, test_df_encoded = one_hot_encode(train_df, val_df, test_df, st.session_state['columns_to_encode'])
        st.write("Categorical columns one-hot encoded successfully.")

        # Preparing the final feature set
        feature_columns = ['interest', 'Impressions', 'Spent'] + [col for col in train_df_encoded.columns if col.endswith('_onehot')]
        train_df_final = train_df_encoded[feature_columns + ['conv2']]
        val_df_final = val_df_encoded[feature_columns + ['conv2']]
        test_df_final = test_df_encoded[feature_columns + ['conv2']]
        st.write("Final feature set prepared.")

        # Preparing features and target variables
        st.session_state['X_train'], st.session_state['y_train'] = prepare_features_and_target(train_df_final, target_column='conv2', drop_columns=[])
        st.session_state['X_val'], st.session_state['y_val'] = prepare_features_and_target(val_df_final, target_column='conv2', drop_columns=[])
        st.session_state['X_test'], st.session_state['y_test'] = prepare_features_and_target(test_df_final, target_column='conv2', drop_columns=[])
        st.write("Features and target variables prepared.")

        # Debugging: Ensure that the variables are correctly set
        st.write("X_train shape:", st.session_state['X_train'].shape if st.session_state['X_train'] is not None else "None")
        st.write("y_train shape:", st.session_state['y_train'].shape if st.session_state['y_train'] is not None else "None")
        st.write("X_val shape:", st.session_state['X_val'].shape if st.session_state['X_val'] is not None else "None")
        st.write("y_val shape:", st.session_state['y_val'].shape if st.session_state['y_val'] is not None else "None")
        st.write("X_test shape:", st.session_state['X_test'].shape if st.session_state['X_test'] is not None else "None")
        st.write("y_test shape:", st.session_state['y_test'].shape if st.session_state['y_test'] is not None else "None")

    # Train Models Button
    if st.button('Train Models'):
        if st.session_state['X_train'] is not None and st.session_state['y_train'] is not None and \
           st.session_state['X_val'] is not None and st.session_state['y_val'] is not None and \
           st.session_state['X_test'] is not None and st.session_state['y_test'] is not None:
            # Training models using RidgeClassifierCV
            st.session_state['model'], predictions, probabilities = fit_and_predict_ridgeclassifier(
                st.session_state['X_train'], st.session_state['y_train'],
                st.session_state['X_val'], st.session_state['y_val'],
                st.session_state['X_test'], st.session_state['y_test']
            )
            st.write("Model training completed.")

            # Getting and displaying metrics
            metrics_results = get_metrics(
                st.session_state['y_train'], st.session_state['y_val'], st.session_state['y_test'],
                predictions, probabilities
            )

            for set_name, results in metrics_results.items():
                st.write(f"### Metrics for {set_name.capitalize()} Set")
                st.write(results["AUC"])
                st.write("Classification Report:")
                st.json(results["Classification Report"])
                st.write("Confusion Matrix:")
                plot_confusion_matrix(results["Confusion Matrix"], title=f'Confusion Matrix - {set_name.capitalize()} Set')
        else:
            st.write("Data not processed correctly. Please process the data before training the model.")

    # Input New Data for Prediction Section
    st.write("Input new data for prediction")
    # Create input fields for relevant features
    impression = st.number_input('Impressions', value=10000.0)
    spent = st.number_input('Spent', value=50.0)
    interest = st.number_input('Interest', value=10.0)

    if st.button('Predict Conversion Rate'):
        if st.session_state['model'] is not None:
            # Prepare the input data for prediction
            input_data = pd.DataFrame({
                'interest': [interest],
                'Impressions': [impression],
                'Spent': [spent]
            })

            # One-hot encode categorical columns if necessary
            for column in st.session_state['columns_to_encode']:
                encoded_cols = pd.get_dummies([column], prefix=column)
                input_data = pd.concat([input_data, encoded_cols], axis=1)

            # Align columns with the training data
            input_data = input_data.reindex(columns=st.session_state['X_train'].columns, fill_value=0)

            # Check for NaNs in the input data
            st.write("Input data for prediction:")
            st.write(input_data)
            st.write("Check for NaNs:", input_data.isna().sum())

            # Ensure no missing values are present
            input_data.fillna(0, inplace=True)

            # Perform prediction using the trained model
            prediction = st.session_state['model'].predict(input_data)

            # Display the predicted conversion rate
            st.write(f"Predicted Conversion Rate: {prediction[0]}")
        else:
            st.write("Please train the model first.")
