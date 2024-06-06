import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def add_conversion_flags(df):
    df['conv1'] = np.where(df['Total_Conversion'] != 0, 1, 0)
    df['conv2'] = np.where(df['Approved_Conversion'] != 0, 1, 0)
    return df

def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    adjusted_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(train_val_df, test_size=adjusted_val_size, random_state=random_state)
    return train_df, val_df, test_df

def column_types(df):
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    return numerical_columns, categorical_columns

def scale_numerical_columns(train_df, val_df, test_df, numerical_columns):
    scaler = StandardScaler()
    numerical_cols = [col for col in numerical_columns if col not in ['conv1', 'conv2']]
    train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
    val_df[numerical_cols] = scaler.transform(val_df[numerical_cols])
    test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])
    return train_df, val_df, test_df

def one_hot_encode(train_df, val_df, test_df, columns, suffix='_onehot'):
    """
    One-hot encodes the specified columns in the DataFrames using sklearn's OneHotEncoder,
    and retains the original columns with encoded columns having a specified suffix.

    Parameters:
    train_df (pd.DataFrame): The training DataFrame.
    val_df (pd.DataFrame): The validation DataFrame.
    test_df (pd.DataFrame): The test DataFrame.
    columns (list): List of columns to one-hot encode.
    suffix (str): The suffix to add to the one-hot encoded columns.

    Returns:
    pd.DataFrame: The training DataFrame with one-hot encoded columns.
    pd.DataFrame: The validation DataFrame with one-hot encoded columns.
    pd.DataFrame: The test DataFrame with one-hot encoded columns.
    """
    for column in columns:
        combined_df = pd.concat([train_df[[column]], val_df[[column]], test_df[[column]]], axis=0)
        encoder = OneHotEncoder(sparse_output=False)

        encoder.fit(combined_df)

        encoded_train_cols = encoder.transform(train_df[[column]])
        encoded_val_cols = encoder.transform(val_df[[column]])
        encoded_test_cols = encoder.transform(test_df[[column]])

        encoded_train_df = pd.DataFrame(encoded_train_cols, columns=[f"{column}_{cat}{suffix}" for cat in encoder.categories_[0]])
        encoded_val_df = pd.DataFrame(encoded_val_cols, columns=[f"{column}_{cat}{suffix}" for cat in encoder.categories_[0]])
        encoded_test_df = pd.DataFrame(encoded_test_cols, columns=[f"{column}_{cat}{suffix}" for cat in encoder.categories_[0]])

        train_df = pd.concat([train_df.reset_index(drop=True), encoded_train_df.reset_index(drop=True)], axis=1)
        val_df = pd.concat([val_df.reset_index(drop=True), encoded_val_df.reset_index(drop=True)], axis=1)
        test_df = pd.concat([test_df.reset_index(drop=True), encoded_test_df.reset_index(drop=True)], axis=1)

    return train_df, val_df, test_df

def prepare_features_and_target(df, target_column='conv2', drop_columns=['conv1']):
    X = df.drop(columns=drop_columns + [target_column])
    y = df[target_column]
    return X, y
