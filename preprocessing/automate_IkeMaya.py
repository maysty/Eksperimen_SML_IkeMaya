
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

def preprocess_data(
    input_path,
    output_path
):
    # Load data
    df = pd.read_csv(input_path)

    # Drop duplicates
    df = df.drop_duplicates()

    # Feature grouping
    num_cols = ["age", "trestbps", "chol", "thalch", "oldpeak"]

    cat_cols = [
        "sex", "cp", "fbs", "restecg", "exang",
        "slope", "ca", "thal", "dataset"
    ]

    # Missing value handling
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encoding
    binary_cols = ["sex", "fbs", "exang", "dataset"]
    for col in binary_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    ordinal_cols = ["cp", "slope", "thal", "ca"]
    df[ordinal_cols] = OrdinalEncoder().fit_transform(df[ordinal_cols])

    # Scaling
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Preprocessing selesai. File tersimpan di:", output_path)


if __name__ == "__main__":
    INPUT_PATH = "/content/drive/MyDrive/Eksperimen_SML_IkeMaya/heart_raw/heart.csv"
    OUTPUT_PATH = "/content/drive/MyDrive/Eksperimen_SML_IkeMaya/preprocessing/heart_preprocessing/heart_clean.csv"

    preprocess_data(INPUT_PATH, OUTPUT_PATH)
