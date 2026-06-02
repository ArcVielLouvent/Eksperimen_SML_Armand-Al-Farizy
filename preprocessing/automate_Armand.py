import pandas as pd
import os


def load_data(file_path):
    print("Membaca data raw...")
    return pd.read_csv(file_path)


def clean_data(df):
    print("Melakukan data cleaning...")
    df_clean = df.dropna()
    df_clean = df_clean.drop_duplicates()
    return df_clean


def save_data(df, output_path):
    print(f"Menyimpan data bersih ke {output_path}...")
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    raw_path = "../dataset_raw/data_raw.csv"
    output_dir = "dataset_preprocessing"
    os.makedirs(output_dir, exist_ok=True)
    clean_path = f"{output_dir}/data_clean.csv"

    if os.path.exists(raw_path):
        df_raw = load_data(raw_path)
        df_cleaned = clean_data(df_raw)
        save_data(df_cleaned, clean_path)
        print("Pipeline Preprocessing Selesai!")
    else:
        print(f"Error: File {raw_path} tidak ditemukan.")
