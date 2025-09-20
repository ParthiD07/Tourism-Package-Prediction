# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Parthi07/Tourism-Package-Prediction/tourism.csv"
tour_data = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier columns
tour_data.drop(columns=['Unnamed: 0', 'CustomerID'], inplace=True)

# Treating the incorrect ("Fe Male") value in Gender column
# Clean formatting: strip spaces and standardize case
tour_data['Gender'] = tour_data['Gender'].str.strip().str.title()

print("\n Gender value counts before correction:")
print(tour_data['Gender'].value_counts().to_string())

# Replace incorrect "Fe Male" with "Female"
tour_data.loc[tour_data["Gender"] == "Fe Male", "Gender"] = "Female"

print("\n Gender value counts after correction:")
print(tour_data['Gender'].value_counts().to_string())

target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = tour_data.drop(columns=[target_col])
y = tour_data[target_col]

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

X_train.to_csv("X_train.csv",index=False)
X_test.to_csv("X_test.csv",index=False)
y_train.to_csv("y_train.csv",index=False)
y_test.to_csv("y_test.csv",index=False)

files= ["X_train.csv","X_test.csv","y_train.csv","y_test.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="Parthi07/Tourism-Package-Prediction",
        repo_type="dataset",
    )
