# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import OneHotEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/akskhare/Tourism-Packages/tourism_data.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier
df.drop(columns=['CustomerID'], inplace=True)

# Encoding the categorical columns using OneHotEncoder
categorical_cols = ['TypeofContact','Occupation','Gender','ProductPitched','MaritalStatus','Designation']
onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Fit and transform the categorical columns
X_categorical = df[categorical_cols]
X_encoded = onehot_encoder.fit_transform(X_categorical)

# Create a DataFrame with the one-hot encoded features
encoded_feature_names = onehot_encoder.get_feature_names_out(categorical_cols)
X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_feature_names, index=df.index)

# Drop the original categorical columns and concatenate with the encoded ones
df = df.drop(columns=categorical_cols)
df = pd.concat([df, X_encoded_df], axis=1)

target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="akskhare/Tourism-Packages",
        repo_type="dataset",
    )
