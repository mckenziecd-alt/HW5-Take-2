#AI.ML.HW5P2 from HW4 Code

# Imports
import os
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.AllChem import GetMorganGenerator, GetRDKitFPGenerator
from rdkit import DataStructs
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
path = r"C:\Users\d0m1n\Desktop\VSCode\AI.ML.ENGR\AI.ML_HW5P2\HW5P2\Model_Evaluations\Lipophilicity.csv"
df = pd.read_csv(path)
df.head()

# Label Columns
SMILES_COL = 'smiles'
TARGET_COL = 'exp'

# Convert SMILES to Molecular Fingerprints
df['Mol'] = df[SMILES_COL].apply(Chem.MolFromSmiles)

# Clean Data
df = df.dropna(subset=['Mol', TARGET_COL])
y = df[TARGET_COL].values

#import from Generate_Fingerprints.py
from Generate_Fingerprints import morgan_fp, maccs_fp

# Generate Features
X_morgan = np.vstack(df['Mol'].apply(morgan_fp))
X_maccs = np.vstack(df['Mol'].apply(maccs_fp))

# Split Dataset
X_morgan_train, X_morgan_test, y_train, y_test = train_test_split(
    X_morgan, y, test_size=0.2, random_state=42
    )
X_maccs_train, X_maccs_test, _, _ = train_test_split(
    X_maccs, y, test_size=0.2, random_state=42
)

# Scale Features
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

# MLP Regressor Parameters
mlp_params = {
    'hidden_layer_sizes': (100, 50),
    'activation': 'relu',
    'solver': 'adam',
    'max_iter': 500,
    'random_state': 42
}

# Morgan Fingerprint Model
morgan_model = MLPRegressor(**mlp_params)
morgan_model.fit(X_morgan_train, y_train)

# Generate Predictions
morgan_predictions_scaled = morgan_model.predict(X_morgan_test)
morgan_predictions_scaled = morgan_predictions_scaled.reshape(-1, 1)
morgan_predictions = y_scaler.inverse_transform(morgan_predictions_scaled).ravel()

# Evaluate Morgan Model
morgan_rmse = np.sqrt(mean_squared_error(y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel(), morgan_predictions))
morgan_r2 = r2_score(y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel(), morgan_predictions)

print(f"Morgan Fingerprint Model - RMSE: {morgan_rmse:.4f}, R2: {morgan_r2:.4f}")

# MACCS Keys Model
maccs_model = MLPRegressor(**mlp_params)
maccs_model.fit(X_maccs_train, y_train)

# Generate Predictions
maccs_predictions_scaled = maccs_model.predict(X_maccs_test)
maccs_predictions_scaled = maccs_predictions_scaled.reshape(-1, 1)
maccs_predictions = y_scaler.inverse_transform(maccs_predictions_scaled).ravel()

# Evaluate MACCS Model
maccs_rmse = np.sqrt(mean_squared_error(y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel(), maccs_predictions))
maccs_r2 = r2_score(y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel(), maccs_predictions)
print(f"MACCS Keys Model - RMSE: {maccs_rmse:.4f}, R2: {maccs_r2:.4f}")

# Get Environment
os.getenv("CONDA_DEFAULT_ENV")
print(os.getenv("CONDA_DEFAULT_ENV"))