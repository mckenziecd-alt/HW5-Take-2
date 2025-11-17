#Generate Fingerprints
# Load imports
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
import numpy as np

# Generate Morgan Fingerprint Generator
def morgan_fp(mol, radius=2, fpSize=2048):
    if mol is None:
        return np.zeros((fpSize,), dtype=np.int8)
    generator = AllChem.GetMorganGenerator(radius=radius, fpSize=fpSize)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((fpSize,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# Generate MACCS Keys
def maccs_fp(mol):
    if mol is None:
        return np.zeros((167,), dtype=np.int8)
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((167,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr