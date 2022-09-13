import numpy as np
from dscribe.descriptors import ACSF
from rdkit import Chem
from rdkit.Chem import AllChem,Descriptors
from ase.build import molecule
from ase import Atoms as ASE_Atoms
from rdkit.ML.Descriptors import MoleculeDescriptors

def get_key_position(smi):    
    mol = Chem.MolFromSmiles(smi)
    mol2=AllChem.AddHs(mol)
    AllChem.EmbedMolecule(mol2)
    AllChem.MMFFOptimizeMolecule(mol2)
    positions = mol2.GetConformer().GetPositions()
    atoms = mol2.GetAtoms()
    atom_weights = [atom.GetMass() for atom in atoms]
    atom_weights = np.array([atom_weights,atom_weights,atom_weights]).T 
    weighted_pos = positions*atom_weights
    weight_center = np.round(weighted_pos.sum(axis=0)/atom_weights.sum(axis=0)[0],decimals=8)
    distance=[]
    for i in positions:
        distmat_from_key_atom = np.sqrt(np.sum((i - weight_center)**2,axis=0))
        distance.append(distmat_from_key_atom)
    nearest_atom_index = np.argmin(distance)
    return nearest_atom_index 
def get_acsf_dict(smi_set,species):    
    acsf_dict={}
    for i in smi_set:
        tmp_mol = AllChem.AddHs(Chem.MolFromSmiles(i))
        AllChem.EmbedMolecule(tmp_mol)
        AllChem.MMFFOptimizeMolecule(tmp_mol)
        period_table = Chem.GetPeriodicTable()
        positions = tmp_mol.GetConformer().GetPositions()
        atom_types = [period_table.GetElementSymbol(atom.GetAtomicNum()) for atom in tmp_mol.GetAtoms()]
        atoms = ASE_Atoms(symbols=atom_types,positions=positions)
        acsf = ACSF(
            species=species,
            rcut=6.0,
            g2_params=[[1, 1], [1, 2], [1, 3]],
            g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
        )
        acsf_molecule = acsf.create(atoms, positions=[get_key_position(i)])
        acsf_dict[i]=acsf_molecule
    return acsf_dict 


def GetRdkitDescriptorsdict(smi_set):
    des_dict={}   
    descs = [desc_name[0] for desc_name in Descriptors._descList]
    desc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
    for smiles in smi_set:
        descriptor = np.array(desc_calc.CalcDescriptors(Chem.MolFromSmiles(smiles)))
        des_dict[smiles]=descriptor
    return des_dict


def get_fp_dict(smi_set,radius=2,nBits=2048,useChirality=True):
    fp_dict={}
    for i in smi_set:
        tmp_mol = AllChem.AddHs(Chem.MolFromSmiles(i))
        AllChem.EmbedMolecule(tmp_mol)
        AllChem.MMFFOptimizeMolecule(tmp_mol)    
        fp = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(tmp_mol,radius=radius,nBits=nBits,useChirality=useChirality)
        fp_dict[i]=np.array(list(map(eval,list(fp.ToBitString()))))
    return fp_dict

from sklearn.preprocessing import OneHotEncoder
def get_oh(tem_smiles):    
    oh_enc = OneHotEncoder()
    tem_oh_array=np.array([i if isinstance(i,str) else '' for i in tem_smiles]).reshape(-1,1)
    tem_oh_desc=oh_enc.fit_transform(tem_oh_array).toarray()
    return tem_oh_desc 

def des_std(des_array):
    react_feat_all=des_array[:,des_array.max(axis=0) != des_array.min(axis=0)]
    react_feat_all=(react_feat_all-react_feat_all.min(axis=0))/(react_feat_all.min(axis=0)-react_feat_all.max(axis=0))
    return react_feat_all

