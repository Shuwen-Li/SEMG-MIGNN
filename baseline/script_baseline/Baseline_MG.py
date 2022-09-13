import os
import torch
import numpy as np
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures,AllChem
from rdkit.Chem.rdmolops import FastFindRings
from collections import defaultdict
from script_baseline.SEMG import Get_DGL
def get_mg_nodes_feat(mol):
    
    atom_feats_dict = defaultdict(list)
    is_donor = defaultdict(int)
    is_acceptor = defaultdict(int)
    fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    mol.UpdatePropertyCache()
    FastFindRings(mol)
    mol_featurizer = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    mol_feats = mol_featurizer.GetFeaturesForMol(mol)
    mol_conformers = mol.GetConformers()
    assert len(mol_conformers) == 1
    geom = mol_conformers[0].GetPositions()
    for i in range(len(mol_feats)):
        if mol_feats[i].GetFamily() == 'Donor':
            node_list = mol_feats[i].GetAtomIds()
            for u in node_list:
                is_donor[u] = 1
        elif mol_feats[i].GetFamily() == 'Acceptor':
            node_list = mol_feats[i].GetAtomIds()
            for u in node_list:
                is_acceptor[u] = 1
    num_atoms = mol.GetNumAtoms()
    atoms = mol.GetAtoms()
    AllChem.ComputeGasteigerCharges(mol)
    atom_types = [atom.GetAtomicNum() for atom in atoms]
    rdkit_period_table = Chem.GetPeriodicTable()
    radius = np.array([rdkit_period_table.GetRvdw(item) for item in atom_types])
    charge=[atom.GetDoubleProp('_GasteigerCharge') for atom in atoms]
    conformer = mol.GetConformer()
    positions = conformer.GetPositions()
    h_u = []    
    for u in range(num_atoms):
        ato = mol.GetAtomWithIdx(u)
        symbol = ato.GetSymbol()
        atom_type = ato.GetAtomicNum()
        aromatic = ato.GetIsAromatic()
        hybridization = ato.GetHybridization()
        num_h = ato.GetTotalNumHs()
        atom_feats_dict['pos'].append(torch.FloatTensor(geom[u]))
        atom_feats_dict['node_type'].append(atom_type)   
        h_u += [
            int(symbol == x) for x in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']
        ]
        h_u.append(atom_type)
        h_u.append(is_acceptor[u])
        h_u.append(is_donor[u])
        h_u.append(int(aromatic))
        h_u += [
            int(hybridization == x)
            for x in (Chem.rdchem.HybridizationType.SP,
                      Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3)
        ]
        h_u.append(num_h)
    h_u=np.array(h_u).reshape(-1,15)
    return h_u

def Get_Baseline_MG_feat(file_list):#SEMG
    
    file_graph_dict = {}
    for index, tmp_file in enumerate(file_list):
        tmp_fn = tmp_file.split('/')[-1].split('.')[0]
        tmp_g = Get_DGL(tmp_file)
        file_graph_dict[tmp_fn] = tmp_g
        mol = Chem.MolFromMolFile(tmp_file,removeHs=False)
        num_atoms = mol.GetNumAtoms()
        print(tmp_file)
        atom_feats= get_mg_nodes_feat(mol)
        atom_feats = torch.tensor(atom_feats,dtype=torch.float32)
        tmp_g.ndata['feat'] = atom_feats

    return file_graph_dict

def PackMat(arr_list):
    N = len(arr_list)
    M = max([np.array(arr_list[i]).shape[0] for i in range(len(arr_list))])
    pack_mat = np.zeros([N, M, 15])
    for i in range(N):
        for j in range(len(arr_list[i])):
            pack_mat[i, :len(arr_list[i]), :] = arr_list[i][j]
    return pack_mat

def Get_Baseline_MG_feat_pack(file_list):
    all_atom_feats=[]
    feat_name_dict = {}
    for index, tmp_file in enumerate(file_list):
        tmp_fn = tmp_file.split('/')[-1].split('.')[0]
        feat_name_dict[tmp_fn] = index
        mol = Chem.MolFromMolFile(tmp_file,removeHs=False)
        num_atoms = mol.GetNumAtoms()
        print(tmp_file)
        atom_feats=[]
        for i in range(num_atoms):
            feat= get_mg_nodes_feat(mol)
            atom_feats.append(feat)
        all_atom_feats.append(atom_feats)
    all_atom_feats=np.array(all_atom_feats)
    all_atom_feats = PackMat(all_atom_feats)   
    return all_atom_feats,feat_name_dict

