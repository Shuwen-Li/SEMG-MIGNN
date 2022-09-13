from rdkit import Chem
import networkx as nx
import numpy as np
import dgl,torch
import sys
sys.path.append('..') 
from script.utils import CalcEDLocal,SPMS


desc_n=10
desc_m=20
def Get_DGL(sdf_file='',rdkit_mol=None,spms_acc=desc_n,elec_acc=7,removeHs=False,spms_radius=4):
    
    if sdf_file != '':
        if removeHs:
            mol = Chem.MolFromMolFile(sdf_file,removeHs=True,sanitize=False)
        else:
            mol = Chem.MolFromMolFile(sdf_file,removeHs=False,sanitize=False)
    elif rdkit_mol != None:
        mol = rdkit_mol
        
    atoms = mol.GetAtoms()
    atom_symbols = [atom.GetSymbol() for atom in atoms]
    bonds = mol.GetBonds()
    edges = [[bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()] for bond in bonds]
    pos = mol.GetConformer().GetPositions()
    
    spms_desc = []
    for key_atom_num in range(len(atom_symbols)):
        spms_ = SPMS(rdkit_mol=mol,desc_n=spms_acc,desc_m=spms_acc,key_atom_num=[key_atom_num],sphere_radius=spms_radius)
        spms_.GetSphereDescriptors()
        desc = spms_.sphere_descriptors.reshape(-1,)
        spms_desc.append(desc)
    spms_desc=np.array(spms_desc)
    ed = CalcEDLocal(mol,grid_num=elec_acc,basis='6-311g',skip_cusp=True)
    elec_desc,ele_pos = ed()
    elec_desc=elec_desc.reshape(-1,elec_acc**(3))
    spms_desc=spms_desc.reshape(-1,desc_n*desc_m)
    spms_elec_desc = np.concatenate([spms_desc,elec_desc],axis=1)
    
    G = nx.Graph()
    G.add_nodes_from(list(range(0,len(atom_symbols))))
    G.add_edges_from(edges)
    if dgl.__version__ == '0.4.3post2':
        dgl_g = dgl.graph(G)
    else:
        dgl_g=dgl.from_networkx(G)
    all_edges_0,all_edges_1 = dgl_g.all_edges()
    feat = torch.tensor(spms_elec_desc,dtype=torch.float32)
    dist = np.array([np.linalg.norm(pos[int(all_edges_0[i])]-pos[int(all_edges_1[i])]) for i in range(len(all_edges_0))]).reshape(-1,1)
    dist = torch.tensor(dist,dtype=torch.float32)
    
    dgl_g.edata['feat'] = dist
    dgl_g.ndata['feat'] = feat
    return dgl_g

def Get_SEMG(files):    
    file_graph_dict = {}
    all_desc_max = []
    all_desc_min = []
    for tmp_file in files:
        tmp_file_name = tmp_file.split('/')[-1].split('.')[0]
        tmp_g = Get_DGL(tmp_file)
        file_graph_dict[tmp_file_name] = tmp_g
        tmp_desc_max = np.array(tmp_g.ndata['feat']).max(axis=0)
        tmp_desc_min = np.array(tmp_g.ndata['feat']).min(axis=0)
        all_desc_max.append(tmp_desc_max)
        all_desc_min.append(tmp_desc_min)
        print(tmp_file_name)
    all_desc_max = np.array(all_desc_max)
    all_desc_min = np.array(all_desc_min)
    desc_max = all_desc_max.max(axis=0)
    desc_min = all_desc_min.min(axis=0)
    for key in file_graph_dict:
        file_graph_dict[key].ndata['feat_std'] = (file_graph_dict[key].ndata['feat'] - torch.tensor(desc_min))/(torch.tensor(desc_max) - torch.tensor(desc_min))
    return file_graph_dict    