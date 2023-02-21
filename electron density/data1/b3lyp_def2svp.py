from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto,scf,dft,lib
import numpy as np
from pyscf.dft import numint
import time
import pandas as pd
from pyscf.tools import cubegen
import sys

#all_basis=['def2-svp','def2-tzvpd','def2-qzvpp','cc-pv5z','aug-cc-pvtz','aug-cc-pvqz-ri']#'3-21G',
#all_df_name=['lda','b3lyp','m06-2x','m06','wb97XD','PBE0','M06l','pbe','tpss']
#tem_file='./data2/data2_sdf_files/thiol_0.sdf'
unit_convert = 0.529177    # 1 Bohr = 0.529117 A
def cal_dm(df_name,basis,tem_file,grid_num,file_dir,skip_cusp=True,coord_ang=True,charge_judge=False):
    print(tem_file)
    start = time.time()
    basis_df_time=[]
    #basis_df_dm={}
    rdkit_period_table = Chem.GetPeriodicTable()
    
    file_name=tem_file.split('/')[-1].split('.')[0]
    mol=Chem.MolFromXYZFile(tem_file)

    symbols = [tmp_atom.GetSymbol() for tmp_atom in mol.GetAtoms()]
    positions = mol.GetConformer().GetPositions()
    mol_string = ''
    for i in range(len(positions)):
        mol_string += '%5s %15f %15f %15f;'%(symbols[i],positions[i][0],positions[i][1],positions[i][2])
    if charge_judge:
        AllChem.ComputeGasteigerCharges(mol)
        atoms = mol.GetAtoms()
        charge = np.array([atom.GetDoubleProp('_GasteigerCharge') for atom in atoms]).sum()
        py_mol = gto.M(atom=mol_string,charge=int(round(charge)),basis=basis)
    else:

        py_mol = gto.M(atom=mol_string,basis=basis)

    radius = np.array([rdkit_period_table.GetRvdw(item) for item in symbols])
    all_grid_coords = []
    all_density = []
    mf_df = dft.RKS(py_mol,xc=df_name)#.run()
    mf_df.conv_tol = 1e-6
    mf_df.max_cycle = 50
    mf_df.level_shift = 0.1
    mf_df.diis_space = 12
    mf_df.max_memory = 40000
    mf_df.run()
    for at_id in range(len(radius)):
        d = radius[at_id]*2
        if grid_num % 2 == 1:
            n_range = list(range(round(-(grid_num-1)/2),round((grid_num+1)/2)))
            grid_a = [n*d/grid_num for n in n_range]
            cubic_array = np.array([[x,y,z] for x in grid_a for y in grid_a for z in grid_a])
        else:
            n_range = list(range(-round(grid_num/2),round(grid_num/2)))
            grid_a = [d/2/grid_num + n*d/grid_num for n in n_range]
            cubic_array = np.array([[x,y,z] for x in grid_a for y in grid_a for z in grid_a])   ## 电荷分布网格中心计算
        grid_point_coord = (cubic_array+positions[at_id]) / unit_convert ## coord unit Bohr
        all_grid_coords.append(grid_point_coord)
        if df_name == 'lda':
            ao_value = numint.eval_ao(py_mol, grid_point_coord) # AO value and its gradients
            ed = numint.eval_rho2(py_mol, ao_value, mo_coeff=mf_df.mo_coeff,mo_occ=mf_df.mo_occ, xctype='LDA')
        else:
            ao_value = numint.eval_ao(py_mol, grid_point_coord, deriv=1) # AO value and its gradients
            ed_tem = numint.eval_rho2(py_mol, ao_value, mo_coeff=mf_df.mo_coeff,mo_occ=mf_df.mo_occ, xctype='GGA')
            ed=ed_tem[0]
        if grid_num % 2 == 1 and skip_cusp:
            ed[grid_num*grid_num*grid_num//2] = 0
        all_density.append(ed)
    all_grid_coords = np.array(all_grid_coords)
    all_density = np.array(all_density)
    grid_coords_def2 = all_grid_coords.reshape(len(radius),grid_num,grid_num,grid_num,3)
    if coord_ang:
        grid_coords_def2 = grid_coords_def2 * unit_convert
        all_grid_coords = all_grid_coords * unit_convert
    grid_ed_def2 = all_density.reshape(len(radius),grid_num,grid_num,grid_num)
    #basis_df_dm['%s_%s'%(basis,df_name)] = [grid_ed,grid_coords_def2]
    end = time.time()
    mf_df.chkfile = file_dir+'%s_%s_%s.chk'%(basis,df_name,file_name)
    basis_df_time.append([basis,df_name,end-start,mf_df.chkfile])
    pd.DataFrame(basis_df_time).to_csv(file_dir+'%s_%s_%s_time.csv'%(basis,df_name,file_name))
    cubegen.density(py_mol, file_dir+"%s_%s_%s.cube"%(basis,df_name,file_name), mf_df.make_rdm1())
    lib.chkfile.dump(mf_df.chkfile, "dm", grid_ed_def2)
    np.save( file_dir+"%s_%s_%s.npy"%(basis,df_name,file_name),grid_ed_def2)
    print(basis,df_name,end-start)



cal_dm(df_name='b3lyp',basis='def2-svp',tem_file='/public3/home/scg5914/software-scg5914/LSW/1/data/data1_xyz_std_b3lyp/'+sys.argv[1],file_dir='/public3/home/scg5914/software-scg5914/LSW/1/data/b3lyp_def2svp_data1_save_files/',grid_num=7)