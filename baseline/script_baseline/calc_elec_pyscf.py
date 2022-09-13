# -*- coding: utf-8 -*-

from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto,scf
import numpy as np
from pyscf.dft import numint
rdkit_period_table = Chem.GetPeriodicTable()
unit_convert = 0.529177    # 1 Bohr = 0.529117 A
class CalcEDLocal():
    def __init__(self,mol,grid_num=5,basis='3-21G',GTOval='GTOval',skip_cusp=True,coord_ang=True,charge_judge=False):
        self.mol = mol
        self.grid_num = grid_num
        self.basis = basis
        self.GTOval = GTOval
        self.skip_cusp = skip_cusp
        self.coord_ang = coord_ang
        self.charge_judge = charge_judge
    def __call__(self):
        grid_num = self.grid_num
        symbols = [tmp_atom.GetSymbol() for tmp_atom in self.mol.GetAtoms()]
        positions = self.mol.GetConformer().GetPositions()
        mol_string = ''
        for i in range(len(positions)):
            mol_string += '%5s %15f %15f %15f;'%(symbols[i],positions[i][0],positions[i][1],positions[i][2])
        if self.charge_judge:
            AllChem.ComputeGasteigerCharges(self.mol)
            atoms = self.mol.GetAtoms()
            charge = np.array([atom.GetDoubleProp('_GasteigerCharge') for atom in atoms]).sum()
            py_mol = gto.M(atom=mol_string,charge=int(round(charge)),basis=self.basis)
        else:
            
            py_mol = gto.M(atom=mol_string,basis=self.basis)
        mf = scf.HF(py_mol)
        dm = mf.get_init_guess()

        radius = np.array([rdkit_period_table.GetRvdw(item) for item in symbols])
        all_grid_coords = []
        all_density = []
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
            ao = py_mol.eval_ao(self.GTOval, grid_point_coord)
            ed = numint.eval_rho(py_mol,ao,dm)   # 'Electron density in real space (e/Bohr^3)'
            if grid_num % 2 == 1 and self.skip_cusp:
                ed[grid_num*grid_num*grid_num//2] = 0
            all_density.append(ed)
        all_grid_coords = np.array(all_grid_coords)
        all_density = np.array(all_density)
        grid_coords = all_grid_coords.reshape(len(radius),grid_num,grid_num,grid_num,3)
        if self.coord_ang:
            grid_coords = grid_coords * unit_convert
            all_grid_coords = all_grid_coords * unit_convert
        grid_ed = all_density.reshape(len(radius),grid_num,grid_num,grid_num)
        self.all_density = all_density
        self.all_grid_coords = all_grid_coords
        return grid_ed,grid_coords