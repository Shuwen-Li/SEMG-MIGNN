# -*- coding: utf-8 -*-
from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
import numpy as np
import pandas as pd
from copy import deepcopy
import dgl
import tensorflow as tf

precision = 8
#sphere_radius unit:A
class SPMS():
    def __init__(self,sdf_file='',rdkit_mol=None,key_atom_num=None,sphere_radius=None,desc_n=40,desc_m=40,
                 orientation_standard=True,first_point_index_list=None,second_point_index_list=None,third_point_index_list=None):
        
        self.sdf_file = sdf_file
        self.sphere_radius = sphere_radius
        if key_atom_num != None:
            key_atom_num = list(np.array(key_atom_num,dtype=np.int)-1)
            self.key_atom_num = key_atom_num
        else:
            self.key_atom_num = []
        self.desc_n = desc_n
        self.desc_m = desc_m
        self.orientation_standard = orientation_standard
        self.first_point_index_list = first_point_index_list
        self.second_point_index_list = second_point_index_list
        self.third_point_index_list = third_point_index_list
        rdkit_period_table = Chem.GetPeriodicTable()
        if sdf_file == '':
            mol = rdkit_mol
        else:
            mol = Chem.MolFromMolFile(sdf_file,removeHs=False,sanitize=False)
        conformer = mol.GetConformer()
        positions = conformer.GetPositions()
        atoms = mol.GetAtoms()
        atom_types = [atom.GetAtomicNum() for atom in atoms]
        atom_symbols = [rdkit_period_table.GetElementSymbol(item) for item in atom_types]
        atom_weights = [atom.GetMass() for atom in atoms]     
        atom_weights = np.array([atom_weights,atom_weights,atom_weights]).T
        weighted_pos = positions*atom_weights
        weight_center = np.round(weighted_pos.sum(axis=0)/atom_weights.sum(axis=0)[0],decimals=precision)
        radius = np.array([rdkit_period_table.GetRvdw(item) for item in atom_types]) # van der Waals radius
        volume = 4/3*np.pi*pow(radius,3)
        self.positions = positions
        self.weight_center = weight_center
        self.radius = radius
        self.volume = volume
        self.atom_types = atom_types
        self.atom_symbols = atom_symbols
        self.rdkit_period_table = rdkit_period_table
        self.atom_weight = atom_weights
    def _Standarlize_Geomertry_Input(self,origin_positions):
        
        if self.key_atom_num == []:
            key_atom_position = deepcopy(self.weight_center)
            key_atom_position = key_atom_position.reshape(1,3)
            distmat_from_key_atom = np.sqrt(np.sum((origin_positions - key_atom_position)**2,axis=1))
            farest_atom_index = np.argmax(distmat_from_key_atom)
            distmat_from_key_atom[np.argmin(distmat_from_key_atom)] = np.max(distmat_from_key_atom)
            nearest_atom_index = np.argmin(distmat_from_key_atom)
            second_key_atom_index = nearest_atom_index
            third_key_atom_index = farest_atom_index
            second_atom_position = deepcopy(origin_positions[second_key_atom_index])
            second_atom_position = second_atom_position.reshape(1,3)
            third_atom_position = deepcopy(origin_positions[third_key_atom_index])
            third_atom_position = third_atom_position.reshape(1,3)
            append_positions = np.concatenate([origin_positions,key_atom_position,second_atom_position,third_atom_position])
            
        else:
            key_atom_num = self.key_atom_num
            if len(key_atom_num) == 1:
                key_atom_position = deepcopy(origin_positions[key_atom_num[0]])
                key_atom_position = key_atom_position.reshape(1,3)
                distmat_from_key_atom = np.sqrt(np.sum((origin_positions - key_atom_position)**2,axis=1))
                farest_atom_index = np.argmax(distmat_from_key_atom)
                distmat_from_key_atom[np.argmin(distmat_from_key_atom)] = np.max(distmat_from_key_atom)
                nearest_atom_index = np.argmin(distmat_from_key_atom)
                second_key_atom_index = nearest_atom_index
                third_key_atom_index = farest_atom_index
                second_atom_position = deepcopy(origin_positions[second_key_atom_index])
                second_atom_position = second_atom_position.reshape(1,3)
                third_atom_position = deepcopy(origin_positions[third_key_atom_index])
                third_atom_position = third_atom_position.reshape(1,3)
                append_positions = np.concatenate([origin_positions,key_atom_position,second_atom_position,third_atom_position])

            elif len(key_atom_num) >= 2:
                key_atom_position = origin_positions[key_atom_num].mean(axis=0)
                key_atom_position = key_atom_position.reshape(1,3)
                distmat_from_key_atom = np.sqrt(np.sum((origin_positions - key_atom_position)**2,axis=1))
                farest_atom_index = np.argmax(distmat_from_key_atom)
                distmat_from_key_atom[np.argmin(distmat_from_key_atom)] = np.max(distmat_from_key_atom)
                nearest_atom_index = np.argmin(distmat_from_key_atom)
                second_key_atom_index = nearest_atom_index
                third_key_atom_index = farest_atom_index
                second_atom_position = deepcopy(origin_positions[second_key_atom_index])
                second_atom_position = second_atom_position.reshape(1,3)
                third_atom_position = deepcopy(origin_positions[third_key_atom_index])
                third_atom_position = third_atom_position.reshape(1,3)
                append_positions = np.concatenate([origin_positions,key_atom_position,second_atom_position,third_atom_position])
                
        OldCoord = np.c_[append_positions, np.ones(len(append_positions))]      
        first_atom_coord = OldCoord[-3][0:3]        
        second_atom_coord = OldCoord[-2][0:3]
        Xv =  second_atom_coord-first_atom_coord
        Xv_xy = Xv.copy()
        Xv_xy[2] = 0
        X_v = np.array([Xv[0],0,0])
        Z_v = np.array([0,0,1])
        alpha = np.arccos(Xv_xy[0:3].dot(
                X_v[0:3])/(np.sqrt(Xv_xy[0:3].dot(Xv_xy[0:3]))*np.sqrt(X_v[0:3].dot(X_v[0:3]))))
        beta = np.arccos(Xv[0:3].dot(
                Z_v)/(np.sqrt(Xv[0:3].dot(Xv[0:3]))*np.sqrt(Z_v.dot(Z_v))))
        
        if Xv_xy[1]*Xv_xy[0] > 0:
            alpha = -alpha
        if Xv[0] < 0:
            beta = -beta    
        def T_M(a):
            T_M = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [
                           0, 0, 1, 0], [a[0], a[1], a[2], 1]])
            return T_M
        
        def RZ_alpha_M(alpha):
            RZ_alpha_M = np.array([[np.cos(alpha), np.sin(
                alpha), 0, 0], [-np.sin(alpha), np.cos(alpha), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            return RZ_alpha_M
        
        def RY_beta_M(beta):
            RY_beta_M = np.array([[np.cos(beta), 0, np.sin(beta), 0], [
                                 0, 1, 0, 0], [-np.sin(beta), 0, np.cos(beta), 0], [0, 0, 0, 1]])
            return RY_beta_M
        
        a = -first_atom_coord         
        new_xyz_coord1 = OldCoord.dot(T_M(a)).dot(
            RZ_alpha_M(alpha)).dot(RY_beta_M(beta))                    
        third_atom_coord = new_xyz_coord1[-1][0:3]
        second_atom_coord = new_xyz_coord1[-2][0:3]
        Xy = third_atom_coord - second_atom_coord
        Y_v = np.array([0, 1, 0])
        gamma = np.arccos(Xy.dot(Y_v)/(np.sqrt(Xy.dot(Xy))*np.sqrt(Y_v.dot(Y_v))))
        if Xy[0] < 0:
            gamma = -gamma
        NewCoord = new_xyz_coord1.dot(RZ_alpha_M(gamma))        
        third_atom_coord = NewCoord[-1][0:3]
        third_XY = third_atom_coord[0:2]
        axis_y_2d = np.array([0,1])
        sita = np.arccos(third_XY.dot(axis_y_2d)/(np.sqrt(third_XY.dot(third_XY))*np.sqrt(axis_y_2d.dot(axis_y_2d))))
        if third_XY[0]*third_XY[1] < 0:
            sita = -sita
        NewCoord0 = NewCoord.dot(RZ_alpha_M(sita))
        NewCoord1 = np.around(np.delete(NewCoord0, 3, axis=1), decimals=precision)           
        NewCoord2 = NewCoord1[:-3]
        New3Points = NewCoord1[-3:]

        return NewCoord2,New3Points        
    def _Standarlize_Geomertry_Input_Less(self,origin_positions):
        assert len(self.key_atom_num) <= 2, 'two many key atoms'
        key_atom_num = self.key_atom_num
        if len(origin_positions) == 1:
            return np.array([[0.0,0.0,0.0]])
        
        elif len(origin_positions) == 2:
            dist = np.sqrt(np.sum((origin_positions[0] - origin_positions[1])**2))
            if key_atom_num == [] or len(key_atom_num)==2:
                return np.array([[0.0,0.0,dist/2],
                                 [0.0,0.0,-dist/2]])
            elif len(key_atom_num) == 1:
                coord = np.array([[0.0,0.0,dist],
                                  [0.0,0.0,dist]])
                coord[key_atom_num[0]] = np.array([0.0,0.0,0.0])                                 
                return coord            
        
    def _Customized_Coord_Standard(self,positions,first_point_index_list,second_point_index_list,third_point_index_list):
        def T_M(a):            ### translation
            T_M = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [
                           0, 0, 1, 0], [a[0], a[1], a[2], 1]])
            return T_M
        
        def RZ_alpha_M(alpha):
            RZ_alpha_M = np.array([[np.cos(alpha), np.sin(
                alpha), 0, 0], [-np.sin(alpha), np.cos(alpha), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            return RZ_alpha_M
        
        def RY_beta_M(beta):
            RY_beta_M = np.array([[np.cos(beta), 0, np.sin(beta), 0], [
                                 0, 1, 0, 0], [-np.sin(beta), 0, np.cos(beta), 0], [0, 0, 0, 1]])
            return RY_beta_M
        def RX_gamma_M(gamma):
            RX_gamma_M = np.array([[1,0,0,0],[0,np.cos(gamma),np.sin(gamma),0],[0,-np.sin(gamma),np.cos(gamma),0],[0,0,0,1]])
            return RX_gamma_M
        
        first_point_index_list = [item-1 for item in first_point_index_list]
        second_point_index_list = [item-1 for item in second_point_index_list]
        third_point_index_list = [item-1 for item in third_point_index_list]
        OldCoord = np.c_[positions, np.ones(len(positions))]
        first_point_coord = np.mean(OldCoord[first_point_index_list],axis=0)[0:3]
        second_point_coord = np.mean(OldCoord[second_point_index_list],axis=0)[0:3]
        Xv =  second_point_coord-first_point_coord
        Xv_xy = Xv.copy()
        Xv_xy[2] = 0
        Y_v_neg = np.array([0,-1,0])
        Y_v_pos = np.array([0,1,0])
        alpha = np.arccos(Xv_xy[0:3].dot(Y_v_neg[0:3])/((np.sqrt(Xv_xy[0:3].dot(Xv_xy[0:3])))*(np.sqrt(Y_v_neg[0:3].dot(Y_v_neg[0:3])))))       
        a = -first_point_coord
        
        if Xv_xy[0] > 0:
            alpha = -alpha
        new_xyz_coord = OldCoord.dot(T_M(a))     ### translation done
        new_xyz_coord1 = new_xyz_coord.dot(RZ_alpha_M(alpha))
        first_point_coord1 = np.mean(new_xyz_coord1[first_point_index_list],axis=0)[0:3]
        second_point_coord1 = np.mean(new_xyz_coord1[second_point_index_list],axis=0)[0:3]
        Xv1 =  second_point_coord1-first_point_coord1
        Xv1_yz = Xv1.copy()
        Xv1_yz[0] = 0
        gamma = np.pi-np.arccos(Xv1_yz[0:3].dot(Y_v_pos)/((np.sqrt(Xv1_yz[0:3].dot(Xv1_yz[0:3])))*(np.sqrt(Y_v_pos[0:3].dot(Y_v_pos[0:3])))))
        if Xv1[2] < 0:
            gamma = -gamma
        new_xyz_coord2 = new_xyz_coord1.dot(RX_gamma_M(gamma))    ### put one point at the negative y axis        
        ### rotate around y axis
        third_point_coord = np.mean(new_xyz_coord2[third_point_index_list],axis=0)[0:3]
        Xv3 = third_point_coord.copy()
        Xv3_xz = Xv3.copy()
        Xv3_xz[1] = 0
        X_v_pos = np.array([1,0,0])
        beta = np.arccos(Xv3_xz[0:3].dot(X_v_pos[0:3])/((np.sqrt(Xv3_xz[0:3].dot(Xv3_xz[0:3])))*(np.sqrt(X_v_pos[0:3].dot(X_v_pos[0:3])))))
        if Xv3[2] > 0:
            beta = -beta
        new_xyz_coord3 = new_xyz_coord2.dot(RY_beta_M(beta))
        return new_xyz_coord3[:,0:3]       
    
    def _Standarlize_Geomertry(self):
        if self.orientation_standard == True:
            if len(self.positions) > 2:
                new_positions,new_3points = self._Standarlize_Geomertry_Input(self.positions)
            else:
                new_positions = self._Standarlize_Geomertry_Input_Less(self.positions)
                new_3points = np.array([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
            if self.key_atom_num != None:
                bias_move = np.array([0.000001,0.000001,0.000001])
                new_positions += bias_move
                new_3points += bias_move
            new_geometric_center,new_weight_center,new_weight_center_2 = new_3points[0],new_3points[1],new_3points[2]
            self.new_positions = new_positions
            self.new_geometric_center,self.new_weight_center,self.new_weight_center_2 = new_geometric_center,new_weight_center,new_weight_center_2        
        elif self.orientation_standard == False:
            new_positions = self.positions
            self.new_positions = self.positions              
        elif self.orientation_standard == "Customized":
            new_positions = self._Customized_Coord_Standard(self.positions,self.first_point_index_list,self.second_point_index_list,self.third_point_index_list)
            self.new_positions = new_positions
        distances = np.sqrt(np.sum(new_positions**2,axis=1))
        self.distances = distances
        distances_plus_radius = distances + self.radius
        sphere_radius = np.ceil(distances_plus_radius.max())
        if self.sphere_radius == None:
            self.sphere_radius = sphere_radius
        
    def _polar2xyz(self,r,theta,fi):
        x = r*np.sin(theta)*np.cos(fi)
        y = r*np.sin(theta)*np.sin(fi)
        z = r*np.cos(theta)
        return np.array([x,y,z])
    
    def _xyz2polar(self,x,y,z):
        # theta 0-pi
        # fi 0-2pi
        r = np.sqrt(x**2+y**2+z**2)
        theta = np.arcsin(np.sqrt(x**2+y**2)/r)
        fi = np.arctan(y/x)
        if z < 0:
            theta = np.pi - theta
        if x < 0 and y > 0:
            fi = np.pi + fi
        elif x < 0 and y < 0:
            fi = np.pi + fi
        elif x > 0 and y < 0:
            fi = 2*np.pi + fi
        return np.array([r,theta,fi])
    
    def Writegjf(self,file_path):
        try:
            new_positions = self.new_positions
        except:
            self._Standarlize_Geomertry()
            new_positions = self.new_positions
        atom_types = self.atom_types
        coord_string = ''
        for at_ty,pos in zip(atom_types,new_positions):
            coord_string += '%10d %15f %15f %15f \n'%(at_ty,pos[0],pos[1],pos[2])
        string = '#p\n\nT\n\n0 1\n' + coord_string + '\n'
        with open(file_path,'w') as fw:
            fw.writelines(string)           
            
    def Writexyz(self,file_path):
        try:
            new_positions = self.new_positions
        except:
            self._Standarlize_Geomertry()
            new_positions = self.new_positions
        atom_symbols = self.atom_symbols
        
        atom_num = len(atom_symbols)
        coord_string = '%d\ntitle\n'%atom_num
        for at_sy,pos in zip(atom_symbols,new_positions):
            coord_string += '%10s %15f %15f %15f \n'%(at_sy,pos[0],pos[1],pos[2])
        with open(file_path,'w') as fw:
            fw.writelines(coord_string)
    def GetSphereDescriptors(self):
        self._Standarlize_Geomertry()
        new_positions = self.new_positions        
        radius = self.radius
        sphere_radius = self.sphere_radius
        N = self.desc_n
        M = self.desc_m
        delta_theta = 1/N * np.pi
        delta_fi = 1/M * np.pi
        theta_screenning = np.array([item*delta_theta for item in range(1,N+1)])
        self.theta_screenning = theta_screenning
        fi_screenning = np.array([item*delta_fi for item in range(1,M*2+1)])
        self.fi_screenning = fi_screenning
        PHI, THETA = np.meshgrid(fi_screenning, theta_screenning)
        x = sphere_radius*np.sin(THETA)*np.cos(PHI)
        y = sphere_radius*np.sin(THETA)*np.sin(PHI)
        z = sphere_radius*np.cos(THETA)
        mesh_xyz = np.array([[x[i][j],y[i][j],z[i][j]] for i in range(theta_screenning.shape[0]) for j in range(fi_screenning.shape[0])])
        self.mesh_xyz = mesh_xyz
        psi = np.linalg.norm(new_positions,axis=1)
        atom_vec = deepcopy(new_positions)
        self.psi = psi
        all_cross = []
        for j in range(atom_vec.shape[0]): 
            all_cross.append(np.cross(atom_vec[j].reshape(-1,3),mesh_xyz,axis=1)) 
        all_cross = np.array(all_cross)
        all_cross = all_cross.transpose(1,0,2)
        self.all_cross = all_cross
        mesh_xyz_h = np.linalg.norm(all_cross,axis=2)/sphere_radius

        dot = np.dot(mesh_xyz,atom_vec.T)
        atom_vec_norm = np.linalg.norm(atom_vec,axis=1).reshape(-1,1)
        mesh_xyz_norm = np.linalg.norm(mesh_xyz,axis=1).reshape(-1,1)
        self.mesh_xyz_norm = mesh_xyz_norm
        self.atom_vec_norm = atom_vec_norm       
        orthogonal_mesh = dot/np.dot(mesh_xyz_norm,atom_vec_norm.T)       
        self.mesh_xyz_h = mesh_xyz_h       
        self.orthogonal_mesh = orthogonal_mesh        
        cross_det = mesh_xyz_h <= radius
        orthogonal_det = np.arccos(orthogonal_mesh) <= np.pi*0.5
        double_correct = np.array([orthogonal_det,cross_det]).all(axis=0)
        double_correct_index = np.array(np.where(double_correct==True)).T
        self.double_correct_index = double_correct_index
        d_1 = np.zeros(mesh_xyz_h.shape)
        d_2 = np.zeros(mesh_xyz_h.shape)
        for item in double_correct_index:            
            d_1[item[0]][item[1]] = max( (psi[item[1]]**2 - mesh_xyz_h[item[0]][item[1]]**2) ,0)**0.5
            d_2[item[0]][item[1]]=(radius[item[1]]**2 - mesh_xyz_h[item[0]][item[1]]**2)**0.5
        self.d_1 = d_1
        self.d_2 = d_2       
        sphere_descriptors = sphere_radius - d_1 - d_2
        sphere_descriptors_compact = sphere_descriptors.min(1)
        sphere_descriptors_reshaped = sphere_descriptors_compact.reshape(PHI.shape)
        sphere_descriptors_reshaped = sphere_descriptors_reshaped.round(precision)       
        if len(self.key_atom_num) == 1:
            sphere_descriptors_init = np.zeros((theta_screenning.shape[0],fi_screenning.shape[0])) + sphere_radius - self.radius[self.key_atom_num[0]]
            sphere_descriptors_final = np.min(np.concatenate([sphere_descriptors_reshaped.reshape(theta_screenning.shape[0],fi_screenning.shape[0],1),sphere_descriptors_init.reshape(theta_screenning.shape[0],fi_screenning.shape[0],1)],axis=2),axis=2)
        else:
            sphere_descriptors_final = sphere_descriptors_reshaped        
        self.PHI = PHI
        self.THETA = THETA
        self.sphere_descriptors = sphere_descriptors_final
        
    def GetLocalSphereDescriptors(self):
        
        if len(self.key_atom_num) != 1:
            raise ValueError('the number of key atom should be 1')        
        self._Standarlize_Geomertry()
        new_positions = self.new_positions 
        sphere_radius = self.sphere_radius
        local_positions = self.new_positions[self.distances + self.radius<=self.sphere_radius]
        local_radius = self.radius[self.distances + self.radius<=self.sphere_radius]
        local_distances = self.distances[self.distances + self.radius<=self.sphere_radius]      
        self.local_positions = local_positions
        self.local_radius = local_radius
        self.local_distances = local_distances
        self.local_key_atom = [np.argmin(local_distances)]        
        if len(local_positions) < 2:
            raise ValueError('the sphere radius is too small')
        N = self.desc_n
        M = self.desc_m
        delta_theta = 1/N * np.pi
        delta_fi = 1/M * np.pi
        theta_screenning = np.array([item*delta_theta for item in range(1,N+1)])
        self.theta_screenning = theta_screenning
        fi_screenning = np.array([item*delta_fi for item in range(1,M*2+1)])
        self.fi_screenning = fi_screenning
        PHI, THETA = np.meshgrid(fi_screenning, theta_screenning)        
        x = sphere_radius*np.sin(THETA)*np.cos(PHI)
        y = sphere_radius*np.sin(THETA)*np.sin(PHI)
        z = sphere_radius*np.cos(THETA)
        mesh_xyz = np.array([[x[i][j],y[i][j],z[i][j]] for i in range(theta_screenning.shape[0]) for j in range(fi_screenning.shape[0])])
        self.mesh_xyz = mesh_xyz
        psi = np.linalg.norm(local_positions,axis=1)
        atom_vec = deepcopy(local_positions)
        self.psi = psi
        all_cross = []
        for j in range(atom_vec.shape[0]):
            all_cross.append(np.cross(atom_vec[j].reshape(-1,3),mesh_xyz,axis=1)) 
        all_cross = np.array(all_cross)
        all_cross = all_cross.transpose(1,0,2)
        self.all_cross = all_cross
        mesh_xyz_h = np.linalg.norm(all_cross,axis=2)/sphere_radius
        
        dot = np.dot(mesh_xyz,atom_vec.T)
        atom_vec_norm = np.linalg.norm(atom_vec,axis=1).reshape(-1,1)
        mesh_xyz_norm = np.linalg.norm(mesh_xyz,axis=1).reshape(-1,1)
        self.mesh_xyz_norm = mesh_xyz_norm
        self.atom_vec_norm = atom_vec_norm    
        orthogonal_mesh = dot/np.dot(mesh_xyz_norm,atom_vec_norm.T)        
        self.mesh_xyz_h = mesh_xyz_h        
        self.orthogonal_mesh = orthogonal_mesh        
        cross_det = mesh_xyz_h <= local_radius
        orthogonal_det = np.arccos(orthogonal_mesh) <= np.pi*0.5
        double_correct = np.array([orthogonal_det,cross_det]).all(axis=0)
        double_correct_index = np.array(np.where(double_correct==True)).T
        self.double_correct_index = double_correct_index
        d_1 = np.zeros(mesh_xyz_h.shape)
        d_2 = np.zeros(mesh_xyz_h.shape)
        for item in double_correct_index:            
            d_1[item[0]][item[1]] = max( (psi[item[1]]**2 - mesh_xyz_h[item[0]][item[1]]**2) ,0)**0.5
            d_2[item[0]][item[1]]=(local_radius[item[1]]**2 - mesh_xyz_h[item[0]][item[1]]**2)**0.5
        self.d_1 = d_1
        self.d_2 = d_2        
        sphere_descriptors = sphere_radius - d_1 - d_2
        sphere_descriptors_compact = sphere_descriptors.min(1)
        sphere_descriptors_reshaped = sphere_descriptors_compact.reshape(PHI.shape)
        sphere_descriptors_reshaped = sphere_descriptors_reshaped.round(precision)        
        if len(self.local_key_atom) == 1:
            sphere_descriptors_init = np.zeros((theta_screenning.shape[0],fi_screenning.shape[0])) + sphere_radius - local_radius[self.local_key_atom[0]]
            sphere_descriptors_final = np.min(np.concatenate([sphere_descriptors_reshaped.reshape(theta_screenning.shape[0],fi_screenning.shape[0],1),sphere_descriptors_init.reshape(theta_screenning.shape[0],fi_screenning.shape[0],1)],axis=2),axis=2)
        else:
            sphere_descriptors_final = sphere_descriptors_reshaped       
        self.PHI = PHI
        self.THETA = THETA
        self.local_sphere_descriptors = sphere_descriptors_final

#rdkit_period_table = Chem.GetPeriodicTable()


from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto,scf
import numpy as np
from pyscf.dft import numint

class CalcEDLocal():
    def __init__(self,mol,grid_num=7,basis='6-311g',GTOval='GTOval',unit_convert = 0.529177, # 1 Bohr = 0.529117 A
                 skip_cusp=True,coord_ang=True,charge_judge=False):#smi
        self.mol = mol
        #self.smi = smi
        self.grid_num = grid_num
        self.basis = basis
        self.GTOval = GTOval
        self.unit_convert=unit_convert
        self.skip_cusp = skip_cusp
        self.coord_ang = coord_ang
        self.charge_judge = charge_judge
    def __call__(self):
        #smi=self.smi
        grid_num = self.grid_num
        symbols = [tmp_atom.GetSymbol() for tmp_atom in self.mol.GetAtoms()]
        positions = self.mol.GetConformer().GetPositions()
        mol_string = ''
        for i in range(len(positions)):
            mol_string += '%5s %15f %15f %15f;'%(symbols[i],positions[i][0],positions[i][1],positions[i][2])
        if self.charge_judge:
            '''if 'Ir'or'Rh' in smi:
                charge=1
            else:'''
            AllChem.ComputeGasteigerCharges(self.mol)
            atoms = self.mol.GetAtoms()
            charge = np.array([atom.GetDoubleProp('_GasteigerCharge') for atom in atoms]).sum()
            py_mol = gto.M(atom=mol_string,charge=int(round(charge)),basis=self.basis)
        else:
            
            py_mol = gto.M(atom=mol_string,basis=self.basis)
        mf = scf.HF(py_mol)
        dm = mf.get_init_guess()
        rdkit_period_table = Chem.GetPeriodicTable()
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
            grid_point_coord = (cubic_array+positions[at_id]) / self.unit_convert ## coord unit Bohr
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
            grid_coords = grid_coords * self.unit_convert
            all_grid_coords = all_grid_coords * self.unit_convert
        grid_ed = all_density.reshape(len(radius),grid_num,grid_num,grid_num)
        self.all_density = all_density
        self.all_grid_coords = all_grid_coords
        return grid_ed,grid_coords
 
class Calc_SPMS_Elec():
    def __init__(self,file_list,mol_dir,spms_acc = 10,elec_acc = 7,sphere_radius = 4,align='Origin'):#smi
        self.spms_acc = spms_acc
        self.elec_acc = elec_acc
        self.mol_dir=mol_dir
        self.sphere_radius = sphere_radius
        self.file_list = file_list
        self.align=align
    def MolSPMS(self,mol,spms_acc):
        atom_nums = len(mol.GetAtoms())
        mol_spms = []
        for i in range(atom_nums):
            spms = SPMS(rdkit_mol=mol,key_atom_num=[i+1],desc_n=self.spms_acc,desc_m=self.spms_acc,sphere_radius=self.sphere_radius)
            spms.GetLocalSphereDescriptors()
            mol_spms.append(spms.local_sphere_descriptors)
        mol_spms = np.array(mol_spms)
        return mol_spms
    def PackCub(self,arr_list):
        N = len(arr_list)
        M = max([arr_list[i].shape[0] for i in range(len(arr_list))])
        pack_cub = np.zeros([N,M,arr_list[0].shape[1],arr_list[0].shape[2],arr_list[0].shape[3]])
        for i in range(N):
            pack_cub[i,:len(arr_list[i]),:,:,:] = arr_list[i]
        return pack_cub
    def PackMat(self,arr_list):
        N = len(arr_list)
        M = max([arr_list[i].shape[0] for i in range(len(arr_list))])
        pack_mat = np.zeros([N,M,arr_list[0].shape[1],arr_list[0].shape[2]])
        for i in range(N):
            pack_mat[i,:len(arr_list[i]),:,:] = arr_list[i]
        return pack_mat
    
    def calc_spms_elec(self,file_list,mol_dir):


        all_spms = []
        all_elec_desc = []
        index_name_dict = {}
        for index,tmp_file in enumerate(self.file_list):
            tmp_fn = tmp_file.split('/')[-1].split('.')[0]
            index_name_dict[tmp_fn] = index

            tmp_mol = Chem.MolFromMolFile(tmp_file,removeHs=False)
            atom_nums = len(tmp_mol.GetAtoms())
            mol_spms = self.MolSPMS(tmp_mol,self.spms_acc)
            if self.align=='True':
                with open(self.mol_dir+'smiles_file_dict.csv','r') as fr:
                    lines = fr.readlines()
                name_smiles_dict = {tmp_line.strip().split(',')[1]:tmp_line.strip().split(',')[0] for tmp_line in lines}        
                match_idx=pd.read_csv(self.mol_dir+'match_index.csv')
                smiles_match_idx_map = {smi:list(map(eval,str(idx).split('-'))) for smi,idx in zip(match_idx['SMILES'].to_list(),
                                                 match_idx['match_index'].to_list())}            
                smi=name_smiles_dict[tmp_fn]
                match_index = smiles_match_idx_map[smi]
                align_index=match_index+list(set(list(range(atom_nums)))-set(match_index))
                mol_spms = mol_spms[align_index]       #排序
                ed = CalcEDLocal(tmp_mol,grid_num=self.elec_acc,basis='6-311g',skip_cusp=True)
                elec_desc,_ = ed()
                elec_desc = elec_desc[align_index]
                all_spms.append(mol_spms)
                all_elec_desc.append(elec_desc)
            elif self.align=='False':
                argsort_index = np.argsort(np.mean(np.mean(mol_spms,axis=1),axis=1))
                mol_spms = mol_spms[argsort_index]       #排序
                ed = CalcEDLocal(tmp_mol,grid_num=self.elec_acc,basis='6-311g',skip_cusp=True)
                elec_desc,_ = ed()
                elec_desc = elec_desc[argsort_index]
                all_spms.append(mol_spms)
                all_elec_desc.append(elec_desc)
            elif self.align=='Origin':
                #argsort_index = np.argsort(np.mean(np.mean(mol_spms,axis=1),axis=1))
                #mol_spms = mol_spms#[argsort_index]       #排序
                ed = CalcEDLocal(tmp_mol,grid_num=self.elec_acc,basis='6-311g',skip_cusp=True)
                elec_desc,_ = ed()
                elec_desc = elec_desc#[argsort_index]
                all_spms.append(mol_spms)
                all_elec_desc.append(elec_desc)                
        all_spms = self.PackMat(all_spms)
        all_elec_desc = self.PackCub(all_elec_desc)
        return all_spms,all_elec_desc,index_name_dict       

def Scaler(arr,method='minmax'):
    if method == 'minmax':
        return (arr - arr.min())/(arr.max()-arr.min())
    elif method == 'z-score':
        return (arr - arr.mean())/arr.var()
    elif method == 'log_minmax':
        arr = np.log(arr)
        return (arr - arr.min())/(arr.max()-arr.min())
