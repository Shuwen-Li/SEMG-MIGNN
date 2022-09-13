import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow import keras
from tensorflow.keras.layers import Dense, Multiply

class Graph_DataLoader(Sequence):
    def __init__(self,spms_mat,elec_vec,label_std,batch_size=32,shuffle=True,predict=False):
        self.spms_mat = spms_mat
        self.elec_vec = elec_vec
        self.label_std = label_std
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.predict = predict
        if self.predict:
            self.shuffle = False
            
        self.on_epoch_end()
    def __len__(self):
        return int(np.ceil(len(self.spms_mat)/self.batch_size))
    def __getitem__(self,index):
        tmp_spms_mat = self.spms_mat[index*self.batch_size:(index+1)*self.batch_size]
        tmp_elec_vec = self.elec_vec[index*self.batch_size:(index+1)*self.batch_size]
        tmp_label = self.label_std[index*self.batch_size:(index+1)*self.batch_size]
        if not self.predict:
            return [tf.convert_to_tensor(np.array(tmp_spms_mat)),tf.convert_to_tensor(np.array(tmp_elec_vec))],tf.convert_to_tensor(np.array(tmp_label))
        else:
            return [tf.convert_to_tensor(np.array(tmp_spms_mat)),tf.convert_to_tensor(np.array(tmp_elec_vec))],tf.convert_to_tensor(np.array(tmp_label))
    def on_epoch_end(self):
        if self.shuffle:
            zipped = list(zip(self.spms_mat,self.elec_vec,self.label_std))
            np.random.shuffle(zipped)
            self.spms_mat,self.elec_vec,self.label_std = zip(*zipped)
            
class MIGNN_model1(keras.Model):
    def __init__(self,lig_size,add_size,base_size,ar_ha_size,hidden_size=256,linear_depth=10,
                 atom_attention=4,inter_attention=1,end_attention=1,spms_number=32,ele_number=32,
                 inter_len=8,final_act='sigmoid'):
        super(MIGNN_model1,self).__init__()
        self.lig_size = lig_size
        self.add_size = add_size
        self.base_size = base_size
        self.ar_ha_size = ar_ha_size
        self.hidden_size = hidden_size
        self.linear_depth = linear_depth
        self.atom_attention=atom_attention
        self.inter_attention=inter_attention
        self.end_attention=end_attention
        self.final_act = final_act
        self.spms_number=spms_number
        self.ele_number=ele_number
        self.norm_lig = keras.layers.BatchNormalization()
        self.norm_add = keras.layers.BatchNormalization()
        self.norm_base = keras.layers.BatchNormalization()
        self.norm_ar_ha = keras.layers.BatchNormalization()
        self.lig_spms_init_layer = keras.layers.Dense(hidden_size,
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.lig_spms_hidden_layer = keras.layers.Dense(hidden_size,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.lig_spms_final_layer = keras.layers.Dense(1,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.lig_spms_attention_dense = Dense(lig_size, activation='softmax', name='attention_vec')
        self.lig_spms_attention_mul =  Multiply()
        self.lig_elec_init_layer = keras.layers.Dense(hidden_size,
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.lig_elec_hidden_layer = keras.layers.Dense(hidden_size,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.lig_elec_final_layer = keras.layers.Dense(1,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.lig_elec_attention_dense = Dense(lig_size, activation='softmax', name='attention_vec')
        self.lig_elec_attention_mul =  Multiply()           
        self.add_spms_init_layer = keras.layers.Dense(hidden_size,
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.add_spms_hidden_layer = keras.layers.Dense(hidden_size,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.add_spms_final_layer = keras.layers.Dense(1,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.add_spms_attention_dense = Dense(add_size, activation='softmax', name='attention_vec')
        self.add_spms_attention_mul =  Multiply()        
        self.add_elec_init_layer = keras.layers.Dense(hidden_size,
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.add_elec_hidden_layer = keras.layers.Dense(hidden_size,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.add_elec_final_layer = keras.layers.Dense(1,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.add_elec_attention_dense = Dense(add_size, activation='softmax', name='attention_vec')
        self.add_elec_attention_mul =  Multiply()          
        self.base_spms_init_layer = keras.layers.Dense(hidden_size,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.base_spms_hidden_layer = keras.layers.Dense(hidden_size,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.base_spms_final_layer = keras.layers.Dense(1,kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.base_spms_attention_dense = Dense(base_size, activation='softmax', name='attention_vec')
        self.base_spms_attention_mul =  Multiply()            
        self.base_elec_init_layer = keras.layers.Dense(hidden_size,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.base_elec_hidden_layer = keras.layers.Dense(hidden_size,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.base_elec_final_layer = keras.layers.Dense(1,kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.base_elec_attention_dense = Dense(base_size, activation='softmax', name='attention_vec')
        self.base_elec_attention_mul =  Multiply()          
        self.ar_ha_spms_init_layer = keras.layers.Dense(hidden_size,
                                                        kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.ar_ha_spms_hidden_layer = keras.layers.Dense(hidden_size,
                                                          kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.ar_ha_spms_final_layer = keras.layers.Dense(1,kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.ar_ha_spms_attention_dense = Dense(ar_ha_size, activation='softmax', name='attention_vec')
        self.ar_ha_spms_attention_mul =  Multiply()                
        self.ar_ha_elec_init_layer = keras.layers.Dense(hidden_size,
                                                        kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.ar_ha_elec_hidden_layer = keras.layers.Dense(hidden_size,
                                                          kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.ar_ha_elec_final_layer = keras.layers.Dense(1,kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.ar_ha_elec_attention_dense = Dense(ar_ha_size, activation='softmax', name='attention_vec')
        self.ar_ha_elec_attention_mul =  Multiply()           
        
        self.maxpooling_2d = keras.layers.MaxPool2D(2)
        self.maxpooling_3d = keras.layers.MaxPool3D(2)
        self.flatten = keras.layers.Flatten()
        self.super_spms_attention_dense = Dense(self.spms_number, activation='softmax', name='attention_vec')
        self.super_spms_attention_mul =  Multiply() 
        self.super_ele_attention_dense = Dense(self.ele_number, activation='softmax', name='attention_vec')
        self.super_ele_attention_mul =  Multiply() 
        
        self.conv2d_1 = keras.layers.Conv2D(16,(3,5))
        self.conv2d_2 = keras.layers.Conv2D(16,(2,3))
        self.conv2d_3 = keras.layers.Conv2D(1,(16,16))
        self.conv2d_4 = keras.layers.Conv2D(1,(8,8))
        self.conv2d_5 = keras.layers.Conv2D(1,(8,8))
        self.conv2d_6 = keras.layers.Conv2D(1,(16,16))
        self.conv2d_7 = keras.layers.Conv2D(1,(8,8))
        self.conv2d_8 = keras.layers.Conv2D(1,(8,8))
        self.conv3d = keras.layers.Conv3D(16,4)
        self.fc_1 = keras.layers.Dense(256)
        self.fc_2 = keras.layers.Dense(256)
        self.fc_3 = keras.layers.Dense(1,activation='sigmoid')
        self.fc_4 = keras.layers.Dense(1)
        self.fc_5 = keras.layers.Dense(32)
        self.fc_6 = keras.layers.Dense(32)
        
        self.x_attention_dense = Dense(434, activation='softmax', name='attention_vec')#322 434
        self.x_attention_mul =  Multiply() 
    def call(self,input_):
        spms_x = input_[0]
        elec_x = input_[1]
        number_batch=len(spms_x)
        lig_x = spms_x[:,:,:,:self.lig_size]
        add_x = spms_x[:,:,:,self.lig_size:self.lig_size+self.add_size]
        base_x = spms_x[:,:,:,self.lig_size+self.add_size:self.lig_size+self.add_size+self.base_size]
        ar_ha_x = spms_x[:,:,:,self.lig_size+self.add_size+self.base_size:]
        lig_elec_x = elec_x[:,:,:,:,:self.lig_size]
        add_elec_x = elec_x[:,:,:,:,self.lig_size:self.lig_size+self.add_size]
        base_elec_x = elec_x[:,:,:,:,self.lig_size+self.add_size:self.lig_size+self.add_size+self.base_size]
        ar_ha_elec_x = elec_x[:,:,:,:,self.lig_size+self.add_size+self.base_size:]
        for i in range(self.atom_attention):
            lig_x_=self.lig_spms_attention_dense(lig_x)
            lig_x=self.lig_spms_attention_mul([lig_x,lig_x_])    
            add_x_=self.add_spms_attention_dense(add_x)
            add_x=self.add_spms_attention_mul([add_x,add_x_])  
            base_x_=self.base_spms_attention_dense(base_x)
            base_x=self.base_spms_attention_mul([base_x,base_x_])  
            ar_ha_x_=self.ar_ha_spms_attention_dense(ar_ha_x)
            ar_ha_x=self.ar_ha_spms_attention_mul([ar_ha_x,ar_ha_x_])  
            lig_elec_x_=self.lig_elec_attention_dense(lig_elec_x)
            lig_elec_x=self.lig_spms_attention_mul([lig_elec_x,lig_elec_x_])
            add_elec_x_=self.add_elec_attention_dense(add_elec_x)
            add_elec_x=self.add_spms_attention_mul([add_elec_x,add_elec_x_])
            base_elec_x_=self.base_elec_attention_dense(base_elec_x)
            base_elec_x=self.base_spms_attention_mul([base_elec_x,base_elec_x_])
            ar_ha_elec_x_=self.ar_ha_elec_attention_dense(ar_ha_elec_x)
            ar_ha_elec_x=self.ar_ha_spms_attention_mul([ar_ha_elec_x,ar_ha_elec_x_])
        lig_x = self.lig_spms_init_layer(lig_x)
        for i in range(self.linear_depth):
            lig_x = self.lig_spms_hidden_layer(lig_x)
        lig_x = self.lig_spms_final_layer(lig_x)
        lig_x = self.norm_lig(lig_x)
        lig_x = tf.nn.tanh(lig_x)
        add_x = self.add_spms_init_layer(add_x)
        for i in range(self.linear_depth):
            add_x = self.add_spms_hidden_layer(add_x)
        add_x = self.add_spms_final_layer(add_x)
        add_x = self.norm_add(add_x)
        add_x = tf.nn.tanh(add_x)
        base_x = self.base_spms_init_layer(base_x)
        for i in range(self.linear_depth):
            base_x = self.base_spms_hidden_layer(base_x)
        base_x = self.base_spms_final_layer(base_x)
        base_x = self.norm_base(base_x)
        base_x = tf.nn.tanh(base_x)
        ar_ha_x = self.ar_ha_spms_init_layer(ar_ha_x)
        for i in range(self.linear_depth):
            ar_ha_x = self.ar_ha_spms_hidden_layer(ar_ha_x)
        ar_ha_x = self.ar_ha_spms_final_layer(ar_ha_x)
        ar_ha_x = self.norm_ar_ha(ar_ha_x)
        ar_ha_x = tf.nn.tanh(ar_ha_x)
        react_spms_mat = tf.concat([lig_x,add_x,base_x,ar_ha_x],axis=3)
        react_spms_x = self.conv2d_2(self.maxpooling_2d(self.conv2d_1(react_spms_mat)))
        react_spms_x = self.flatten(react_spms_x)
        lig_elec_x = self.lig_elec_init_layer(lig_elec_x)
        for i in range(self.linear_depth):
            lig_elec_x = self.lig_elec_hidden_layer(lig_elec_x)
        lig_elec_x = self.lig_elec_final_layer(lig_elec_x)
        lig_elec_x = tf.nn.tanh(lig_elec_x)
        add_elec_x = self.add_elec_init_layer(add_elec_x)
        for i in range(self.linear_depth):
            add_elec_x = self.add_elec_hidden_layer(add_elec_x)
        add_elec_x = self.add_elec_final_layer(add_elec_x)
        add_elec_x = tf.nn.tanh(add_elec_x)
        base_elec_x = self.base_elec_init_layer(base_elec_x)
        for i in range(self.linear_depth):
            base_elec_x = self.base_elec_hidden_layer(base_elec_x)
        base_elec_x = self.base_elec_final_layer(base_elec_x)
        base_elec_x = tf.nn.tanh(base_elec_x)
        ar_ha_elec_x = self.ar_ha_elec_init_layer(ar_ha_elec_x)
        for i in range(self.linear_depth):
            ar_ha_elec_x = self.ar_ha_elec_hidden_layer(ar_ha_elec_x)
        ar_ha_elec_x = self.ar_ha_elec_final_layer(ar_ha_elec_x)
        ar_ha_elec_x = tf.nn.tanh(ar_ha_elec_x)
        react_elec_x = tf.concat([lig_elec_x,add_elec_x,base_elec_x,ar_ha_elec_x],axis=4)
        react_elec_x = self.flatten(self.maxpooling_3d(self.conv3d(react_elec_x)))
        react_spms_x_=react_spms_x
        react_elec_x_=react_elec_x     
        react_spms_x=self.fc_5(react_spms_x)
        react_elec_x=self.fc_6(react_elec_x)
        super_spms=tf.reshape(react_spms_x,[number_batch,1,self.spms_number])
        super_ele=tf.reshape(react_elec_x,[number_batch,1,self.ele_number])
        for i in range(self.inter_attention):
            super_spms_=self.super_spms_attention_dense(super_spms)
            super_spms=self.super_spms_attention_mul([super_spms,super_spms_])    
            super_ele_=self.super_ele_attention_dense(super_ele)
            super_ele=self.super_ele_attention_mul([super_ele,super_ele_])
        react_spms_x=tf.matmul(tf.reshape(react_spms_x,[number_batch,self.spms_number,1]),super_spms)
        react_spms_x=tf.reshape(react_spms_x,[number_batch,self.spms_number,self.spms_number,1])
        react_spms_x=self.flatten(self.conv2d_5(self.conv2d_4(self.conv2d_3(react_spms_x))))
        react_spms_x=tf.reshape(react_spms_x,[number_batch,-1])
        react_elec_x=tf.matmul(tf.reshape(react_elec_x,[number_batch,self.ele_number,1]),super_ele)
        react_elec_x=tf.reshape(react_elec_x,[number_batch,self.ele_number,self.ele_number,1])
        react_elec_x=self.flatten(self.conv2d_8(self.conv2d_7(self.conv2d_6(react_elec_x))))        
        react_elec_x=tf.reshape(react_elec_x,[number_batch,-1])
        
        #if self.interaction=='True':
        x = tf.concat([react_spms_x_,react_elec_x_,react_spms_x,react_elec_x],axis=1)
        for i in range(self.end_attention):
            x_=self.x_attention_dense(x)
            x=self.x_attention_mul([x,x_])  
        if self.final_act == 'sigmoid':
            x = self.fc_3(self.fc_2(self.fc_1(x)))
        else:
            x = self.fc_4(self.fc_2(self.fc_1(x)))

        return x
class MIGNN_model2(keras.Model):
    def __init__(self,cat_size,imine_size,thiol_size,hidden_size=256,linear_depth=10,atom_attention=4,
                 inter_attention=1,end_attention=1,inter_len=8,spms_number=32,ele_number=32,fc_size=256,final_act='sigmoid'):

        super(MIGNN_model2,self).__init__()
        self.cat_size = cat_size
        self.imine_size = imine_size
        self.thiol_size = thiol_size
        self.hidden_size = hidden_size
        self.linear_depth = linear_depth
        self.atom_attention=atom_attention
        self.inter_attention=inter_attention
        self.end_attention=end_attention
        self.inter_len=inter_len
        self.final_act = final_act
        self.spms_number=spms_number
        self.ele_number=ele_number
        
        self.norm_cat = keras.layers.BatchNormalization()
        self.norm_imine = keras.layers.BatchNormalization()
        self.norm_thiol = keras.layers.BatchNormalization()
        self.norm_ar_ha = keras.layers.BatchNormalization()    
        self.cat_spms_init_layer = keras.layers.Dense(hidden_size,
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.cat_spms_hidden_layer = keras.layers.Dense(hidden_size,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.cat_spms_final_layer = keras.layers.Dense(1,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.cat_spms_attention_dense = Dense(cat_size, activation='softmax', name='attention_vec')
        self.cat_spms_attention_mul =  Multiply()
        self.cat_elec_init_layer = keras.layers.Dense(hidden_size,
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.cat_elec_hidden_layer = keras.layers.Dense(hidden_size,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.cat_elec_final_layer = keras.layers.Dense(1,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)       
        self.cat_elec_attention_dense = Dense(cat_size, activation='softmax', name='attention_vec')
        self.cat_elec_attention_mul =  Multiply()        
        self.imine_spms_init_layer = keras.layers.Dense(hidden_size,
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.imine_spms_hidden_layer = keras.layers.Dense(hidden_size,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.imine_spms_final_layer = keras.layers.Dense(1,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.imine_spms_attention_dense = Dense(imine_size, activation='softmax', name='attention_vec')
        self.imine_spms_attention_mul =  Multiply()
        
        self.imine_elec_init_layer = keras.layers.Dense(hidden_size,
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.imine_elec_hidden_layer = keras.layers.Dense(hidden_size,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.imine_elec_final_layer = keras.layers.Dense(1,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.imine_elec_attention_dense = Dense(imine_size, activation='softmax', name='attention_vec')
        self.imine_elec_attention_mul =  Multiply()  
        self.thiol_spms_init_layer = keras.layers.Dense(hidden_size,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.thiol_spms_hidden_layer = keras.layers.Dense(hidden_size,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.thiol_spms_final_layer = keras.layers.Dense(1,kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.thiol_spms_attention_dense = Dense(thiol_size, activation='softmax', name='attention_vec')
        self.thiol_spms_attention_mul =  Multiply()        
        self.thiol_elec_init_layer = keras.layers.Dense(hidden_size,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.thiol_elec_hidden_layer = keras.layers.Dense(hidden_size,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.thiol_elec_final_layer = keras.layers.Dense(1,kernel_initializer=tf.random_normal_initializer(stddev=0.1),use_bias=False)
        self.thiol_elec_attention_dense = Dense(thiol_size, activation='softmax', name='attention_vec')
        self.thiol_elec_attention_mul =  Multiply()  

        self.super_spms_attention_dense = Dense(self.spms_number, activation='softmax', name='attention_vec')
        self.super_spms_attention_mul =  Multiply() 
        self.super_ele_attention_dense = Dense(self.ele_number, activation='softmax', name='attention_vec')
        self.super_ele_attention_mul =  Multiply() 
        self.conv2d_1 = keras.layers.Conv2D(16,(3,5))
        self.conv2d_2 = keras.layers.Conv2D(16,(2,3))
        self.conv2d_3 = keras.layers.Conv2D(1,(16,16))
        self.conv2d_4 = keras.layers.Conv2D(1,(8,8))
        self.conv2d_5 = keras.layers.Conv2D(1,(inter_len,inter_len))
        self.conv2d_6 = keras.layers.Conv2D(1,(16,16))
        self.conv2d_7 = keras.layers.Conv2D(1,(8,8))
        self.conv2d_8 = keras.layers.Conv2D(1,(inter_len,inter_len))
        self.conv3d = keras.layers.Conv3D(16,4)
        self.fc_1 = keras.layers.Dense(fc_size)
        self.fc_2 = keras.layers.Dense(fc_size)
        self.fc_3 = keras.layers.Dense(1,activation='sigmoid')
        self.fc_4 = keras.layers.Dense(1)
        self.fc_5 = keras.layers.Dense(32)
        self.fc_6 = keras.layers.Dense(32)
        
        self.maxpooling_2d = keras.layers.MaxPool2D(2)
        self.maxpooling_3d = keras.layers.MaxPool3D(2)
        self.flatten = keras.layers.Flatten()        
        self.x_attention_dense = Dense(434, activation='softmax', name='attention_vec')
        self.x_attention_mul =  Multiply()         
    def call(self,input_):
        spms_x = input_[0]
        elec_x = input_[1]
        number_batch=len(spms_x)
        cat_x = spms_x[:,:,:,:self.cat_size]
        imine_x = spms_x[:,:,:,self.cat_size:self.cat_size+self.imine_size]
        thiol_x = spms_x[:,:,:,self.cat_size+self.imine_size:]
        cat_elec_x = elec_x[:,:,:,:,:self.cat_size]
        imine_elec_x = elec_x[:,:,:,:,self.cat_size:self.cat_size+self.imine_size]
        thiol_elec_x = elec_x[:,:,:,:,self.cat_size+self.imine_size:]
        for i in range(self.atom_attention):
            cat_x_=self.cat_spms_attention_dense(cat_x)
            cat_x=self.cat_spms_attention_mul([cat_x,cat_x_])    
            imine_x_=self.imine_spms_attention_dense(imine_x)
            imine_x=self.imine_spms_attention_mul([imine_x,imine_x_])  
            thiol_x_=self.thiol_spms_attention_dense(thiol_x)
            thiol_x=self.thiol_spms_attention_mul([thiol_x,thiol_x_])   
            cat_elec_x_=self.cat_elec_attention_dense(cat_elec_x)
            cat_elec_x=self.cat_elec_attention_mul([cat_elec_x,cat_elec_x_])
            imine_elec_x_=self.imine_elec_attention_dense(imine_elec_x)
            imine_elec_x=self.imine_elec_attention_mul([imine_elec_x,imine_elec_x_])
            thiol_elec_x_=self.thiol_elec_attention_dense(thiol_elec_x)
            thiol_elec_x=self.thiol_elec_attention_mul([thiol_elec_x,thiol_elec_x_])  
        cat_x = self.cat_spms_init_layer(cat_x)
        for i in range(self.linear_depth):
            cat_x = self.cat_spms_hidden_layer(cat_x)
        cat_x = self.cat_spms_final_layer(cat_x)
        cat_x = self.norm_cat(cat_x)
        cat_x = tf.nn.tanh(cat_x)
        imine_x = self.imine_spms_init_layer(imine_x)
        for i in range(self.linear_depth):
            imine_x = self.imine_spms_hidden_layer(imine_x)
        imine_x = self.imine_spms_final_layer(imine_x)
        imine_x = self.norm_imine(imine_x)
        imine_x = tf.nn.tanh(imine_x)
        thiol_x = self.thiol_spms_init_layer(thiol_x)
        for i in range(self.linear_depth):
            thiol_x = self.thiol_spms_hidden_layer(thiol_x)   
        thiol_x = self.thiol_spms_final_layer(thiol_x)
        thiol_x = self.norm_thiol(thiol_x)
        thiol_x = tf.nn.tanh(thiol_x)
        react_spms_mat = tf.concat([cat_x,imine_x,thiol_x],axis=3)
        react_spms_x = self.conv2d_2(self.maxpooling_2d(self.conv2d_1(react_spms_mat)))
        react_spms_x = self.flatten(react_spms_x)    
        cat_elec_x = self.cat_elec_init_layer(cat_elec_x)
        for i in range(self.linear_depth):
            cat_elec_x = self.cat_elec_hidden_layer(cat_elec_x)
        cat_elec_x = self.cat_elec_final_layer(cat_elec_x)
        cat_elec_x = tf.nn.tanh(cat_elec_x)
        imine_elec_x = self.imine_elec_init_layer(imine_elec_x)
        for i in range(self.linear_depth):
            imine_elec_x = self.imine_elec_hidden_layer(imine_elec_x)
        imine_elec_x = self.imine_elec_final_layer(imine_elec_x)
        imine_elec_x = tf.nn.tanh(imine_elec_x)
        thiol_elec_x = self.thiol_elec_init_layer(thiol_elec_x)
        for i in range(self.linear_depth):
            thiol_elec_x = self.thiol_elec_hidden_layer(thiol_elec_x)
        thiol_elec_x = self.thiol_elec_final_layer(thiol_elec_x)
        thiol_elec_x = tf.nn.tanh(thiol_elec_x)
        react_elec_x = tf.concat([cat_elec_x,imine_elec_x,thiol_elec_x],axis=4)
        react_elec_x = self.flatten(self.maxpooling_3d(self.conv3d(react_elec_x)))
        react_spms_x_=react_spms_x
        react_elec_x_=react_elec_x
        react_spms_x=self.fc_5(react_spms_x)
        react_elec_x=self.fc_6(react_elec_x)
        super_spms=tf.reshape(react_spms_x,[number_batch,1,self.spms_number])
        super_ele=tf.reshape(react_elec_x,[number_batch,1,self.ele_number])
        for i in range(self.inter_attention):
            super_spms_=self.super_spms_attention_dense(super_spms)
            super_spms=self.super_spms_attention_mul([super_spms,super_spms_])    
            super_ele_=self.super_ele_attention_dense(super_ele)
            super_ele=self.super_ele_attention_mul([super_ele,super_ele_])
        react_spms_x=tf.matmul(tf.reshape(react_spms_x,[number_batch,self.spms_number,1]),super_spms)
        react_spms_x=tf.reshape(react_spms_x,[number_batch,self.spms_number,self.spms_number,1])
        react_spms_x=self.flatten(self.conv2d_5(self.conv2d_4(self.conv2d_3(react_spms_x))))
        react_spms_x=tf.reshape(react_spms_x,[number_batch,-1])
        react_elec_x=tf.matmul(tf.reshape(react_elec_x,[number_batch,self.ele_number,1]),super_ele)
        react_elec_x=tf.reshape(react_elec_x,[number_batch,self.ele_number,self.ele_number,1])
        react_elec_x=self.flatten(self.conv2d_8(self.conv2d_7(self.conv2d_6(react_elec_x))))        
        react_elec_x=tf.reshape(react_elec_x,[number_batch,-1])
        x = tf.concat([react_spms_x_,react_elec_x_,react_spms_x,react_elec_x],axis=1)
        for i in range(self.end_attention):
            x_=self.x_attention_dense(x)
            x=self.x_attention_mul([x,x_]) 
        if self.final_act == 'sigmoid':
            x = self.fc_3(self.fc_2(self.fc_1(x)))
        else:
            x = self.fc_4(self.fc_2(self.fc_1(x)))
        return x