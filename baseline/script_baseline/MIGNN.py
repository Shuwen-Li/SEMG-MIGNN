import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Dense, Multiply

from tensorflow.keras.utils import Sequence
class Graph_DataLoader(Sequence):
    def __init__(self,feat,label_std,batch_size=32,shuffle=True,predict=False):
        self.feat = feat
        self.label_std = label_std
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.predict = predict
        if self.predict:
            self.shuffle = False
            
        self.on_epoch_end()
    def __len__(self):
        return int(np.ceil(len(self.feat)/self.batch_size))
    def __getitem__(self,index):
        tmp_feat_mat = self.feat[index*self.batch_size:(index+1)*self.batch_size]
        tmp_label = self.label_std[index*self.batch_size:(index+1)*self.batch_size]
        if not self.predict:
            return tf.convert_to_tensor(np.array(tmp_feat_mat)),tf.convert_to_tensor(np.array(tmp_label))
        else:
            return tf.convert_to_tensor(np.array(tmp_feat_mat)),tf.convert_to_tensor(np.array(tmp_label))
    def on_epoch_end(self):
        if self.shuffle:
            zipped = list(zip(self.feat,self.label_std))
            np.random.shuffle(zipped)
            self.feat,self.label_std = zip(*zipped)
            
def z_score(arr):
    return (arr - tf.reduce_mean(arr))/tf.red(arr)
def max_min(arr):
    return (arr - np.min(arr))/(np.max(arr)-np.min(arr))
class MIGNN1(keras.Model):
    def __init__(self,lig_size=107,add_size=36,base_size=56,ar_ha_size=18,hidden_size=15,depth=2,
                 attention_depth=1,scaler_func=z_score,attention_depth_inter=1,inter_len=8,end_attention=1,fc_size=256):
        super(MIGNN1,self).__init__()
        self.lig_size = lig_size
        self.add_size = add_size
        self.base_size = base_size
        self.ar_ha_size = ar_ha_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.inter_len=inter_len
        self.attention_depth=attention_depth
        self.end_attention=end_attention
        self.attention_depth_inter=attention_depth_inter
        
        self.scaler_func = scaler_func
        self.norm_lig = keras.layers.BatchNormalization()
        self.norm_add = keras.layers.BatchNormalization()
        self.norm_base = keras.layers.BatchNormalization()
        self.norm_ar_ha = keras.layers.BatchNormalization()
        
        self.lig_feat_init_layer = keras.layers.Dense(hidden_size,
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                      use_bias=False
                                                     )
        self.lig_feat_hidden_layer = keras.layers.Dense(hidden_size,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                        use_bias=False
                                                       )
        self.lig_feat_final_layer = keras.layers.Dense(1,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                       use_bias=False
                                                      )
        self.lig_feat_attention_dense = Dense(lig_size, activation='softmax', name='attention_vec')
        self.lig_feat_attention_mul =  Multiply()
        
        self.add_feat_init_layer = keras.layers.Dense(hidden_size,
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                      use_bias=False
                                                     )
        self.add_feat_hidden_layer = keras.layers.Dense(hidden_size,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                        use_bias=False
                                                       )
        self.add_feat_final_layer = keras.layers.Dense(1,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                       use_bias=False
                                                      )
        self.add_feat_attention_dense = Dense(add_size, activation='softmax', name='attention_vec')
        self.add_feat_attention_mul =  Multiply()
        
        self.base_feat_init_layer = keras.layers.Dense(hidden_size,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                        use_bias=False
                                                       )
        self.base_feat_hidden_layer = keras.layers.Dense(hidden_size,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                          use_bias=False
                                                         )
        self.base_feat_final_layer = keras.layers.Dense(1,
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                         use_bias=False
                                                        )
        self.base_feat_attention_dense = Dense(base_size, activation='softmax', name='attention_vec')
        self.base_feat_attention_mul =  Multiply()
        
        self.ar_ha_feat_init_layer = keras.layers.Dense(hidden_size,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                        use_bias=False
                                                       )
        self.ar_ha_feat_hidden_layer = keras.layers.Dense(hidden_size,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                          use_bias=False
                                                         )
        self.ar_ha_feat_final_layer = keras.layers.Dense(1,
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                         use_bias=False
                                                        )
        self.ar_ha_feat_attention_dense = Dense(ar_ha_size, activation='softmax', name='attention_vec')
        self.ar_ha_feat_attention_mul =  Multiply()        
        self.flatten = keras.layers.Flatten()
        self.fc_1 = keras.layers.Dense(256)
        self.fc_2 = keras.layers.Dense(256)
        self.fc_3 = keras.layers.Dense(1,activation='sigmoid')
        self.fc_5 = keras.layers.Dense(32)
        
        
        self.feat_number=32
        self.super_feat_attention_dense = Dense(self.feat_number, activation='softmax', name='attention_vec')
        self.super_feat_attention_mul =  Multiply() 
        
        self.conv2d_3 = keras.layers.Conv2D(1,(16,16))
        self.conv2d_4 = keras.layers.Conv2D(1,(8,8))
        self.conv2d_5 = keras.layers.Conv2D(1,(self.inter_len,self.inter_len))
        self.x_attention_dense = Dense(226, activation='softmax', name='attention_vec')
        self.x_attention_mul =  Multiply()           
        
    def call(self,input_):
        number_batch=len(input_)
        #input_=input_[0]
        lig_x = input_[:,:self.lig_size,:]
        add_x = input_[:,self.lig_size:self.lig_size+self.add_size,:]
        base_x = input_[:,self.lig_size+self.add_size:self.lig_size+self.add_size+self.base_size,:]
        ar_ha_x = input_[:,self.lig_size+self.add_size+self.base_size:,:]

        lig_x=tf.reshape(lig_x,[-1,15,self.lig_size])
        add_x=tf.reshape(add_x,[-1,15,self.add_size])        
        base_x=tf.reshape(base_x,[-1,15,self.base_size])  
        ar_ha_x=tf.reshape(ar_ha_x,[-1,15,self.ar_ha_size])          
        for i in range(self.attention_depth):
            lig_x_=self.lig_feat_attention_dense(lig_x)
            lig_x=self.lig_feat_attention_mul([lig_x,lig_x_])    
            add_x_=self.add_feat_attention_dense(add_x)
            add_x=self.add_feat_attention_mul([add_x,add_x_])  
            base_x_=self.base_feat_attention_dense(base_x)
            base_x=self.base_feat_attention_mul([base_x,base_x_])
            ar_ha_x_=self.ar_ha_feat_attention_dense(ar_ha_x)
            ar_ha_x=self.ar_ha_feat_attention_mul([ar_ha_x,ar_ha_x_])            
        lig_x=tf.reshape(lig_x,[-1,self.lig_size,15])
        add_x=tf.reshape(add_x,[-1,self.add_size,15])        
        base_x=tf.reshape(base_x,[-1,self.base_size,15])          
        ar_ha_x=tf.reshape(ar_ha_x,[-1,self.ar_ha_size,15])         
      
        lig_x = self.lig_feat_init_layer(lig_x)
        for i in range(self.depth):
            lig_x = self.lig_feat_hidden_layer(lig_x)
        lig_x = (lig_x - tf.reduce_min(lig_x))/(tf.reduce_max(lig_x)-tf.reduce_min(lig_x))
        lig_x = self.lig_feat_final_layer(lig_x)
        lig_x = self.norm_lig(lig_x)
        lig_x = tf.nn.tanh(lig_x)
        
        add_x = self.add_feat_init_layer(add_x)
        for i in range(self.depth):
            add_x = self.add_feat_hidden_layer(add_x)
        add_x = (add_x - tf.reduce_min(add_x))/(tf.reduce_max(add_x)-tf.reduce_min(add_x))
        add_x = self.add_feat_final_layer(add_x)
        add_x = self.norm_add(add_x)
        add_x = tf.nn.tanh(add_x)
        
        base_x = self.base_feat_init_layer(base_x)
        for i in range(self.depth):
            base_x = self.base_feat_hidden_layer(base_x)
        base_x = (base_x - tf.reduce_min(base_x))/(tf.reduce_max(base_x)-tf.reduce_min(base_x))
        base_x = self.base_feat_final_layer(base_x)
        base_x = self.norm_base(base_x)
        base_x = tf.nn.tanh(base_x)
        
        ar_ha_x = self.ar_ha_feat_init_layer(ar_ha_x)
        for i in range(self.depth):
            ar_ha_x = self.ar_ha_feat_hidden_layer(ar_ha_x)
        ar_ha_x = (ar_ha_x - tf.reduce_min(ar_ha_x))/(tf.reduce_max(ar_ha_x)-tf.reduce_min(ar_ha_x))
        ar_ha_x = self.ar_ha_feat_final_layer(ar_ha_x)
        ar_ha_x = self.norm_ar_ha(ar_ha_x)
        ar_ha_x = tf.nn.tanh(ar_ha_x)       
        react_feat_mat = tf.concat([lig_x,add_x,base_x,ar_ha_x],axis=1)        
        react_feat_mat=tf.reshape(react_feat_mat,[number_batch,217])
        react_feat_mat_=react_feat_mat
        react_feat_mat=self.fc_5(react_feat_mat)        
        super_feat=tf.reshape(react_feat_mat,[number_batch,1,self.feat_number])
        for i in range(self.attention_depth_inter):
            super_feat_=self.super_feat_attention_dense(super_feat)
            super_feat=self.super_feat_attention_mul([super_feat,super_feat_])    
        react_feat_mat=tf.matmul(tf.reshape(react_feat_mat,[number_batch,self.feat_number,1]),super_feat)
        react_feat_mat=tf.reshape(react_feat_mat,[number_batch,self.feat_number,self.feat_number,1])
        react_feat_mat=self.conv2d_5(self.conv2d_4(self.conv2d_3(react_feat_mat)))
        react_feat_mat=tf.reshape(react_feat_mat,[number_batch,-1])        
        x = tf.concat([react_feat_mat_,react_feat_mat],axis=1)
        x=tf.reshape(x,[number_batch,226])
        for i in range(self.end_attention):
            x_=self.x_attention_dense(x)
            x=self.x_attention_mul([x,x_])         
        x = self.fc_3(self.fc_2(self.fc_1(x))) 
        return x

class MIGNN2(keras.Model):
    def __init__(self,cat_size=184,imine_size=33,thiol_size=19,hidden_size=256,depth=2,attention_depth=3
                 ,scaler_func=z_score,attention_depth_inter=2,inter_len=8,end_attention=1,fc_size=256 ):  
        super(MIGNN2,self).__init__()
        self.cat_size = cat_size
        self.imine_size = imine_size
        self.thiol_size = thiol_size
        self.hidden_size = hidden_size
        self.fc_size=fc_size
        self.depth = depth
        self.attention_depth=attention_depth
        self.scaler_func = scaler_func
        self.norm_cat = keras.layers.BatchNormalization()
        self.norm_imine = keras.layers.BatchNormalization()
        self.norm_thiol = keras.layers.BatchNormalization()

        self.inter_len=inter_len
        self.attention_depth=attention_depth
        self.end_attention=end_attention
        self.attention_depth_inter=attention_depth_inter
        
        self.cat_feat_init_layer = keras.layers.Dense(hidden_size,
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                      use_bias=False
                                                     )
        self.cat_feat_hidden_layer = keras.layers.Dense(hidden_size,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                        use_bias=False
                                                       )
        self.cat_feat_final_layer = keras.layers.Dense(1,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                       use_bias=False
                                                      )
        self.cat_feat_attention_dense = Dense(cat_size, activation='softmax', name='attention_vec')
        self.cat_feat_attention_mul =  Multiply()
        
        self.imine_feat_init_layer = keras.layers.Dense(hidden_size,
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                      use_bias=False
                                                     )
        self.imine_feat_hidden_layer = keras.layers.Dense(hidden_size,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                        use_bias=False
                                                       )
        self.imine_feat_final_layer = keras.layers.Dense(1,
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                       use_bias=False
                                                      )
        self.imine_feat_attention_dense = Dense(imine_size, activation='softmax', name='attention_vec')
        self.imine_feat_attention_mul =  Multiply()
        
        self.thiol_feat_init_layer = keras.layers.Dense(hidden_size,
                                     kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                        use_bias=False
                                                       )
        self.thiol_feat_hidden_layer = keras.layers.Dense(hidden_size,
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                          use_bias=False
                                                         )
        self.thiol_feat_final_layer = keras.layers.Dense(1,
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.1),
                                                         use_bias=False
                                                        )
        self.thiol_feat_attention_dense = Dense(thiol_size, activation='softmax', name='attention_vec')
        self.thiol_feat_attention_mul =  Multiply()
        
        self.flatten = keras.layers.Flatten()
        self.fc_1 = keras.layers.Dense(128)
        self.fc_2 = keras.layers.Dense(128)
        self.fc_3 = keras.layers.Dense(1,activation='sigmoid')
        self.fc_5 = keras.layers.Dense(32)             
        self.feat_number=32
        self.super_feat_attention_dense = Dense(self.feat_number, activation='softmax', name='attention_vec')
        self.super_feat_attention_mul =  Multiply()         
        self.conv2d_3 = keras.layers.Conv2D(1,(16,16))
        self.conv2d_4 = keras.layers.Conv2D(1,(8,8))
        self.conv2d_5 = keras.layers.Conv2D(1,(inter_len,inter_len))
        self.x_attention_dense = Dense(245, activation='softmax', name='attention_vec')
        self.x_attention_mul =  Multiply()        
    def call(self,input_):
        #input_=input_[0]
        number_batch=len(input_)
        cat_x = input_[:,:self.cat_size,:]
        imine_x = input_[:,self.cat_size:self.cat_size+self.imine_size,:]
        thiol_x = input_[:,self.cat_size+self.imine_size:self.cat_size+self.imine_size+self.thiol_size,:]
        
        cat_x=tf.reshape(cat_x,[-1,15,self.cat_size])
        imine_x=tf.reshape(imine_x,[-1,15,self.imine_size])        
        thiol_x=tf.reshape(thiol_x,[-1,15,self.thiol_size])        
        for i in range(self.attention_depth):
            cat_x_=self.cat_feat_attention_dense(cat_x)
            cat_x=self.cat_feat_attention_mul([cat_x,cat_x_])    
            imine_x_=self.imine_feat_attention_dense(imine_x)
            imine_x=self.imine_feat_attention_mul([imine_x,imine_x_])  
            thiol_x_=self.thiol_feat_attention_dense(thiol_x)
            thiol_x=self.thiol_feat_attention_mul([thiol_x,thiol_x_])         
        cat_x=tf.reshape(cat_x,[-1,self.cat_size,15])
        imine_x=tf.reshape(imine_x,[-1,self.imine_size,15])        
        thiol_x=tf.reshape(thiol_x,[-1,self.thiol_size,15])         
        
        cat_x = self.cat_feat_init_layer(cat_x)
        for i in range(self.depth):
            cat_x = self.cat_feat_hidden_layer(cat_x)
        cat_x = (cat_x - tf.reduce_min(cat_x))/(tf.reduce_max(cat_x)-tf.reduce_min(cat_x))
        cat_x = self.cat_feat_final_layer(cat_x)
        cat_x = self.norm_cat(cat_x)
        cat_x = tf.nn.tanh(cat_x)
        
        imine_x = self.imine_feat_init_layer(imine_x)
        for i in range(self.depth):
            imine_x = self.imine_feat_hidden_layer(imine_x)
        imine_x = (imine_x - tf.reduce_min(imine_x))/(tf.reduce_max(imine_x)-tf.reduce_min(imine_x))
        imine_x = self.imine_feat_final_layer(imine_x)
        imine_x = self.norm_imine(imine_x)
        imine_x = tf.nn.tanh(imine_x)
        
        thiol_x = self.thiol_feat_init_layer(thiol_x)
        for i in range(self.depth):
            thiol_x = self.thiol_feat_hidden_layer(thiol_x)
        thiol_x = (thiol_x - tf.reduce_min(thiol_x))/(tf.reduce_max(thiol_x)-tf.reduce_min(thiol_x))
        thiol_x = self.thiol_feat_final_layer(thiol_x)
        thiol_x = self.norm_thiol(thiol_x)
        thiol_x = tf.nn.tanh(thiol_x)
        react_feat_mat = tf.concat([cat_x,imine_x,thiol_x],axis=1)        
        react_feat_mat=tf.reshape(react_feat_mat,[number_batch,236])
        react_feat_mat_=react_feat_mat
        react_feat_mat=self.fc_5(react_feat_mat)
        
        super_feat=tf.reshape(react_feat_mat,[number_batch,1,self.feat_number])
        for i in range(self.attention_depth_inter):
            super_feat_=self.super_feat_attention_dense(super_feat)
            super_feat=self.super_feat_attention_mul([super_feat,super_feat_])    
        
        react_feat_mat=tf.matmul(tf.reshape(react_feat_mat,[number_batch,self.feat_number,1]),super_feat)
        react_feat_mat=tf.reshape(react_feat_mat,[number_batch,self.feat_number,self.feat_number,1])
        react_feat_mat=self.flatten(self.conv2d_5(self.conv2d_4(self.conv2d_3(react_feat_mat))))
        react_feat_mat=tf.reshape(react_feat_mat,[number_batch,-1])        
        x = tf.concat([react_feat_mat_,react_feat_mat],axis=1)
        x=tf.reshape(x,[number_batch,245])
        for i in range(self.end_attention):
            x_=self.x_attention_dense(x)
            x=self.x_attention_mul([x,x_])          
        x = self.fc_3(self.fc_2(self.fc_1(x)))
        return x