import dgl
import torch 
from torch import nn
import torch.nn.functional as F
class GCN1(nn.Module):
    def __init__(self, in_features, hidden_features, out_features,multi_graph_opt='mean',output_opt='mean',layer=3):
        super().__init__()      
        
        self.conv1 = dgl.nn.GraphConv(in_features, hidden_features)
        self.conv2 = dgl.nn.GraphConv(hidden_features, hidden_features)
        self.conv3 = dgl.nn.GraphConv(hidden_features, hidden_features)  
        self.fc1   = nn.Linear(304,128) 
        self.fc2   = nn.Linear(128, 64)
        self.fc3   = nn.Linear(64, 1)
        self.multi_graph_opt = multi_graph_opt
        self.output_opt = output_opt
        self.layer = layer
    def forward(self,g1,x1,g2,x2,g3,x3,g4,x4):
        
        if self.layer == 3:
            x1 = F.relu(self.conv1(g1, x1))
            x1 = F.relu(self.conv2(g1, x1))
            x1 = F.relu(self.conv3(g1, x1))

            x2 = F.relu(self.conv1(g2, x2))
            x2 = F.relu(self.conv2(g2, x2))
            x2 = F.relu(self.conv3(g2, x2))

            x3 = F.relu(self.conv1(g3, x3))
            x3 = F.relu(self.conv2(g3, x3))
            x3 = F.relu(self.conv3(g3, x3))

            x4 = F.relu(self.conv1(g4, x4))
            x4 = F.relu(self.conv2(g4, x4))
            x4 = F.relu(self.conv3(g4, x4))
        elif self.layer == 2:
            x1 = F.relu(self.conv1(g1, x1))
            x1 = F.relu(self.conv2(g1, x1))

            x2 = F.relu(self.conv1(g2, x2))
            x2 = F.relu(self.conv2(g2, x2))

            x3 = F.relu(self.conv1(g3, x3))
            x3 = F.relu(self.conv2(g3, x3))

            x4 = F.relu(self.conv1(g4, x4))
            x4 = F.relu(self.conv2(g4, x4))
        elif self.layer == 1:
            x1 = F.relu(self.conv1(g1, x1))
            
            x2 = F.relu(self.conv1(g2, x2))

            x3 = F.relu(self.conv1(g3, x3))

            x4 = F.relu(self.conv1(g4, x4))
        
        if self.multi_graph_opt == 'mean':
            x = torch.mean(torch.cat([x1,x2,x3,x4],axis=0),axis=0)          
        elif self.multi_graph_opt == 'sum':
            x = torch.sum(torch.cat([x1,x2,x3,x4],axis=0),axis=0)
        elif self.multi_graph_opt == 'max':
            x = torch.max(torch.cat([x1,x2,x3,x4],axis=0),axis=0).values         
        
        if self.output_opt == 'mean':
            x = F.torch.mean(x)
        elif self.output_opt == 'sum':
            x = F.torch.sum(x)
        elif self.output_opt == 'max':
            x = F.torch.max(x)            
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.sigmoid(self.fc3(x))        
        return x
    
class GCN2(nn.Module):
    def __init__(self, in_features, hidden_features, out_features,multi_graph_opt='max',output_opt='mean',layer=3):
        super().__init__() 
        self.conv1 = dgl.nn.GraphConv(in_features, hidden_features) 
        self.conv2 = dgl.nn.GraphConv(hidden_features, hidden_features)
        self.conv3 = dgl.nn.GraphConv(hidden_features, hidden_features) 
        self.fc1   = nn.Linear(304,128) 
        self.fc2   = nn.Linear(128, 64)
        self.fc3   = nn.Linear(64, 1)
        self.multi_graph_opt = multi_graph_opt
        self.output_opt = output_opt
        self.layer =layer

    def forward(self,g1,x1,g2,x2,g3,x3):
        if self.layer == 3:
            x1 = F.relu(self.conv1(g1, x1))
            x1 = F.relu(self.conv2(g1, x1))
            x1 = F.relu(self.conv3(g1, x1))

            x2 = F.relu(self.conv1(g2, x2))
            x2 = F.relu(self.conv2(g2, x2))
            x2 = F.relu(self.conv3(g2, x2))

            x3 = F.relu(self.conv1(g3, x3))
            x3 = F.relu(self.conv2(g3, x3))
            x3 = F.relu(self.conv3(g3, x3))
        elif self.layer == 2:
            x1 = F.relu(self.conv1(g1, x1))
            x1 = F.relu(self.conv2(g1, x1))

            x2 = F.relu(self.conv1(g2, x2))
            x2 = F.relu(self.conv2(g2, x2))

            x3 = F.relu(self.conv1(g3, x3))
            x3 = F.relu(self.conv2(g3, x3))
        elif self.layer == 1:
            x1 = F.relu(self.conv1(g1, x1))
            x2 = F.relu(self.conv1(g2, x2))
            x3 = F.relu(self.conv1(g3, x3))
        if self.multi_graph_opt == 'mean':
            x = torch.mean(torch.cat([x1,x2,x3],axis=0),axis=0)         
        elif self.multi_graph_opt == 'sum':
            x = torch.sum(torch.cat([x1,x2,x3],axis=0),axis=0)  
        elif self.multi_graph_opt == 'max':
            x = torch.max(torch.cat([x1,x2,x3],axis=0),axis=0).values
            
        if self.output_opt == 'mean':
            x = F.torch.mean(x)
        elif self.output_opt == 'sum':
            x = F.torch.sum(x)
        elif self.output_opt == 'max':
            x = F.torch.max(x)            
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.sigmoid(self.fc3(x))
        
        return x
    
    
