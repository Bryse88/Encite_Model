from torch_geometric.nn import HGTConv
import torch.nn as nn
import torch.nn.functional as F

class HGTModel(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()
        
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in graph.node_types:
            in_channels = graph[node_type].x.size(1)
            self.lin_dict[node_type] = torch.nn.Linear(in_channels, hidden_channels)
        
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, graph.metadata(), 
                          num_heads, group='sum')
            self.convs.append(conv)
        
        self.lin = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x_dict, edge_index_dict):
        # Initial projection
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()
        
        # Multi-layer message passing
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            for node_type in x_dict.keys():
                x_dict[node_type] = F.relu(x_dict[node_type])
        
        # Output projection
        return {k: self.lin(v) for k, v in x_dict.items()}