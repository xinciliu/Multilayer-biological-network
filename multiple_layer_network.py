import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class multilayer_bionetwork(BaseModel):
    def __init__(self, node_num, type_num, emb_dim, gcn_layersize):
        super().__init__()
        self.node_num = node_num
        self.adj = adj #adj for gcn
        self.emb_dim = emb_dim
        self.value_embedding = nn.Embedding(node_num+1, emb_dim, padding_idx=0)
        self.type_embedding = nn.Embedding(type_num+1, emb_dim, padding_idx=0)
        self.gcn = GCN(nfeat=gcn_layersize[0], nhid=gcn_layersize[1],
                       nclass=gcn_layersize[2], dropout=dropout)
        #try BILSYM, set up four layers 
        self.BILSTM = nn.LSTM(input_size=emb_dim*2, hidden_size=emb_dim, \
        num_layers=4, batch_first=True, bidirectional=True)
        self.node_attention = attention_mechiansm(nn.Linear(in_features=emb_dim, out_features=1, bias=False), nn.Softmax(dim=1), True)
        self.path_attention = attention_mechiansm(nn.Linear(in_features=emb_dim, out_features=1, bias=False), nn.Softmax(dim=1), False)
        self.output_linear = nn.Linear(in_features=emb_dim, out_features=1)

    def forward(self, path_feature, type_feature, lengths, mask, gcn=True):
        """
        """
        #try use gcn embedding to embed path_feature
        #build up all node embedding initially + concat to same dimension
        total_node = torch.LongTensor([list(range(self.node_num+1))]).to(path_feature.device)
        total_node_embeded = self.value_embedding(total_node).squeeze()
         #if use gcn embeded
        if gcn:
            gcn_value_embedding = self.gcn.forward(x=total_node_embeded, adj=self.adj.to(path_feature.device))
        else:
            gcn_value_embedding = totle_node_embeded
        #path embedding - use gcn 
        batch, path_num, path_len = path_feature.size()
        path_feature = path_feature.view(batch*path_num, path_len)
        path_embedding = gcn_value_embedding[path_feature]
        #type embedding
        type_feature = type_feature.view(batch*path_num, path_len)
        type_embedding = self.type_embedding(type_feature).squeeze()
        #all-feature = concat path_feature + type_feature
        feature = torch.cat((path_embedding, type_embedding), 2)
        feature = torch.transpose(feature, dim0=0, dim1=1)
        feature = utils.rnn.pack_padded_sequence(feature, \
        lengths=list(lengths.view(batch*path_num).data), enforce_sorted=False)
        #BILSTM network output
        bilstm_out, _ = self.BILSTM(feature)
        bilstm_out, _ = utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=path_len)
        #BILSTM output - node attention
        mask = mask.view(batch*path_num, path_len)
        output_path_embedding, node_weight_normalized = self.node_attention.attention(lstm_out, mask)
        node_weight_normalized = node_weight_normalized.view(batch, path_num, path_len)
        output_path_embedding = output_path_embedding.view(batch, path_num, self.emb_dim)
        #Path attention
        output_embedding, path_weight_normalized = self.path_attention.attention(output_path_embedding)
        output = self.output_linear(output_embedding)
        return output, node_weight_normalized, path_weight_normalized


class attention_mechiansm(object):
    """
    use for attention
    """
    def __init__(self, linear_algorithm, softmax_algorithm, if_maskedfill):
        self.linear_algorithm = linear_algorithm
        self.softmax_algorithm = softmax_algorithm
        self.if_maskedfill = if_maskedfill
    
    def attention(self, input, mask=0):
        weight = self.linear_algorithm(input)
        weight = weight.squeeze() 
        if if_maskedfill:
            weight = weight.masked_fill(mask==0, torch.tensor(-1e9))
        weight_normalized = self.softmax_algorithm(weight) 
        weight_expand = torch.unsqueeze(weight_normalized, dim=2) 
        input_weighted = (input * weight_expand).sum(dim=1) 
        return input_weighted, weight_normalized


class GCN(nn.Module):
    def  __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):    
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
