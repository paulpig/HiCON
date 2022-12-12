
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum
import torch
from torch_geometric.nn import GCNConv
# from torch_geometric.nn import LGConv
import pdb
from .MetaUnit_v1 import MetaWeightNet
import os

"""
参考代码: https://github.com/blindsubmission1/PEAGNN/
"""
class PEABaseChannel(torch.nn.Module):
    def reset_parameters(self):
        for module in self.gnn_layers:
            module.reset_parameters()

    def _edge_sampling(self, edge_index, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices]

    def forward(self, x, edge_index_list, mess_dropout=True):
        """
            x: (N, dim)
            edge_index_list: lisf of tensor (edge_num, 2)
        """
        assert len(edge_index_list) == self.num_steps
        # x_final_emb = x
        x_final_emb_list = [x]
        for step_idx in range(self.num_steps - 1):
            # pdb.set_trace()
            edge_index_dropout = edge_index_list[step_idx]
            x = F.relu(self.gnn_layers[step_idx](x, edge_index_dropout)) # idx is wrong ?
            """message dropout"""
            if mess_dropout:
                x = self.dropout(x)
            x = F.normalize(x)
            # x_final_emb = x_final_emb + x
            x_final_emb_list.append(x)

        edge_index_dropout = edge_index_list[-1]
        x = self.gnn_layers[-1](x, edge_index_dropout) # get last layer.
        x = F.normalize(x) # adding this component is better.
        return x

class PEAGCNChannel(PEABaseChannel):
    def __init__(self, num_steps, emb_dim, repr_dim, hidden_size, mess_dropout_rate=0.1):
        super(PEAGCNChannel, self).__init__()
        self.num_steps = num_steps
        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

        self.gnn_layers = torch.nn.ModuleList()
        if num_steps == 1:
            self.gnn_layers.append(GCNConv(emb_dim, repr_dim)) # dd
            # self.gnn_layers.append(LGConv())
        else:
            self.gnn_layers.append(GCNConv(emb_dim, hidden_size))
            # self.gnn_layers.append(LGConv())
            for i in range(num_steps - 2):
                self.gnn_layers.append(GCNConv(hidden_size, hidden_size))
                # self.gnn_layers.append(LGConv())
            self.gnn_layers.append(GCNConv(hidden_size, repr_dim))
            # self.gnn_layers.append(LGConv())
        self.reset_parameters()
    

class Aggregator(nn.Module):
    def __init__(self, n_users, emb_size):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.relation_w = nn.Linear(emb_size, emb_size, bias=True)

    def forward(self, entity_emb, user_emb,
                edge_index, edge_type, interact_mat,
                weight):
        """
        weight: (n_relation, channel)
        """
        n_entities = entity_emb.shape[0]
        # pdb.set_trace()
         # step1: kg aggregation based on KG which excludes user-item pairs.
        head, tail = edge_index
        edge_relation_emb = weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        # ------------calculate attention weights ---------------
        neigh_relation_emb_weight = self.calculate_sim_hrt(entity_emb[head], entity_emb[tail], 
                                                                    weight[edge_type - 1]) #(pair_num, 1)
        neigh_relation_emb_weight = neigh_relation_emb_weight.expand(neigh_relation_emb.shape[0],
                                                                    neigh_relation_emb.shape[1]) #(pair_num, dim)
        
        # neigh_relation_emb_tmp = torch.matmul(neigh_relation_emb_weight, neigh_relation_emb)
        neigh_relation_emb_weight = scatter_softmax(neigh_relation_emb_weight, index=head, dim=0) # 对head_idx聚类后, softmax;
        neigh_relation_emb = torch.mul(neigh_relation_emb_weight, neigh_relation_emb) #(pair_num, dim)
        entity_agg = scatter_sum(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0) #(entities, dim)

        #测试一下效果如何;
        # pdb.set_trace()
        entity_agg += torch.sparse.mm(interact_mat.t(), user_emb)

        # step2: user aggregation based on the interactive graph and all relations. Why?
        user_agg = torch.sparse.mm(interact_mat, entity_emb) #(u, dim)

        # # remove this code, which not includes in the paper.
        # # user_agg = user_agg + user_emb * user_agg
        # score = torch.mm(user_emb, weight.t()) #(u, n_relation)
        # score = torch.softmax(score, dim=-1) #(u, n_relation）
        # user_agg = user_agg + (torch.mm(score, weight)) * user_agg

        return entity_agg, user_agg

    def forward_hierarchy(self, entity_emb, user_emb,
                edge_index, edge_type, interact_mat,
                weight):
        """
        results are not good, because of the lack of graph topology.
        How to effectively use the graph topology ?
        weight: (n_relation, channel)
        """
        n_entities = entity_emb.shape[0]
        # pdb.set_trace()
        # step1: kg aggregation based on KG which excludes user-item pairs.
        head, tail = edge_index
        edge_relation_emb = weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1), (1, 120)

        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        # ------------calculate attention weights ---------------
        neigh_relation_emb_weight = self.calculate_sim_hrt(entity_emb[head], entity_emb[tail],
                                                           weight[edge_type - 1])  # (pair_num, 1)

        # ddddd = scatter_sum(src=(neigh_relation_emb_weight > 0 ), index=head, dim=0)
        relation_count = scatter_sum(src=torch.ones_like(neigh_relation_emb_weight), index=head, dim_size=n_entities, dim=0) # (ddddd>1).sum(), 7074
        # pdb.set_trace()

        neigh_relation_emb_weight = neigh_relation_emb_weight.expand(-1, weight.shape[0])  # (pair_num, relation_num)
        relations_one_hot = F.one_hot((edge_type - 1), num_classes=weight.shape[0]) #(pair_num, relation_num)
        neigh_relation_emb_weight = neigh_relation_emb_weight * relations_one_hot #(pair_num, relation_num)

        neigh_relation_emb_weight = neigh_relation_emb_weight.unsqueeze(-1).expand(-1, -1, neigh_relation_emb.shape[1]) #(pair_num, relation_num, dim)

        relation_aware_emb_list = []
        # relation_counnt_list = []
        for i in range(1, weight.shape[0]):
            neigh_relation_emb_weight_s = neigh_relation_emb_weight[:,i,:] #(pair_num, dim)
            # convert zero to -inf
            neigh_relation_emb_weight_s = neigh_relation_emb_weight_s + (neigh_relation_emb_weight_s == 0) * -1.e20
            neigh_relation_emb_weight_s = scatter_softmax(neigh_relation_emb_weight_s, index=head, dim=0)  # 对head_idx聚类后, softmax;
            # pdb.set_trace()
            mask_relation_w = (neigh_relation_emb_weight[:, i, :] != 0) # (pair_num, dim)
            neigh_relation_emb_weight_s = neigh_relation_emb_weight_s * mask_relation_w #remove no relations entity.
            neigh_relation_emb = torch.mul(neigh_relation_emb_weight_s, neigh_relation_emb)  # (pair_num, dim)
            entity_agg = scatter_sum(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)  # (entities, dim)
            # relation_count = scatter_sum(src=mask_relation_w, index=head, dim_size=n_entities, dim=0)  # (entities, dim)

            # relation_counnt_list.append(relation_count)
            relation_aware_emb_list.append(entity_agg)

        # entity_agg = torch.stack(relation_aware_emb_list, dim=1).sum(dim=1) # which is better ?
        # entity_agg = torch.stack(relation_aware_emb_list, dim=1).mean(dim=1) #
        # pdb.set_trace()
        entity_agg = torch.stack(relation_aware_emb_list, dim=1).sum(dim=1, keepdim=True) / relation_count.unsqueeze(-1) # mean
        entity_agg = entity_agg.squeeze(1)
        entity_agg = self.relation_w(entity_agg) # is useful: 加上FC后, 至少能达到baseline的效果了, 必须添加, 效果很差;

        # step2: user aggregation based on the interactive graph and all relations. Why?
        user_agg = torch.sparse.mm(interact_mat, entity_emb)  # (u, dim)

        # # remove this code, which not includes in the paper.
        # # user_agg = user_agg + user_emb * user_agg
        # score = torch.mm(user_emb, weight.t()) #(u, n_relation)
        # score = torch.softmax(score, dim=-1) #(u, n_relation）
        # user_agg = user_agg + (torch.mm(score, weight)) * user_agg

        return entity_agg, user_agg
    def calculate_sim_hrt(self, entity_emb_head, entity_emb_tail, relation_emb):

        tail_relation_emb = entity_emb_tail * relation_emb
        tail_relation_emb = tail_relation_emb.norm(dim=1, p=2, keepdim=True)
        head_relation_emb = entity_emb_head * relation_emb
        head_relation_emb = head_relation_emb.norm(dim=1, p=2, keepdim=True)
        # pdb.set_trace()
        att_weights = torch.matmul(head_relation_emb.unsqueeze(dim=1), tail_relation_emb.unsqueeze(dim=2)).squeeze(dim=-1)
        att_weights = att_weights ** 2
        return att_weights

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_users,
                  n_relations, interact_mat,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1, device=None):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind
        self.topk = 10
        self.lambda_coeff = 0.5
        self.temperature = 0.2
        # self.device = torch.device("cuda:" + str(0))
        self.device = device
        initializer = nn.init.xavier_uniform_
        weight = initializer(torch.empty(n_relations - 1, channel))
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]

        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users, emb_size=weight.shape[-1]))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        """
        randomly remove edges from KG.
        Args:
            edge_index:
            edge_type:
            rate:

        Returns:

        """
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        """
        Args:
            x:
            rate:

        Returns:

        """
        noise_shape = x._nnz() # number non zero elements.
        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device) # [0, 1)
        dropout_mask = torch.floor(random_tensor).type(torch.bool) # [0.5, 1.5)
        i = x._indices() # remove half edges.
        v = x._values()
        i = i[:, dropout_mask]
        v = v[dropout_mask]
        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate)) # keep values sum to 1.

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, user_emb, entity_emb, edge_index, edge_type,
                interact_mat, mess_dropout=True, node_dropout=False, groupby_relation=False, is_training=True):

        """node dropout"""
        if node_dropout and is_training: # remove edges.
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)
        
        #step1: generate the item-item graph.
        # origin_item_adj = self.build_adj(entity_emb, self.topk)

        #step2: based on the hetegonerous graph and interactive graph to get user embs and entity embs.
        # entity_res_emb = entity_emb  # [n_entity, channel]
        # user_res_emb = user_emb  # [n_users, channel]
        entity_res_emb_list = [entity_emb]
        user_res_emb_list = [user_emb]
        for i in range(len(self.convs)):
            if groupby_relation == False:
                entity_emb, user_emb = self.convs[i](entity_emb, user_emb,
                                                     edge_index, edge_type, interact_mat,
                                                     self.weight)
            else:
                entity_emb, user_emb = self.convs[i].forward_hierarchy(entity_emb, user_emb,
                                                     edge_index, edge_type, interact_mat,
                                                     self.weight)
            """message dropout"""
            if mess_dropout and is_training:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)
            # pdb.set_trace() # torch.norm(user_emb, dim=1), torch.mean(torch.norm(user_emb, dim=1))
            """result emb"""
            entity_res_emb_list.append(entity_emb)
            user_res_emb_list.append(user_emb)
            # entity_res_emb = torch.add(entity_res_emb, entity_emb)
            # user_res_emb = torch.add(user_res_emb, user_emb)

        # entity_res_emb = torch.mean(torch.stack(entity_res_emb_list, dim=1), dim=1)
        # user_res_emb = torch.mean(torch.stack(user_res_emb_list, dim=1), dim=1)

        entity_res_emb = torch.sum(torch.stack(entity_res_emb_list, dim=1), dim=1) # the sum operation is better.
        user_res_emb = torch.sum(torch.stack(user_res_emb_list, dim=1), dim=1)

        #step3: update item-item graph
        # item_adj = (1 - self.lambda_coeff) * self.build_adj(entity_res_emb,
        #            self.topk) + self.lambda_coeff * origin_item_adj

        # return entity_res_emb, user_res_emb, item_adj
        return entity_res_emb, user_res_emb

    def build_adj(self, context, topk):
        # Purpose: construct similarity adj matrix, not userful.
        n_entity = context.shape[0]
        # step 1: select top k elements from the predicted item-item graph.
        context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True)).cpu()
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        knn_val, knn_ind = torch.topk(sim, topk, dim=-1)
        # adj_matrix = (torch.zeros_like(sim)).scatter_(-1, knn_ind, knn_val)
        knn_val, knn_ind = knn_val.to(self.device), knn_ind.to(self.device)
        y = knn_ind.reshape(-1)
        x = torch.arange(0, n_entity).unsqueeze(dim=-1).to(self.device)
        x = x.expand(n_entity, topk).reshape(-1)
        indice = torch.cat((x.unsqueeze(dim=0), y.unsqueeze(dim=0)), dim=0)
        value = knn_val.reshape(-1)
        adj_sparsity = torch.sparse.FloatTensor(indice.data, value.data, torch.Size([n_entity, n_entity])).to(self.device)

        #step2: normalized laplacian adj
        rowsum = torch.sparse.sum(adj_sparsity, dim=1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt_value = d_inv_sqrt._values()
        x = torch.arange(0, n_entity).unsqueeze(dim=0).to(self.device)
        x = x.expand(2, n_entity)
        d_mat_inv_sqrt_indice = x
        d_mat_inv_sqrt = torch.sparse.FloatTensor(d_mat_inv_sqrt_indice, d_mat_inv_sqrt_value, torch.Size([n_entity, n_entity]))
        L_norm = torch.sparse.mm(torch.sparse.mm(d_mat_inv_sqrt, adj_sparsity.to_dense()), d_mat_inv_sqrt.to_dense())
        
        return L_norm



class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, ckg_graph_list, metapath_steps, adj_mat, user_metapath_idxes=None, item_metapath_idxes=None):
        super(Recommender, self).__init__()
        """
        ckg_graph_list: metapath_num * metapath_step * (edges_num, 2)
        """
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
                                                                      else torch.device("cpu")

        # pdb.set_trace()
        self.adj_mat = adj_mat
        self.graph = graph
        self.edge_index, self.edge_type = self._get_edges(graph)

        self.user_metapath_idxes = user_metapath_idxes
        self.item_metapah_idxes = item_metapath_idxes
        self.meta_path_tensor_list = self._get_edges_v2(ckg_graph_list)

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.gcn = self._init_model()
        self.lightgcn_layer = 2
        self.n_item_layer = 1
        self.alpha = 0.2
        self.fc1 = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                )
        self.fc2 = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                )
        self.fc3 = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                )
        self.fc4 = nn.Sequential(
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                nn.ReLU(),
                nn.Linear(self.emb_size, self.emb_size, bias=True),
                )
        self.fc5 = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size, bias=True),
        )
        self.fc6 = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size, bias=True),
        )
        self.atten_w = nn.Linear(self.emb_size, self.emb_size, bias=True)
        self.fc_meta_w = nn.Linear(self.emb_size, self.emb_size, bias=True)

        self.metapath_steps = metapath_steps

        # meta unit
        self.meta_unit = MetaWeightNet(self.emb_size)

    def initDevice(self):
        # Create channels
        # uiu, iui, [iai]*4, [uiaiu]*4, [iuiai]*4, [iaiui]*4, uiuiu
        # metapath_steps = [2] * 2 + [2] *4 + [4] * 13
        # 为每个meta构建一个GCN模型;
        self.pea_channels = torch.nn.ModuleList()
        for idx, num_steps in enumerate(self.metapath_steps):  # for test.
            # num_steps, num_nodes, dropout, emb_dim, repr_dim, hidden_size
            if idx in self.user_metapath_idxes:
                #根据metapath的长度来构建;
                self.pea_channels.append(
                    PEAGCNChannel(num_steps, self.emb_size, self.emb_size, self.emb_size).to(self.device))
            else:
                self.pea_channels.append(
                    PEAGCNChannel(num_steps, self.emb_size, self.emb_size, self.emb_size).to(self.device))

        # v2: share meta-path-based gcn models.
        # self.pea_channels_user = torch.nn.ModuleList()
        # self.pea_channels_item = torch.nn.ModuleList()
        #
        # self.pea_channels_user.append(
        #             PEAGCNChannel(2, self.emb_size, self.emb_size, self.emb_size).to(self.device))
        # self.pea_channels_user.append(
        #     PEAGCNChannel(4, self.emb_size, self.emb_size, self.emb_size).to(self.device))
        # self.pea_channels_item.append(
        #     PEAGCNChannel(2, self.emb_size, self.emb_size, self.emb_size).to(self.device))
        # self.pea_channels_item.append(
        #     PEAGCNChannel(4, self.emb_size, self.emb_size, self.emb_size).to(self.device))

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_relations=self.n_relations,
                         interact_mat=self.interact_mat,
                         ind=self.ind,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate,
                         device=self.device)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def _get_edges_v2(self, graph_list):
        """
        params:
            ckg_graph_list: metapath_num * metapath_step * (edges_num, 2)
        return:
            list of list of tensor (2, edge_num).
        """
        meta_path_tensor_list = []
        for idx, meta_path_list in enumerate(graph_list):
            if idx in self.user_metapath_idxes:
                meta_path_tensor_list.append([torch.tensor(meta_path).long().t().to(self.device) for meta_path in meta_path_list]) # 2 * edges_num;
            else:
                meta_path_tensor_list.append([torch.tensor(meta_path).long().t().to(self.device) for meta_path in meta_path_list]) # 2 * edges_num;
        return meta_path_tensor_list

    def forward(
        self,
        batch=None,
        is_training=True):
        user = batch['users']
        item = batch['items']
        labels = batch['labels']
        add_self_augment = False # augment hete_graph by dropping edges.
        add_metpath = True  # remove metpath-based embeddings.

        # step1: hetegorious graph
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb, user_gcn_emb, item_adj = self.gcn(user_emb,
        entity_gcn_emb, user_gcn_emb = self.gcn(user_emb,
                                 item_emb,
                                 self.edge_index, #(item + entity)
                                 self.edge_type,
                                 self.interact_mat,
                                 mess_dropout=self.mess_dropout,
                                 node_dropout=True,
                                 groupby_relation=False,
                                 is_training=is_training)
        hete_u_e = user_gcn_emb[user]
        hete_i_e = entity_gcn_emb[item]


        if add_self_augment == True:
            entity_gcn_emb_aug, user_gcn_emb_aug = self.gcn(user_emb,
                                                    item_emb,
                                                    self.edge_index,  # (item + entity)
                                                    self.edge_type,
                                                    self.interact_mat,
                                                    mess_dropout=self.mess_dropout,
                                                    node_dropout=True,
                                                    groupby_relation=False,
                                                    is_training=is_training)

            hete_u_e_aug = user_gcn_emb_aug[user]
            hete_i_e_aug = entity_gcn_emb_aug[item]


        # step2: obtain high-order item representations based on the item-item graph.
        # i_h = item_emb
        # for i in range(self.n_item_layer):
        #     i_h = torch.sparse.mm(item_adj, i_h)
        # i_h = F.normalize(i_h, p=2, dim=1)
        # i_e_1 = i_h[item]

        # step3: create the collabotive matrix (user+item, user+item), and obtain user and item representations based on collabotive signals.
        # interact_mat_new = self.interact_mat
        # indice_old = interact_mat_new._indices()
        # value_old = interact_mat_new._values()
        # x = indice_old[0, :]
        # y = indice_old[1, :]
        # x_A = x
        # y_A = y + self.n_users
        # x_A_T = y + self.n_users
        # y_A_T = x
        # x_new = torch.cat((x_A, x_A_T), dim=-1)
        # y_new = torch.cat((y_A, y_A_T), dim=-1)
        # indice_new = torch.cat((x_new.unsqueeze(dim=0), y_new.unsqueeze(dim=0)), dim=0)
        # value_new = torch.cat((value_old, value_old), dim=-1)
        # interact_graph = torch.sparse.FloatTensor(indice_new, value_new, torch.Size(
        #     [self.n_users + self.n_entities, self.n_users + self.n_entities]))
        # user_lightgcn_emb, item_lightgcn_emb = self.light_gcn(user_emb, item_emb, interact_graph)
        # collab_u_e = user_lightgcn_emb[user]
        # collab_i_e = item_lightgcn_emb[item]

        # step4: metapath-based GCN,
        if add_metpath == True:
            def generate_metapath_representation(keep_rate=0.4):
                # TODO: consider the relation-aware aggregation.
                metapath_idx = None
                # self.channel_aggr = 'att'
                self.channel_aggr = 'mean'
                # pdb.set_trace()
                # meta_path_tensor_list: list of list of tensor (edge_num, 2).
                # metapath_based_embed = [module(self.all_embed, self.meta_path_tensor_list[idx]).unsqueeze(1) for idx, module in enumerate(self.pea_channels)]

                # random select pea_channels # 类似于gumbel softmax自动筛选meta-paths, 根据用户喜好个性化筛选;
                # keep_rate = 0.4 # best for book datasets.
                # keep_rate = 1.0
                # keep_num = 1
                self.user_metapath_sample_idxes = np.random.choice(self.user_metapath_idxes, size=int(len(self.user_metapath_idxes) * keep_rate), replace=False)
                self.item_metapah_sample_idxes = np.random.choice(self.item_metapah_idxes, size=int(len(self.item_metapah_idxes) * keep_rate), replace=False)

                # self.user_metapath_sample_idxes = np.random.choice(len(self.user_metapath_idxes),
                #                                                    size=keep_num,
                #                                                    replace=False)
                # self.item_metapah_sample_idxes = np.random.choice(len(self.item_metapah_idxes),
                #                                                   size=keep_num,
                #                                                   replace=False)

                random_indices = np.concatenate([self.user_metapath_sample_idxes, self.item_metapah_sample_idxes], axis=0)
                # pdb.set_trace()
                # replace with hete rep;
                hete_output_emb = torch.cat((user_gcn_emb, entity_gcn_emb), dim=0)

                metapath_based_embed = dict()
                for idx in random_indices:
                    if idx in self.user_metapath_sample_idxes:
                        # metapath_based_embed[idx] = self.pea_channels[idx](self.all_embed,
                        #                                                    self.meta_path_tensor_list[idx]).unsqueeze(1)

                        # use metapath_based_embed as input; the results ???
                        metapath_based_embed[idx] = self.pea_channels[idx](hete_output_emb,
                                                                           self.meta_path_tensor_list[idx]).unsqueeze(1)

                        # shared meta-path based gcn models. results are not good ??
                        # if len(self.meta_path_tensor_list[idx]) == 2:
                        #     metapath_based_embed[idx] = self.pea_channels_user[0](self.all_embed,
                        #                                                        self.meta_path_tensor_list[idx]).unsqueeze(1)
                        # else:
                        #     metapath_based_embed[idx] = self.pea_channels_user[1](self.all_embed,
                        #                                                             self.meta_path_tensor_list[idx]).unsqueeze(1)
                    else:
                        # metapath_based_embed[idx] = self.pea_channels[idx](self.all_embed,
                        #                                                    self.meta_path_tensor_list[idx]).unsqueeze(1)

                        # use metapath_based_embed as input; the results ???
                        metapath_based_embed[idx] = self.pea_channels[idx](hete_output_emb,
                                                                           self.meta_path_tensor_list[idx]).unsqueeze(1)

                        # if len(self.meta_path_tensor_list[idx]) == 2:
                        #     metapath_based_embed[idx] = self.pea_channels_item[0](self.all_embed,
                        #                                                           self.meta_path_tensor_list[idx]).unsqueeze(1)
                        # else:
                        #     metapath_based_embed[idx] = self.pea_channels_item[1](self.all_embed,
                        #                                                           self.meta_path_tensor_list[idx]).unsqueeze(1)


                # pdb.set_trace()
                if self.user_metapath_idxes is not None:
                    metapath_user_emb_l = [value for idx, value in metapath_based_embed.items() if idx in self.user_metapath_sample_idxes]
                    # cor_scores = 0.0
                    # for i in range(3): # OOM; how  dd
                    #     cor_scores += self._create_distance_correlation(metapath_user_emb_l[i].squeeze(1), metapath_user_emb_l[i+1].squeeze(1)).sum() #[scalar]
                    # pdb.set_trace() # test
                    metapath_user_emb_l = torch.cat(metapath_user_emb_l, dim=1)

                if self.item_metapah_idxes is not None:
                    # metapath_item_emb_l = [metapath_based_embed[idx] for idx in range(len(metapath_based_embed)) if idx in self.item_metapah_sample_idxes]
                    metapath_item_emb_l = [value for idx, value in metapath_based_embed.items() if
                                           idx in self.item_metapah_sample_idxes]
                    metapath_item_emb_l = torch.cat(metapath_item_emb_l, dim=1)

                # pdb.set_trace()
                if self.channel_aggr == "mean":
                    # metapath_user_emb = F.normalize(self.fc_meta_w(metapath_user_emb_l.sum(dim=1)), dim=-1)
                    # metapath_item_emb = F.normalize(self.fc_meta_w(metapath_item_emb_l.sum(dim=1)), dim=-1)
                    metapath_user_emb = metapath_user_emb_l.mean(dim=1)
                    metapath_item_emb = metapath_item_emb_l.mean(dim=1)

                    meta_u_e = metapath_user_emb[:self.n_users, :][user]
                    meta_i_e = metapath_item_emb[self.n_users:, :][item]
                # elif self.channel_aggr == 'att': #TODO: based on relation representations, results are not good.
                #     batch_metapath_user_emb = metapath_user_emb_l[:self.n_users, :, :][user]
                #     batch_metapath_item_emb = metapath_item_emb_l[self.n_users:, :, :][item]
                #     metapath_user_emb_l_mlp_f = self.atten_w(batch_metapath_user_emb) #(bs, R1, dim)
                #     metapath_item_emb_l_mlp_f = self.atten_w(batch_metapath_item_emb) #(bs, R2, dim)
                #     # pdb.set_trace()
                #     metapath_user_emb_l_norm, metapath_item_emb_l_norm = F.normalize(metapath_user_emb_l_mlp_f, dim=-1), F.normalize(metapath_item_emb_l_mlp_f, dim=-1)
                #     user_lightgcn_emb_norm, item_lightgcn_emb_norm = F.normalize(self.atten_w(collab_u_e), dim=-1), F.normalize(self.atten_w(collab_i_e), dim=-1)
                #
                #     att_user_w = torch.bmm(metapath_user_emb_l_norm, user_lightgcn_emb_norm.unsqueeze(-1)) #(n_users, R, 1)
                #     att_item_w = torch.bmm(metapath_item_emb_l_norm, item_lightgcn_emb_norm.unsqueeze(-1)) #(n_items, R, 1)
                #     # pdb.set_trace()
                #     att_user_w = F.softmax(att_user_w, dim=1)
                #     att_item_w = F.softmax(att_item_w, dim=1)
                #     meta_u_e = torch.sum(batch_metapath_user_emb * att_user_w, dim=1)
                #     meta_i_e = torch.sum(batch_metapath_item_emb * att_item_w, dim=1)
                return meta_u_e, meta_i_e, metapath_user_emb[:self.n_users, :], metapath_item_emb[self.n_users:, :]

            if is_training:
                meta_u_e, meta_i_e, _, _ = generate_metapath_representation(keep_rate=0.4)
                # meta_u_e_aug, meta_i_e_aug = generate_metapath_representation(keep_rate=0.4)
            else:
                meta_u_e, meta_i_e, total_meta_user_emb, total_meta_item_emb = generate_metapath_representation(keep_rate=1.0)
                # meta_u_e_aug, meta_i_e_aug = generate_metapath_representation(keep_rate=0.4)

        #step5: construct losses, which contain BCE and contrastive loss.
        # # loss_contrast = 0
        # loss_contrast = self.alpha * self.calculate_loss(i_e_1, i_e_2)
        # # i_e_1 = i_e_1 + i_e_2
        # loss_contrast = loss_contrast + ((1-self.alpha)/2)*self.calculate_loss_1(i_e_2, i_e)
        # loss_contrast = loss_contrast + ((1-self.alpha)/2)*self.calculate_loss_2(u_e_2, u_e)
        #
        # u_e = torch.cat((u_e, u_e), dim=-1)
        # i_e = torch.cat((i_e, i_e_1), dim=-1)
        # i_e_1 = i_e_1 + i_e_2
        # item_1 = item_emb[item]
        # user_1 = user_emb[user]

        loss_contrast = 0.
        # add norm, the result is very bad.
        # hete_u_e, hete_i_e = F.normalize(hete_u_e), F.normalize(hete_i_e)
        # meta_u_e, meta_i_e = F.normalize(meta_u_e), F.normalize(meta_i_e) # metapath is not useful ?


        # pdb.set_trace() # torch.mean(torch.norm((hete_u_e + hete_u_e_aug) / 2.0, dim=1)), 不同tensor之间的模长差距大, hete_u_e的模长过大.
        # torch.mean(torch.norm(collab_u_e, dim=1))
        # u_e = torch.cat((hete_u_e, meta_u_e, collab_u_e), dim=-1) #模长对最终模型的性能有很大的影响, 让模长的长度尽可能是相同的.
        # i_e = torch.cat((hete_i_e, meta_i_e, collab_i_e), dim=-1)

        # i_e = torch.cat((hete_i_e, collab_i_e, meta_i_e), dim=-1) # transpose is not good, 0.8645 and 0.7858;

        # u_e = torch.cat((hete_u_e, collab_u_e), dim=-1)  # 效果有下降, 验证metapath-aware representations是有效果的;
        # i_e = torch.cat((hete_i_e, collab_i_e), dim=-1)

        # u_e = torch.cat((meta_u_e, collab_u_e), dim=-1)  # 验证hete_u_e是否有效呢, 0.8435, 0.7422; 相比于只考虑interactive graph的效果还要差;
        # i_e = torch.cat((meta_u_e, collab_i_e), dim=-1)

        # u_e = torch.cat((hete_u_e, collab_u_e), dim=-1)  # 验证按照首先按照关系聚合的方式是否有效呢？ doing.
        # i_e = torch.cat((hete_i_e, collab_i_e), dim=-1)
        # pdb.set_trace()

        # u_e = torch.cat(((hete_u_e + hete_u_e_aug)/2.0, collab_u_e), dim=-1) # 联合不同的聚合方式, 是否有效呢?
        # i_e = torch.cat(((hete_i_e + hete_i_e_aug)/2.0, collab_i_e), dim=-1)

        # u_e = torch.cat(((hete_u_e + hete_u_e_aug) / 2.0, meta_u_e, collab_u_e), dim=-1) # 联合不同的聚合方式, 是否有效呢?
        # i_e = torch.cat(((hete_i_e + hete_i_e_aug) / 2.0, meta_i_e, collab_i_e), dim=-1)

        # hete_u_e_average = (hete_u_e + hete_u_e_aug) / 2.0
        # hete_i_e_average = (hete_i_e + hete_i_e_aug) / 2.0
        # meta_u_e_average = (meta_u_e + meta_u_e_aug)/2.0
        # meta_i_e_average = (meta_i_e + meta_i_e_aug)/2.0


        # self gate;
        # v1: concat_share_weight: test_auc:0.8730, test_f1:0.7967
        # v2: add_share_weight: test_auc:0.8761, test_f1:0.7923
        # v3: add_independent_weight: test_auc:0.8728, test_f1:0.7907

        # collab_u_e, collab_i_e, hete_u_e_average, hete_i_e_average, meta_u_e_average, meta_i_e_average = \
        #                                         self.meta_unit(collab_u_e, collab_i_e, hete_u_e_average, hete_i_e_average, meta_u_e_average, meta_i_e_average, is_share=False)



        # add
        # u_e = hete_u_e_average + meta_u_e_average + collab_u_e
        # i_e = hete_i_e_average + meta_i_e_average + collab_i_e

        # u_e = torch.cat((hete_u_e, meta_u_e, collab_u_e),
        #                 dim=-1)  # results are not good.
        # i_e = torch.cat((hete_i_e, meta_i_e, collab_i_e),
        #                 dim=-1)
        # pdb.set_trace()

        # u_e = torch.cat((meta_u_e, collab_u_e), dim=-1)  # 验证按照首先按照关系聚合的方式是否有效呢？ doing.
        # i_e = torch.cat((meta_i_e, collab_i_e), dim=-1)
        # pdb.set_trace()
        # add disentangled losses
        # loss_contrast = self._create_distance_correlation(hete_i_e, self.fc1(meta_i_e)).sum()
        # loss_contrast += self._create_distance_correlation(hete_u_e, self.fc1(meta_u_e)).sum()



        # scores_meta = (meta_u_e * meta_i_e).sum(dim=1)
        # scores_colla = (collab_u_e * collab_i_e).sum(dim=1)
        # scores_hete = (((hete_u_e + hete_u_e_aug) / 2.0) * ((hete_i_e + hete_i_e_aug) / 2.0)).sum(dim=1)
        # pdb.set_trace()


        # 如何让模型自己筛选关注的特征, 通过meta Unit方式构建
        # bce_loss + emb_loss + cl_w * loss_contrast, scores, bce_loss, emb_loss

        # v1: concat version.
        # batch_size = collab_u_e.shape[0]
        # colla_scores = (collab_u_e * collab_i_e).sum(dim=1)
        # heta_scores = (hete_u_e_average * hete_i_e_average).sum(dim=1)
        # meta_scores = (meta_u_e_average * meta_i_e_average).sum(dim=1)
        #
        # total_scores = colla_scores + heta_scores + meta_scores
        # total_scores = torch.sigmoid(total_scores)
        # criteria = nn.BCELoss()
        # bce_loss = criteria(total_scores, labels.float())
        #
        # regularizer = (torch.norm(collab_u_e) ** 2
        #                + torch.norm(collab_i_e) ** 2
        #                + torch.norm(hete_u_e_average) ** 2
        #                + torch.norm(hete_i_e_average) ** 2
        #                + torch.norm(meta_u_e_average) ** 2
        #                + torch.norm(meta_i_e_average) ** 2) / 1
        # emb_loss = self.decay * regularizer / batch_size
        #
        # total_loss = bce_loss + emb_loss
        # total_bce_loss = bce_loss
        # total_emb_loss = emb_loss

        # v2: 独立建模损失损失.
        # colla_total_loss, colla_scores, colla_bce_loss, colla_emb_loss = self.create_bpr_loss_single(collab_u_e, collab_i_e, labels)
        # hete_total_loss, hete_scores, hete_bce_loss, hete_emb_loss = self.create_bpr_loss_single(hete_u_e_average, hete_i_e_average, labels)
        # meta_total_loss, meta_scores, meta_bce_loss, meta_emb_loss = self.create_bpr_loss_single(meta_u_e_average, meta_i_e_average, labels)
        # # merge
        # total_loss = 1.0 * colla_total_loss + hete_total_loss + meta_total_loss
        # total_scores = (colla_scores + hete_scores + meta_scores) / 3.0
        # total_bce_loss = colla_bce_loss + hete_bce_loss + meta_bce_loss
        # total_emb_loss = colla_emb_loss + hete_emb_loss + meta_emb_loss
        # return total_loss, total_scores, total_bce_loss, total_emb_loss

        hete_u_e_average = hete_u_e
        hete_i_e_average = hete_i_e
        meta_u_e_average = meta_u_e
        meta_i_e_average = meta_i_e

        # concat
        # u_e = torch.cat((hete_u_e_average, meta_u_e_average, collab_u_e), dim=-1)  # results are good. dd
        # i_e = torch.cat((hete_i_e_average, meta_i_e_average, collab_i_e), dim=-1)

        # remove interaction-based representations,,
        u_e = torch.cat((hete_u_e_average, meta_u_e_average), dim=-1)  # remove lightgcn
        i_e = torch.cat((hete_i_e_average, meta_i_e_average), dim=-1)

        # KGAT设置不同dropout编码到表征的对比学习;
        # loss_contrast += self.calculate_loss(hete_u_e, hete_u_e_aug, self.fc5)
        # loss_contrast += self.calculate_loss(hete_i_e, hete_i_e_aug, self.fc6)

        # 不同KG encoder编码的表征之间的对比学习;
        loss_contrast += self.calculate_loss(hete_u_e, meta_u_e, self.fc1)  # different KG-enhanced reps.
        loss_contrast += self.calculate_loss(hete_i_e, meta_i_e, self.fc2)

        # loss_contrast += self.calculate_loss(meta_u_e, meta_u_e_aug, self.fc3)  # low and high reps induced from the UI graph.
        # loss_contrast += self.calculate_loss(meta_i_e, meta_i_e_aug, self.fc4)
        if is_training == False:
            self.user_all_embeddings = torch.cat((user_gcn_emb, total_meta_user_emb), dim=-1)
            self.entity_all_embeddings = torch.cat((entity_gcn_emb, total_meta_item_emb), dim=-1)

        return self.create_bpr_loss(u_e, i_e, labels, loss_contrast, cl_w=0.05)


    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def calculate_loss(self, A_embedding, B_embedding, fc):
        tau = 0.6    # default = 0.8
        f_exp = lambda x: torch.exp(x / tau)
        A_embedding = fc(A_embedding)
        B_embedding = fc(B_embedding)
        refl_sim = f_exp(self.sim(A_embedding, A_embedding))
        between_sim = f_exp(self.sim(A_embedding, B_embedding))

        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        
        # symmetry cl loss
        refl_sim_1 = f_exp(self.sim(B_embedding, B_embedding))
        between_sim_1 = f_exp(self.sim(B_embedding, A_embedding))
        loss_2 = -torch.log(
            between_sim_1.diag()
            / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
        ret = (loss_1 + loss_2) * 0.5
        ret = ret.mean()
        return ret

    def light_gcn(self, user_embedding, item_embedding, adj):
        ego_embeddings = torch.cat((user_embedding, item_embedding), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.lightgcn_layer):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            side_embeddings = F.normalize(side_embeddings) # add normalization, add the normalization is better.
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False) # mean的效果更好;
        # all_embeddings = all_embeddings.sum(dim=1, keepdim=False) # sum的效果不好;
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_entities], dim=0)
        return u_g_embeddings, i_g_embeddings

    def _create_distance_correlation(self, X1, X2):
        """
        Args:
            X1: (batchsize, dim)
            X2: (batchsize, dim)

        Returns:
        """
        def _create_centered_distance(X):
            '''
                X: (batchsize, dim)
                Used to calculate the distance matrix of N samples.
                (However how could tf store a HUGE matrix with the shape like 70000*70000*4 Bytes????)
            '''
            # calculate the pairwise distance of X
            # .... A with the size of [batch_size, embed_size/n_factors]
            # .... D with the size of [batch_size, batch_size]
            # X = tf.math.l2_normalize(XX, axis=1)
            R = torch.square(X).sum(dim=1, keepdim=True) # (bs, 1)
            # pdb.set_trace()
            # self.thre_scalar = torch.FloatTensor([0.0]).expand_as(R).to(self.device)
            ddd = R - 2 * torch.matmul(X, torch.transpose(X, 0, 1)) + torch.transpose(R, 0, 1)
            ddd = (ddd > 0.0) * ddd
            # pdb.set_trace()
            D = torch.sqrt(ddd + 1e-8) #(bs, bs)

            # # calculate the centered distance of X
            # # .... D with the size of [batch_size, batch_size]
            D = D - D.mean(dim=0, keepdim=True) - D.mean(dim=1, keepdim=True) + D.mean()
            return D

        def _create_distance_covariance(D1, D2):
            # calculate distance covariance between D1 and D2
            n_samples = float(D1.size()[0])
            ddd = (D1 * D2).sum() / (n_samples * n_samples)
            ddd = (ddd > 0.0) * ddd
            dcov = torch.sqrt(ddd + 1e-8)
            # dcov = tf.sqrt(tf.maximum(tf.reduce_sum(D1 * D2)) / n_samples
            return dcov

        D1 = _create_centered_distance(X1)
        D2 = _create_centered_distance(X2)
        # pdb.set_trace()
        dcov_12 = _create_distance_covariance(D1, D2)
        dcov_11 = _create_distance_covariance(D1, D1)
        dcov_22 = _create_distance_covariance(D2, D2)

        # calculate the distance correlation
        ddd = dcov_11 * dcov_22
        ddd = (ddd > 0.0) * ddd
        dcor = dcov_12 / (torch.sqrt(ddd) + 1e-10)
        # return tf.reduce_sum(D1) + tf.reduce_sum(D2)
        return dcor

    def create_bpr_loss(self, users, items, labels, loss_contrast, cl_w=0.01):
        batch_size = users.shape[0]
        scores = (items * users).sum(dim=1)
        scores = torch.sigmoid(scores)
        criteria = nn.BCELoss()
        bce_loss = criteria(scores, labels.float())
        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        # cor_loss = self.sim_decay * cor
        # cl_loss_w = 0.01
        # pdb.set_trace()
        return bce_loss + emb_loss + cl_w*loss_contrast, scores, bce_loss, emb_loss

    def create_bpr_loss_single(self, users, items, labels):
        batch_size = users.shape[0]
        scores = (items * users).sum(dim=1)
        scores = torch.sigmoid(scores)
        criteria = nn.BCELoss()
        bce_loss = criteria(scores, labels.float())
        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return bce_loss + emb_loss, scores, bce_loss, emb_loss

    # def generate(self):
    #     user_emb = self.all_embed[:self.n_users, :]
    #     item_emb = self.all_embed[self.n_users:, :]
    #     entity_gcn_emb, user_gcn_emb, item_adj = self.gcn(user_emb,
    #                                                       item_emb,
    #                                                       self.edge_index,
    #                                                       self.edge_type,
    #                                                       self.interact_mat,
    #                                                       mess_dropout=self.mess_dropout,
    #                                                       node_dropout=self.node_dropout)
    #
    #     interact_mat_new = torch.sparse.mm(self.interact_mat, item_adj)
    #     indice_old = interact_mat_new._indices()
    #     value_old = interact_mat_new._values()
    #     x = indice_old[0, :]
    #     y = indice_old[1, :]
    #     x_A = x
    #     y_A = y + self.n_users
    #     x_A_T = y + self.n_users
    #     y_A_T = x
    #     x_new = torch.cat((x_A, x_A_T), dim=-1)
    #     y_new = torch.cat((y_A, y_A_T), dim=-1)
    #     indice_new = torch.cat((x_new.unsqueeze(dim=0), y_new.unsqueeze(dim=0)), dim=0)
    #     value_new = torch.cat((value_old, value_old), dim=-1)
    #     interact_graph = torch.sparse.FloatTensor(indice_new, value_new, torch.Size(
    #         [self.n_users + self.n_entities, self.n_users + self.n_entities]))
    #     user_lightgcn_emb, item_lightgcn_emb = self.light_gcn(user_emb, item_emb, interact_graph)
    #     u_e = torch.cat((user_gcn_emb, user_lightgcn_emb), dim=-1)
    #     i_e = torch.cat((entity_gcn_emb, item_lightgcn_emb), dim=-1)
    #     return i_e, u_e