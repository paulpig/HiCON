from turtle import pos
import numpy as np
from tqdm import tqdm
import networkx as nx
import scipy.sparse as sp
import torch
import os
import matplotlib.pyplot as plt
import scipy
import random
from time import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import pdb

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0
interact_r_idx = 0
inverse_interact_r_idx = 0
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)
max_entity_len = 0


def read_cf(file_name):
    try:
        inter_mat = np.load(file_name + '.npy')
    except:
        inter_mat = list()
        lines = open(file_name, "r").readlines()
        max_user, max_item = 0, 0
        user2items = defaultdict(list)
        for l in lines:
            tmps = l.strip()
            inters = [int(i) for i in tmps.split(" ")]
            u_id, pos_ids = inters[0], inters[1:]

            # filter 10-core
            # if len(pos_ids) < 10:
            #     continue

            max_user = max(max_user, u_id)

            pos_ids = list(set(pos_ids))
            for i_id in pos_ids:
                inter_mat.append([u_id, i_id, 1])
                max_item = max(max_item, i_id)
            user2items[u_id] = pos_ids
        # pdb.set_trace()
        pos_len = len(inter_mat)
        for idx in range(pos_len):
            if idx % 10000 == 0:
                print("{}/{}".format(idx, pos_len))
            neg_item_idx = np.random.choice(max_item, 1, replace=False)[0]
            while neg_item_idx in user2items[inter_mat[idx][0]]:
                neg_item_idx = np.random.choice(max_item, 1, replace=False)[0]
            # pdb.set_trace()
            inter_mat.append([u_id, neg_item_idx, 0])

        inter_mat = np.array(inter_mat)
        # shuffle
        np.random.shuffle(inter_mat)
        np.save(file_name + '.npy', inter_mat)
    return inter_mat

def read_cf_new():
    # reading rating file
    rating_file = 'data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)

    # rating_np_origin = rating_np
    # rating_np_label = rating_np.take([2], axis=1)
    # indix_click = np.where(rating_np_label == 1)
    # rating_np = rating_np.take(indix_click[0], axis=0)
    # rating_np = rating_np.take([0, 1], axis=1)

    test_ratio = 0.2
    n_ratings = rating_np.shape[0]
    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * test_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    # test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left)

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    # test_data = rating_np[test_indices]

    train_rating = rating_np[train_indices]
    ui_adj = generate_ui_adj(rating_np, train_rating)
    return train_data, eval_data, ui_adj

def generate_ui_adj(rating, train_rating):
    #ui_adj = sp.dok_matrix((n_user, n_item), dtype=np.float32)
    n_user, n_item = len(set(rating[:, 0])), len(set(rating[:, 1]))
    ui_adj_orign = sp.coo_matrix(
        (train_rating[:, 2], (train_rating[:, 0], train_rating[:, 1])), shape=(n_user, n_item)).todok()

    # ui_adj = sp.dok_matrix((n_user+n_item, n_user+n_item), dtype=np.float32)
    # ui_adj[:n_user, n_user:] = ui_adj_orign
    # ui_adj[n_user:, :n_user] = ui_adj_orign.T
    ui_adj = sp.bmat([[None, ui_adj_orign],
                    [ui_adj_orign.T, None]], dtype=np.float32)
    ui_adj = ui_adj.todok()
    print('already create user-item adjacency matrix', ui_adj.shape)
    return ui_adj

def remap_item(train_data, eval_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(eval_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(eval_data[:, 1])) + 1

    eval_data_label = eval_data.take([2], axis=1)
    indix_click = np.where(eval_data_label == 1)
    eval_data = eval_data.take(indix_click[0], axis=0) # select the samples whose labels are 1.

    eval_data = eval_data.take([0, 1], axis=1)
    train_data = train_data.take([0, 1], axis=1)
    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in eval_data:
        test_user_set[int(u_id)].append(int(i_id))


def read_triplets(file_name, inverse_r=False):
    global n_entities, n_relations, n_nodes, max_entity_len, interact_r_idx, inverse_interact_r_idx

    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)

    max_entity_len = max(can_triplets_np[:, 1]) + 1 # shift num

    # if args.inverse_r:
    if inverse_r:
        # get triplets with inverse direction like <entity, is-aspect-of, item>
        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]
        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max_entity_len
        # consider two additional relations --- 'interact' and 'be interacted'
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 2
        interact_r_idx, inverse_interact_r_idx = 0, 1 #idx of interactive relations
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 2
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        # consider two additional relations --- 'interact'.
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1 #[1, 60]
        triplets = can_triplets_np.copy()

    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1  # not including users
    # pdb.set_trace() # np.sort(triplets[:, 2])[:40]
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets

# heterogeneous graph excludes user-item pairs.
def build_graph(train_data, triplets):
    ckg_graph = nx.MultiDiGraph()
    rd = defaultdict(list)
    train_data = train_data.take([0, 1], axis=1)
    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        rd[0].append([u_id, i_id])

    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):
        ckg_graph.add_edge(h_id, t_id, key=r_id)
        rd[r_id].append([h_id, t_id])
    return ckg_graph, rd

# heterogeneous graph includes user-item pairs.
def build_graph_v2(train_data, triplets):
    global n_users, interact_r_idx, inverse_interact_r_idx
    ckg_graph = nx.MultiDiGraph()
    rd = defaultdict(list)
    train_data = train_data.take([0, 1], axis=1)
    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        ckg_graph.add_edge(u_id, i_id+n_users, key=interact_r_idx)
        ckg_graph.add_edge(i_id+n_users, u_id, key=inverse_interact_r_idx)
        rd[0].append([u_id, i_id])

    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):
        ckg_graph.add_edge(h_id+n_users, t_id+n_users, key=r_id)
        rd[r_id].append([h_id, t_id])
    # pdb.set_trace() # triplets[:,1].min(); max: 121, min: 2;
    return ckg_graph, rd


def build_graph_metapath(train_data, triplets):
    """
    return 
        ckg_graph_list: 'uiu, iui, iai, uiaiu, iuiai, iaiui, uiuiu';

    """
    i2e_relation_num = n_relations - 1 # -1 for the interactive relations.
    # v1: uiu, iui, iai, uiaiu, iuiai, iaiui, uiuiu
    meta_path_num = 4
    u2i_idx_edges, i2u_idx_edges = [], []
    e2i_idx_edges_list = defaultdict(list) # 60*2
    i2e_idx_edges_list = defaultdict(list) # 60*2

    ckg_graph_list = [[] for _ in range(i2e_relation_num * meta_path_num + 3)] # 60*4+2

    # head_r_tail = defaultdict(dict)
    # rd = defaultdict(list)
    train_data = train_data.take([0, 1], axis=1)
    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        # rd[0].append([u_id, i_id])
        u2i_idx_edges.append([u_id, i_id])
        i2u_idx_edges.append([i_id, u_id])

    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True): # r_id: [1, 60]
        i2e_idx_edges_list[r_id].append([h_id, t_id])
        e2i_idx_edges_list[r_id].append([t_id, h_id])

    ckg_graph_list[0] = [u2i_idx_edges, i2u_idx_edges] # uiu
    ckg_graph_list[1] = [i2u_idx_edges, u2i_idx_edges] # iui
    ckg_graph_list[-1] = [u2i_idx_edges, i2u_idx_edges, u2i_idx_edges, i2u_idx_edges] # uiuiu

    for idx in range(1, i2e_relation_num+1): # 0 for the interactive relatioin.
        pos_idx = (idx - 1) * meta_path_num + 2
        # pdb.set_trace() #max(list(i2e_idx_edges_list.keys()))
        # iai
        ckg_graph_list[pos_idx] = [i2e_idx_edges_list[idx], e2i_idx_edges_list[idx]]
        # uiaiu
        ckg_graph_list[pos_idx + 1] = [u2i_idx_edges, i2e_idx_edges_list[idx], e2i_idx_edges_list[idx], i2u_idx_edges]
        # iuiai
        ckg_graph_list[pos_idx + 2] = [i2u_idx_edges, u2i_idx_edges, i2e_idx_edges_list[idx], e2i_idx_edges_list[idx]]
        # iaiui
        ckg_graph_list[pos_idx + 3] = [i2e_idx_edges_list[idx], e2i_idx_edges_list[idx], i2u_idx_edges, u2i_idx_edges]
    
    return ckg_graph_list

# result: test_auc:0.8807, test_f1:0.7984
def build_graph_metapath_v2(train_data, triplets):
    """
    metapath types: uiu, uiuiu, uiaiu. iui, iai, iuiai, iaiui.
    return 
        ckg_graph_list: 'uiu, iui, iai, uiaiu, iuiai, iaiui, uiuiu';

    """
    i2e_relation_num = (n_relations - 1) // 2 # -1 for the interactive relations.
    
    meta_path_num = 4
    u2i_idx_edges, i2u_idx_edges = [], []
    e2i_idx_edges_list = defaultdict(list) # 60*2
    i2e_idx_edges_list = defaultdict(list) # 60*2

    ckg_graph_list = [[] for _ in range(i2e_relation_num * meta_path_num + 3)] # 60*4+2

    # head_r_tail = defaultdict(dict)
    # rd = defaultdict(list)
    train_data = train_data.take([0, 1], axis=1)
    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        # rd[0].append([u_id, i_id])
        i_id = i_id + n_users # support for the symmetric interactive graph.
        u2i_idx_edges.append([u_id, i_id])
        i2u_idx_edges.append([i_id, u_id])

    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True): # r_id: [1, 60]
        h_id = h_id + n_users  # support for the hetegenorous graph.
        t_id = t_id + n_users  # support for the hetegenorous graph.
        i2e_idx_edges_list[r_id].append([h_id, t_id])
        e2i_idx_edges_list[r_id].append([t_id, h_id])
        # test,aa = min([i2e_idx_edges_list[i] for i in range(1, i2e_relation_num)])
        # [i for i in range(1, n_relations)]
    # min_l, max_l = [], []
    # for i in range(1, i2e_relation_num):
    #     # pdb.set_trace()
    #     a = np.array(i2e_idx_edges_list[i])[:,0].min()
    #     min_l.append(a)
    #     b = np.array(i2e_idx_edges_list[i])[:,0].max()
    #     max_l.append(b)
    # print(min(min_l), max(max_l)) test
    user_metpath_idxes, item_metpath_idxes = [], []
    ckg_graph_list[0] = [u2i_idx_edges, i2u_idx_edges] # uiu
    ckg_graph_list[1] = [i2u_idx_edges, u2i_idx_edges] # iui
    ckg_graph_list[-1] = [u2i_idx_edges, i2u_idx_edges, u2i_idx_edges, i2u_idx_edges] # uiuiu

    user_metpath_idxes.extend([0, len(ckg_graph_list)-1])
    item_metpath_idxes.append(1)

    for idx in range(1, i2e_relation_num+1): # 0 for the interactive relatioin.
        pos_idx = (idx - 1) * meta_path_num + 2
        # pdb.set_trace() #max(list(i2e_idx_edges_list.keys()))
        # iai
        # if max(np.array(i2e_idx_edges_list[idx])[:,1]) <= (n_items + n_users):
        #     pdb.set_trace()
        ckg_graph_list[pos_idx] = [i2e_idx_edges_list[idx], e2i_idx_edges_list[idx]]
        # uiaiu
        ckg_graph_list[pos_idx + 1] = [u2i_idx_edges, i2e_idx_edges_list[idx], e2i_idx_edges_list[idx], i2u_idx_edges]
        # iuiai
        ckg_graph_list[pos_idx + 2] = [i2u_idx_edges, u2i_idx_edges, i2e_idx_edges_list[idx], e2i_idx_edges_list[idx]]
        # iaiui
        ckg_graph_list[pos_idx + 3] = [i2e_idx_edges_list[idx], e2i_idx_edges_list[idx], i2u_idx_edges, u2i_idx_edges]

        user_metpath_idxes.append(pos_idx + 1)
        item_metpath_idxes.extend([pos_idx, pos_idx + 2, pos_idx + 3])

    return ckg_graph_list, user_metpath_idxes, item_metpath_idxes

# result: test_auc:0.8762, test_f1:0.7946
def build_graph_metapath_v3(train_data, triplets):
    """
    metapath types: iaiu, iaiuiu;  uiai, uiuiai,
    return
        ckg_graph_list: 'iaiu, iaiuiu;  uiai, uiuiai';

    """
    i2e_relation_num = (n_relations - 1) // 2  # -1 for the interactive relations.

    meta_path_num = 4
    u2i_idx_edges, i2u_idx_edges = [], []
    e2i_idx_edges_list = defaultdict(list)  # 60*2
    i2e_idx_edges_list = defaultdict(list)  # 60*2

    ckg_graph_list = [[] for _ in range(i2e_relation_num * meta_path_num)]  # 60*4

    # head_r_tail = defaultdict(dict)
    # rd = defaultdict(list)
    train_data = train_data.take([0, 1], axis=1)
    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        # rd[0].append([u_id, i_id])
        i_id = i_id + n_users  # support for the symmetric interactive graph.
        u2i_idx_edges.append([u_id, i_id])
        i2u_idx_edges.append([i_id, u_id])

    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):  # r_id: [1, 60]
        h_id = h_id + n_users  # support for the hetegenorous graph.
        t_id = t_id + n_users  # support for the hetegenorous graph.
        i2e_idx_edges_list[r_id].append([h_id, t_id])
        e2i_idx_edges_list[r_id].append([t_id, h_id])

    for idx in range(1, i2e_relation_num + 1):  # 0 for the interactive relation.
        pos_idx = (idx - 1) * meta_path_num
        # iaiu
        ckg_graph_list[pos_idx] = [i2e_idx_edges_list[idx], e2i_idx_edges_list[idx], i2u_idx_edges]
        # iaiuiu
        ckg_graph_list[pos_idx + 1] = [i2e_idx_edges_list[idx], e2i_idx_edges_list[idx], i2u_idx_edges,
                                       u2i_idx_edges, i2u_idx_edges]
        # uiai
        ckg_graph_list[pos_idx + 2] = [u2i_idx_edges, i2e_idx_edges_list[idx], e2i_idx_edges_list[idx]]
        # uiuiai
        ckg_graph_list[pos_idx + 3] = [u2i_idx_edges, i2u_idx_edges, u2i_idx_edges,
                                       i2e_idx_edges_list[idx], e2i_idx_edges_list[idx]]

    return ckg_graph_list


# result:
def build_graph_metapath_v4(train_data, triplets):
    """
    metapath types: uiu, uiuiu, uiaiu. iui, iai, iuiai, iaiui. aia, aiuia, aiaia.
    return
        ckg_graph_list: 'uiu, iui, iai, uiaiu, iuiai, iaiui, uiuiu';

    """
    i2e_relation_num = (n_relations - 1) // 2  # -1 for the interactive relations.

    # meta_path_num = 4
    meta_path_num = 7
    u2i_idx_edges, i2u_idx_edges = [], []
    e2i_idx_edges_list = defaultdict(list)  # 60*2
    i2e_idx_edges_list = defaultdict(list)  # 60*2

    ckg_graph_list = [[] for _ in range(i2e_relation_num * meta_path_num + 3)]  # 60*4+2

    # head_r_tail = defaultdict(dict)
    # rd = defaultdict(list)
    train_data = train_data.take([0, 1], axis=1)
    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        # rd[0].append([u_id, i_id])
        i_id = i_id + n_users  # support for the symmetric interactive graph.
        u2i_idx_edges.append([u_id, i_id])
        i2u_idx_edges.append([i_id, u_id])

    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):  # r_id: [1, 60]
        h_id = h_id + n_users  # support for the hetegenorous graph.
        t_id = t_id + n_users  # support for the hetegenorous graph.
        i2e_idx_edges_list[r_id].append([h_id, t_id])
        e2i_idx_edges_list[r_id].append([t_id, h_id])

    ckg_graph_list[0] = [u2i_idx_edges, i2u_idx_edges]  # uiu
    ckg_graph_list[1] = [i2u_idx_edges, u2i_idx_edges]  # iui
    ckg_graph_list[-1] = [u2i_idx_edges, i2u_idx_edges, u2i_idx_edges, i2u_idx_edges]  # uiuiu

    for idx in range(1, i2e_relation_num + 1):  # 0 for the interactive relatioin.
        pos_idx = (idx - 1) * meta_path_num + 2
        # pdb.set_trace() #max(list(i2e_idx_edges_list.keys()))
        # iai
        ckg_graph_list[pos_idx] = [i2e_idx_edges_list[idx], e2i_idx_edges_list[idx]]
        # uiaiu
        ckg_graph_list[pos_idx + 1] = [u2i_idx_edges, i2e_idx_edges_list[idx], e2i_idx_edges_list[idx], i2u_idx_edges]
        # iuiai
        ckg_graph_list[pos_idx + 2] = [i2u_idx_edges, u2i_idx_edges, i2e_idx_edges_list[idx], e2i_idx_edges_list[idx]]
        # iaiui
        ckg_graph_list[pos_idx + 3] = [i2e_idx_edges_list[idx], e2i_idx_edges_list[idx], i2u_idx_edges, u2i_idx_edges]
        # aia
        ckg_graph_list[pos_idx + 4] = [e2i_idx_edges_list[idx], i2e_idx_edges_list[idx]]
        # aiuia
        ckg_graph_list[pos_idx + 5] = [e2i_idx_edges_list[idx], i2u_idx_edges, u2i_idx_edges, i2e_idx_edges_list[idx]]
        # aiaia.
        ckg_graph_list[pos_idx + 6] = [e2i_idx_edges_list[idx], i2e_idx_edges_list[idx], e2i_idx_edges_list[idx], i2e_idx_edges_list[idx]]
    return ckg_graph_list


# result:
def build_graph_metapath_v5(train_data, triplets):
    """
    metapath types: i-u, e-i-u, e-i, i-e-i.
    return
        ckg_graph_list: 'i-u, e-i-u, e-i, i-e-i'.
    """
    i2e_relation_num = (n_relations - 1) // 2  # -1 for the interactive relations.

    meta_path_num = 3
    u2i_idx_edges, i2u_idx_edges = [], []
    e2i_idx_edges_list = defaultdict(list)  # 60*2
    i2e_idx_edges_list = defaultdict(list)  # 60*2

    ckg_graph_list = [[] for _ in range(i2e_relation_num * meta_path_num + 1)]  # 60*4+2

    # head_r_tail = defaultdict(dict)
    # rd = defaultdict(list)
    train_data = train_data.take([0, 1], axis=1)
    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        # rd[0].append([u_id, i_id])
        i_id = i_id + n_users  # support for the symmetric interactive graph.
        u2i_idx_edges.append([u_id, i_id])
        i2u_idx_edges.append([i_id, u_id])

    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):  # r_id: [1, 60]
        h_id = h_id + n_users  # support for the hetegenorous graph.
        t_id = t_id + n_users  # support for the hetegenorous graph.
        i2e_idx_edges_list[r_id].append([h_id, t_id])
        e2i_idx_edges_list[r_id].append([t_id, h_id])
        # test,aa = min([i2e_idx_edges_list[i] for i in range(1, i2e_relation_num)])
        # [i for i in range(1, n_relations)]
    # print(min(min_l), max(max_l))
    user_metpath_idxes, item_metpath_idxes = [], []
    ckg_graph_list[0] = [i2u_idx_edges]  # iu
    user_metpath_idxes.extend([0])

    for idx in range(1, i2e_relation_num + 1):  # 0 for the interactive relatioin.
        pos_idx = (idx - 1) * meta_path_num + 1
        # e-i-u
        ckg_graph_list[pos_idx] = [e2i_idx_edges_list[idx], i2u_idx_edges]
        # e-i
        ckg_graph_list[pos_idx + 1] = [e2i_idx_edges_list[idx]]
        # i-e-i
        ckg_graph_list[pos_idx + 2] = [i2e_idx_edges_list[idx], e2i_idx_edges_list[idx]]

        user_metpath_idxes.append(pos_idx)
        item_metpath_idxes.extend([pos_idx + 1, pos_idx + 2])

    return ckg_graph_list, user_metpath_idxes, item_metpath_idxes


def build_sparse_relational_graph(relation_dict):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    adj_mat_list = []
    print("Begin to build sparse relation matrix ...")
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0:
            cf = np_mat.copy()
            cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
            vals = [1.] * len(cf)
            adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
        else:
            vals = [1.] * len(np_mat)
            adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes))
        # pdb.set_trace() #sp.linalg.norm(A-A.T, scipy.Inf) < 1e-8:
        adj_mat_list.append(adj)

    norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
    mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]
    # interaction: user->item, [n_users, n_entities]
    norm_mat_list[0] = norm_mat_list[0].tocsr()[:n_users, n_users:].tocoo()
    # norm_mat_list[0] = norm_mat_list[0].tocoo()
    mean_mat_list[0] = mean_mat_list[0].tocsr()[:n_users, n_users:].tocoo()

    return adj_mat_list, norm_mat_list, mean_mat_list


def build_sparse_relational_graph_right_norm(relation_dict):
    """
    修正col and row-wise norm
    Args:
        relation_dict:
    Returns:

    """
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    adj_mat_list = []
    adj_i_r_list = []
    print("Begin to build sparse relation matrix ...")
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0:
            cf = np_mat.copy()
            cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
            vals = [1.] * len(cf) * 2
            # adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
            adj = sp.coo_matrix(
                (vals, (np.concatenate([cf[:, 0], cf[:, 1]], axis=-1), np.concatenate([cf[:, 1], cf[:, 0]], axis=-1))),
                shape=(n_nodes, n_nodes))
            adj_i_r_list.append([])
        else:
            vals = [1.] * len(np_mat)
            adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes))

            # i-r matrix
            y_idx = [r_id] * len(np_mat)
            adj_i_r = sp.coo_matrix((vals, (np_mat[:, 0] + n_users, y_idx)), shape=(n_nodes, n_relations))
            adj_i_r_list.append(adj_i_r)
        # pdb.set_trace() #sp.linalg.norm(A-A.T, scipy.Inf) < 1e-8:
        adj_mat_list.append(adj)

    norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
    mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]
    # interaction: user->item, [n_users, n_entities]
    norm_mat_list[0] = norm_mat_list[0].tocsr()[:n_users, n_users:].tocoo()
    # norm_mat_list[0] = norm_mat_list[0].tocoo()
    mean_mat_list[0] = mean_mat_list[0].tocsr()[:n_users, n_users:].tocoo()

    return adj_mat_list, norm_mat_list, mean_mat_list, adj_i_r_list

# conduct a symmetric user-item interactive graph.
def build_sparse_relational_graph_v2(relation_dict):
    """
    Args:
        relation_dict:

    Returns:
        user-item graph: (n_users + n_entites, n_users + n_entities)
    """
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    adj_mat_list = []
    print("Begin to build sparse relation matrix ...")
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0:
            cf = np_mat.copy()
            cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
            vals = [1.] * len(cf) * 2
            # pdb.set_trace() # np.concatenate([cf[:, 0], cf[:, 1]], axis=-1)
            # symmetric matrix
            adj = sp.coo_matrix((vals, (np.concatenate([cf[:, 0], cf[:, 1]], axis=-1), np.concatenate([cf[:, 1], cf[:, 0]], axis=-1))), shape=(n_nodes, n_nodes))
        else:
            vals = [1.] * len(np_mat)
            adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes))
        # pdb.set_trace() #sp.linalg.norm(adj-adj.T, scipy.Inf) < 1e-8:
        adj_mat_list.append(adj)

    norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
    mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]
    # interaction: user->item, [n_users, n_entities]
    # norm_mat_list[0] = norm_mat_list[0].tocsr()[:n_users, n_users:].tocoo()
    norm_mat_list[0] = norm_mat_list[0].tocoo()
    # mean_mat_list[0] = mean_mat_list[0].tocsr()[:n_users, n_users:].tocoo()
    mean_mat_list[0] = mean_mat_list[0].tocoo()

    return adj_mat_list, norm_mat_list, mean_mat_list

def count_relations(triplets):
    rela_d = defaultdict(float)
    for item in triplets:
        rela_d[item[1]] += 1.
    return rela_d

def conductHomoGraph(relation_dict):
    """
    从异构图转化为同构图, 不考虑关系.
    Returns:
    """
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    adj_mat_list = []
    adj_i_r_list = []
    print("Begin to build sparse relation matrix ...")
    total_idx_list = []
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0: #交互信息从非对称转化为对称
            cf = np_mat.copy()
            cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
            total_idx_list.append(cf)

            inv_cf = cf.copy()
            inv_cf[:, 0] = cf[:, 1]
            inv_cf[:, 1] = cf[:, 0]
            total_idx_list.append(inv_cf)
        else: # KG之前处理为对称了.
            cf = np_mat.copy()
            cf[:, 0] = cf[:, 0] + n_users
            cf[:, 1] = cf[:, 1] + n_users
            total_idx_list.append(cf)

    total_cf = np.concatenate(total_idx_list, axis=0) #(bs, 2)
    vals = [1.] * len(total_cf)
    adj_mat = sp.coo_matrix(
        (vals, (total_cf[:, 0], total_cf[:, 1])),
        shape=(n_nodes, n_nodes))
    norm_adj_mat = _bi_norm_lap(adj_mat)
    return norm_adj_mat


def conductMetaPathMatrix(train_data, triplets):
    i2e_relation_num = (n_relations - 1) // 2  # -1 for the interactive relations.

    meta_path_num = 4
    u2i_idx_edges, i2u_idx_edges = [], []
    e2i_idx_edges_list = defaultdict(list)  # 60*2
    i2e_idx_edges_list = defaultdict(list)  # 60*2
    e2i_idx_edges_list_np = defaultdict(list)  # 60*2
    i2e_idx_edges_list_np = defaultdict(list)  # 60*2

    ckg_graph_list = [[] for _ in range(i2e_relation_num * meta_path_num + 3)]  # 60*4+2

    # head_r_tail = defaultdict(dict)
    # rd = defaultdict(list)
    train_data = train_data.take([0, 1], axis=1)
    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        # rd[0].append([u_id, i_id])
        i_id = i_id + n_users  # support for the symmetric interactive graph.
        u2i_idx_edges.append([u_id, i_id])
        i2u_idx_edges.append([i_id, u_id])

    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):  # r_id: [1, 60]
        h_id = h_id + n_users  # support for the hetegenorous graph.
        t_id = t_id + n_users  # support for the hetegenorous graph.
        i2e_idx_edges_list[r_id].append([h_id, t_id])
        e2i_idx_edges_list[r_id].append([t_id, h_id])

    # convert matrix
    u2i_idx_edges_np = np.array(u2i_idx_edges)
    i2u_idx_edges_np = np.array(i2u_idx_edges)
    ui_ = sp.coo_matrix((np.ones(u2i_idx_edges_np.shape[0]), (u2i_idx_edges_np[:, 0], u2i_idx_edges_np[:, 1])), shape=(n_users, n_entities)).toarray()
    iu_ = sp.coo_matrix((np.ones(i2u_idx_edges_np.shape[0]), (i2u_idx_edges_np[:, 0], i2u_idx_edges_np[:, 1])), shape=(n_entities, n_users)).toarray()

    for r_idx, data in i2e_idx_edges_list.items():
        data = np.array(data)
        e2i_idx_edges_list_np[r_idx] = sp.coo_matrix((np.ones(data.shape[0]), (data[:, 0], data[:, 1])), shape=(n_entities, n_entities)).toarray()
        data = np.array(e2i_idx_edges_list[r_idx])
        i2e_idx_edges_list_np[r_idx] = sp.coo_matrix((np.ones(data.shape[0]), (data[:, 0], data[:, 1])), shape=(n_entities, n_entities)).toarray()

    # conduct
    user_metpath_idxes, item_metpath_idxes = [], []
    ckg_graph_list[0] = sp.coo_matrix(np.matmul(ui_, iu_) > 0) # uiu
    ckg_graph_list[1] = sp.coo_matrix(np.matmul(iu_, ui_) > 0) # iui
    # [u2i_idx_edges, i2u_idx_edges, u2i_idx_edges, i2u_idx_edges]  # uiuiu
    ckg_graph_list[-1] = sp.coo_matrix(np.matmul(np.matmul(np.matmul(ui_, iu_), ui_), iu_) > 0)

    user_metpath_idxes.extend([0, len(ckg_graph_list) - 1])
    item_metpath_idxes.append(1)

    for idx in range(1, i2e_relation_num + 1):  # 0 for the interactive relatioin.
        pos_idx = (idx - 1) * meta_path_num + 2
        # pdb.set_trace() #max(list(i2e_idx_edges_list.keys()))
        # iai
        ckg_graph_list[pos_idx] = sp.coo_matrix(np.matmul(i2e_idx_edges_list_np[idx], e2i_idx_edges_list_np[idx]) > 0)
        # uiaiu
        ckg_graph_list[pos_idx + 1] = sp.coo_matrix(np.matmul(np.matmul(np.matmul(ui_, i2e_idx_edges_list_np[idx]), e2i_idx_edges_list_np[idx]), iu_) > 0)
        # ckg_graph_list[pos_idx + 1] = [u2i_idx_edges, i2e_idx_edges_list[idx], e2i_idx_edges_list[idx], i2u_idx_edges])
        # iuiai
        ckg_graph_list[pos_idx + 2] = sp.coo_matrix(
            np.matmul(np.matmul(np.matmul(iu_, ui_), i2e_idx_edges_list_np[idx]), e2i_idx_edges_list_np[idx]) > 0)
        # ckg_graph_list[pos_idx + 2] = [i2u_idx_edges, u2i_idx_edges, i2e_idx_edges_list[idx], e2i_idx_edges_list[idx]]
        # iaiui
        ckg_graph_list[pos_idx + 3] = sp.coo_matrix(
            np.matmul(np.matmul(np.matmul(i2e_idx_edges_list_np[idx], e2i_idx_edges_list_np[idx]), iu_), ui_) > 0)
        # ckg_graph_list[pos_idx + 3] = [i2e_idx_edges_list[idx], e2i_idx_edges_list[idx], i2u_idx_edges, u2i_idx_edges]

        user_metpath_idxes.append(pos_idx + 1)
        item_metpath_idxes.extend([pos_idx, pos_idx + 2, pos_idx + 3])

    return ckg_graph_list, user_metpath_idxes, item_metpath_idxes

def load_data(model_args):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'

    print('reading train and test user-item set ...')
    # v1, contains train.txt and test.txt.
    # train_cf = read_cf(directory + 'train.txt')
    # eval_cf = read_cf(directory + 'test.txt')
    # v2, contains rating.txt. dd
    train_cf, eval_cf, ui_adj = read_cf_new()
    remap_item(train_cf, eval_cf)

    # pdb.set_trace()
    print('combinating train_cf and kg data ...')

    triplets = read_triplets(directory + 'kg_final.txt', inverse_r=False)
    inverse_triplets = read_triplets(directory + 'kg_final.txt', inverse_r=True)
    #relation counts
    # rela_d = count_relations(triplets)
    # print(sorted(rela_d.items(), key=lambda x: x[1]))
    # pdb.set_trace()

    graph, relation_dict = build_graph(train_cf, inverse_triplets) # not include user-item paris
    # graph, relation_dict = build_graph_v2(train_cf, inverse_triplets) # construct the heterogeneous graph that includes user-item pairs.

    # different meta-paths
    ckg_graph_list, user_metpath_idxes, item_metpath_idxes = build_graph_metapath_v2(train_cf, triplets) # list of metapath_steps (N) of idx_list ([:, 2]) # current best result.
    # ckg_graph_list, user_metpath_idxes, item_metpath_idxes = build_graph_metapath_v5(train_cf, triplets)
    # ckg_graph_list = build_graph_metapath_v3(train_cf, triplets)  # list of metapath_steps (N) of idx_list ([:, 2])
    # ckg_graph_list = build_graph_metapath_v4(train_cf, triplets)  # list of metapath_steps (N) of idx_list ([:, 2])

    metapath_steps = [len(item) for item in ckg_graph_list]
    # pdb.set_trace()

    print('building the adj mat ...')
    # adj_mat_list, norm_mat_list, mean_mat_list = build_sparse_relational_graph(relation_dict) # get an unsymmetric graph.
    adj_mat_list, norm_mat_list, mean_mat_list, _ = build_sparse_relational_graph_right_norm(relation_dict) # get an unsymmetric graph, but correct normalization.
    # adj_mat_list, norm_mat_list, mean_mat_list = build_sparse_relational_graph_v2(relation_dict) # get a symmetric interactive graph

    adj_norm_graph = conductHomoGraph(relation_dict)

    # ckg_graph_metapath_list, _, _ = conductMetaPathMatrix(train_cf, triplets)

    # # count relation distribution interacted by users
    # _, relation_dict_t = build_graph(train_cf, triplets)
    # adj_mat_list_t, _, _, adj_i_r_list_t = build_sparse_relational_graph_right_norm(relation_dict_t)
    # adj_u_i = adj_mat_list_t[0]
    # adj_i_r = adj_i_r_list_t[1]
    # for idx in range(2, len(adj_i_r_list_t)):
    #     adj_i_r += adj_i_r_list_t[idx]
    # def _convert_sp_mat_to_sp_tensor(X):
    #     coo = X.tocoo()
    #     i = torch.LongTensor([coo.row, coo.col])
    #     v = torch.from_numpy(coo.data).float()
    #     return torch.sparse.FloatTensor(i, v, coo.shape)
    # adj_u_e = torch.sparse.mm(_convert_sp_mat_to_sp_tensor(adj_u_i), _convert_sp_mat_to_sp_tensor(adj_i_r).to_dense()) # adj_u_i.toarray()[:n_users, n_users:], adj_i_r.toarray()[n_users:, :]
    # # adj_u_e_norm = torch.div(adj_u_e, adj_u_e.sum(1, keepdims=True))
    # # pdb.set_trace()
    # adj_u_e_norm = adj_u_e.numpy()
    # adj_u_e_norm = adj_u_e_norm[:n_users, 0:n_relations//2]
    #
    # max_count = adj_u_e_norm.max(axis=1)
    # adj_u_e_norm[adj_u_e_norm == 0] = float(np.inf)
    # min_count = adj_u_e_norm.min(axis=1)
    # ddd = [((max_count - min_count) > i).sum()/n_users for i in range(20)]


    # adj_scores = adj_u_e_norm.sum(0)
    # adj_count = (adj_u_e_norm > 0.).sum(0)
    # adj_w = adj_scores / adj_count
    # adj_w[np.isinf(adj_w)] = 0.
    # adj_w[np.isnan(adj_w)] = 0.

    #统计用户感知到不同关系的数量差.
    # plt.plot([i for i in range(20)], ddd)
    # plt.xlabel("Max different value of relation number")
    # plt.ylabel("user ratio")
    # # plt.hist(ddd)
    # plt.savefig("vis_yelp.png")


    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations)
    }
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }
    return train_cf, eval_cf, user_dict, n_params, graph, ckg_graph_list, metapath_steps, \
           [adj_mat_list, norm_mat_list, mean_mat_list], user_metpath_idxes, item_metpath_idxes

