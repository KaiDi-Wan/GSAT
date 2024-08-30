import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mlp import MLP

class GATLayer(nn.Module):
    def __init__(self, input_dim, output_dim, heads=1):
        super(GATLayer, self).__init__()
        self.gat = nn.MultiheadAttention(embed_dim=output_dim, num_heads=heads)

    def forward(self, x):
        x = x.unsqueeze(0)  # Add batch dimension
        out, _ = self.gat(x, x, x)
        return out.squeeze(0)

# GMS-N: using Node-splitting CVIG
class GMS(nn.Module):
    def __init__(self, args):
        super(GMS, self).__init__()
        self.args = args

        self.heads = 4
        self.adj_list = {}
        self.dfn = None
        self.cnt = 0
        self.L_unpack_childindices = []
        self.L_unpack_faindices = []
        self.L_unpack_nontreeindices = []
        self.cnt_child = 0
        self.cnt_nontree = 0

        self.dim = args.dim

        self.init_ts = torch.ones(1)
        self.init_ts.requires_grad = False

        self.L_init = nn.Linear(1, args.dim)
        self.C_init = nn.Linear(1, args.dim)

        #self.L_msg = GATLayer(self.args.dim, self.args.dim, self.heads)
        self.L_msg_child = GATLayer(self.args.dim, self.args.dim, self.heads)
        self.L_msg_parent = GATLayer(self.args.dim, self.args.dim, self.heads)
        self.L_msg_non_tree = GATLayer(self.args.dim, self.args.dim, self.heads)

        #self.C_msg = GATLayer(self.args.dim, self.args.dim, self.heads)
        self.C_msg_child = GATLayer(self.args.dim, self.args.dim, self.heads)
        self.C_msg_parent = GATLayer(self.args.dim, self.args.dim, self.heads)
        self.C_msg_non_tree = GATLayer(self.args.dim, self.args.dim, self.heads)

        self.L_update = nn.LSTM(self.args.dim * 2, self.args.dim)
        self.C_update = nn.LSTM(self.args.dim, self.args.dim)

        self.var_vote = MLP(self.args.dim * 2, self.args.dim, 1)

    
    def traverse_graph(self, ts_L_unpack_indices, n_lits):
        # 获取 vlit 和 clause 的索引
        vlit_indices, clause_indices = ts_L_unpack_indices

        # 遍历所有的边
        for vlit, clause in zip(vlit_indices, clause_indices):
            vlit = vlit.item()
            clause = clause.item()

            # 更新邻接列表
            if vlit not in self.adj_list:
                self.adj_list[vlit] = []
            if clause not in self.adj_list:
                self.adj_list[clause + n_lits] = []

            self.adj_list[vlit].append(clause + n_lits)
            self.adj_list[clause + n_lits].append(vlit)
    
    def dfs(self, x, fa, n_lits):
        for y in self.adj_list[x]:
            p = min(x, y)
            q = max(x, y) - n_lits
            if self.dfn[y].item() == 0:
                self.cnt = self.cnt + 1
                self.dfn[y] = self.cnt
                self.L_unpack_childindices.append([p, q])
                self.cnt_child += 1 
                self.dfs(y, x, n_lits)
            elif y == fa: self.L_unpack_faindices.append([p, q])
            else: 
                self.L_unpack_nontreeindices.append([p, q])
                self.cnt_nontree += 1
                
                

    def forward(self, problem):
        n_vars = problem.n_vars
        n_lits = problem.n_lits
        n_clauses = problem.n_clauses
        n_probs = len(problem.objective)
        n_offsets = problem.iclauses_offset

        ts_L_unpack_childindices = torch.Tensor(self.L_unpack_childindices).t().long()  
        ts_L_unpack_faindices = torch.Tensor(self.L_unpack_faindices).t().long()  
        ts_L_unpack_nontreeindices = torch.Tensor(self.L_unpack_nontreedices).t().long()  
         
        ts_L_unpack_indices = torch.Tensor(problem.L_unpack_indices).t().long()        
        self.traverse_graph(ts_L_unpack_indices, problem.n_lits)
        self.dfn = torch.zeros(n_lits + n_clauses, dtype=torch.int)


        init_ts = self.init_ts.cuda()
        L_init = self.L_init(init_ts).view(1, 1, -1)
        L_init = L_init.repeat(1, n_lits, 1)
        C_init = self.C_init(init_ts).view(1, 1, -1)
        C_init = C_init.repeat(1, n_clauses, 1)

        L_state = (L_init, torch.zeros(1, n_lits, self.args.dim).cuda())
        C_state = (C_init, torch.zeros(1, n_clauses, self.args.dim).cuda())
        #L_unpack = torch.sparse_coo_tensor(ts_L_unpack_indices, torch.ones(problem.n_cells),
        #                                    torch.Size([n_lits, n_clauses])).cuda()
        
        L_unpack_child = torch.sparse_coo_tensor(ts_L_unpack_childindices, torch.ones(self.cnt_child),
                                            torch.Size([n_lits, n_clauses])).cuda()
        L_unpack_fa = torch.sparse_coo_tensor(ts_L_unpack_faindices, torch.ones(n_lits),
                                            torch.Size([n_lits, n_clauses])).cuda()
        L_unpack_nontree = torch.sparse_coo_tensor(ts_L_unpack_nontreeindices, torch.ones(self.cnt_nontree),
                                            torch.Size([n_lits, n_clauses])).cuda()

        for _ in range(self.args.n_rounds):
            L_hidden = L_state[0].squeeze(0)
            
            #LC_msg = torch.sparse.mm(L_unpack.t(), L_pre_msg)

            # 根据边的类型分别计算消息
            L_pre_msg_child = self.L_msg_child(L_hidden)
            L_pre_msg_parent = self.L_msg_parent(L_hidden)
            L_pre_msg_non_tree = self.L_msg_non_tree(L_hidden)

            LC_msg_child = torch.sparse.mm(L_unpack_child.t(), L_pre_msg_child)
            LC_msg_parent = torch.sparse.mm(L_unpack_fa.t(), L_pre_msg_parent)
            LC_msg_non_tree = torch.sparse.mm(L_unpack_nontree.t(), L_pre_msg_non_tree)
            
            LC_msg = LC_msg_child + LC_msg_parent + LC_msg_non_tree

            _, C_state = self.C_update(LC_msg.unsqueeze(0), C_state)
            
            C_hidden = C_state[0].squeeze(0)
            #C_pre_msg = self.C_msg(C_hidden)
            
            C_pre_msg_child = self.C_msg_child(C_hidden)
            C_pre_msg_parent = self.C_msg_parent(C_hidden)
            C_pre_msg_non_tree = self.C_msg_non_tree(C_hidden)
            
            CL_msg_child = torch.sparse.mm(L_unpack_child, C_pre_msg_child)
            CL_msg_parent = torch.sparse.mm(L_unpack_fa, C_pre_msg_parent)
            CL_msg_non_tree = torch.sparse.mm(L_unpack_nontree, C_pre_msg_non_tree)
            
            CL_msg = CL_msg_child + CL_msg_parent + CL_msg_non_tree
            _, L_state = self.L_update(torch.cat(
                [CL_msg, self.flip(L_state[0].squeeze(0), n_vars)], dim=1).unsqueeze(0), L_state)

        logits = L_state[0].squeeze(0)
        clauses = C_state[0].squeeze(0)

        x = torch.cat((logits[:n_vars, :], logits[n_vars:, :]), dim=1)
        var_vote = self.var_vote(x)

        return var_vote

    def flip(self, msg, n_vars):
        return torch.cat([msg[n_vars:2 * n_vars, :], msg[:n_vars, :]], dim=0)
