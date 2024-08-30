import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from config import parser

args = parser.parse_args()
if args.model == 'GMS_N':
    from gms_n import GMS
else:
    from gms_e import GMS

from train import evaluate
from data_maker_maxsat import *
from mk_problem import Problem

##### START #####
net = GMS(args)
net = net.cuda()

task_name = args.task_name + '_n' + str(args.n_vars) + '_ep' + str(args.epochs) + \
            '_nr' + str(args.n_rounds) + '_d' + str(args.dim)
log_file = open(os.path.join(args.log_dir, task_name + '.log'), 'a+')
detail_log_file = open(os.path.join(args.log_dir, task_name + '_detail.log'), 'a+')

test = None
with open(os.path.join(args.data_dir, 'test', args.test_file), 'rb') as f:
    test = pickle.load(f)

loss_fn = nn.BCELoss()
optim = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=1e-10)
sigmoid = nn.Sigmoid()

best_var_acc = 0.0
best_obj_diff = 999999.0
start_epoch = 0

if args.restore is not None:
    print('restore from', args.restore, flush=True)
    print('restore from', args.restore, file=log_file, flush=True)
    print('restore from', args.restore, file=detail_log_file, flush=True)
    model = torch.load(args.restore)
    start_epoch = model['epoch']
    best_var_acc = model['var_acc']
    best_obj_diff = model['obj_diff']
    net.load_state_dict(model['state_dict'])

test_bar = tqdm(test)
VTP, VTN, VFN, VFP = torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long(), torch.zeros(1).long()
obj_diff = 0.0
total_obj_diff = []
    
net.eval()

def GMS_Solve_cnf(cnf_file):
    solution = None
    header, iclauses = parse_dimacs(cnf_file)
    n_vars, n_clauses = int(header[2]), int(header[3])
    if solution is None:
        solution = [0] * n_vars  # 如果没有解信息，假设解是全零，具体情况根据需要调整
    objective = [0] * n_vars  # 假设目标函数是全零，具体情况根据需要调整
    n_cells_per_batch = [len(clause) for clause in iclauses]
    all_dimacs = ""  # 这里假设dimacs信息为空，具体根据需要填充
    prob = Problem(n_vars, iclauses, objective, solution, n_cells_per_batch, all_dimacs)
    #prob = torch.tensor(prob)
    #prob.to('cuda')
    net(prob)
    var_outputs = net(prob)
    var_outputs = sigmoid(var_outputs).view(-1)
    #var_target = torch.Tensor(sum(prob.solution, [])).cuda().float()  # 2d-list -> 1d-list
    var_preds = torch.where(var_outputs > 0.5, torch.ones(var_outputs.shape).cuda(),
                            torch.zeros(var_outputs.shape).cuda())
    print(f"len(var_preds): {len(var_preds)}")
    c = 0
    for x in var_preds:
        print(x)
        c = c + 1
        if c > len(prob.objective): break
    unsat_count = 0
    for iclause in iclauses:
        sat = 0
        for var in iclause:
            if var > 0 and var_preds[var - 1] == 1:
                sat = 1
                break
            if var < 0 and var_preds[-var - 1] == 0:
                sat = 1
                break
        if sat == 0:
            unsat_count += 1
    print(f"unsat_count: {unsat_count}")
GMS_Solve_cnf("./raw_data_uf/s2v60c600/s2v60c600-19997.cnf")

'''
for _, prob in enumerate(test_bar):
    optim.zero_grad()
    var_outputs = net(prob)
    var_outputs = sigmoid(var_outputs).view(-1)
    var_target = torch.Tensor(sum(prob.solution, [])).cuda().float()  # 2d-list -> 1d-list

    valid_var_outputs = []
    for index in range(len(prob.objective)):
        valid_var_outputs.append(var_outputs[int(index * var_outputs.shape[0] / len(prob.objective)):int(
            (index + 1) * var_outputs.shape[0] / len(prob.objective))])
    valid_var_outputs = torch.cat(valid_var_outputs, dim=0)

    var_loss = loss_fn(valid_var_outputs, var_target)
    tot_loss = var_loss
    desc = 'tot loss: %.4f, var loss: %.4f ' % (tot_loss.item(), var_loss.item())
    
    var_preds = torch.where(valid_var_outputs > 0.5, torch.ones(valid_var_outputs.shape).cuda(),
                            torch.zeros(valid_var_outputs.shape).cuda())
    cur_obj_diff = evaluate(prob, var_preds)
    total_obj_diff.extend(cur_obj_diff)
    obj_diff = np.mean(total_obj_diff)

    VTP += (var_preds.eq(1) & var_target.eq(1)).cpu().sum()
    VTN += (var_preds.eq(0) & var_target.eq(0)).cpu().sum()
    VFN += (var_preds.eq(0) & var_target.eq(1)).cpu().sum()
    VFP += (var_preds.eq(1) & var_target.eq(0)).cpu().sum()
    VTOT = VTP + VTN + VFN + VFP

    print(f"len(var_preds): {len(var_preds)}")
    c = 0
    for x in var_preds:
        print(x)
        c = c + 1
        if c > len(prob.objective): break
'''   