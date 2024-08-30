'''
import torch
import torch.nn as nn
import torch.optim as optim
from gms_n import GMS

class ContinuousMaxSATLoss(nn.Module):
    def __init__(self):
        super(ContinuousMaxSATLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, problem, preds):
        # 确保 bool_array 是布尔类型
        bool_array = torch.zeros(problem.n_clauses, dtype=torch.bool, device=preds.device)
        initial_loss = torch.tensor(problem.n_clauses, dtype=torch.float, requires_grad=True, device=preds.device)  # 初始化损失值

        # 获取文字和子句的索引
        lits = problem.L_unpack_indices[:, 0].to(preds.device)
        clauses = problem.L_unpack_indices[:, 1].to(preds.device)

        # 打印调试信息
        print(f"lits: {lits}")
        print(f"preds size: {preds.size()}")
        print(f"problem.n_vars: {problem.n_vars}")
        print(f"max lit: {torch.max(lits)}")
        print(f"max clause: {torch.max(clauses)}")
        
        # 检查索引是否越界
        assert torch.all(lits < preds.size(0)), f"Index out of bounds in `lits`: {lits[lits >= preds.size(0)]}"
        assert torch.all(clauses < problem.n_clauses), f"Index out of bounds in `clauses`: {clauses[clauses >= problem.n_clauses]}"

        # 计算子句是否满足
        try:
            satisfied = ((lits < problem.n_vars) & (preds[lits] >= 0.5)) | ((lits >= problem.n_vars) & (preds[lits - problem.n_vars] < 0.5))
        except RuntimeError as e:
            print(f"Error while calculating satisfied: {e}")
            print(f"lits: {lits}")
            print(f"preds size: {preds.size()}")
            raise

        # 更新 bool_array
        bool_array = torch.index_add(bool_array.float(), 0, clauses, satisfied.float()).bool()

        # 计算损失
        loss = initial_loss - bool_array.sum().float()

        return loss

# 示例的 prob 对象，包含子句
class Prob:
    def __init__(self, n_vars, clauses, L_unpack_indices):
        self.n_vars = n_vars
        self.clauses = clauses
        self.n_clauses = len(clauses)  # 添加 n_clauses 属性
        self.L_unpack_indices = torch.tensor(L_unpack_indices).cuda()

# 创建一个 prob 实例，假设有若干子句，每个子句包含变量和符号
prob = Prob(n_vars=3, clauses=[[(0, 1), (1, -1)], [(1, 1), (2, -1)], [(0, -1), (2, 1)]],
            L_unpack_indices=[[0, 0], [1, 0], [1, 1], [2, 1], [0, 2], [2, 2]])

# 假设 args 已经定义
class Args:
    def __init__(self):
        self.dim = 128
        self.n_rounds = 10
        self.learning_rate = 1e-4

args = Args()

# 创建模型实例
model = GMS(args).cuda()

# 定义优化器
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

# 定义余弦退火学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# 创建自定义连续近似的无监督损失函数实例
loss_fn = ContinuousMaxSATLoss()

# 假设我们有一个数据加载器
train_loader = [torch.randn(10, 10) for _ in range(100)]  # 示例训练数据加载器

# 训练循环
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # 启用训练模式
    for data in train_loader:
        optimizer.zero_grad()  # 清零梯度
        preds = model(prob)  # 前向传播
        
        # 使用自定义连续近似的无监督损失函数计算损失
        loss = loss_fn(problem=prob, preds=preds)
        
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
    
    # 更新学习率
    scheduler.step()
    
    # 打印当前学习率和损失
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}/{num_epochs}, Learning Rate: {current_lr}, Loss: {loss.item()}")
'''

'''
import torch
import torch.nn as nn

# 创建一个 LSTM 实例
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)

# 输入序列 (batch_size, seq_len, input_size)
input_seq = torch.randn(5, 3, 10)  # (batch_size=5, seq_len=3, input_size=10)

# 前向传播
output, (hn, cn) = lstm(input_seq)

print(output.shape)  # 输出的形状
print(hn.shape)      # 最后一层隐藏状态的形状
print(cn.shape)      # 最后一层单元状态的形状
'''

import torch
print(torch.cuda.is_available())

