from DatasetBuilder import NB_rldata
import os
import numpy as np
import torch

# import torch
# 
# 假设action是一个(batch_size, 4)的张量
action = torch.tensor([[0.5429, 0.0000, 0.0000, 0.0000],
                       [0.0000, 0.1234, 0.0000, 0.0000]])

# 找到每个向量中的最大值及其索引
values, indices = torch.max(action, dim=1)

# 创建一个与action形状相同的零张量作为one-hot基础
action1 = torch.zeros_like(action)

# 使用indices在正确的位置上放置1以创建one-hot向量
action1.scatter_(1, indices.unsqueeze(1), 1.0)

# 将values变形为(batch_size, 1)以匹配action2的形状
action2 = values.unsqueeze(1)

print("action1 (one-hot):", action1)
print("action2 (max values):", action2)



import numpy as np

def split_data(S, Sa, A, reward, window, predn):
    """
    Splits the data into sequences of length 'window' with S' being offset by 'predn' units from S.
    
    Parameters:
    - S: State array
    - Sa: Additional state information array
    - A: Action array
    - reward: Reward array
    - window: Length of the window for both S and S'
    - predn: The number of units S' is offset from S
    
    Returns:
    Tuple containing the split data for S, Sa, A, S', reward, and done signal.
    """
    # Adjust arrays to ensure they can be split correctly
    St = S[:-predn,...]
    Sat = Sa[:-predn,...]
    At = A[:-predn,...]
    Stn = S[predn:predn + window,...]
    Rtn = reward[predn:predn + window,...]
    
    # Initialize lists to store final sequences
    St_final, Sat_final, At_final, Stn_final, Rtn_final, done_column_final = [],[],[],[],[],[]
    
    # Calculate the maximum index for iteration based on the window and predn values
    max_index = len(St) - window + 1
    
    for start_idx in range(max_index):
        end_idx = start_idx + window
        St_final.append(St[start_idx:end_idx])
        Sat_final.append(Sat[start_idx:end_idx])
        At_final.append(At[start_idx:end_idx])
        Stn_final.append(Stn[start_idx:end_idx])
        Rtn_final.append(Rtn[start_idx:end_idx])
        
        # Create done column for each sequence
        done_column = np.zeros((window, 1), dtype=np.float32)
        if end_idx >= len(S) - predn:
            done_column[-1, 0] = 1  # Mark the last state as done
        done_column_final.append(done_column)
    
    return St_final, Sat_final, At_final, Stn_final, Rtn_final, done_column_final


# 创建示例数据
S = np.random.rand(100, 4)  # 假设有100个样本，每个样本4个特征
Sa = np.random.rand(100, 2)  # 额外的状态信息
A = np.random.rand(100, 3)  # 动作数组
reward = np.random.rand(100, 1)  # 奖励数组

# 设置窗口大小和预测偏移
window = 3
predn = 2

# 调用函数进行数据切分
St_final, Sat_final, At_final, Stn_final, Rtn_final, done_column_final = split_data(S, Sa, A, reward, window, predn)

# 打印结果的一部分进行检查
print(f"St_final shape: {St_final}")
print(f"Stn_final shape: {Stn_final}")
print(f"Done column sample: {done_column_final[0]}")