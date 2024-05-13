import torch.nn as nn
import torch.nn.functional as F
import torch
from dqnnet import ParamActor,Qnet
import copy
import os
import numpy as np
import pickle
def decorrelation_loss(A, B):
    """
    A: Input tensor of shape (bs, 4) with one-hot encoding.
    B: Input tensor of shape (bs, 1) with scalar values.
    """
    # 确保B是浮点数，以便进行后续计算
    # 计算加权平均值

    #B (bs,1) ->repeat ->  (bs,4)
    B = B.repeat(1, 4)

    # 使用A作为权重，与B点乘得到加权的B值
    weighted_BfromA = A * B # (bs, 4)
    # 按照batch维度求和，得到每个类别的B值之和
    sum_B_per_class = weighted_BfromA.sum(dim=0)
    # 求出每个类别的加权平均值
    weight_per_class = A.sum(dim=0)
    # 避免除0错误，使用clamp将权重限制在非零的最小值上
    weight_per_class_clamped = weight_per_class+1 # (4,)
    WeightAvg_A = sum_B_per_class / weight_per_class_clamped # (4,)
    

    # (bs, 4) -> (4,)
    Var_BfromA = weighted_BfromA.var(dim=0) # 计算每个类别的B值的方差 (4,)
    loss = WeightAvg_A.var()+Var_BfromA.var()
    # loss = WeightAvg_A.var()
    
    return loss

def clipandnorm_reward(reward, min_reward=-105, max_reward=0):
    """
    将奖励裁剪到指定的范围。

    参数:
    - reward: 原始奖励值，可以是一个数值或一个PyTorch张量。
    - min_reward: 原始奖励的最小值。
    - max_reward: 原始奖励的最大值。

    返回:
    - 裁剪后的奖励值。
    """
    # 首先判断reward是否小于min_reward
    clipped_reward = torch.where(reward < min_reward, torch.tensor(min_reward), reward)
    
    # 将原始奖励标准化到[0, 1]
    normalized_reward = (clipped_reward - min_reward) / (max_reward - min_reward)
    
    return normalized_reward-1

def add_noise_to_onehot(onehot_vectors, noise_level=0.06):
    """
    在One-Hot向量上添加噪声。

    参数:
    - onehot_vectors: 一个shape为(bs, 4)的Tensor，其中包含One-Hot向量。
    - noise_level: 噪声水平，表示添加到向量的随机噪声的比例。

    返回:
    - 带有噪声的向量。
    """
    # 确保输入是float类型，以便进行噪声添加操作
    onehot_vectors = onehot_vectors.float()
    
    # 生成与onehot_vectors形状相同的随机噪声
    noise = torch.rand_like(onehot_vectors) * noise_level
    
    # 向One-Hot向量添加噪声
    noisy_vectors = onehot_vectors + noise
    
    # 重新标准化，确保每个向量的元素和为1
    noisy_vectors = noisy_vectors / noisy_vectors.sum(dim=1, keepdim=True)
    
    return noisy_vectors

def construct_noise_vector(vectors):
    """
    构造一个噪声向量，该向量从输入向量的非零元素中等概率随机采样。
    
    参数:
    vectors - 输入的向量，维度为 (bs, 4)。
    
    返回:
    噪声向量，维度也为 (bs, 4)。
    """
    bs, _ = vectors.shape  # 获取batch size和其他维度信息
    noise_vectors = torch.zeros_like(vectors,device=vectors.device)  # 初始化噪声向量为0
    non_zero_values = vectors[:][vectors[:] > 0]
    for i in range(bs):
        # 提取非零值
        # 从非零值中随机采样，替换整个向量
        sampled_values = non_zero_values[torch.randint(0, len(non_zero_values), (4,))]
        noise_vectors[i] = sampled_values
    
    return noise_vectors

def construct_noise_vector_idpbatch(vectors):
    """
    构造一个噪声向量，该向量从输入向量的非零元素中等概率随机采样。从自己所在的索引位置进行采样
    
    参数:
    vectors - 输入的向量，维度为 (bs, 4)。
    
    返回:
    噪声向量，维度也为 (bs, 4)。
    """
    bs, _ = vectors.shape  # 获取batch size和其他维度信息
    noise_vectors = torch.zeros_like(vectors,device=vectors.device)  # 初始化噪声向量为0
    
    """
    [
        [0, 0, 0.4, 0],
        [0, 2, 0, 0],
        [0, 0.7, 0, 0],
        [0, 0, 0, 0.5],
        [0.3, 0, 0, 0]
    ]
    ->
    [
        []
    ]
    """
    


class dqn_env:
    '''
    一个用于装载参数的daddpg的环境
    '''
    def __init__(self,config:dict):
        self.config = config
        self.device = config["device"]
        phase_map =[ #(4,2,12)
                [
                    [0, 0, 0,    0, 1, 0,    0, 0, 0,    0, 0, 0],
                    [0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 1, 0]
                ],
                [
                    [0, 1, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],
                    [0, 0, 0,    0, 0, 0,    0, 1, 0,    0, 0, 0]
                ],
                [
                    [0, 0, 0,    1, 0, 0,    0, 0, 0,    0, 0, 0],
                    [0, 0, 0,    0, 0, 0,    0, 0, 0,    1, 0, 0],
                ],
                [
                    [1, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],
                    [0, 0, 0,    0, 0, 0,    1, 0, 0,    0, 0, 0],
                ]
            ]
        self.phase_map = torch.tensor(phase_map).float().to(self.device)
        self.d_state = config.get("d_state",9)
        self.d_hidden = config.get("d_hidden",128)
        self.d_phasehidden = config.get("d_phasehidden",64)
        self.n_phase = config.get("n_phase",4)
        self.n_select = config.get("n_select",2)
        self.normal_factor = config.get("normal_factor",5.0)
        self.clip_gradss = config.get("clip_gradss",0)
        self.d_actionemb,self.mu,self.beta,self.tau = \
                                    config.get("d_actionemb",4),\
                                    config.get("mu",0),\
                                    config.get("beta",1),\
                                    config.get("tau",0.1)
        self.d_vtrans,\
        self.d_vhidden,\
        self.d_v =  config.get("d_vtrans",48),\
                    config.get("d_vhidden",64),\
                    config.get("d_v",1)
        self.clip_action_min = config.get("clip_action_min",0.18)
        self.clip_action_max = config.get("clip_action_max",1.0) 
        self.action_max = config.get("action_max",30)

        self.clip_grad_critic = config.get("clip_grad_critic",30)
        self.clip_grad_actor = config.get("clip_grad_actor",10)

        self.pactor = ParamActor(
            self.d_state,
            self.d_hidden,
            self.d_phasehidden,
            self.n_phase,
            self.n_select,
            self.phase_map,
            actionmin=self.clip_action_min,
            actionmax=self.clip_action_max
        ).to(self.device)
        self.pactor_target = copy.deepcopy(self.pactor).to(self.device)
        self.qnet = Qnet(
            self.d_state,
            self.d_hidden,
            self.d_phasehidden,
            self.d_vtrans,
            self.d_vhidden,
            self.n_phase,
            self.n_select,
            self.phase_map,
            self.d_actionemb,K=4
        ).to(self.device)
        self.qnet_target = copy.deepcopy(self.qnet).to(self.device)
        print("clip_action_min:",self.clip_action_min,"clip_action_max:",self.clip_action_max)
        print("action_max:",self.action_max)
        self.train0_gradarray = []
        self.train1_gradarray = []
        self.train2_gradarray = []
        self.train3_gradarray = []
        self.train_gradarray = []
    def gradarrayoutput(self,savepath):
        nparray = np.array(self.train_gradarray)
        nparray0 = np.array(self.train0_gradarray)
        nparray1 = np.array(self.train1_gradarray)
        nparray2 = np.array(self.train2_gradarray)
        nparray3 = np.array(self.train3_gradarray)
        self.train_gradarray = []
        self.train0_gradarray = []
        self.train1_gradarray = []
        self.train2_gradarray = []
        self.train3_gradarray = []
        with open(savepath,"wb") as f:
            pickle.dump([nparray,nparray0,nparray1,nparray2,nparray3],f)
            
    def _hook_grad(self,grad):
        self.train_gradarray.append(grad.cpu().numpy())
    def config_optimizer(self,learning_rate=1e-4,weight_decay=1e-4):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.pactor_optimizer = torch.optim.AdamW(self.pactor.parameters(),lr=self.learning_rate,weight_decay=self.weight_decay)
        self.qnet_optimizer = torch.optim.AdamW(self.qnet.parameters(),lr=self.learning_rate,weight_decay=self.weight_decay)
    def config_optimizer_idp(self,lr_actor=1e-4,lr_critic=1e-4,weight_decay=1e-4):
        # self.learning_rate = learning_rate
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay
        self.pactor_optimizer = torch.optim.AdamW(self.pactor.parameters(),lr=self.lr_actor,weight_decay=self.weight_decay)
        self.qnet_optimizer = torch.optim.AdamW(self.qnet.parameters(),lr=self.lr_critic,weight_decay=self.weight_decay)

    def train(self,BatchExperience,policy_freq:int,total_it:int,gamma=0.8,tau=0.3):
        '''
        {'Sc': 'float', 'Ac': 'float', 'A': 'float', 'Sn': 'float', 'An': 'float', 'Rn': 'float', 'done_column': 'float'}
        '''
        # states, actions, rewards, next_states, dones = BatchExperience["St"], BatchExperience["At"], BatchExperience["Rtn"], BatchExperience["Stn"], BatchExperience["done_column"]
        states,prevactions,actions, rewards, next_states,next_prevactions,dones = BatchExperience["Sc"],BatchExperience["Ac"],BatchExperience["A"], BatchExperience["Rn"], BatchExperience["Sn"],BatchExperience["An"],BatchExperience["done_column"]
        bs = rewards.shape[0]

        # -------action prepare --------
        values, srcindices = torch.max(actions, dim=1)
        action_parameters = torch.zeros_like(actions)
        action_parameters.scatter_(1, srcindices.unsqueeze(1), 1.0)
        stdaction= values.unsqueeze(1)
        randomnoise = construct_noise_vector(actions)
        # 在srcindices的位置上用0掩盖噪声：
        randomnoise.scatter_(1, srcindices.unsqueeze(1), 0) # (bs, 4)



        # -------prevaction prepare --------
        values, indices = torch.max(prevactions, dim=1)
        prevaction_parameters = torch.zeros_like(prevactions,dtype=torch.int32)
        prevaction_parameters.scatter_(1, indices.unsqueeze(1), 1)

        values, indices = torch.max(next_prevactions, dim=1)
        next_prevaction_parameters = torch.zeros_like(next_prevactions,dtype=torch.int32)
        next_prevaction_parameters.scatter_(1, indices.unsqueeze(1), 1)
        #-------------------------------

        target_v = None
        with torch.no_grad():
            action_param = self.pactor_target.forward(next_states,next_prevaction_parameters,next_prevactions)
            tmp_next_q = self.qnet_target.forward(next_states,next_prevaction_parameters,next_prevactions,action_param)
            # qonehot,actionv = self.pactor_target.forward(next_states,next_prevaction_parameters)
            target_v = rewards + gamma * (tmp_next_q.max(dim=-1,keepdim=True))[0]
        y_expected = target_v
        v = self.qnet.forward(states,prevaction_parameters,prevactions,actions+randomnoise).gather(1,srcindices.unsqueeze(1))
        qloss = F.mse_loss(v,target_v)
        self.qnet_optimizer.zero_grad()
        qloss.backward()
        if self.clip_grad_critic > 0:
            torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), self.clip_grad_critic)
        self.qnet_optimizer.step()
        if total_it % policy_freq == 0:
            new_actionparam = self.pactor.forward(states,prevaction_parameters,prevactions)
            new_actionparam.reg
            new_q_pred = self.qnet.forward(states,prevaction_parameters,prevactions,new_actionparam) #(bs,4)
            policy_loss = -new_q_pred.sum().mean() #(bs,4) -> (bs,1) -> mean
            self.pactor_optimizer.zero_grad()
            policy_loss.backward()
            if self.clip_grad_actor > 0:
                torch.nn.utils.clip_grad_norm_(self.pactor.parameters(), self.clip_grad_actor)
            self.pactor_optimizer.step()
            self.soft_update(self.pactor,self.pactor_target,tau)
            self.soft_update(self.qnet,self.qnet_target,tau)
    def train_without_update(self,BatchExperience,policy_freq:int,total_it:int,gamma=0.8):
        '''
        {'Sc': 'float', 'Ac': 'float', 'A': 'float', 'Sn': 'float', 'An': 'float', 'Rn': 'float', 'done_column': 'float'}
        '''
        # states, actions, rewards, next_states, dones = BatchExperience["St"], BatchExperience["At"], BatchExperience["Rtn"], BatchExperience["Stn"], BatchExperience["done_column"]
        states,prevactions,actions, rewards, next_states,next_prevactions,dones = BatchExperience["Sc"],BatchExperience["Ac"],BatchExperience["A"], BatchExperience["Rn"], BatchExperience["Sn"],BatchExperience["An"],BatchExperience["done_column"]
        bs = rewards.shape[0]

        # -------action prepare --------
        values, srcindices = torch.max(actions, dim=1)
        action_parameters = torch.zeros_like(actions)
        action_parameters.scatter_(1, srcindices.unsqueeze(1), 1.0)
        stdaction= values.unsqueeze(1)
        randomnoise = construct_noise_vector(actions)
        # 在srcindices的位置上用0掩盖噪声：
        randomnoise.scatter_(1, srcindices.unsqueeze(1), 0) # (bs, 4)



        # -------prevaction prepare --------
        values, indices = torch.max(prevactions, dim=1)
        prevaction_parameters = torch.zeros_like(prevactions,dtype=torch.int32)
        prevaction_parameters.scatter_(1, indices.unsqueeze(1), 1)

        values, indices = torch.max(next_prevactions, dim=1)
        next_prevaction_parameters = torch.zeros_like(next_prevactions,dtype=torch.int32)
        next_prevaction_parameters.scatter_(1, indices.unsqueeze(1), 1)
        #-------------------------------

        target_v = None
        with torch.no_grad():
            action_param = self.pactor_target.forward(next_states,next_prevaction_parameters,next_prevactions)
            tmp_next_q = self.qnet_target.forward(next_states,next_prevaction_parameters,next_prevactions,action_param)
            # qonehot,actionv = self.pactor_target.forward(next_states,next_prevaction_parameters)
            target_v = rewards + gamma * (tmp_next_q.max(dim=-1,keepdim=True))[0]
        y_expected = target_v
        v = self.qnet.forward(states,prevaction_parameters,prevactions,actions+randomnoise).gather(1,srcindices.unsqueeze(1))
        qloss = F.mse_loss(v,target_v)
        self.qnet_optimizer.zero_grad()
        qloss.backward()
        if self.clip_grad_critic > 0:
            torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), self.clip_grad_critic)
        self.qnet_optimizer.step()
        if total_it % policy_freq == 0:
            new_actionparam = self.pactor.forward(states,prevaction_parameters,prevactions)
            new_q_pred = self.qnet.forward(states,prevaction_parameters,prevactions,new_actionparam) #(bs,4)
            policy_loss = -new_q_pred.sum().mean() #(bs,4) -> (bs,1) -> mean
            self.pactor_optimizer.zero_grad()
            policy_loss.backward()
            if self.clip_grad_actor > 0:
                torch.nn.utils.clip_grad_norm_(self.pactor.parameters(), self.clip_grad_actor)
            self.pactor_optimizer.step()
    def test(self,BatchExperience_test,gamma):
        states,prevactions,actions, rewards, next_states,next_prevactions,dones = BatchExperience_test["Sc"],BatchExperience_test["Ac"],BatchExperience_test["A"], BatchExperience_test["Rn"], BatchExperience_test["Sn"],BatchExperience_test["An"],BatchExperience_test["done_column"]
        batchsize = rewards.shape[0]

        # -------action prepare --------
        values, srcindices = torch.max(actions, dim=1)
        action_parameters = torch.zeros_like(actions)
        action_parameters.scatter_(1, srcindices.unsqueeze(1), 1.0)
        stdaction= values.unsqueeze(1)
        # -------prevaction prepare --------
        values, indices = torch.max(prevactions, dim=1)
        prevaction_parameters = torch.zeros_like(prevactions,dtype=torch.int32)
        prevaction_parameters.scatter_(1, indices.unsqueeze(1), 1)
        values, indices = torch.max(next_prevactions, dim=1)
        next_prevaction_parameters = torch.zeros_like(next_prevactions,dtype=torch.int32)
        next_prevaction_parameters.scatter_(1, indices.unsqueeze(1), 1)
        #-------------------------------
        with torch.no_grad():
            action_param = self.pactor_target.forward(next_states,next_prevaction_parameters,next_prevactions)
            tmp_next_q = self.qnet_target.forward(next_states,next_prevaction_parameters,next_prevactions,action_param)
            target_v = rewards + gamma * (tmp_next_q.max(dim=-1,keepdim=True))[0]
        # y_expected = target_v
        v = self.qnet.forward(states,prevaction_parameters,prevactions,actions).gather(1,srcindices.unsqueeze(1))
        qloss = F.mse_loss(v,target_v)
        return qloss.item()
    def choose_action(self,data:np.ndarray,phasenumber:int):
        '''
        states (seq,node,feature) -> (1,seq,node,feature)
        '''
        states,prevactions = data
        torchstates = torch.from_numpy(states).float().to(self.device)
        torchactions = torch.from_numpy(prevactions).float().to(self.device)
        #(seq,node,feature) -> (1,seq,node,feature)
        torchstates = torchstates.unsqueeze(0)
        torchactions = torchactions.unsqueeze(0)
        values, srcindices = torch.max(torchactions, dim=1)
        action_parameters = torch.zeros_like(torchactions,dtype=torch.int32)
        action_parameters.scatter_(1, srcindices.unsqueeze(1), 1)
        actionselect = -1
        value = -1
        selectv = -1
        with torch.no_grad():
            action_param = self.pactor.forward(torchstates,action_parameters,torchactions)
            qvalue = self.qnet.forward(torchstates,action_parameters,torchactions,action_param)
            value,index = qvalue.max(dim=-1)  # actionselect (bs=1,1)
            selectv = action_param[0,index.item()].item()
            actionselect = index.item()

        return actionselect+1,int(round(selectv*self.action_max))
    @staticmethod
    def soft_update(local_model,target_model,tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    @staticmethod
    def hard_update(local_model,target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def save(self,save_path_dir:str):
        if not os.path.exists(save_path_dir):
            os.makedirs(save_path_dir)
        actormodel_path = os.path.join(save_path_dir,"actor.pth")
        criticmodel_path = os.path.join(save_path_dir,"critic.pth")
        actormodel_target_path = os.path.join(save_path_dir,"actor_target.pth")
        criticmodel_target_path = os.path.join(save_path_dir,"critic_target.pth")
        torch.save(self.pactor.state_dict(),actormodel_path)
        torch.save(self.qnet.state_dict(),criticmodel_path)
        torch.save(self.pactor_target.state_dict(),actormodel_target_path)
        torch.save(self.qnet_target.state_dict(),criticmodel_target_path)
        print("save success!")
    def load(self,load_path_dir:str):
        actormodel_path = os.path.join(load_path_dir,"actor.pth")
        criticmodel_path = os.path.join(load_path_dir,"critic.pth")
        actormodel_target_path = os.path.join(load_path_dir,"actor_target.pth")
        criticmodel_target_path = os.path.join(load_path_dir,"critic_target.pth")
        self.pactor.load_state_dict(torch.load(actormodel_path))
        self.qnet.load_state_dict(torch.load(criticmodel_path))
        self.pactor_target.load_state_dict(torch.load(actormodel_target_path))
        self.qnet_target.load_state_dict(torch.load(criticmodel_target_path))
        print("load success!")


if __name__ == '__main__':
    def test_for_daddpg_env():
        config = {
            "device": torch.device("cuda"),
            "v_out_dim": 1,
            "d_state": 28,
            "d_mixer_f": 60,
            "d_mixer_mid": 64,
            "n_seq": 4,
            "n_node": 5,
            "d_hidden": 128,
            "d_action": 1,
            "d_action_parameters": 4,
            "d_action_mid": 32,
            "d_action_parameters_mid": 64,
            "d_critic_transtart": 56,
            "d_vmid": 144,
            "d_v": 1,
            "clip_grad_critic": 5,
            "clip_grad_actor": 5,
            "mu": 0,
            "beta": 1,
            "tau_gumbel": 0.2,
            "batchsize": 64,
            "sigma_actionnoise": 0.07,
            "clip_action_min": 0.34 ,
            "clip_action_max": 1.0
        }
        env = daddpg_env(config)
        # env.config_optimizer(learning_rate=0.001,weight_decay=0.0001)
        print(env.actor)
        print(env.critic)
        print(env.actor_target)
        print(env.critic_target)
        from DatasetBuilder import NBintersection_rldata
        config2 = {
            "batch_size":32,
            "train_rate":0.9,
            "eval_rate":0.1,
            "rawdata_path":"dataoffline2/",
            "cache_file_path":"cache_data/anon_3_4_jinan_real_2500_cachenew.npz",
            "stateseq_window":4,
            "stateseq_predn":1,
            "a":1,
            "b":0.3
        }
        dataset = NBintersection_rldata(config2)
        train_dataloader,eval_dataloader = dataset.get_data()

        trainit = 0
        
        # traincritic_cit = 0
        # for traincritic_it in range(1):
        #     criticloss = []
        #     for batch in train_dataloader:
        #         batch.to_tensor(torch.device("cuda"))
        #         res = env.train_onlycritic(batch)
        #         criticloss.append(res)
        #         traincritic_cit += 1
        #         if traincritic_cit % 100==0:
        #             print("traincritic_cit:",traincritic_cit,"criticloss:",sum(criticloss)/len(criticloss))
        #             criticloss = []
        #         if traincritic_cit > 16000:
        #             break

            
        
        # env.soft_update(env.critic,env.critic_target,1)
        env.config_optimizer(learning_rate=0.0001,weight_decay=0.0001)   
        testendit = 50000
        for i in range(60):
            ct = i+1
            gamma = (0.2 if ct*0.01 > 0.2 else ct*0.01)
            # tau = 0.15 if 0.99-i*0.05 < 0.15 else 0.99-i*0.05
            tau = 0.1
            freq = 14 if 30-i*2 < 14 else 30-i*2
            control_param = 1
            print("gamma:",gamma,"tau:",tau)
            print("freq:",freq,"control_param:",control_param)
            for batch in train_dataloader:
                batch.to_tensor(torch.device("cuda"))
                env.train(batch,freq,trainit,gamma,tau,control_param=control_param)
                trainit += 1
                # print("trainit:",trainit)
            print("epoch:",i)
            testloss = []
            for batch in eval_dataloader:
                batch.to_tensor(torch.device("cuda"))
                testlossres = env.test(batch,gamma)
                testloss.append(testlossres)
            print("testloss:",sum(testloss)/len(testloss))

            import time
            currenttimestr = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            savepath = "model_save/daddpg/"+currenttimestr
            env.save(savepath)
    test_for_daddpg_env()