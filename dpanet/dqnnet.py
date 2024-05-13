import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
# --------Attention------------------
    
class Simplified_SelfSupperAttention(nn.Module):
    def __init__(self,K,D,n_v,d_out):
        '''
        :param K: K组
        :param D: 输入维度 (D = K * d_k)
        :param n_v: 查询序列长度
        :param d_out: 输出维度
        '''
        super(Simplified_SelfSupperAttention, self).__init__()
        self.K = K
        self.D = D
        self.n_v = n_v
        self.d_out = d_out
        self.d_k = self.D // self.K
        self.Wq = nn.Parameter(torch.empty(self.D,self.D))
        self.Wa = nn.Parameter(torch.empty(self.n_v,self.n_v))
        self.fo = nn.Linear(K * self.d_k, d_out, bias=False)
        self.init_weight()
    def init_weight(self):
        A = torch.randn(self.Wq.shape)
        U, _, V = torch.svd(A, some=True)
        if U.shape[1] < V.shape[0]:
            U = torch.cat((U,torch.zeros(U.shape[0],V.shape[0]-U.shape[1])),dim=1)
        elif V.shape[0] < U.shape[1]:
            V = torch.cat((V,torch.zeros(U.shape[1]-V.shape[0],V.shape[1])),dim=0)
        Wq_init = U @ V.t()
        self.Wq.data = Wq_init
        #对 fo 进行初始化
        init.xavier_uniform_(self.fo.weight)
        self.Wa.data = torch.randn(self.n_v,self.n_v)+torch.eye(self.n_v)
    def forward(self,X):
        '''
        :param X (...,n_v,D)
        '''
        other_shape = X.shape[:-2]
        Qtmp = torch.einsum('...qd,df->...qf', X, self.Wq) #-> (...,n_v,D)

        # (...,n_v,D) -> (...,n_v,K,d_k) -> (...,K,n_v,d_k)
        target_shape = other_shape + (self.n_v,self.K,self.d_k)
        Qtmp = Qtmp.view(*target_shape).transpose(-3,-2) #-> (...,K,n_v,d_k)

        
        Ktmp = X.view(*target_shape).transpose(-3,-2) #-> (...,K,n_v,d_k) -> (...,K,d_k,n_v)
        # 计算注意力得分 
        attention = torch.einsum('...qf,...fk->...qk', Qtmp, Ktmp.transpose(-2,-1)) / np.sqrt(self.d_k)
        attention = torch.softmax(attention, dim=-1)
        
        Vtmp = torch.einsum('ji,...id->...jd',self.Wa,X) #-> (...,n_v,D)
        #-> (...,n_v,D) -> (...,n_v,K,d_k) -> (...,K,n_v,d_k)
        Vtmp = Vtmp.view(*target_shape).transpose(-3,-2)
        # 应用注意力得分到value上
        O = torch.einsum('...qk,...kd->...qd', attention, Vtmp) #-> (...,K,n_v,d_k)
        # 合并K组结果
        target_shape = other_shape + (self.n_v,self.K*self.d_k)
        O = O.transpose(-3,-2).contiguous().view(*target_shape) #-> (...,n_v,K*d_k)
        # 通过输出全连接层
        O = self.fo(O) #-> (...,n_v,d_out)
        return O

class SuperAttention_Mutihead(nn.Module):
    def __init__(self,d_q,d_k,d_v,n_v,d_out,n_heads):
        super(SuperAttention_Mutihead, self).__init__()
        self.n_heads = n_heads
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.n_v = n_v
        self.d_out = d_out
        self.Wq = nn.Parameter(torch.empty(n_heads,d_q,d_k))
        self.Wa = nn.Parameter(torch.empty(n_heads,n_v,n_v))
        self.fo = nn.Linear(n_heads*d_v,d_out,bias=False)
        self.init_weight()
    def init_weight(self):
        #对 Wq 进行初始化，确保Wq的每一个head的秩为min(d_q,d_k)
        for i in range(self.n_heads):
            A = torch.randn(self.Wq[i].shape)
            U, _, V = torch.svd(A, some=True)
            if U.shape[1] < V.shape[0]:
                U = torch.cat((U,torch.zeros(U.shape[0],V.shape[0]-U.shape[1])),dim=1)
            elif V.shape[0] < U.shape[1]:
                V = torch.cat((V,torch.zeros(U.shape[1]-V.shape[0],V.shape[1])),dim=0)
            Wq_init = U @ V.t()
            self.Wq.data[i] = Wq_init
        #对 fo 进行初始化
        init.xavier_uniform_(self.fo.weight)

        # n_heads个 eyes(n_v) + noise
        self.Wa.data = torch.randn(self.n_heads,self.n_v,self.n_v)+torch.eye(self.n_v)
    def forward(self,Q,K,V):
        '''
        :param Q: (batch_size, n_q, d_q)
        :param K: (batch_size, n_k, d_k)
        :param V: (batch_size, n_k, d_v)
        
        '''
        bs,n_q = Q.shape[:2]
        nhQ = Q.unsqueeze(1).repeat(1,self.n_heads,1,1)
        nhK = K.unsqueeze(1).repeat(1,self.n_heads,1,1)
        nhV = V.unsqueeze(1).repeat(1,self.n_heads,1,1)
        S = torch.einsum("bhqi,hij,bhkj->bhqk",nhQ,self.Wq,nhK)/np.sqrt(self.d_k)
        Atten = torch.softmax(S,dim=-1)

        # V ->repeat-> (bs,nh,nk,dv) 
        # V' = AV
        V = torch.einsum("bhik,hij->bhjk",nhV,self.Wa)
        V = Atten @ V
        # O = SV
        V = V.permute(0,2,1,3).contiguous().view(bs,n_q,self.n_heads*self.d_v)
        O = self.fo(V)
        return O
# ----------------------------------



class MLP(nn.Module):
    def __init__(self,d_in,d_hidden,d_out,fc1bias=True,fc2bias=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_in,d_hidden,bias=fc1bias)
        self.fc2 = nn.Linear(d_hidden,d_out,bias=fc2bias)
        self.gelu = nn.GELU()
        self.init_weight()
    def init_weight(self):
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
    def forward(self,X):
        '''
        :param X: (bs,d_in)
        '''
        return self.fc2(self.gelu(self.fc1(X)))


class StateEncoder(nn.Module):
    def __init__(self,d_f,d_hidden,n_select,phase_map,K=4,hiddenexpand=2):
        super(StateEncoder, self).__init__()
        self.d_f = d_f
        self.d_hidden = d_hidden
        # self.d_actionemb = 8
        # self.cur_phase_embedding = nn.Embedding(2, self.d_actionemb)
        self.fc1 = nn.Linear(d_f,hiddenexpand*d_hidden) #(bs,n_lane,d_f) -> (bs,n_lane,d_hidden)
        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU()
        self.fc2 = nn.Linear(hiddenexpand*d_hidden,d_hidden) #(bs,n_lane,d_f) -> (bs,n_lane,d_hidden)
        self.phase_map:torch.Tensor = phase_map #(n_phase,n_select,n_lane)
        # self.laneMHA = Simplified_SelfSupperAttention(4,d_hidden,n_lane,d_hidden)
        self.onephaseMHA = Simplified_SelfSupperAttention(4,d_hidden,n_select,d_hidden)
        self.init_weight()
    def init_weight(self):
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
    def forward(self,X):
        X = self.sigmoid(self.fc2(self.leakyrelu(self.fc1(X))))
        H = torch.einsum('psl,blf->bpsf',self.phase_map,X) #(bs,n_phase,n_select,d_hidden)
        H = self.onephaseMHA(H) #(bs,n_phase,n_select,d_hidden) -> (bs,n_phase,n_select,d_hidden)
        return H
    

class DeulingQnetModule(nn.Module):
    def __init__(self,d_acthidden,n_action):
        super(DeulingQnetModule,self).__init__()
        '''
        (bs,n_action,d_acthidden) -> (bs,n_action) Qvalue
        '''

        self.d_acthidden = d_acthidden
        self.n_action = n_action
        # self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.fc1_adv = nn.Linear(d_acthidden,d_acthidden)
        self.fc2_adv = nn.Linear(d_acthidden,1) # Q adv

        self.fc1_val = nn.Linear(d_acthidden,d_acthidden)
        self.fc2_val = nn.Linear(d_acthidden*n_action,1) # Q value

        self.init_weight()
    def init_weight(self):
        init.xavier_uniform_(self.fc1_adv.weight)
        init.xavier_uniform_(self.fc2_adv.weight)
        init.xavier_uniform_(self.fc1_val.weight)
        init.xavier_uniform_(self.fc2_val.weight)
    def forward(self,X):
        '''
        :param X: (bs,n_action,d_acthidden)
        '''
        bs = X.shape[0]
        adv = self.fc2_adv(self.gelu(self.fc1_adv(X))) #(bs,n_action,1)
        adv = adv.view(bs,self.n_action) #(bs,n_action)
        adv = adv - adv.mean(dim=-1,keepdim=True) #(bs,n_action)
        val = self.fc2_val(
                self.gelu(self.fc1_val(X)).view(bs,-1)
            ) #(bs,1)
        return val + adv,val,adv

class GeneralEncoder(nn.Module):
    '''
    (bs,n_lane,d_f),(bs,n_phase) -> (bs,n_phase,d_actionhidden)
    '''
    def __init__(self,d_f,d_hidden,d_actionhidden,n_action,n_select,phase_map,d_actionemb=8,K=4,hiddenexpand=3):
        super(GeneralEncoder, self).__init__()
        self.d_f = d_f
        self.d_hidden = d_hidden
        self.d_actionhidden = d_actionhidden
        self.d_actionemb = d_actionemb
        self.n_action = n_action
        self.n_select = n_select
        self.K = K
        self.hiddenexpand = hiddenexpand
        self.cur_phase_embedding = nn.Embedding(2, self.d_actionemb)
        self.cur_phasev_mlp = MLP(1,1+8,self.d_actionemb,fc2bias=False)
        self.stateenc = StateEncoder(d_f,d_hidden,n_select,phase_map,K=self.K,hiddenexpand=self.hiddenexpand)
        self.d_fusion = d_hidden+self.d_actionemb*2
        self.mutiphaseMHA = SuperAttention_Mutihead(self.d_fusion,self.d_fusion,self.d_fusion,n_action,d_actionhidden,self.K)
        self.paramgg = nn.Parameter(torch.Tensor(2))
        self.init_weight()
    def init_weight(self):
        init.uniform_(self.paramgg,a=0.5,b=0.6) #初始化参数 让两个参数相差不大
    def forward(self,X,cur_phase,cur_phasev):
        '''
        :param X: (bs,n_lane,d_f)
        :param cur_phase: (bs,n_action)
        :param cur_phasev: (bs,n_action)
        '''
        bs = X.shape[0]
        cur_phase_emb = self.cur_phase_embedding(cur_phase) #(bs,n_action,d_actionemb)
        cur_phasev = cur_phasev.unsqueeze(-1)
        cur_phasev = self.cur_phasev_mlp(cur_phasev) #(bs,n_action,d_actionemb)


        H = self.stateenc(X)
        H_skip = H.mean(dim=-2,keepdim=True).squeeze(-2)
        H = torch.cat((H_skip,cur_phase_emb,cur_phasev),dim=-1)
        H = self.mutiphaseMHA(H,H,H)
        p_attention = torch.softmax(self.paramgg, dim=0)
        return H * p_attention[0] + H_skip * p_attention[1]


class ParamActor(nn.Module):
    def __init__(self,d_f,d_hidden,d_actionhidden,n_action,n_select,phase_map,d_actionemb=8,actionmin=0.2,actionmax=1):
        super(ParamActor, self).__init__()
        self.d_f = d_f
        self.d_hidden = d_hidden
        self.d_actionhidden = d_actionhidden
        self.n_action = n_action
        self.d_actionemb = d_actionemb
        self.encoder = GeneralEncoder(d_f,d_hidden,d_actionhidden,n_action,n_select,phase_map,self.d_actionemb)
        # self.actdecoder = DeulingQnetModule(d_actionhidden,n_action)
        # self.param_mlp1 = MLP(d_actionhidden,d_actionhidden+16,d_actionhidden,fc2bias=False)
        self.param_mlp2 = MLP(d_actionhidden,d_actionhidden+8,1,fc2bias=False)
        self.sigmoid = nn.Sigmoid()
        self.actionRange = [actionmin,actionmax]
        self.actionRangeV = actionmax-actionmin
    def forward(self,X,cur_phase,cur_phasev):
        '''
        :param X: (bs,n_lane,d_f)
        :param cur_phase: (bs,n_action)
        :param cur_phasev: (bs,n_action)
        '''
        H_skip = self.encoder(X,cur_phase,cur_phasev) #(bs,n_action,d_actionhidden)
        actionv = self.param_mlp2(H_skip) #(bs,n_action,1)
        actionv = actionv.squeeze(-1) #(bs,n_action)
        actionv = self.sigmoid(actionv) * self.actionRangeV + self.actionRange[0]
        return actionv
        


class Qnet(nn.Module):
    def __init__(self,d_f,d_hidden,d_actionhidden,d_vtrans,d_vhidden,n_action,n_select,phase_map,d_actionemb=8,K=4):
        super(Qnet, self).__init__()
        self.d_f = d_f
        self.d_hidden = d_hidden
        self.d_actionhidden = d_actionhidden
        self.n_action = n_action
        self.d_actionemb = d_actionemb
        self.d_vtrans = d_vtrans
        self.d_vhidden = d_vhidden
        self.d_v = n_action
        self.K = K
        self.relu = nn.ReLU()
        self.encoder = GeneralEncoder(d_f,d_hidden,d_actionhidden,n_action,n_select,phase_map,self.d_actionemb)
        self.actionmlp = MLP(1,d_actionemb,d_actionemb,fc2bias=False)
        self.actionmlp2 = MLP(1,d_actionemb,d_actionemb,fc2bias=False)
        self.d_fusion = d_actionhidden+self.d_actionemb
        self.MHA1 = SuperAttention_Mutihead(self.d_fusion,self.d_fusion,self.d_fusion,n_action,d_actionhidden,2)
        self.MH2 = Simplified_SelfSupperAttention(self.K,self.d_actionhidden,n_action,d_vtrans)
        self.d_flat = n_action*d_vtrans
        self.mlpout1 = MLP(self.d_flat,self.d_flat+8,d_vhidden,fc2bias=False)
        self.mlpout2 = MLP(d_vhidden,d_vhidden+8,n_action,fc2bias=False)
        self.paramgg = nn.Parameter(torch.Tensor(2))
        self.init_weight()
    def init_weight(self):
        init.uniform_(self.paramgg,a=0.5,b=0.6) #初始化参数 让两个参数相差不大  


    def forward(self,X,cur_phase,cur_phasev,actionv):
        '''
        :param X: (bs,n_lane,d_f)
        :param cur_phase: (bs,n_action) stdonehot
        :param cur_phasev: (bs,n_action)
        :param actionv: (bs,n_action)

        '''
        bs,_ = X.shape[:2]
        H_skip = self.encoder(X,cur_phase,cur_phasev) #(bs,n_action,d_actionhidden)
        actionv = actionv.unsqueeze(-1) #(bs,n_action,1)
        newactionH = self.actionmlp(actionv)
        H = torch.cat((H_skip,newactionH),dim=-1) #(bs,n_action,d_fusion)
        H = self.MHA1(H,H,H) #(bs,n_action,d_actionhidden)
        p_attention = torch.softmax(self.paramgg, dim=0)
        H = H * p_attention[0] + H_skip * p_attention[1] #确保相加信息归一化
        H = self.MH2(H) #(bs,n_action,d_vtrans)
        H = H.view(bs,-1)
        H = self.mlpout1(H)
        V = self.mlpout2(H) #(bs,1)
        return -self.relu(V)














    