import pickle
import numpy as np
import os
import torch
from torch.utils.data import Dataset,DataLoader
import copy
from tools import ensure_dir

class Batch(object):

    def __init__(self, feature_name):
        """Summary of class here

        Args:
            feature_name (dict): key is the corresponding feature's name, and
                the value is the feature's data type
        """
        self.data = {}
        self.feature_name = feature_name
        for key in feature_name:
            self.data[key] = []

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        else:
            raise KeyError('{} is not in the batch'.format(key))

    def __setitem__(self, key, value):
        if key in self.data:
            self.data[key] = value
        else:
            raise KeyError('{} is not in the batch'.format(key))

    def append(self, item):
        """
        append a new item into the batch

        Args:
            item (list): 一组输入，跟feature_name的顺序一致，feature_name即是这一组输入的名字
        """
        if len(item) != len(self.feature_name):
            raise KeyError('when append a batch, item is not equal length with feature_name')
        for i, key in enumerate(self.feature_name):
            self.data[key].append(item[i])

    def to_tensor(self, device):
        """
        将数据self.data转移到device上

        Args:
            device(torch.device): GPU/CPU设备
        """
        for key in self.data:
            if self.feature_name[key] == 'int':
                self.data[key] = torch.LongTensor(np.array(self.data[key])).to(device)
            elif self.feature_name[key] == 'float':
                self.data[key] = torch.FloatTensor(np.array(self.data[key])).to(device)
            else:
                raise TypeError(
                    'Batch to_tensor, only support int, float but you give {}'.format(self.feature_name[key]))

    def to_ndarray(self):
        for key in self.data:
            if self.feature_name[key] == 'int':
                self.data[key] = np.array(self.data[key])
            elif self.feature_name[key] == 'float':
                self.data[key] = np.array(self.data[key])
            else:
                raise TypeError(
                    'Batch to_ndarray, only support int, float but you give {}'.format(self.feature_name[key]))
class ListDataset(Dataset):
    def __init__(self, data):
        """
        data: 必须是一个 list
        """
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def decode_stddata_intersection(data:np.ndarray,a:float,b:float):
    '''
    data:  (?,265)  np.ndarray
    a: self weight
    b: other weight
    '''
    nseq,d_f = data.shape
    Swait = data[:,:60]
    Sspeed = data[:,60:120]
    Slasta = data[:,120:140]
    #Swait.reshape(nseq,5,-1),Sspeed.reshape(nseq,5,-1),Slasta.reshape(nseq,5,-1)
    S = np.concatenate([Swait.reshape(nseq,5,-1),Sspeed.reshape(nseq,5,-1),Slasta.reshape(nseq,5,-1)],axis=-1) # (nseq,5,？)
    A = data[:,140:144]
    Snext_wait = data[:,144:204]
    Snext_speed = data[:,204:264]
    Snext = np.concatenate([Snext_wait.reshape(nseq,5,-1),Snext_speed.reshape(nseq,5,-1)],axis=-1) # (nseq,5,？)
    done = data[:,-1]
    # Snext_wait0 (?,1)
    Snext_wait0 = Snext_wait[:,0,:].sum(axis=-1).sum(axis=-1)
    Snext_waitOther = Snext_wait[:,1:,:].sum(axis=-1).sum(axis=-1)
    reward = a * Snext_wait0 + b * Snext_waitOther
    return S,A,reward,Snext,done

def decode_stddata_lane(data:np.ndarray,a:float=1,b:float=1):
    '''
    data:  (?,265)  np.ndarray
    a: self weight
    b: other weight
    '''
    nseq,d_f = data.shape
    Swait = data[:,:60]
    Sspeed = data[:,60:120]
    Slasta = data[:,120:140]
    #Swait.reshape(nseq,5,-1),Sspeed.reshape(nseq,5,-1),Slasta.reshape(nseq,5,-1)
    S = np.concatenate([Swait.view(nseq,60,-1),Sspeed.view(nseq,60,-1)],axis=-1) # (nseq,60,？)
    Sa = Slasta.view(nseq,5,-1)
    A = data[:,140:144]
    Snext_wait = data[:,144:204]
    Snext_speed = data[:,204:264]
    Snext = np.concatenate([Snext_wait.view(nseq,60,-1),Snext_speed.view(nseq,60,-1)],axis=-1) # (nseq,60,？)
    done = data[:,-1]

    # Snext_wait0 (?,1)
    Snext_wait0 = Snext_wait[:,0:12].sum(axis=-1)
    Snext_waitOther = Snext_wait[:,12:].sum(axis=-1)
    reward = a * Snext_wait0 + b * Snext_waitOther
    return S,Sa,A,reward,Snext,done

def decode_simdata_intersection(data:np.ndarray):
    '''
    data:  (?,265)  np.ndarray
    a: self weight
    b: other weight
    '''
    nseq,d_f = data.shape
    Swait = data[:,:60]
    Sspeed = data[:,60:120]
    Slasta = data[:,120:140]
    #Swait.reshape(nseq,5,-1),Sspeed.reshape(nseq,5,-1),Slasta.reshape(nseq,5,-1)
    S = np.concatenate([Swait.reshape(nseq,5,-1),Sspeed.reshape(nseq,5,-1),Slasta.reshape(nseq,5,-1)],axis=-1) # (nseq,5,？)
    A = data[:,140:144]
    return S,A

def decode_simdata_lane(data:np.ndarray):
    '''
    data:  (?,265)  np.ndarray
    a: self weight
    b: other weight
    '''
    nseq,d_f = data.shape
    Swait = data[:,:60]
    Sspeed = data[:,60:120]
    Slasta = data[:,120:140]
    #Swait.reshape(nseq,5,-1),Sspeed.reshape(nseq,5,-1),Slasta.reshape(nseq,5,-1)
    S = np.concatenate([Swait.reshape(nseq,60,-1),Sspeed.reshape(nseq,60,-1)],axis=-1) # (nseq,60,？)
    Sa = Slasta.reshape(nseq,5,-1)
    A = data[:,140:144]
    return S,Sa,A

def generate_dataloader(train_data,eval_data,feature_name,batch_size,num_workers, shuffle=True,pad_with_last_sample=False):
    level_2_num = 0
    if len(train_data) > 0:
        level_2_num = len(train_data[0])
    else:
        raise ValueError('train_data is empty')
    if pad_with_last_sample:
        # num_padding = (batch_size - (len(train_data) % batch_size)) % batch_size
        train_num_padding = (batch_size - (len(train_data) % batch_size)) % batch_size
        train_data_padding2 = np.empty((train_num_padding,level_2_num), dtype=object)

        eval_num_padding = (batch_size - (len(eval_data) % batch_size)) % batch_size
        eval_data_padding2 = np.empty((eval_num_padding,level_2_num), dtype=object)

        for i in range(train_num_padding):
            for j in range(level_2_num):
                train_data_padding2[i,j] = train_data[-1][j]
        for i in range(eval_num_padding):
            for j in range(level_2_num):
                eval_data_padding2[i,j] = eval_data[-1][j]
        
        train_data_raw = np.empty((len(train_data),level_2_num), dtype=object)
        eval_data_raw = np.empty((len(eval_data),level_2_num), dtype=object)
        for i in range(len(train_data)):
            for j in range(level_2_num):
                train_data_raw[i,j] = train_data[i][j]
        for i in range(len(eval_data)):
            for j in range(level_2_num):
                eval_data_raw[i,j] = eval_data[i][j]
        train_data = np.concatenate([train_data_raw, train_data_padding2], axis=0)
        eval_data = np.concatenate([eval_data_raw, eval_data_padding2], axis=0)
    train_dataset = ListDataset(train_data)
    eval_dataset = ListDataset(eval_data)
    def collator(indices):
        batch = Batch(feature_name)
        for item in indices:
            batch.append(copy.deepcopy(item))
        return batch
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collator)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collator)

    return train_dataloader, eval_dataloader
class NB_rldata:
    def __init__(self,config:dict):
        self.config = config
        self.feature_name = {
            "St": "float",
            "Sat": "float",
            "At": "float",
            "Stn": "float",
            "Rtn": "float",
            "done_column": "float",
        }
        # self.rel_file = config["rel_file"] #没有关联信息
        self.rawdata_path = config["rawdata_path"]
        self.cache_file_path = config["cache_file_path"]
        self.batch_size = config["batch_size"]
        self.train_rate = self.config.get("train_rate",0.9)
        self.eval_rate = self.config.get("eval_rate",0.1)
        self.input_window = self.config.get('input_window', 4)
        self.output_window = self.config.get('output_window', 2)

        self.stateseq_window = self.config.get("stateseq_window",4)
        self.stateseq_predn = self.config.get("stateseq_predn",1)

        self.cache_dataset = self.config.get("cache_dataset",True)
        self.data_files = [os.path.join(self.rawdata_path, file) for file in os.listdir(self.rawdata_path)]
        self.a_reward:float = self.config.get("a",1.0)
        self.b_reward:float = self.config.get("b",1.0)
        self.num_workers = self.config.get("num_workers",2) #表示dataloader的num_workers
        self.data = None #表示数据是否已经加载
    def _load_cache_train_val_test(self):
        cat_data = np.load(self.cache_file_path)
        St_train = cat_data['St_train']
        Sat_train = cat_data['Sat_train']
        At_train = cat_data['At_train']
        Stn_train = cat_data['Stn_train']
        Rtn_train = cat_data['Rtn_train']
        done_column_train = cat_data['done_column_train']
        St_eval = cat_data['St_eval']
        Sat_eval = cat_data['Sat_eval']
        At_eval = cat_data['At_eval']
        Stn_eval = cat_data['Stn_eval']
        Rtn_eval = cat_data['Rtn_eval']
        done_column_eval = cat_data['done_column_eval']
        return St_train, Sat_train, At_train, Stn_train, Rtn_train, done_column_train,St_eval, Sat_eval, At_eval, Stn_eval, Rtn_eval, done_column_eval
    def _generate_input_data2(self,S:np.ndarray,Sa:np.ndarray,A:np.ndarray,reward:np.ndarray):
        '''
        处理单个数据文件
        '''
        St = S[:-1,...]
        Sat = Sa[:-1,...]
        At = A[:-1,...]
        Stn = S[1:,...]
        Rtn = reward[1:,...]
        Stn_size = Stn.shape[0]
        done_column = np.zeros((Stn_size,1),dtype=np.float32)
        done_column[-1,0] = 1
        num_samples = St.shape[0]
        x_offsets = np.sort(np.concatenate((np.arange(-self.input_window + 1, 1, 1),)))
        y_offsets = np.sort(np.arange(1, self.output_window + 1, 1))
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))
        St_finial, Sat_finial, At_finial, Stn_finial, Rtn_finial, done_column_finial = [],[],[],[],[],[]
        for t in range(min_t,max_t):
            St_finial.append(St[t + x_offsets])
            Sat_finial.append(Sat[t + x_offsets])
            At_finial.append(At[t + x_offsets])
            Stn_finial.append(Stn[t + y_offsets])
            Rtn_finial.append(Rtn[t + y_offsets])
            done_column_finial.append(done_column[t + y_offsets])
        return St_finial, Sat_finial, At_finial, Stn_finial, Rtn_finial, done_column_finial
    def _generate_input_data(self,S:np.ndarray,Sa:np.ndarray,A:np.ndarray,reward:np.ndarray):
        """
        切分数据函数。
        
        参数:
        - data: 输入数据，假设是一个NumPy数组。
        - window: 窗口大小。
        - predn: 预测偏移量。
        
        返回:
        - St, Stn: 分别为S和S'的数据列表，每个列表中的元素都是长度为window的数据。
        """
        St_finial, Sat_finial, At_finial, Stn_finial, Rtn_finial, done_column_finial = [],[],[],[],[],[]
        num_samples = S.shape[0]
        max_t = num_samples - self.stateseq_window - self.stateseq_predn + 1
        done_column = np.zeros((max_t,1),dtype=np.float32)
        done_column[-1,0] = 1
        window = self.stateseq_window
        pn = self.stateseq_predn
        for t in range(max_t):
            St_finial.append(S[t:t + window])
            Sat_finial.append(Sa[t:t + window])
            At_finial.append(A[t+window-1])
            Stn_finial.append(S[t + pn:t + window + pn])
            Rtn_finial.append([-1*reward[t + window - 1]])
            done_column_finial.append(done_column[t])
        
        return St_finial, Sat_finial, At_finial, Stn_finial, Rtn_finial, done_column_finial
    def _rewardfunction(self,data:np.ndarray):
        '''
        S: (nseq,5,?) np.ndarray
        '''
        Swait = data[:,:60]
        Swait0 = Swait[:,:12].sum(axis=-1)
        SwaitOther = Swait[:,12:60].sum(axis=-1)
        reward = self.a_reward * Swait0 + self.b_reward * SwaitOther
        return reward
    
    def _generate_data(self):
        '''
        处理self.data_files的所有数据文件
        '''
        data_files = []
        if isinstance(self.data_files,list):
            data_files = self.data_files.copy()
        else:
            data_files = [self.data_files].copy()
        
        St_list, Sat_list, At_list, Stn_list, Rtn_list, done_column_list = [],[],[],[],[],[]
        for filepath in data_files:
            dst_data = None
            with open(filepath,'rb') as f:
                dst_data = pickle.load(f)
            State,Statela,Action = decode_simdata_lane(dst_data)
            Reward = self._rewardfunction(dst_data)
            St_finial, Sat_finial, At_finial, Stn_finial, Rtn_finial, done_column_finial = self._generate_input_data(State,Statela,Action,Reward)
            St_list.append(St_finial)
            Sat_list.append(Sat_finial)
            At_list.append(At_finial)
            Stn_list.append(Stn_finial)
            Rtn_list.append(Rtn_finial)
            done_column_list.append(done_column_finial)
        St_list = np.concatenate(St_list)
        Sat_list = np.concatenate(Sat_list)
        At_list = np.concatenate(At_list)
        Stn_list = np.concatenate(Stn_list)
        Rtn_list = np.concatenate(Rtn_list)
        done_column_list = np.concatenate(done_column_list)
        return St_list, Sat_list, At_list, Stn_list, Rtn_list, done_column_list
    def _split_train_val_test(self,St:np.ndarray,Sat:np.ndarray,At:np.ndarray,Stn:np.ndarray,Rtn:np.ndarray,done_column:np.ndarray):
        num_samples = St.shape[0]
        num_train = int(num_samples * self.train_rate)
        indices = np.random.permutation(num_samples)
        train_indices = indices[:num_train]
        eval_indices = indices[num_train:]

        St_train, Sat_train, At_train, Stn_train, Rtn_train, done_column_train = St[train_indices], Sat[train_indices], At[train_indices], Stn[train_indices], Rtn[train_indices], done_column[train_indices]

        St_eval, Sat_eval, At_eval, Stn_eval, Rtn_eval, done_column_eval = St[eval_indices], Sat[eval_indices], At[eval_indices], Stn[eval_indices], Rtn[eval_indices], done_column[eval_indices]
        if self.cache_dataset:
            cachedir = os.path.dirname(self.cache_file_path)
            ensure_dir(cachedir)
            np.savez(self.cache_file_path,
                    St_train=St_train, 
                    Sat_train=Sat_train, 
                    At_train=At_train, 
                    Stn_train=Stn_train, 
                    Rtn_train=Rtn_train, 
                    done_column_train=done_column_train,
                    St_eval=St_eval, 
                    Sat_eval=Sat_eval, 
                    At_eval=At_eval, 
                    Stn_eval=Stn_eval, 
                    Rtn_eval=Rtn_eval, 
                    done_column_eval=done_column_eval)
        return St_train, Sat_train, At_train, Stn_train, Rtn_train, done_column_train,St_eval, Sat_eval, At_eval, Stn_eval, Rtn_eval, done_column_eval
    
    def _generate_train_val_test(self):
        St, Sat, At, Stn, Rtn, done_column = self._generate_data()
        return self._split_train_val_test(St,Sat,At,Stn,Rtn,done_column)
    def get_data(self):
        St_train, Sat_train, At_train, Stn_train, Rtn_train, done_column_train = [],[],[],[],[],[]
        St_eval, Sat_eval, At_eval, Stn_eval, Rtn_eval, done_column_eval = [],[],[],[],[],[]
        # train_memory = []
        # eval_memory = []
        if self.data is None:
            if self.cache_dataset and os.path.exists(self.cache_file_path):
                St_train, Sat_train, At_train, Stn_train, Rtn_train, done_column_train,St_eval, Sat_eval, At_eval, Stn_eval, Rtn_eval, done_column_eval = self._load_cache_train_val_test()
            else:
                St_train, Sat_train, At_train, Stn_train, Rtn_train, done_column_train,St_eval, Sat_eval, At_eval, Stn_eval, Rtn_eval, done_column_eval = self._generate_train_val_test()
        train_data = list(zip(St_train, Sat_train, At_train, Stn_train, Rtn_train, done_column_train))
        eval_data = list(zip(St_eval, Sat_eval, At_eval, Stn_eval, Rtn_eval, done_column_eval))
        self.train_dataloader,self.eval_dataloader = generate_dataloader(train_data,eval_data,self.feature_name,self.batch_size,self.num_workers,shuffle=True,pad_with_last_sample=True)
        return self.train_dataloader,self.eval_dataloader



effectpressure_phasemask = [
    [
        [0, 0, 0,    0, 1, 0,    0, 0, 0,    0, 1, 0],
        [0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],
        [0, 0, 0,    0, 0, 0,    0, 0, 0,    -0.5,-0.5,0],
        [0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],
        [0, 0, 0,    -0.5,-0.5, 0,    0, 0, 0,    0, 0, 0],
    ],
    [
        [0, 1, 0,    0, 0, 0,    0, 1, 0,    0, 0, 0],
        [0, 0, 0,    0, 0, 0,    -0.5,-0.5,0,    0, 0, 0],
        [0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],
        [-0.5,-0.5,0,    0, 0, 0,    0, 0, 0,    0, 0, 0],
        [0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],
    ],
    [
        [0, 0, 0,    1, 0, 0,    0, 0, 0,    1, 0, 0],
        [0, 0, 0,    0, 0, 0,    -0.5,-0.5,0,    0, 0, 0],
        [0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],
        [-0.5,-0.5,0,    0, 0, 0,    0, 0, 0,    0, 0, 0],
        [0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],
    ],
    [
        [1, 0, 0,    0, 0, 0,    1, 0, 0,    0, 0, 0],
        [0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],
        [0, 0, 0,    0, 0, 0,    0, 0, 0,    -0.5,-0.5,0],
        [0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],
        [0, 0, 0,    -0.5,-0.5, 0,    0, 0, 0,    0, 0, 0],
    ],

    [
        [0, 0, 0,    0, 0, 0,    0, 0, 0,    1, 1, 0],
        [0, 0, 0,    0, 0, 0,    -0.5,-0.5,0,    0, 0, 0],
        [0, 0, 0,    0, 0, 0,    0, 0, 0,    -0.5,-0.5,0],
        [0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],
        [0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],
    ],
    [
        [0, 0, 0,    1, 1, 0,    0, 0, 0,    0, 0, 0],
        [0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],
        [0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],
        [-0.5,-0.5,0,    0, 0, 0,    0, 0, 0,    0, 0, 0],
        [0, 0, 0,    -0.5,-0.5, 0,    0, 0, 0,    0, 0, 0],
    ],
    [
        [0, 0, 0,    0, 0, 0,    1, 1, 0,    0, 0, 0],
        [0, 0, 0,    0, 0, 0,    -0.5,-0.5,0,    0, 0, 0],
        [0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],
        [0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],
        [0, 0, 0,    -0.5,-0.5, 0,    0, 0, 0,    0, 0, 0],
    ],
    [
        [1, 1, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],
        [0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],
        [0, 0, 0,    0, 0, 0,    0, 0, 0,    -0.5,-0.5,0],
        [-0.5,-0.5,0,    0, 0, 0,    0, 0, 0,    0, 0, 0],
        [0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],
    ]
]

neighbourSelect = \
    [0, 0, 0,    0, 0, 0,    1,1,0,    0, 0, 0  ,  
     0, 0, 0,    0, 0, 0,    0, 0, 0,    1,1,0  ,  
     1,1,0,    0, 0, 0,    0, 0, 0,    0, 0, 0  ,  
     0, 0, 0,    1,1, 0,    0, 0, 0,    0, 0, 0],

class NBintersection_rldata:
    def __init__(self,config:dict):
        self.config = config
        self.feature_name = {
            "St": "float",
            "At": "float",
            "Stn": "float",
            "Rtn": "float",
            "done_column": "float",
        }
        # self.rel_file = config["rel_file"] #没有关联信息
        self.rawdata_path = self.config.get("rawdata_path","./rawdata/")
        self.cache_file_path = self.config.get("cache_file_path","./cache.npz")
        self.batch_size = config["batch_size"]
        self.train_rate = self.config.get("train_rate",0.9)
        self.eval_rate = self.config.get("eval_rate",0.1)
        self.input_window = self.config.get('input_window', 4)
        self.output_window = self.config.get('output_window', 2)
        self.stateseq_window = self.config.get("stateseq_window",4)
        self.stateseq_predn = self.config.get("stateseq_predn",1)
        self.cache_dataset = self.config.get("cache_dataset",True)
        
        self.a_reward:float = self.config.get("a",1.0)
        self.b_reward:float = self.config.get("b",1.0)
        self.neighbour_selectmatrix:np.ndarray = np.array(neighbourSelect) # (48,)
        print(self.a_reward,self.b_reward)
        self.num_workers = self.config.get("num_workers",2) #表示dataloader的num_workers
        self.data = None #表示数据是否已经加载
    def _load_cache_train_val_test(self):
        cat_data = np.load(self.cache_file_path)
        St_train = cat_data['St_train']
        At_train = cat_data['At_train']
        Stn_train = cat_data['Stn_train']
        Rtn_train = cat_data['Rtn_train']
        done_column_train = cat_data['done_column_train']
        St_eval = cat_data['St_eval']
        At_eval = cat_data['At_eval']
        Stn_eval = cat_data['Stn_eval']
        Rtn_eval = cat_data['Rtn_eval']
        done_column_eval = cat_data['done_column_eval']
        return St_train, At_train, Stn_train, Rtn_train, done_column_train,St_eval, At_eval, Stn_eval, Rtn_eval, done_column_eval
    def _generate_input_data2(self,S:np.ndarray,A:np.ndarray,reward:np.ndarray):
        '''
        处理单个数据文件
        '''
        St = S[:-1,...]
        At = A[:-1,...]
        Stn = S[1:,...]
        Rtn = reward[1:,...]
        Stn_size = Stn.shape[0]
        done_column = np.zeros((Stn_size,1),dtype=np.float32)
        done_column[-1,0] = 1
        num_samples = St.shape[0]
        x_offsets = np.sort(np.concatenate((np.arange(-self.input_window + 1, 1, 1),)))
        y_offsets = np.sort(np.arange(1, self.output_window + 1, 1))
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))
        St_finial, At_finial, Stn_finial, Rtn_finial, done_column_finial = [],[],[],[],[]
        for t in range(min_t,max_t):
            St_finial.append(St[t + x_offsets])
            At_finial.append(At[t + x_offsets])
            Stn_finial.append(Stn[t + y_offsets])
            Rtn_finial.append(Rtn[t + y_offsets])
            done_column_finial.append(done_column[t + y_offsets])
        return St_finial, At_finial, Stn_finial, Rtn_finial, done_column_finial
    def _generate_input_data(self,S:np.ndarray,A:np.ndarray,reward:np.ndarray):
        """
        切分数据函数。
        
        参数:
        - data: 输入数据，假设是一个NumPy数组。
        - window: 窗口大小。
        - predn: 预测偏移量。
        
        返回:
        - St, Stn: 分别为S和S'的数据列表，每个列表中的元素都是长度为window的数据。
        """
        St_finial, At_finial, Stn_finial, Rtn_finial, done_column_finial = [],[],[],[],[]
        num_samples = S.shape[0]
        max_t = num_samples - self.stateseq_window - self.stateseq_predn + 1
        done_column = np.zeros((max_t,1),dtype=np.float32)
        done_column[-1,0] = 1
        window = self.stateseq_window
        pn = self.stateseq_predn
        for t in range(max_t):
            St_finial.append(S[t:t + window])
            At_finial.append(A[t + window - 1])
            Stn_finial.append(S[t + pn:t + window + pn])
            Rtn_finial.append([-1*reward[t + window - 1]])
            done_column_finial.append(done_column[t])
        
        return St_finial, At_finial, Stn_finial, Rtn_finial, done_column_finial

        

    def _rewardfunction(self,data:np.ndarray):
        '''
        S: (nseq,5,?) np.ndarray
        '''
        Swait = data[:,:60]
        Swait0 = Swait[:,:12].sum(axis=-1)
        SwaitOther = Swait[:,12:60].sum(axis=-1)
        reward = self.a_reward * Swait0 + self.b_reward * SwaitOther
        return reward
    def _rewardfunctionV2(self,data:np.ndarray):
        Swait = data[:,:60]
        Swait0 = Swait[:,:12].sum(axis=-1)
        # swaitother = neighbourSelectmatrix X Swait[:,12:]   (1,48) @ (n,48) -> (n,)
        # self.neighbour_selectmatrix -> (1,48)   self.neighbour_selectmatrix.reshape(1,48) -> (1,48)
        SwaitOther =  (Swait[:,12:] @ self.neighbour_selectmatrix.reshape(48,1)).reshape(-1)
        reward = self.a_reward * Swait0 + self.b_reward * SwaitOther
        return reward

    
    def _generate_data(self):
        '''
        处理self.data_files的所有数据文件
        '''
        data_files = []
        self.data_files = [os.path.join(self.rawdata_path, file) for file in os.listdir(self.rawdata_path)]
        if isinstance(self.data_files,list):
            data_files = self.data_files.copy()
        else:
            data_files = [self.data_files].copy()
        
        St_list, At_list, Stn_list, Rtn_list, done_column_list = [],[],[],[],[]
        for filepath in data_files:
            dst_data = None
            with open(filepath,'rb') as f:
                dst_data = pickle.load(f)
            State,Action = decode_simdata_intersection(dst_data)
            Reward = self._rewardfunctionV2(dst_data)
            St_finial, At_finial, Stn_finial, Rtn_finial, done_column_finial = self._generate_input_data(State,Action,Reward)
            St_list.append(St_finial)
            At_list.append(At_finial)
            Stn_list.append(Stn_finial)
            Rtn_list.append(Rtn_finial)
            done_column_list.append(done_column_finial)
        St_list = np.concatenate(St_list)
        At_list = np.concatenate(At_list)
        Stn_list = np.concatenate(Stn_list)
        Rtn_list = np.concatenate(Rtn_list)
        done_column_list = np.concatenate(done_column_list)
        return St_list, At_list, Stn_list, Rtn_list, done_column_list
    def _generate_data_from_ndarray(self,dst_data:np.ndarray):
        '''
        处理self.data_files的所有数据文件
        '''
        St_list, At_list, Stn_list, Rtn_list, done_column_list = [],[],[],[],[]
        State,Action = decode_simdata_intersection(dst_data)
        Reward = self._rewardfunctionV2(dst_data)
        St_finial, At_finial, Stn_finial, Rtn_finial, done_column_finial = self._generate_input_data(State,Action,Reward)
        St_list.append(St_finial)
        At_list.append(At_finial)
        Stn_list.append(Stn_finial)
        Rtn_list.append(Rtn_finial)
        done_column_list.append(done_column_finial)
        St_list = np.concatenate(St_list)
        At_list = np.concatenate(At_list)
        Stn_list = np.concatenate(Stn_list)
        Rtn_list = np.concatenate(Rtn_list)
        done_column_list = np.concatenate(done_column_list)
        return St_list, At_list, Stn_list, Rtn_list, done_column_list
    def _split_train_val_test(self,St:np.ndarray,At:np.ndarray,Stn:np.ndarray,Rtn:np.ndarray,done_column:np.ndarray):
        num_samples = St.shape[0]
        num_train = int(num_samples * self.train_rate)
        indices = np.random.permutation(num_samples)
        train_indices = indices[:num_train]
        eval_indices = indices[num_train:]

        St_train, At_train, Stn_train, Rtn_train, done_column_train = St[train_indices], At[train_indices], Stn[train_indices], Rtn[train_indices], done_column[train_indices]

        St_eval, At_eval, Stn_eval, Rtn_eval, done_column_eval = St[eval_indices], At[eval_indices], Stn[eval_indices], Rtn[eval_indices], done_column[eval_indices]
        if self.cache_dataset:
            cachedir = os.path.dirname(self.cache_file_path)
            ensure_dir(cachedir)
            np.savez(self.cache_file_path,
                    St_train=St_train, 
                    At_train=At_train, 
                    Stn_train=Stn_train, 
                    Rtn_train=Rtn_train, 
                    done_column_train=done_column_train,
                    St_eval=St_eval, 
                    At_eval=At_eval, 
                    Stn_eval=Stn_eval, 
                    Rtn_eval=Rtn_eval, 
                    done_column_eval=done_column_eval)
        return St_train, At_train, Stn_train, Rtn_train, done_column_train,St_eval, At_eval, Stn_eval, Rtn_eval, done_column_eval
    
    def _generate_train_val_test(self):
        St, At, Stn, Rtn, done_column = self._generate_data()
        return self._split_train_val_test(St,At,Stn,Rtn,done_column)
    def _generate_train_val_test_fromndarray(self,dst_data):
        St, At, Stn, Rtn, done_column = self._generate_data_from_ndarray(dst_data)
        return self._split_train_val_test(St,At,Stn,Rtn,done_column)
    def get_data(self):
        St_train, At_train, Stn_train, Rtn_train, done_column_train = [],[],[],[],[]
        St_eval, At_eval, Stn_eval, Rtn_eval, done_column_eval = [],[],[],[],[]
        # train_memory = []
        # eval_memory = []
        if self.data is None:
            if self.cache_dataset and os.path.exists(self.cache_file_path):
                St_train, At_train, Stn_train, Rtn_train, done_column_train,St_eval, At_eval, Stn_eval, Rtn_eval, done_column_eval = self._load_cache_train_val_test()
            else:
                St_train, At_train, Stn_train, Rtn_train, done_column_train,St_eval, At_eval, Stn_eval, Rtn_eval, done_column_eval = self._generate_train_val_test()
        train_data = list(zip(St_train, At_train, Stn_train, Rtn_train, done_column_train))
        eval_data = list(zip(St_eval, At_eval, Stn_eval, Rtn_eval, done_column_eval))
        self.train_dataloader,self.eval_dataloader = generate_dataloader(train_data,eval_data,self.feature_name,self.batch_size,self.num_workers,shuffle=True,pad_with_last_sample=True)
        return self.train_dataloader,self.eval_dataloader
    def get_data_fromndarray(self,StAt:np.ndarray):
        St_train, At_train, Stn_train, Rtn_train, done_column_train = [],[],[],[],[]
        St_eval, At_eval, Stn_eval, Rtn_eval, done_column_eval = [],[],[],[],[]
        # train_memory = []
        # eval_memory = []
        St_train, At_train, Stn_train, Rtn_train, done_column_train,St_eval, At_eval, Stn_eval, Rtn_eval, done_column_eval = self._generate_train_val_test_fromndarray(StAt)
        train_data = list(zip(St_train, At_train, Stn_train, Rtn_train, done_column_train))
        eval_data = list(zip(St_eval, At_eval, Stn_eval, Rtn_eval, done_column_eval))
        self.train_dataloader,self.eval_dataloader = generate_dataloader(train_data,eval_data,self.feature_name,self.batch_size,self.num_workers,shuffle=True,pad_with_last_sample=True)
        return self.train_dataloader,self.eval_dataloader



class RLdata:
    '''
    Sc: 
        - nseq =1: S(?,lane,d_feature)
        - nseq>=2: S(?,nseq,lane,d_feature)

    Ac:
        - nseq =1: A(?,d_action)
        - nseq>=2: A(?,nseq,d_action)
    
    A:
        - A(?,d_action)

    Sn:
        - nseq =1: S(?,lane,d_feature)
        - nseq>=2: S(?,nseq,lane,d_feature)

    An:
        - nseq =1: A(?,d_action)
        - nseq>=2: A(?,nseq,d_action)

    Rn:
        - R(?)
    
    done_column:
        - done_column(?)
    '''
    def __init__(self,config:dict):

        self.config = config
        self.feature_name = {
            "Sc": "float",
            "Ac": "float",
            "A": "float",
            "Sn": "float",
            "An": "float",
            "Rn": "float",
            "done_column": "float",
        }
        self.rawdata_path = self.config.get("rawdata_path","./rawdata/")
        self.cache_file_path = self.config.get("cache_file_path","./cache.npz")
        self.batch_size = config["batch_size"]
        self.train_rate = self.config.get("train_rate",0.95)
        self.eval_rate = self.config.get("eval_rate",0.05)
        self.stateseq_window = self.config.get("stateseq_window",4) #表示窗口大小
        self.cache_dataset = self.config.get("cache_dataset",False)
        self.num_workers = self.config.get("num_workers",2)
        self.action_dim = self.config.get("action_dim",4)
        self.default_action = self.config.get("default_action",np.ndarray((self.action_dim,)))
        self.data = None #表示数据是否已经加载
    def _generate_input_data(self,S:np.ndarray,A:np.ndarray,reward:np.ndarray):
        """
        向A的前面填充一个默认的动作，使得A的长度比S的长度多1

        填充前：S0->A0->S1->A1->S2->A2->...  S[n] -> A[n]

        填充后：Ad->S0->A0->S1->A1->S2->A2->... A[n] -> S[n] -> A[n+1]


        """
        Scfinial, Acfinial, Afinial, Snfinial, Anfinial, Rnfinial, done_column_finial = [],[],[],[],[],[],[]
        num_samples = S.shape[0]
        # A:(?,d_action)  default_action:(d_action,)
        A = np.concatenate([self.default_action[np.newaxis,:],A],axis=0) #(?+1,d_action)
        if self.stateseq_window>1: #这一分支将会考虑stateseq_window和stateseq_predn
            max_t = num_samples - self.stateseq_window
            done_column = np.zeros((max_t,1),dtype=np.float32)
            done_column[-1,0] = 1
            window = self.stateseq_window
            for t in range(max_t):
                Scfinial.append(S[t:t + window])
                Acfinial.append(A[t:t + window])
                Afinial.append(A[t + window])
                Snfinial.append(S[t + 1:t + window + 1])
                Anfinial.append(A[t + 1:t + window + 1])
                Rnfinial.append([reward[t + window]])
                done_column_finial.append(done_column[t])
        else: #这一分支不会考虑stateseq_window，最后输出的数据也是没有n_seq维度的数据。
            max_t = num_samples - 1
            done_column = np.zeros((max_t,1),dtype=np.float32)
            done_column[-1,0] = 1
            for t in range(max_t):
                Scfinial.append(S[t])
                Acfinial.append(A[t])
                Afinial.append(A[t+1])
                Snfinial.append(S[t+1])
                Anfinial.append(A[t+1])
                Rnfinial.append([reward[t+1]])
                done_column_finial.append(done_column[t])
        return Scfinial, Acfinial, Afinial, Snfinial, Anfinial, Rnfinial, done_column_finial
    def _rewardfunction(self,data:np.ndarray):
        '''
        S: (...？) 其中最后一维度的第一个数值表示排队长度，但是由于reward希望排队长度减少，所以这里取负数，用这个数值sum后得到了一个 （num_sample,）的reward 

        '''
        Swait = data[...,0] # (n_sample,...,1) 

        # (n_sample,...,1)  需要在 除了n_sample维度的其他维度上求和 -> (n_sample,)
        Swaitsum = Swait.sum(axis=tuple(range(1,len(Swait.shape))))
        reward = -Swaitsum
        return reward
    

    def _generate_data(self):
        '''
        处理self.data_files的所有数据文件
        '''
        data_files = []
        self.data_files = [os.path.join(self.rawdata_path, file) for file in os.listdir(self.rawdata_path)]
        if isinstance(self.data_files,list):
            data_files = self.data_files.copy()
        else:
            data_files = [self.data_files].copy()
        Sc_list, Ac_list, A_list, Sn_list, An_list, Rn_list, done_column_list = [],[],[],[],[],[],[]
        for filepath in data_files:
            dst_data = None
            with open(filepath,'rb') as f:
                dst_data = pickle.load(f)
            State,Action = dst_data
            Reward = self._rewardfunction(State)
            Sc_finial, Ac_finial, A_finial, Sn_finial, An_finial, Rn_finial, done_column_finial = self._generate_input_data(State,Action,Reward)
            Sc_list.append(Sc_finial)
            Ac_list.append(Ac_finial)
            A_list.append(A_finial)
            Sn_list.append(Sn_finial)
            An_list.append(An_finial)
            Rn_list.append(Rn_finial)
            done_column_list.append(done_column_finial)
        Sc_list = np.concatenate(Sc_list)
        Ac_list = np.concatenate(Ac_list)
        A_list = np.concatenate(A_list)
        Sn_list = np.concatenate(Sn_list)
        An_list = np.concatenate(An_list)
        Rn_list = np.concatenate(Rn_list)
        done_column_list = np.concatenate(done_column_list)
        return Sc_list, Ac_list, A_list, Sn_list, An_list, Rn_list, done_column_list
    def _generate_data_from_ndarray(self,dst_data:np.ndarray):
        '''
        处理self.data_files的所有数据文件
        '''
        Sc_list, Ac_list, A_list, Sn_list, An_list, Rn_list, done_column_list = [],[],[],[],[],[],[]
        State,Action = dst_data
        Reward = self._rewardfunction(State)
        Sc_finial, Ac_finial, A_finial, Sn_finial, An_finial, Rn_finial, done_column_finial = self._generate_input_data(State,Action,Reward)
        Sc_list.append(Sc_finial)
        Ac_list.append(Ac_finial)
        A_list.append(A_finial)
        Sn_list.append(Sn_finial)
        An_list.append(An_finial)
        Rn_list.append(Rn_finial)
        done_column_list.append(done_column_finial)
        Sc_list = np.concatenate(Sc_list)
        Ac_list = np.concatenate(Ac_list)
        A_list = np.concatenate(A_list)
        Sn_list = np.concatenate(Sn_list)
        An_list = np.concatenate(An_list)
        Rn_list = np.concatenate(Rn_list)
        done_column_list = np.concatenate(done_column_list)
        return Sc_list, Ac_list, A_list, Sn_list, An_list, Rn_list, done_column_list
    def _split_train_val_test(self,Sc:np.ndarray,Ac:np.ndarray,A:np.ndarray,Sn:np.ndarray,An:np.ndarray,Rn:np.ndarray,done_column:np.ndarray):
        num_samples = Sc.shape[0]
        num_train = int(num_samples * self.train_rate)
        indices = np.random.permutation(num_samples)
        train_indices = indices[:num_train]
        eval_indices = indices[num_train:]

        Sc_train, Ac_train, A_train, Sn_train, An_train, Rn_train, done_column_train = Sc[train_indices], Ac[train_indices], A[train_indices], Sn[train_indices], An[train_indices], Rn[train_indices], done_column[train_indices]

        Sc_eval, Ac_eval, A_eval, Sn_eval, An_eval, Rn_eval, done_column_eval = Sc[eval_indices], Ac[eval_indices], A[eval_indices], Sn[eval_indices], An[eval_indices], Rn[eval_indices], done_column[eval_indices]
        if self.cache_dataset:
            cachedir = os.path.dirname(self.cache_file_path)
            ensure_dir(cachedir)
            np.savez(self.cache_file_path,
                    Sc_train=Sc_train, 
                    Ac_train=Ac_train, 
                    A_train=A_train, 
                    Sn_train=Sn_train, 
                    An_train=An_train, 
                    Rn_train=Rn_train, 
                    done_column_train=done_column_train,
                    Sc_eval=Sc_eval, 
                    Ac_eval=Ac_eval, 
                    A_eval=A_eval, 
                    Sn_eval=Sn_eval, 
                    An_eval=An_eval, 
                    Rn_eval=Rn_eval, 
                    done_column_eval=done_column_eval)
        return Sc_train, Ac_train, A_train, Sn_train, An_train, Rn_train, done_column_train,Sc_eval, Ac_eval, A_eval, Sn_eval, An_eval, Rn_eval, done_column_eval
    
    # ----split train val test
    def _generate_train_val_test(self):
        Sc, Ac, A, Sn, An, Rn, done_column = self._generate_data()
        return self._split_train_val_test(Sc,Ac,A,Sn,An,Rn,done_column)

    # ----split train val test
    def _generate_train_val_test_fromndarray(self,dst_data):
        Sc, Ac, A, Sn, An, Rn, done_column = self._generate_data_from_ndarray(dst_data)
        return self._split_train_val_test(Sc,Ac,A,Sn,An,Rn,done_column)
    
    # --------------获取 dataloader--------------
    def get_data(self):
        Sc_train, Ac_train, A_train, Sn_train, An_train, Rn_train, done_column_train = [],[],[],[],[],[],[]
        Sc_eval, Ac_eval, A_eval, Sn_eval, An_eval, Rn_eval, done_column_eval = [],[],[],[],[],[],[]
        # train_memory = []
        # eval_memory = []
        if self.data is None:
            if self.cache_dataset and os.path.exists(self.cache_file_path):
                Sc_train, Ac_train, A_train, Sn_train, An_train, Rn_train, done_column_train,Sc_eval, Ac_eval, A_eval, Sn_eval, An_eval, Rn_eval, done_column_eval = self._load_cache_train_val_test()
            else:
                Sc_train, Ac_train, A_train, Sn_train, An_train, Rn_train, done_column_train,Sc_eval, Ac_eval, A_eval, Sn_eval, An_eval, Rn_eval, done_column_eval = self._generate_train_val_test()
        train_data = list(zip(Sc_train, Ac_train, A_train, Sn_train, An_train, Rn_train, done_column_train))
        eval_data = list(zip(Sc_eval, Ac_eval, A_eval, Sn_eval, An_eval, Rn_eval, done_column_eval))
        self.train_dataloader,self.eval_dataloader = generate_dataloader(train_data,eval_data,self.feature_name,self.batch_size,self.num_workers,shuffle=True,pad_with_last_sample=True)
        return self.train_dataloader,self.eval_dataloader
    def get_data_fromndarray(self,ScAc:np.ndarray):
        Sc_train, Ac_train, A_train, Sn_train, An_train, Rn_train, done_column_train = [],[],[],[],[],[],[]
        Sc_eval, Ac_eval, A_eval, Sn_eval, An_eval, Rn_eval, done_column_eval = [],[],[],[],[],[],[]
        # train_memory = []
        # eval_memory = []
        Sc_train, Ac_train, A_train, Sn_train, An_train, Rn_train, done_column_train,Sc_eval, Ac_eval, A_eval, Sn_eval, An_eval, Rn_eval, done_column_eval = self._generate_train_val_test_fromndarray(ScAc)
        train_data = list(zip(Sc_train, Ac_train, A_train, Sn_train, An_train, Rn_train, done_column_train))
        eval_data = list(zip(Sc_eval, Ac_eval, A_eval, Sn_eval, An_eval, Rn_eval, done_column_eval))
        self.train_dataloader,self.eval_dataloader = generate_dataloader(train_data,eval_data,self.feature_name,self.batch_size,self.num_workers,shuffle=True,pad_with_last_sample=True)
        return self.train_dataloader,self.eval_dataloader
    # ------------------------------------------



        
        



        
        


if __name__=='__main__':
    config = {
        "batch_size":32,
        "train_rate":0.7,
        "eval_rate":0.1,
        "rawdata_path":"dataoffline2/",
        "cache_file_path":"cache_data/anon_3_4_jinan_real_2500_cache103.npz",
        "input_window":4,
        "output_window":1,
    }
    dataset = NB_rldata(config)
    train_dataloader,eval_dataloader = dataset.get_data()
    feature_name = {
        "St": "float",
        "Sat": "float",
        "At": "float",
        "Stn": "float",
        "Rtn": "float",
        "done_column": "float",
    }
    for batch in train_dataloader:
        batch.to_tensor(torch.device("cpu"))
        for key in batch.data:
            print(key)
            print(batch[key].shape)
            if key == "At":
                print(batch[key][0,:])
            print('-----------------')
        # print(batch)
        break
    print("===========================")
    config2 = {
        "batch_size":32,
        "train_rate":0.7,
        "eval_rate":0.1,
        "rawdata_path":"dataoffline2/",
        "cache_file_path":"cache_data/anon_3_4_jinan_real_2500_cache104.npz",
        "stateseq_window":4,
        "stateseq_predn":1,
    }
    dataset2 = NBintersection_rldata(config2)
    train_dataloader2,eval_dataloader = dataset2.get_data()
    feature_name2 = {
        "St": "float",
        "At": "float",
        "Stn": "float",
        "Rtn": "float",
        "done_column": "float",
    }
    for batch in train_dataloader2:
        batch.to_tensor(torch.device("cpu"))
        for key in batch.data:
            print(key)
            print(batch[key].shape)
            print('-----------------')

        # print(batch)
        break
    for batch in train_dataloader2:
        batch.to_tensor(torch.device("cpu"))
        dones = batch['done_column']
        print(dones)
        # break