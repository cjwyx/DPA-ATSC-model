from cityflow_utils import cityflowlib
import cityflow as engine
from run_frame import run_queue,phase_adapter #导入放行模板和放行规则
import random
from typing import List, Tuple, Dict
import pickle
import numpy as np
import torch
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# net=cityflowlib.net("./data/Jinan/3_4/roadnet_3_4.json")
net = cityflowlib.net("./data/Hangzhou/4_4/roadnet_4_4.json")

nodelist = net.tlnodes
nodenum = len(nodelist)
phasetime_max = 20
phasetime_min = 15
yellowtime = 5
epmasked = cityflowlib.effectpressure_phasemaskstd()

# epmaskedmatrix = np.array(epmasked[:4],dtype=np.float32) # (4,60)
epmasked = cityflowlib.maxpressure_phasemask
epmaskedmatrix = np.array(epmasked[:4],dtype=np.float32) # (4,12)
def parse_nodename2id(nodename:str)-> Tuple[int,int]:
    '''
    - nodename: 节点名称  intersection_1_1
    - return: (intersection_idx,intersection_id)
    '''
    intersection_id1 = int(nodename.split("_")[1])
    intersection_id2 = int(nodename.split("_")[2])
    return (intersection_id1,intersection_id2)

savedatadirpath = "./model_save/recorddata_qst3fc10hz/"
def savedataata2pkl(statadata,actiondata,ATT:float,savepath:str):
    savepathdir = os.path.dirname(savepath)
    if not os.path.exists(savepathdir):
        os.makedirs(savepathdir)
    with open(savepath,"wb") as f:
        pickle.dump((statadata,actiondata,ATT),f)

def savedata2pklbytime(statadata,actiondata,ATT:float,savedir:str):
    current_time = time.time()
    currenttimestr = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(current_time)) + '_{:03d}'.format(int((current_time - int(current_time)) * 1000))
    savepath = os.path.join(savedir,currenttimestr+".pkl")
    savedataata2pkl(statadata,actiondata,ATT,savepath)
def savedata2pklbytime_tag(statadata,actiondata,ATT:float,savedir:str):
    current_time = time.time()
    currenttimestr = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(current_time)) + '_{:03d}'.format(int((current_time - int(current_time)) * 1000))
    savepath = os.path.join(savedir,currenttimestr+"_{}.pkl".format(ATT))
    savedataata2pkl(statadata,actiondata,ATT,savepath)

'''

00:
'intersection_1_1'
01:
'intersection_1_2'
02:
'intersection_1_3'
03:
'intersection_2_1'
04:
'intersection_2_2'
05:
'intersection_2_3'
06:
'intersection_3_1'
07:
'intersection_3_2'
08:
'intersection_3_3'
09:
'intersection_4_1'
10:
'intersection_4_2'
11:
'intersection_4_3'
'''

# nodenighbors = {
#     "intersection_1_1":["intersection_1_1","intersection_1_2","intersection_2_1","",""],
#     "intersection_1_2":["intersection_1_2","intersection_1_3","intersection_2_2","intersection_1_1",""],
#     "intersection_1_3":["intersection_1_3","","intersection_2_3","intersection_1_2",""],
#     "intersection_2_1":["intersection_2_1","intersection_2_2","intersection_3_1","","intersection_1_1"],
#     "intersection_2_2":["intersection_2_2","intersection_2_3","intersection_3_2","intersection_2_1","intersection_1_2"],
#     "intersection_2_3":["intersection_2_3","","intersection_3_3","intersection_2_2","intersection_1_3"],
#     "intersection_3_1":["intersection_3_1","intersection_3_2","intersection_4_1","","intersection_2_1"],
#     "intersection_3_2":["intersection_3_2","intersection_3_3","intersection_4_2","intersection_3_1","intersection_2_2"],
#     "intersection_3_3":["intersection_3_3","","intersection_4_3","intersection_3_2","intersection_2_3"],
#     "intersection_4_1":["intersection_4_1","intersection_4_2","","","intersection_3_1"],
#     "intersection_4_2":["intersection_4_2","intersection_4_3","","intersection_4_1","intersection_3_2"],
#     "intersection_4_3":["intersection_4_3","","","intersection_4_2","intersection_3_3"],
# }

# nodenighborsidx = []
# for nodename in nodelist:
#     nodes = nodenighbors[nodename]
#     nodesidx = []
#     for node in nodes:
#         if node == "":
#             nodesidx.append(-1)
#         else:
#             nodesidx.append(nodelist.index(node))
#     nodenighborsidx.append(nodesidx)


class collectband:
    def __init__(self,nodenamelist:List[str],max_runtime:float):
        self.datafilterlist:List[cityflowlib.TnodeDataFilter] = []
        self.anode_space:List[dict] = []
        self.task_space:List[List] = []
        self.state_dst =[]
        self.action_dst = []
        self.prevaction_dst = []
        self.dst =[]
        self.nodelist = nodenamelist
        self.max_runtime = max_runtime
   
    # init-------------
    # 配置segment的切分长度信息，
    def config_segmentinfo_lanelen(self,lanes_length_dict:Dict[str,float],obslen):
        for tnode in self.datafilterlist:
            tnode.set_lanes_length_bydict(lanes_length_dict,obslen)
    # 使用一个节点的数据配置车道数量  
    def config_common_lanesnum(self,TnodeTemplate:cityflowlib.TnodeDataFilter):
        self.lanesnum = len(TnodeTemplate.entring_lanes)
    # 链接一个数据输出位置
    def link_dst(self,datadst:List): 
        del self.dst
        self.dst = datadst

    # 初始化flowrecord (用于passcar flow 记录)
    def init_flowrecord(self):
        for datatnode in self.datafilterlist:
            datatnode.init_flow_ids()
    # ---------------------
    
    



    # Step ------------
    def step(self,connection:engine):
        connection.next_step()
        for item in self.anode_space:
            item["current_runtime"] += 1
    def step_flowrecord(self,connection:engine):
        lane_flow_dict: Dict[str, list] = connection.get_lane_vehicles()
        for datatnode in self.datafilterlist:
            datatnode.update_flow_ids(lane_flow_dict)
        connection.next_step()
        for item in self.anode_space:
            item["current_runtime"] += 1
    # -----------------

    def set_signal(self,connection:engine,intersection_idx:int, phase_number:int,phase_duration:int):
        str_intersection_id = self.nodelist[intersection_idx]
        connection.set_tl_phase(str_intersection_id, phase_number)
        if phase_number !=0:
            self.anode_space[intersection_idx]["phase"] = phase_number
            self.anode_space[intersection_idx]["target_runtime"] = phase_duration
            self.anode_space[intersection_idx]["current_runtime"] = 0
            self.anode_space[intersection_idx]["currentphaseidx"] = self.anode_space[intersection_idx]["phaselist"].index(phase_number)

    # Collect ----------
    def funcollect_q(self,connection:engine,intersection_idx:int,phase_number:int,new_phase:int,phase_duration:int):
        # intersection_idx = intersection_id-1
        waiting_data = connection.get_lane_waiting_vehicle_count()
        reswaiting = self.datafilterlist[intersection_idx].get_waiting_data(waiting_data)
        # self.dst[intersection_idx].append([phase_number,new_phase,phase_duration,reswaiting])
        res_tastate = np.concatenate([reswaiting.reshape(self.lanesnum,-1)],axis=-1,dtype=np.float32)
        self.state_dst[intersection_idx].append(res_tastate)

        rphasetimevec = np.zeros((4,),dtype=np.float32)
        
        newphaseidx = self.anode_space[intersection_idx]["phaselist"].index(new_phase)
        
        newruntimestd = phase_duration/self.max_runtime
        rphasetimevec[newphaseidx] = newruntimestd
        
        self.action_dst[intersection_idx].append(rphasetimevec)

        prevphaseidx = self.anode_space[intersection_idx]["phaselist"].index(phase_number)
        prevphasevec = np.zeros((4,),dtype=np.float32)
        prevphasevec[prevphaseidx] = 1
        self.prevaction_dst[intersection_idx].append(prevphasevec)
    def funcollect_qv(self,connection:engine,intersection_idx:int,phase_number:int,new_phase:int,phase_duration:int):
        ''' 
        - q:waiting queue (12,1)
        - v:flow speed (12,1)
        '''
        # intersection_idx = intersection_id-1
        
        flow_speed_dict: Dict[str, float] = connection.get_vehicle_speed()
        waiting_data = connection.get_lane_waiting_vehicle_count()
        lane_flow_dict: Dict[str, list] = connection.get_lane_vehicles()
        reswaiting = self.datafilterlist[intersection_idx].get_waiting_data(waiting_data)
        resspeed = self.datafilterlist[intersection_idx].get_meanspeed_data(flow_speed_dict,lane_flow_dict)
        res_tastate = np.concatenate([reswaiting.reshape(self.lanesnum,-1),resspeed.reshape(self.lanesnum,-1)],axis=-1,dtype=np.float32)
        self.state_dst[intersection_idx].append(res_tastate)

        rphasetimevec = np.zeros((4,),dtype=np.float32)
        newphaseidx = self.anode_space[intersection_idx]["phaselist"].index(new_phase)
        newruntimestd = phase_duration/self.max_runtime
        rphasetimevec[newphaseidx] = newruntimestd
        self.action_dst[intersection_idx].append(rphasetimevec)


        prevphaseidx = self.anode_space[intersection_idx]["phaselist"].index(phase_number)
        prevphasevec = np.zeros((4,),dtype=np.float32)
        prevphasevec[prevphaseidx] = 1
        self.prevaction_dst[intersection_idx].append(prevphasevec)
    def funcollect_qv_rev(self,connection:engine,intersection_idx:int,phase_number:int,new_phase:int,phase_duration:int):
        ''' 
        - q:waiting queue (12,1)
        - v:flow speed (12,1) 是 正常速度的倒数
        '''
        # intersection_idx = intersection_id-1
        
        flow_speed_dict: Dict[str, float] = connection.get_vehicle_speed()
        waiting_data = connection.get_lane_waiting_vehicle_count()
        lane_flow_dict: Dict[str, list] = connection.get_lane_vehicles()
        reswaiting = self.datafilterlist[intersection_idx].get_waiting_data(waiting_data)
        resspeed = self.datafilterlist[intersection_idx].get_meanspeed_data(flow_speed_dict,lane_flow_dict)
        resspeed = 1/(resspeed+1)
        res_tastate = np.concatenate([reswaiting.reshape(self.lanesnum,-1),resspeed.reshape(self.lanesnum,-1)],axis=-1,dtype=np.float32)
        self.state_dst[intersection_idx].append(res_tastate)

        rphasetimevec = np.zeros((4,),dtype=np.float32)
        newphaseidx = self.anode_space[intersection_idx]["phaselist"].index(new_phase)
        newruntimestd = phase_duration/self.max_runtime
        rphasetimevec[newphaseidx] = newruntimestd
        self.action_dst[intersection_idx].append(rphasetimevec)

        prevphaseidx = self.anode_space[intersection_idx]["phaselist"].index(phase_number)
        prevphasevec = np.zeros((4,),dtype=np.float32)
        prevphasevec[prevphaseidx] = 1
        self.prevaction_dst[intersection_idx].append(prevphasevec)
    def funcollect_qvp_rev(self,connection:engine,intersection_idx:int,phase_number:int,new_phase:int,phase_duration:int):
        '''
        请配合step_flowrecord一起使用
        - q:waiting queue
        - v:flow speed 表示正常速度的倒数
        - p:pass car

        :param connection: engine
        :param intersection_idx: int 表示节点编号
        :param phase_number: int 表示当前相位
        :param new_phase: int 表示新的相位
        :param phase_duration: int 表示新的相位持续时间

        '''
        
        flow_speed_dict: Dict[str, float] = connection.get_vehicle_speed()
        waiting_data = connection.get_lane_waiting_vehicle_count()
        lane_flow_dict: Dict[str, list] = connection.get_lane_vehicles()
        reswaiting = self.datafilterlist[intersection_idx].get_waiting_data(waiting_data)
        resspeed = self.datafilterlist[intersection_idx].get_meanspeed_data(flow_speed_dict,lane_flow_dict)
        resspeed = 1/(resspeed+1)

        self.datafilterlist[intersection_idx].update_flow_ids(lane_flow_dict)
        passcarcount = self.datafilterlist[intersection_idx].get_flow_countdata()


        res_tastate = np.concatenate([reswaiting.reshape(self.lanesnum,-1),resspeed.reshape(self.lanesnum,-1),passcarcount.reshape(self.lanesnum,-1)],axis=-1,dtype=np.float32)


        self.datafilterlist[intersection_idx].reset_flow_ids()


        rphasetimevec = np.zeros((4,),dtype=np.float32)
        newphaseidx = self.anode_space[intersection_idx]["phaselist"].index(new_phase)
        newruntimestd = phase_duration/self.max_runtime
        rphasetimevec[newphaseidx] = newruntimestd
        self.state_dst[intersection_idx].append(res_tastate)
        self.action_dst[intersection_idx].append(rphasetimevec)

        prevphaseidx = self.anode_space[intersection_idx]["phaselist"].index(phase_number)
        prevphasevec = np.zeros((4,),dtype=np.float32)
        prevphasevec[prevphaseidx] = 1
        self.prevaction_dst[intersection_idx].append(prevphasevec)
    def funcollect_qcrp(self,connection:engine,intersection_idx:int,phase_number:int,new_phase:int,phase_duration:int):
        '''
        - q:waiting queue
        - c:flow count
        - r:running count
        - p:pass car

        '''
        flow_speed_dict: Dict[str, float] = connection.get_vehicle_speed()
        waiting_data = connection.get_lane_waiting_vehicle_count()
        lane_flow_dict: Dict[str, list] = connection.get_lane_vehicles()
        lane_vehicle_count:Dict[str,int] = connection.get_lane_vehicle_count()
        reswaiting = self.datafilterlist[intersection_idx].get_waiting_data(waiting_data)
        rescount = self.datafilterlist[intersection_idx].get_count_data(lane_vehicle_count)
        resrunning = self.datafilterlist[intersection_idx].get_count_running_data(flow_speed_dict,lane_flow_dict)

        self.datafilterlist[intersection_idx].update_flow_ids(lane_flow_dict)
        passcarcount = self.datafilterlist[intersection_idx].get_flow_countdata()
        self.datafilterlist[intersection_idx].reset_flow_ids()

        res_tastate = np.concatenate([
            reswaiting.reshape(self.lanesnum,-1),
            rescount.reshape(self.lanesnum,-1),
            resrunning.reshape(self.lanesnum,-1),
            passcarcount.reshape(self.lanesnum,-1) ],axis=-1,dtype=np.float32)

        rphasetimevec = np.zeros((4,),dtype=np.float32)
        newphaseidx = self.anode_space[intersection_idx]["phaselist"].index(new_phase)
        newruntimestd = phase_duration/self.max_runtime
        rphasetimevec[newphaseidx] = newruntimestd

        self.state_dst[intersection_idx].append(res_tastate)
        self.action_dst[intersection_idx].append(rphasetimevec)

        prevphaseidx = self.anode_space[intersection_idx]["phaselist"].index(phase_number)
        prevphasevec = np.zeros((4,),dtype=np.float32)
        prevphasevec[prevphaseidx] = 1
        self.prevaction_dst[intersection_idx].append(prevphasevec)
    def funcollect_qcrps(self,connection:engine,intersection_idx:int,phase_number:int,new_phase:int,phase_duration:int):
        '''
        - q:waiting queue (12,1)
        - c:flow count (12,1)
        - r:running count (12,1)
        - p:pass car (12,1)
        - s:segment data (12,4)
        '''
        flow_speed_dict: Dict[str, float] = connection.get_vehicle_speed()
        waiting_data = connection.get_lane_waiting_vehicle_count()
        lane_flow_dict: Dict[str, list] = connection.get_lane_vehicles()
        lane_vehicle_count:Dict[str,int] = connection.get_lane_vehicle_count()
        vehicle_distance:Dict[str,float] = connection.get_vehicle_distance()
        reswaiting = self.datafilterlist[intersection_idx].get_waiting_data(waiting_data)
        rescount = self.datafilterlist[intersection_idx].get_count_data(lane_vehicle_count)
        resrunning = self.datafilterlist[intersection_idx].get_count_running_data(flow_speed_dict,lane_flow_dict)

        self.datafilterlist[intersection_idx].update_flow_ids(lane_flow_dict)
        passcarcount = self.datafilterlist[intersection_idx].get_flow_countdata()
        self.datafilterlist[intersection_idx].reset_flow_ids()

        ressegment_data= self.datafilterlist[intersection_idx].get_segments_data(lane_flow_dict,vehicle_distance)

        res_tastate = np.concatenate([
            reswaiting.reshape(self.lanesnum,-1),
            rescount.reshape(self.lanesnum,-1),
            resrunning.reshape(self.lanesnum,-1),
            passcarcount.reshape(self.lanesnum,-1),
            ressegment_data.reshape(self.lanesnum,-1)],axis=-1,dtype=np.float32)
    
        rphasetimevec = np.zeros((4,),dtype=np.float32)
        newphaseidx = self.anode_space[intersection_idx]["phaselist"].index(new_phase)
        newruntimestd = phase_duration/self.max_runtime
        rphasetimevec[newphaseidx] = newruntimestd

        self.state_dst[intersection_idx].append(res_tastate)
        self.action_dst[intersection_idx].append(rphasetimevec)

        prevphaseidx = self.anode_space[intersection_idx]["phaselist"].index(phase_number)
        prevphasevec = np.zeros((4,),dtype=np.float32)
        prevphasevec[prevphaseidx] = 1
        self.prevaction_dst[intersection_idx].append(prevphasevec)
    def funcollect_qvcrps_rev(self,connection:engine,intersection_idx:int,phase_number:int,new_phase:int,phase_duration:int):
        '''
        - q:waiting queue (12,1)
        - v:flow speed (12,1) 是 正常速度的倒数
        - c:flow count (12,1) 
        - r:running count (12,1)
        - p:pass car (12,1)
        - s:segment data (12,4)

        '''
        flow_speed_dict: Dict[str, float] = connection.get_vehicle_speed()
        waiting_data = connection.get_lane_waiting_vehicle_count()
        lane_flow_dict: Dict[str, list] = connection.get_lane_vehicles()
        lane_vehicle_count:Dict[str,int] = connection.get_lane_vehicle_count()
        vehicle_distance:Dict[str,float] = connection.get_vehicle_distance()
        reswaiting = self.datafilterlist[intersection_idx].get_waiting_data(waiting_data)
        resspeed = self.datafilterlist[intersection_idx].get_meanspeed_data(flow_speed_dict,lane_flow_dict)
        resspeed = 1/(resspeed+1)
        rescount = self.datafilterlist[intersection_idx].get_count_data(lane_vehicle_count)
        resrunning = self.datafilterlist[intersection_idx].get_count_running_data(flow_speed_dict,lane_flow_dict)

        self.datafilterlist[intersection_idx].update_flow_ids(lane_flow_dict)
        passcarcount = self.datafilterlist[intersection_idx].get_flow_countdata()
        self.datafilterlist[intersection_idx].reset_flow_ids()

        ressegment_data= self.datafilterlist[intersection_idx].get_segments_data(lane_flow_dict,vehicle_distance)

        res_tastate = np.concatenate([
            reswaiting.reshape(self.lanesnum,-1),
            resspeed.reshape(self.lanesnum,-1),
            rescount.reshape(self.lanesnum,-1),
            resrunning.reshape(self.lanesnum,-1),
            passcarcount.reshape(self.lanesnum,-1),
            ressegment_data.reshape(self.lanesnum,-1)],axis=-1,dtype=np.float32)

        rphasetimevec = np.zeros((4,),dtype=np.float32)
        newphaseidx = self.anode_space[intersection_idx]["phaselist"].index(new_phase)
        newruntimestd = phase_duration/self.max_runtime
        rphasetimevec[newphaseidx] = newruntimestd

        self.state_dst[intersection_idx].append(res_tastate)
        self.action_dst[intersection_idx].append(rphasetimevec)

        prevphaseidx = self.anode_space[intersection_idx]["phaselist"].index(phase_number)
        prevphasevec = np.zeros((4,),dtype=np.float32)
        prevphasevec[prevphaseidx] = 1
        self.prevaction_dst[intersection_idx].append(prevphasevec)
    def funcollect_qvcrs_rev(self,connection:engine,intersection_idx:int,phase_number:int,new_phase:int,phase_duration:int):
        '''
        - q:waiting queue (12,1)
        - v:flow speed (12,1) 是 正常速度的倒数
        - c:flow count (12,1) 
        - r:running count (12,1)
        - s:segment data (12,4)

        '''
        flow_speed_dict: Dict[str, float] = connection.get_vehicle_speed()
        waiting_data = connection.get_lane_waiting_vehicle_count()
        lane_flow_dict: Dict[str, list] = connection.get_lane_vehicles()
        lane_vehicle_count:Dict[str,int] = connection.get_lane_vehicle_count()
        vehicle_distance:Dict[str,float] = connection.get_vehicle_distance()
        reswaiting = self.datafilterlist[intersection_idx].get_waiting_data(waiting_data)
        resspeed = self.datafilterlist[intersection_idx].get_meanspeed_data(flow_speed_dict,lane_flow_dict)
        resspeed = 1/(resspeed+1)
        rescount = self.datafilterlist[intersection_idx].get_count_data(lane_vehicle_count)
        resrunning = self.datafilterlist[intersection_idx].get_count_running_data(flow_speed_dict,lane_flow_dict)
        ressegment_data= self.datafilterlist[intersection_idx].get_segments_data(lane_flow_dict,vehicle_distance)

        res_tastate = np.concatenate([
            reswaiting.reshape(self.lanesnum,-1),
            resspeed.reshape(self.lanesnum,-1),
            rescount.reshape(self.lanesnum,-1),
            resrunning.reshape(self.lanesnum,-1),
            ressegment_data.reshape(self.lanesnum,-1)],axis=-1,dtype=np.float32)

        rphasetimevec = np.zeros((4,),dtype=np.float32)
        newphaseidx = self.anode_space[intersection_idx]["phaselist"].index(new_phase)
        newruntimestd = phase_duration/self.max_runtime
        rphasetimevec[newphaseidx] = newruntimestd

        self.state_dst[intersection_idx].append(res_tastate)
        self.action_dst[intersection_idx].append(rphasetimevec)

        prevphaseidx = self.anode_space[intersection_idx]["phaselist"].index(phase_number)
        prevphasevec = np.zeros((4,),dtype=np.float32)
        prevphasevec[prevphaseidx] = 1
        self.prevaction_dst[intersection_idx].append(prevphasevec)
    def funcollect_qrs(self,connection:engine,intersection_idx:int,phase_number:int,new_phase:int,phase_duration:int):
        '''
        - q:waiting queue (12,1)
        - r:running count (12,1)
        - s:segment data (12,4)

        '''
        flow_speed_dict: Dict[str, float] = connection.get_vehicle_speed()
        waiting_data = connection.get_lane_waiting_vehicle_count()
        lane_flow_dict: Dict[str, list] = connection.get_lane_vehicles()
        lane_vehicle_count:Dict[str,int] = connection.get_lane_vehicle_count()
        vehicle_distance:Dict[str,float] = connection.get_vehicle_distance()
        reswaiting = self.datafilterlist[intersection_idx].get_waiting_data(waiting_data)
        resrunning = self.datafilterlist[intersection_idx].get_count_running_data(flow_speed_dict,lane_flow_dict)
        ressegment_data= self.datafilterlist[intersection_idx].get_segments_data(lane_flow_dict,vehicle_distance)

        res_tastate = np.concatenate([
            reswaiting.reshape(self.lanesnum,-1),
            resrunning.reshape(self.lanesnum,-1),
            ressegment_data.reshape(self.lanesnum,-1)],axis=-1,dtype=np.float32)

        rphasetimevec = np.zeros((4,),dtype=np.float32)
        newphaseidx = self.anode_space[intersection_idx]["phaselist"].index(new_phase)
        newruntimestd = phase_duration/self.max_runtime
        rphasetimevec[newphaseidx] = newruntimestd

        self.state_dst[intersection_idx].append(res_tastate)
        self.action_dst[intersection_idx].append(rphasetimevec)

        prevphaseidx = self.anode_space[intersection_idx]["phaselist"].index(phase_number)
        prevphasevec = np.zeros((4,),dtype=np.float32)
        prevphasevec[prevphaseidx] = 1
        self.prevaction_dst[intersection_idx].append(prevphasevec)

    # ----数据ndarray化----
    def compiledate(self):
        internum = len(self.state_dst)
        for i in range(internum):
            self.state_dst[i] = np.array(self.state_dst[i])
            self.action_dst[i] = np.array(self.action_dst[i])
            self.prevaction_dst[i] = np.array(self.prevaction_dst[i])
    # -------------------

    # 提取数据
    def extract_state(self,connection:engine,intersection_idx:int,dataseq_num:int):
        if dataseq_num > 1:
            dataextract_from_stated_dst= []
            dataextract_from_actionprev_dst = []
            for i in range(dataseq_num):
                dataextract_from_stated_dst.append(self.state_dst[intersection_idx][-dataseq_num+i])
                dataextract_from_actionprev_dst.append(self.prevaction_dst[intersection_idx][-dataseq_num+i])
            DataSrcState = np.array(dataextract_from_stated_dst)
            DataSrcActionPrev = np.array(dataextract_from_actionprev_dst)
            return DataSrcState,DataSrcActionPrev
        else:
            DataSrcState = self.state_dst[intersection_idx][-1]
            DataSrcActionPrev = self.prevaction_dst[intersection_idx][-1]
            return DataSrcState,DataSrcActionPrev
    def extract_state2(self,connection:engine,intersection_idx:int,dataseq_num:int):
        if dataseq_num > 1:
            dataextract_from_stated_dst= []
            dataextract_from_actionprev_dst = []
            for i in range(dataseq_num):
                dataextract_from_stated_dst.append(self.state_dst[intersection_idx][-dataseq_num+i])
                dataextract_from_actionprev_dst.append(self.prevaction_dst[intersection_idx][-dataseq_num+i])
            DataSrcState = np.array(dataextract_from_stated_dst)
            DataSrcActionPrev = np.array(dataextract_from_actionprev_dst)
            return DataSrcState,DataSrcActionPrev
        else:
            DataSrcState = self.state_dst[intersection_idx][-1]
            DataSrcActionPrev = self.action_dst[intersection_idx][-2]
            return DataSrcState,DataSrcActionPrev
    def extract_laststate(self,connection:engine,intersection_idx:int):
        DataSrcState = self.state_dst[intersection_idx][-1]
        DataSrcActionPrev = self.prevaction_dst[intersection_idx][-1]
        return DataSrcState,DataSrcActionPrev
    def extract_lastwaitstate(self,connection:engine,intersection_idx):
        DataSrcState = self.state_dst[intersection_idx][-1]
        Swait = DataSrcState[...,0] # (...,12,1)

        # (...,12,1) -> (...,12)
        return Swait    
    def extract_lastwaitstate_imdt(self,connection:engine,intersection_idx):
        waiting_data = connection.get_lane_waiting_vehicle_count()
        reswaiting = self.datafilterlist[intersection_idx].get_waiting_data(waiting_data)
        return reswaiting

    def extractdata_oneagent_simple(self) -> np.ndarray:
        '''
        tutal 
        '''
        internums = len(self.state_dst)
        finial_state = []
        finial_action = []
        finial_actionprev = []
        for i in range(internums):
            Sc = self.state_dst[i]
            Ac = self.action_dst[i]
            Ap = self.prevaction_dst[i]
            finial_state.append(Sc)
            finial_action.append(Ac)
            finial_actionprev.append(Ap)
        finial_state = np.concatenate(finial_state,axis=0)
        finial_action = np.concatenate(finial_action,axis=0)
        finial_actionprev = np.concatenate(finial_actionprev,axis=0)
        return finial_state,finial_action,finial_actionprev
    
    def extract_stateimdt_qvcrs_rev(self,connection:engine,intersection_idx:int,dataseq_num:int):
        '''
        - q:waiting queue (12,1)
        - v:flow speed (12,1) 是 正常速度的倒数
        - c:flow count (12,1) 
        - r:running count (12,1)
        - p:pass car (12,1)
        - s:segment data (12,4)
        '''
        if dataseq_num <=1 :
            #DataSrcState数据只有一个，
            flow_speed_dict: Dict[str, float] = connection.get_vehicle_speed()
            waiting_data = connection.get_lane_waiting_vehicle_count()
            lane_flow_dict: Dict[str, list] = connection.get_lane_vehicles()
            lane_vehicle_count:Dict[str,int] = connection.get_lane_vehicle_count()
            vehicle_distance:Dict[str,float] = connection.get_vehicle_distance()
            reswaiting = self.datafilterlist[intersection_idx].get_waiting_data(waiting_data)
            resspeed = self.datafilterlist[intersection_idx].get_meanspeed_data(flow_speed_dict,lane_flow_dict)
            resspeed = 1/(resspeed+1)
            rescount = self.datafilterlist[intersection_idx].get_count_data(lane_vehicle_count)
            resrunning = self.datafilterlist[intersection_idx].get_count_running_data(flow_speed_dict,lane_flow_dict)
            ressegment_data= self.datafilterlist[intersection_idx].get_segments_data(lane_flow_dict,vehicle_distance)

            res_tastate = np.concatenate([
                reswaiting.reshape(self.lanesnum,-1),
                resspeed.reshape(self.lanesnum,-1),
                rescount.reshape(self.lanesnum,-1),
                resrunning.reshape(self.lanesnum,-1),
                ressegment_data.reshape(self.lanesnum,-1)],axis=-1,dtype=np.float32)
            DataSrcActionPrev = self.action_dst[intersection_idx][-1]


            return res_tastate,DataSrcActionPrev
        raise ValueError("dataseq_num must be greater than 1")
    def extract_stateimdt_qrs(self,connection:engine,intersection_idx:int,dataseq_num:int):
        '''
        - q:waiting queue (12,1)
        - r:running count (12,1)
        - s:segment data (12,4)

        '''
        if dataseq_num <=1 :
            #DataSrcState数据只有一个，
            flow_speed_dict: Dict[str, float] = connection.get_vehicle_speed()
            waiting_data = connection.get_lane_waiting_vehicle_count()
            lane_flow_dict: Dict[str, list] = connection.get_lane_vehicles()
            lane_vehicle_count:Dict[str,int] = connection.get_lane_vehicle_count()
            vehicle_distance:Dict[str,float] = connection.get_vehicle_distance()
            reswaiting = self.datafilterlist[intersection_idx].get_waiting_data(waiting_data)
            resrunning = self.datafilterlist[intersection_idx].get_count_running_data(flow_speed_dict,lane_flow_dict)
            ressegment_data= self.datafilterlist[intersection_idx].get_segments_data(lane_flow_dict,vehicle_distance)

            res_tastate = np.concatenate([
                reswaiting.reshape(self.lanesnum,-1),
                resrunning.reshape(self.lanesnum,-1),
                ressegment_data.reshape(self.lanesnum,-1)],axis=-1,dtype=np.float32)
            DataSrcActionPrev = self.action_dst[intersection_idx][-1]

            return res_tastate,DataSrcActionPrev
        raise ValueError("dataseq_num must be greater than 1")
    # -----------------

    # -----保存数据----------
    def savepkl(self,savepath:str):
        with open(savepath,"wb") as f:
            pickle.dump((self.state_dst,self.action_dst),f)
    # ----------------------
            

class collectband_nighbor:
    def __init__(self,nodenamelist:List[str],atasklist:List[List],max_runtime:float):
        self.datafilterlist:List[cityflowlib.TnodeDataFilter] = []
        self.anode_space:List[dict] = []
        self.task_space:List[List] = atasklist
        self.state_dst =[]
        self.action_dst = []
        self.nodelist = nodenamelist
        self.const_Action = np.array([1,1,1,1])
        self.max_runtime = max_runtime
    def config_segmentinfo_lanelen(self,lanes_length_dict:Dict[str,float]):
        for tnode in self.datafilterlist:
            tnode.set_lanes_length_bydict(lanes_length_dict)
    def step(self,connection:engine):
        connection.next_step()
        for item in self.anode_space:
            item["current_runtime"] += 1
    def set_signal(self,connection:engine,intersection_idx:int, phase_number:int,phase_duration:int):
        str_intersection_id = self.nodelist[intersection_idx]
        connection.set_tl_phase(str_intersection_id, phase_number)
        if phase_number !=0:
            self.anode_space[intersection_idx]["phase"] = phase_number
            self.anode_space[intersection_idx]["target_runtime"] = phase_duration
            self.anode_space[intersection_idx]["current_runtime"] = 0
            self.anode_space[intersection_idx]["currentphaseidx"] = self.anode_space[intersection_idx]["phaselist"].index(phase_number)
    def funcollect_qva(self,connection:engine,intersection_idx:int,phase_number:int,new_phase:int,phase_duration:int):
        # intersection_idx = intersection_id-1
        flow_speed_dict: Dict[str, float] = connection.get_vehicle_speed()
        waiting_data = connection.get_lane_waiting_vehicle_count()
        lane_flow_dict: Dict[str, list] = connection.get_lane_vehicles()
        reswaiting = self.datafilterlist[intersection_idx].get_waiting_data(waiting_data)
        resspeed = self.datafilterlist[intersection_idx].get_meanspeed_data(flow_speed_dict,lane_flow_dict)
        resaction = []
        for anodeidx in self.task_space[intersection_idx]:
            if anodeidx == -1:
                resaction.append(self.const_Action)
            else:
                rphasetimevec = np.zeros((4,),dtype=np.float32)
                currentphaseidx = self.anode_space[anodeidx]["currentphaseidx"]
                rlasttimenorm = max(0,self.anode_space[anodeidx]["target_runtime"]-self.anode_space[anodeidx]["current_runtime"])/self.max_runtime
                rphasetimevec[currentphaseidx] = rlasttimenorm
                resaction.append(rphasetimevec)
        resaction = np.concatenate(resaction,axis=-1,dtype=np.float32)
        res_tastate = np.concatenate([reswaiting,resspeed,resaction],axis=-1,dtype=np.float32)
        rphasetimevec = np.zeros((4,),dtype=np.float32)

        #配置新的相位-> rphasetimevec是一个向量，表示四个相位的时间（标准化）
        newphaseidx = self.anode_space[intersection_idx]["phaselist"].index(new_phase)
        newruntimestd = phase_duration/self.max_runtime
        rphasetimevec[newphaseidx] = newruntimestd
        #-----------------------------------------------------------


        self.state_dst[intersection_idx].append(res_tastate)
        self.action_dst[intersection_idx].append(rphasetimevec)
    def extract_state(self,connection:engine,intersection_idx:int,dataseq_num:int):
        '''
        :param intersection_idx: int,节点编号
        :param dataseq_num: int,从数据库中提取最后dataseq_num个数据

        '''
        dataextract_from_stated_dst= []
        for i in range(dataseq_num):
            dataextract_from_stated_dst.append(self.state_dst[intersection_idx][-dataseq_num+i])

        DataSrcState = np.array(dataextract_from_stated_dst)
        Swait = DataSrcState[:,:60]
        Sspeed = DataSrcState[:,60:120]
        Slasta = DataSrcState[:,120:140]
        S = np.concatenate([Swait.reshape(dataseq_num,5,-1),Sspeed.reshape(dataseq_num,5,-1),Slasta.reshape(dataseq_num,5,-1)],axis=-1) # (dataseq_num,5,？)
        return S
    def extract_lastwaitstate(self,connection:engine,intersection_idx:int):

        DataSrcState = self.state_dst[intersection_idx][-1]
        Swait = DataSrcState[...,:60]
        return Swait
    def savenpz(self,savepath:str):
        '''
        .rand(3, 2)
        >>> test_vector = np.random.rand(4)
        >>> np.savez_compressed('/tmp/123', a=test_array, b=test_vector)
        >>> loaded = np.load('/tmp/123.npz')
        >>> print(np.array_equal(test_array, loaded['a']))
        True
        >>> print(np.array_equal(test_vector, loaded['b']))
        True
        '''
        np.savez_compressed(savepath,state=np.array(self.state_dst),action=np.array(self.action_dst))
    def compiledate(self):
        internum = len(self.state_dst)
        for i in range(internum):
            self.state_dst[i] = np.array(self.state_dst[i])
            self.action_dst[i] = np.array(self.action_dst[i])
    def savepkl(self,savepath:str):
        with open(savepath,"wb") as f:
            pickle.dump((self.state_dst,self.action_dst),f)
    def savedata_oneagent_simple(self,savepath:str):
        '''
        tutal 264

        0-139:S

        140-143:A
        '''
        finial_state = []
        for i in range(nodenum):
            Sc = self.state_dst[i]
            Ac = self.action_dst[i]
            SA = np.concatenate([Sc,Ac],axis=-1)
            finial_state.append(SA)
        finial_state = np.concatenate(finial_state,axis=0)
        with open(savepath,'wb') as f:
            pickle.dump(finial_state,f)
    def extractdata_oneagent_simple(self) -> np.ndarray:
        '''
        tutal 264

        0-139:S

        140-143:A
        '''
        finial_state = []
        for i in range(nodenum):
            Sc = self.state_dst[i]
            Ac = self.action_dst[i]
            SA = np.concatenate([Sc,Ac],axis=-1)
            finial_state.append(SA)
        finial_state = np.concatenate(finial_state,axis=0)
        return finial_state
            
    def savedata_oneagentstd(self,savepath:str):
        '''
        tutal 264

        0-139:S

        140-143:A

        144-263:S'

        264:done
        '''
        finial_state = []
        # finial_action = []

        for i in range(nodenum):
            Sc = self.state_dst[i][:-1,:]
            Ac = self.action_dst[i][:-1,:]
            Sn = self.state_dst[i][1:,:120]
            size1 = Sn.shape[0]
            done_column = np.zeros((size1,1),dtype=np.float32)
            done_column[-1,0] = 1
            SASD = np.concatenate([Sc,Ac,Sn,done_column],axis=-1)
            finial_state.append(SASD)
        finial_state = np.concatenate(finial_state,axis=0)
        with open(savepath,'wb') as f:
            pickle.dump(finial_state,f)



def choose_action(prev_phase_number):
    ruler = random.randint(0,19)
    if ruler <= 5:
        return phase_gen.green_2_next[prev_phase_number],random.randint(phasetime_min,phasetime_max)
    elif ruler<= 8:
        return prev_phase_number,random.randint(phasetime_min,phasetime_max)
    else:
        return phase_gen.random_phase(),random.randint(phasetime_min,phasetime_max)
def choose_action_maxpressure(prev_phase_number):
    ruler = random.randint(0,19)
    if ruler <= 16:
        return phase_gen.green_2_next[prev_phase_number],random.randint(phasetime_min,phasetime_max)
    elif ruler<= 18:
        return prev_phase_number,random.randint(phasetime_min,phasetime_max)
    else:
        return phase_gen.random_phase(),random.randint(phasetime_min,phasetime_max)

def choose_action_epfunc(state:np.ndarray,phasenumber:int):
    '''
    :param state: np.ndarray,shape(?,12,d_f)
    epmask (4,60)
    epmask @ state -> (4,60)@(?,60) -> (?,4) S
    '''
    
    S = np.einsum("ij,...j->...i",epmaskedmatrix,state)
    # argmax(S,dim=-1) -> (?,) return (?,)
    newphaseidx = 0
    #计算出最大值的标量
    maxvalue = np.max(S,axis=-1)
    if phasenumber <= 0:
        newphaseidx =  np.argmax(S,axis=-1)
    else:
        oldphaseidx = phasenumber-1
        if S[oldphaseidx] >= maxvalue:
            newphaseidx =  oldphaseidx
        else:
            newphaseidx =  np.argmax(S,axis=-1)
    return newphaseidx+1

class RemomeryBuffer:
    def __init__(self,maxqueuelen:int = 32):
        self.maxqueuelen = maxqueuelen
        self.queue = []
    def append(self,dstdata_slice):
        if len(self.queue) >= self.maxqueuelen:
            self.queue.pop(0)
        self.queue.append(dstdata_slice)
    def get_dstdata(self):
        Statedata = []
        Actiondata = []
        for i in range(len(self.queue)):
            Statedata.append(self.queue[i][0])
            Actiondata.append(self.queue[i][1])
        Statedata = np.concatenate(Statedata,axis=0)
        Actiondata = np.concatenate(Actiondata,axis=0)
        return (Statedata,Actiondata)
if __name__=='__main__':
    import time
    from dqnenv import dqn_env
    record_dirpath = "labs/qsttmixer3simple_s_hu_hz/records/hzfc_offline_5816newC/"
    current_record_dir = os.path.join(record_dirpath,time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    os.makedirs(current_record_dir,exist_ok=True)
    cfgdict = {
        "laneChange":False,
        "dir":"./data/Hangzhou/4_4/",
        "roadnetFile":"roadnet_4_4.json",
        "flowFile":"anon_4_4_hangzhou_real_5816.json",
        # "flowFile":"anon_4_4_hangzhou_real.json",
        "rlTrafficLight": True
    }

    cfgpath = net.build_cfg_custom(cfgdict,current_record_dir)
    

    traincritic_first = True
    bufferlen = 12

    traincriticnums = 1
    traincriticcollects = 2
    pretrainnums = 6

    normaltrainnums = 7
    # normaltrainepoch = 75
    # normaltraincollects = 2
    # greedystart_idx_table = [2,5,16,30,50]
    # greedyvalue_table = [0.25,0.25,0.25,0.25,0.5]
    # gamma_table = [0.4,0.8,0.8,0.8,0.8]
    # freq_table = [4,4,2,2,1]
    # greedy_default = 0.8
    # gamma_default = 0.8
    # freq_default = 1
    
    normaltrainepoch = 80
    normaltraincollects = 2
    greedystart_idx_table = [2,5,11,20,40]
    greedyvalue_table = [0.0,0.0,0.0,0.0,0.0]
    gamma_table = [0.4,0.8,0.8,0.8,0.8]
    freq_table = [4,4,4,4,4]
    
    greedy_default = 0.0
    gamma_default = 0.8
    freq_default = 2
    
    
    
    config = { 
        "device": torch.device("cuda"),
        "d_state": 6,
        "d_hidden": 24,
        "d_phasehidden": 32,
        "n_phase": 4,
        "n_select": 2,
        "clip_action_min": round(phasetime_min/phasetime_max,8),
        "clip_action_max": 1.0,
        "clip_grad_critic": 10,
        "clip_grad_actor": 10,
        "d_actionemb":8,
        "action_max":phasetime_max,


        "d_vtrans":32,
        "d_vhidden":32,
        "d_v":1,


    }
    config2 = {
        "batch_size":40,
        "train_rate":0.96,
        "eval_rate":0.04,
        "stateseq_window":1
    }
    os.makedirs(current_record_dir,exist_ok=True)
    attfile = open(os.path.join(current_record_dir,"new_summary.csv"),"w")
    attfile.write("round,att,eatt,fvc,unfvc,CompletionRate\n")
    attfile.close()
    
    env = dqn_env(config)
    # env = daddpg_env(config)
    # env.load("./model_save/daddpg_online2/2024_03_14_03_16_19") #load pretrained model
    phase_gen = phase_adapter.phase_generator_single()
    phase_gen.config_green_phases_list([1,2,3,4])
    def choose_action_epsystem(statewait:np.ndarray,phasenumber,eps:float=0.95):
        ruler = random.random()
        if ruler < eps:
            return choose_action_epfunc(statewait,phasenumber),random.randint(phasetime_min,phasetime_max)
        else:
            return phase_gen.random_phase(),random.randint(phasetime_min,phasetime_max)
    
    #-----node with nighbor
    # fulltasklist = []
    # for nodename in nodelist:
    #     currentnodefulldata = []
    #     nodewithnighbor = nodenighbors[nodename]
    #     for subnodename in nodewithnighbor:
    #         enteringlanes = cityflowlib.Rawtaskcollect_enteringlanes(subnodename,3)
    #         currentnodefulldata.extend(enteringlanes)
    #     fulltasklist.append(currentnodefulldata)
    # -----------------
        


    #node without nighbor ---------
    fulltasklist = []
    for nodename in nodelist:
        enteringlanes = cityflowlib.Rawtaskcollect_enteringlanes(nodename,3)
        fulltasklist.append(enteringlanes)
    # -----------------

    datafilterlist = [
        cityflowlib.TnodeDataFilter(fulltask) for fulltask in fulltasklist
    ]
    _,dic_lane_length = net.get_lane_length()
    from DatasetBuilder import RLdata
    import time

    DatasetBuilder = RLdata(config2)
    
    rmb = RemomeryBuffer(bufferlen)
    for datainter in range(bufferlen-traincriticcollects):
        eng = engine.Engine(cfgpath, thread_num=4)
        tq = run_queue.DACS_taskrunframeV2(180,yellow_duration=yellowtime)
        random_phaselist = [random.randint(1,4) for _ in range(nodenum)]
        init_preaction = [[i,random_phaselist[i],0,1] for i in range(nodenum)]
        cb = collectband(nodelist,phasetime_max)
        cb.action_dst = [[] for _ in range(nodenum)]
        cb.state_dst = [[] for _ in range(nodenum)]
        cb.prevaction_dst = [[] for _ in range(nodenum)]
        cb.anode_space = [{"phase":random_phaselist[i],"phaselist":[1,2,3,4],"target_runtime":0,"current_runtime":0,"currentphaseidx":[1,2,3,4].index(random_phaselist[i])} for i in range(nodenum)]

        cb.datafilterlist = datafilterlist
        tq.task_queue.task_queue[0] = init_preaction
        # tq.process_tasks_limit_finialpro_interact_customfunc(eng,3600,cb.set_signal,choose_action,choose_action_epsystem,cb.step_flowrecord,cb.funcollect_qvp,cb.extract_state,cb.extract_lastwaitstate,4,0,4,env.choose_action)
        cb.config_segmentinfo_lanelen(dic_lane_length,100) #配置车道长度
        cb.config_common_lanesnum(datafilterlist[0]) #选择一个节点的车道数量
        cb.init_flowrecord()
        tq.process_tasks_limit_finialpro(eng,3600,cb.set_signal,choose_action,cb.step,cb.funcollect_qrs)

        current_traveltime = eng.get_average_travel_time()
        print(current_traveltime)
        cb.compiledate()
        dstdata_slice,actionslice,actionprevslice = cb.extractdata_oneagent_simple()
        savedata2pklbytime(dstdata_slice,actionslice,current_traveltime,savedatadirpath)
        rmb.append((dstdata_slice,actionslice))

    if traincritic_first == True:
        trainit =  0
        for traincritic in range(traincriticnums):
            dstdata_finial = []
            traveltime = []
            for datainter in range(traincriticcollects):
                eng = engine.Engine(cfgpath, thread_num=4)
                tq = run_queue.DACS_taskrunframeV2(180,yellow_duration=yellowtime)
                random_phaselist = [random.randint(1,4) for _ in range(nodenum)]
                init_preaction = [[i,random_phaselist[i],0,1] for i in range(nodenum)]
                cb = collectband(nodelist,phasetime_max)
                cb.action_dst = [[] for _ in range(nodenum)]
                cb.state_dst = [[] for _ in range(nodenum)]
                cb.prevaction_dst = [[] for _ in range(nodenum)]
                cb.anode_space = [{"phase":random_phaselist[i],"phaselist":[1,2,3,4],"target_runtime":0,"current_runtime":0,"currentphaseidx":[1,2,3,4].index(random_phaselist[i])} for i in range(nodenum)]
                cb.datafilterlist = datafilterlist
                tq.task_queue.task_queue[0] = init_preaction

                cb.config_segmentinfo_lanelen(dic_lane_length,100) #配置车道长度
                cb.config_common_lanesnum(datafilterlist[0]) #选择一个节点的车道数量
                cb.init_flowrecord()
                tq.process_tasks_limit_finialpro_interact_customfunc(eng,3600,cb.set_signal,choose_action,choose_action_epsystem,cb.step,cb.funcollect_qrs,cb.extract_stateimdt_qrs,cb.extract_lastwaitstate_imdt,1,0,1,env.choose_action)
                current_traveltime = eng.get_average_travel_time()
                traveltime.append(current_traveltime)
                print(current_traveltime)
                cb.compiledate()
                dstdata_slice,actionslice,actionprevslice = cb.extractdata_oneagent_simple()
                savedata2pklbytime_tag(dstdata_slice,actionslice,current_traveltime,savedatadirpath)
                rmb.append((dstdata_slice,actionslice))
            print(f"criticepoch:{traincritic},traveltime:{np.mean(traveltime)}","+-({})".format(np.std(traveltime)))
            # dstdata_finial = np.concatenate(dstdata_finial,axis=0)
            dstdata_finial = rmb.get_dstdata()
            train_dataloader,eval_dataloader = DatasetBuilder.get_data_fromndarray(dstdata_finial)
            env.config_optimizer(learning_rate=0.002,weight_decay=0.0001)
            for traintimes in range(pretrainnums):
                for batch in train_dataloader:
                    batch.to_tensor(torch.device("cuda"))
                    # env.train(batch,20,trainit,0.01,0.2)
                    env.train_without_update(batch,2,trainit,0)
                    trainit+=1
                # for batch in eval_dataloader:
                #     batch.to_tensor(torch.device("cuda"))
                #     testlossres = env.test(batch,0)
                #     print("\t testloss:",testlossres)
                #     break
                env.hard_update(env.qnet,env.qnet_target)
                env.hard_update(env.pactor,env.pactor_target)
            testloss = []
            for batch in eval_dataloader:
                batch.to_tensor(torch.device("cuda"))
                testlossres = env.test(batch,0)
                testloss.append(testlossres)
                # print(testlossres)
            print(f"epoch:{traincritic},testloss:{np.mean(testloss)}","+-({})".format(np.std(testloss)))

        import time
        currenttimestr = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        savepath = "model_save/ddpg_online/"+currenttimestr
        env.save(savepath)
    # collect_timestable = [4,4,4,5,6]
    for trainepoch in range(normaltrainepoch):
        collect_times = normaltraincollects
        gamma = gamma_default
        #当trainepoch在greedystart_idx_table中一阶时，设置greedy为对应greedyvalue_table
        greedy = greedy_default
        freq = freq_default
        for idx in range(len(greedystart_idx_table)):
            if trainepoch < greedystart_idx_table[idx]:
                greedy = greedyvalue_table[idx]
                gamma = gamma_table[idx]
                freq = freq_table[idx]
                # collect_times = collect_timestable[idx]
                break
        print(f"greedy:{greedy}")
        print(f"gamma:{gamma}")
        traveltime = []
        # collect_times = 4
        ck =0
        trytime = 0
        while ck < collect_times:
            eng = engine.Engine(cfgpath, thread_num=4)
            tq = run_queue.DACS_taskrunframeV2(180,yellow_duration=yellowtime)
            random_phaselist = [random.randint(1,4) for _ in range(nodenum)]
            init_preaction = [[i,random_phaselist[i],0,1] for i in range(nodenum)]
            cb = collectband(nodelist,phasetime_max)
            cb.action_dst = [[] for _ in range(nodenum)]
            cb.state_dst = [[] for _ in range(nodenum)]
            cb.prevaction_dst = [[] for _ in range(nodenum)]
            cb.anode_space = [{"phase":random_phaselist[i],"phaselist":[1,2,3,4],"target_runtime":0,"current_runtime":0,"currentphaseidx":[1,2,3,4].index(random_phaselist[i])} for i in range(nodenum)]
            cb.datafilterlist = datafilterlist
            tq.task_queue.task_queue[0] = init_preaction
            cb.config_segmentinfo_lanelen(dic_lane_length,100) #配置车道长度
            cb.config_common_lanesnum(datafilterlist[0]) #选择一个节点的车道数量
            cb.init_flowrecord()
            # tq.process_tasks_limit_finialpro_interact(eng,3600,cb.set_signal,choose_action,cb.step,cb.funcollect_qvcrps_rev,cb.extract_state,4,greedy,4,env.choose_action)
            tq.process_tasks_limit_finialpro_interact_customfunc(eng,3600,cb.set_signal,choose_action,choose_action_epsystem,cb.step,cb.funcollect_qrs,cb.extract_stateimdt_qrs,cb.extract_lastwaitstate_imdt,1,greedy,1,env.choose_action)
            current_traveltime = eng.get_average_travel_time()

            traveltime.append(current_traveltime)
            print(current_traveltime)
            cb.compiledate()
            dstdata_slice,actionslice,actionprevslice = cb.extractdata_oneagent_simple()
            savedata2pklbytime_tag(dstdata_slice,actionslice,current_traveltime,savedatadirpath)
            rmb.append((dstdata_slice,actionslice))
            ck+=1
        print(f"epoch:{trainepoch},traveltime:{np.mean(traveltime)}","+-({})".format(np.std(traveltime)))
        # dstdata_finial = np.concatenate(dstdata_finial,axis=0)
        dstdata_finial = rmb.get_dstdata()
        train_dataloader,eval_dataloader = DatasetBuilder.get_data_fromndarray(dstdata_finial)
        trainit = 0
        env.config_optimizer_idp(lr_actor=0.0001,
                                 lr_critic=0.0001,
                                 weight_decay=0.0001)
        
        for traintimes in range(normaltrainnums):
            for batch in train_dataloader:
                batch.to_tensor(torch.device("cuda"))
                # env.train(batch,10,trainit,gamma,0.2)
                env.train_without_update(batch,freq,trainit,gamma)
                trainit += 1
            # for batch in eval_dataloader:
            #     batch.to_tensor(torch.device("cuda"))
            #     testlossres = env.test(batch,gamma)
            #     print("\t testloss:",testlossres)
            #     break
            env.hard_update(env.qnet,env.qnet_target)
            env.hard_update(env.pactor,env.pactor_target)
        testloss = []
        for batch in eval_dataloader:
            batch.to_tensor(torch.device("cuda"))
            testlossres = env.test(batch,gamma)
            testloss.append(testlossres)
            # print(testlossres)
        print(f"epoch:{trainepoch},testloss:{np.mean(testloss)}","+-({})".format(np.std(testloss)))
        print("trainint",trainit)
        for datainter in range(1):
            eng = engine.Engine(cfgpath, thread_num=4)
            tq = run_queue.DACS_taskrunframeV2(180,yellow_duration=yellowtime)
            random_phaselist = [random.randint(1,4) for _ in range(nodenum)]
            init_preaction = [[i,random_phaselist[i],0,1] for i in range(nodenum)]
            cb = collectband(nodelist,phasetime_max)
            cb.action_dst = [[] for _ in range(nodenum)]
            cb.state_dst = [[] for _ in range(nodenum)]
            cb.prevaction_dst = [[] for _ in range(nodenum)]
            cb.anode_space = [{"phase":random_phaselist[i],"phaselist":[1,2,3,4],"target_runtime":0,"current_runtime":0,"currentphaseidx":[1,2,3,4].index(random_phaselist[i])} for i in range(nodenum)]
            cb.datafilterlist = datafilterlist
            tq.task_queue.task_queue[0] = init_preaction
            cb.config_segmentinfo_lanelen(dic_lane_length,100) #配置车道长度
            cb.config_common_lanesnum(datafilterlist[0]) #选择一个节点的车道数量
            cb.init_flowrecord()
            tq.process_tasks_limit_finialpro_interact(eng,3600,cb.set_signal,choose_action,cb.step,cb.funcollect_qrs,cb.extract_stateimdt_qrs,1,greedy-0.08,1,env.choose_action)
            current_traveltime = eng.get_average_travel_time()
            print(current_traveltime)
            cb.compiledate()
            dstdata_slice,actionslice,actionprevslice = cb.extractdata_oneagent_simple()
            savedata2pklbytime_tag(dstdata_slice,actionslice,current_traveltime,savedatadirpath)
            # dstdata_finial.append(dstdata_slice)
            rmb.append((dstdata_slice,actionslice))
        for datainter in range(1):
            eng = engine.Engine(cfgpath, thread_num=4)
            tq = run_queue.DACS_taskrunframeV2(180,yellow_duration=yellowtime)
            random_phaselist = [random.randint(1,4) for _ in range(nodenum)]
            init_preaction = [[i,random_phaselist[i],0,1] for i in range(nodenum)]
            cb = collectband(nodelist,phasetime_max)
            cb.action_dst = [[] for _ in range(nodenum)]
            cb.state_dst = [[] for _ in range(nodenum)]
            cb.prevaction_dst = [[] for _ in range(nodenum)]
            cb.anode_space = [{"phase":random_phaselist[i],"phaselist":[1,2,3,4],"target_runtime":0,"current_runtime":0,"currentphaseidx":[1,2,3,4].index(random_phaselist[i])} for i in range(nodenum)]
            cb.datafilterlist = datafilterlist
            tq.task_queue.task_queue[0] = init_preaction
            cb.config_segmentinfo_lanelen(dic_lane_length,100) #配置车道长度
            cb.config_common_lanesnum(datafilterlist[0]) #选择一个节点的车道数量
            cb.init_flowrecord()
            tq.process_tasks_limit_finialpro_interact(eng,3600,cb.set_signal,choose_action,cb.step,cb.funcollect_qrs,cb.extract_stateimdt_qrs,1,1,1,env.choose_action)
            current_traveltime = eng.get_average_travel_time()
            eatt = eng.get_ended_average_travel_time()
            fvc = eng.get_finished_vehicle_count()
            ufvc = eng.get_unfinished_vehicle_count()
            attfile = open(os.path.join(current_record_dir,"new_summary.csv"),"a")
            attfile.write("{0},{1},{2},{3},{4},{5}\n".format(trainepoch,current_traveltime,eatt,fvc,ufvc,fvc/(fvc+ufvc)))
            attfile.close()

            print("greedy1:",current_traveltime)
            cb.compiledate()
            dstdata_slice,actionslice,actionprevslice = cb.extractdata_oneagent_simple()
            savedata2pklbytime_tag(dstdata_slice,actionslice,current_traveltime,savedatadirpath)
            # dstdata_finial.append(dstdata_slice)
            rmb.append((dstdata_slice,actionslice))
        if (trainepoch+1) % 5 == 0:
            
            currenttimestr = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            savepath = "model_save/mpddpg_onlineSTD/"+currenttimestr

            
            env.save(savepath)
        #查看eng平均旅行时间
        
