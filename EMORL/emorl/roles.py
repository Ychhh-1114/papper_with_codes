import numpy as np
from constant_params import *
from queue import Queue
import math

class Node:
    def __init__(self):
        self.x_pos = np.random.randint(0,X_MAX)
        self.y_pos = np.random.randint(0,Y_MAX)
        self.task_queue = 0

    def gene_task(self):
        gene = np.random.random()
        if gene > 0.5 and self.task_queue != MAX_TASKS:
            self.task_queue += 1



class UAV:
    def __init__(self,node_list,base_station):

        self.x_pos = 200#np.random.randint(0, X_MAX)
        self.y_pos = 200#np.random.randint(0, Y_MAX)
        self.wait_queue = 0  # current N_tu

        ## action space
        self.dis = np.random.randint(0, X_MAX)
        self.angle = (np.random.randint(0,MAX_ANGLE)/180.0)
        self.bt = np.random.random()
        ##############################
        self.curr_Dto = 0      # D_to
        self.curr_slot = 0
        self.curr_collect = 0  # N_tc
        self.curr_delay = 0    # D_t
        self.curr_energy = 0   # E_t

        # assist calculate delay
        self.trans_list = []

        self.velo = self.dis / time_slot # UAV velocity
        self.acc_propulsion_energy = 0 # E_fly
        self.acc_local_energy = 0      # accumulate E_tl
        self.acc_task_collection = 0   #accumulate N_tc

        self.base_station = base_station    # node as basestation

        self.acc_computation_delay = 0  # accumulate local computation delay

        self.node_list = node_list  # node list as SDs

        self.acc_reward = np.array([0.0,0.0,0.0])


    def step(self,action):
        self.angle = action[0]
        self.dis = action[1]
        self.bt =  action[2]

        self.move()

        rt_D = -self.curr_delay
        rt_E = -0.01 * self.curr_energy
        rt_N = self.curr_collect

        if self.x_pos == X_MAX or self.y_pos == Y_MAX:
            rt_D = 4 * rt_D
            rt_E = 4 * rt_E
            rt_N = -2 * rt_N

        self.acc_reward[0] += rt_D
        self.acc_reward[1] += rt_E
        self.acc_reward[2] += rt_N

        return np.array([rt_D,rt_E,rt_N])




    def move(self): #each move conduct once

        ## each node updat task queue
        for node in self.node_list:
            node.gene_task()

        self.x_pos = np.clip(self.x_pos + self.dis * np.cos(self.angle * np.pi),0,X_MAX)
        self.x_pos = np.clip(self.y_pos + self.dis * np.sin(self.angle * np.pi),0,Y_MAX)
        self.velo = self.dis / time_slot
        self.curr_slot += 1

        # each move need solve tasks

        self.move_collect_task()  # collect task
        self.move_delay()         # count delay
        self.move_energy()        # calculate energy consume




    def propulsion_power(self):
        cal1 = P1 * (1 + (3*(self.velo**2)/(U_tip**2)))
        cal2 = P2 * np.sqrt((np.sqrt(1 + np.power(self.velo,4)/(4*np.power(v0,4))) - (self.velo**2)/(2*(v0**2))))
        cal3 = 0.5 * d0 * row * g * A * np.power(self.velo,3)

        return cal1 + cal2 + cal3

    # def move_energy(self): #
    #     self.acc_propulsion_energy += self.propulsion_power() * time_slot

    def move_local_computation_enegy(self):
        N_to = self.bt * self.wait_queue
        self.trans_list.append(N_to)

        N_tl = self.wait_queue - N_to
        fai = (time_slot * f_U)/beta
        return k * min(N_tl,fai) * beta * (f_U**2)

    def move_cal_mu(self):
        def cal_rt():
            x_dis = self.x_pos - self.base_station.x_pos
            y_dis = self.y_pos - self.base_station.y_pos
            dt_UB = np.sqrt(x_dis ** 2 + y_dis ** 2)
            angle_UB = abs(math.degrees(math.atan(y_dis / x_dis)))


            PL = 10 * A0 * np.log(dt_UB) + B0 * (angle_UB - theta0) * np.exp((theta0 - angle_UB) / C0) + eta0

            return PU * (np.power(10, PL / 10)) / sigma2

        mu_t = W * np.log(1 + cal_rt())
        return mu_t

    def move_delay(self):
        """
            未解决的问题：
                对于未来的延时无法在当前的时刻进行正确的判断，即miu_ti如何在在当前的时刻进行计算计算？
                可能得解决方法：
                    在agent与智能体交互的过程中，不在每一个时间步给出奖励，但是记录此刻的带宽，在一个episode完成时，
                    反向通过记录的带宽数据计算出传递数据需要的时延？
            临时解决方法：
                对于未来的UAV-BASESTATION带宽采用当前时刻的带宽进行处理，即：
                设每个时间步的带宽为miu_t,计算出当前时刻的带宽miu_t:对于ti > t,i = t+1,t+2,...有：
                        miu_ti = miu_t
        """

        mu = self.move_cal_mu()
        N_to = self.bt * self.wait_queue
        sum_data = 0
        fai_t = 1
        done = False
        i = self.curr_slot

        while not done:
            if i >= TIME_SLOTS:
                i = TIME_SLOTS - 1
            sum_data += time_slot * mu
            if sum_data < alpha * N_to:
                fai_t += 1
            else:
                done = True
            i += 1
        if sum_data > alpha * N_to:
            sum_before = sum_data - mu
            delay_t = (fai_t - 1) * time_slot + (alpha * N_to - sum_before) / (mu + fai_t)
        else:
            delay_t = time_slot * fai_t

        fai = (time_slot * f_U) / beta
        N_tq = max(self.wait_queue - fai - self.bt * self.wait_queue, 0)
        N_tl = self.wait_queue - self.bt * self.wait_queue
        self.curr_Dto = delay_t
        self.curr_delay = delay_t + ((min(fai,N_tl) * beta)/f_U) + time_slot * N_tq


    def trans_energy(self):
        return self.delay_transmission() * PU

    def move_energy(self):
        self.curr_energy = self.curr_Dto * PU + self.move_local_computation_enegy()

    def move_computation_delay(self):
        fai = (time_slot * f_U) / beta
        N_tq = np.max(self.wait_queue - fai - self.bt * self.wait_queue,0)
        N_tl = self.wait_queue - self.bt * self.wait_queue


        self.acc_computation_delay +=  (np.min(fai,N_tl) * beta)/f_U + time_slot * N_tq


    def move_collect_task(self):

        def cal_dis(node):
            x_dis = self.x_pos - node.x_pos
            y_dis = self.y_pos - node.y_pos
            return np.sqrt(x_dis**2 + y_dis**2)


        Rmax = H * np.tan(MAX_ANGLE)
        N_tc = 0
        for node in self.node_list:
            dis = cal_dis(node)
            if dis <= Rmax:
                N_tc += node.task_queue
                node.task_queue = 0
        fai = (time_slot * f_U) / beta
        N_tq = max(self.wait_queue - fai - self.bt * self.wait_queue,0)

        self.wait_queue = min(N_tq + N_tc,MAX_TASKS)
        self.acc_task_collection += N_tc
        self.curr_collect = N_tc

    def get_obs(self):
        return self.x_pos,self.y_pos,self.wait_queue,self.curr_collect

















