import copy

import numpy as np
from constant_params import *
# from MMPPO import  conduct,Memory
from MMPPO import *
import torch
from roles import *
# from queue import  Queue

action_std = 0.5  # constant std for action distribution (Multivariate Normal)
K_epochs = 80  # update policy for K epochs
eps_clip = 0.2  # clip parameter for PPO
gamma = 0.995  # discount factor

lr = 0.0001  # parameters for Adam optimizer
betas = (0.9, 0.999)
state_dim = 4
action_dim = 3

model_total_id = 0


class Offspring:
    def __init__(self,ppo,w):
        self.ppo = ppo
        self.w = w
        self.r = np.array([0.0,0.0,0.0])


class Evolutionary:
    def __init__(self):
        ## generate n tasks
        w1 = [np.random.uniform(0,1) for i in range(n_tasks)]
        w2 = [np.random.uniform(0,1-v) for v in w1]
        w3 = [1- w1[i] - w2[i] for i in range(n_tasks)]
        self.w = [[w1[i],w2[i],w3[i]] for i in range(n_tasks)]

        self.curr_model_list = [PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip) for i in range(n_tasks)]

        self.offspring_list = []
        self.EP_list = [[] for i in range(n_tasks)]
        self.model_list = [Offspring(self.curr_model_list[i],self.w[i]) for i in range(n_tasks)]

    def process_task_once_get_reward(self,ppo,node_list,base_station):
        uav = UAV(node_list,base_station)
        # uav.node_list = node_list
        # uav.base_station = base_station
        # uav.x_pos = 200
        # uav.y_pos = 200
        running_reward = np.array([0.0, 0.0, 0.0])
        memory = Memory()
        for t in range(TIME_SLOTS):
            state = torch.Tensor(uav.get_obs())
            action = ppo.select_action(state, memory)
            reward = uav.step(action)
            running_reward += reward

        # print(running_reward)
        return running_reward


    def TPU(self,node_list,base_station):
        def generate_weight_Pnum():
            pw1 = [np.random.uniform(0, 1) for i in range(Pnum)]
            pw2 = [np.random.uniform(0, 1 - v) for v in pw1]
            pw3 = [np.random.uniform(0, 1 - pw1[i] - pw2[i]) for i in range(Pnum)]
            pw = [[pw1[i], pw2[i], pw3[i]] for i in range(Pnum)]
            return  np.array(pw)


        pw = generate_weight_Pnum()
        Pnum_queue = [[] for i in range(Pnum)]
        total_models = []
        for model in self.model_list:
            total_models.append(model)
        for model in self.offspring_list:
            total_models.append(model)

        for model in total_models:
            ppo = model.ppo
            reward = self.process_task_once_get_reward(ppo,node_list,base_station)
            if reward[0] == 0 and reward[1] == 0 and reward[2] == 0:  #促使UAV不做空转，什么都不做的奖励值最小,给予一个-inf
                reward = [-np.inf,-np.inf,0]
            model.r = reward
            max_value = 0
            index = 0
            for i,w in enumerate(pw):
                coef_reward = reward[0] * pw[i][0] + reward[1] * pw[i][1] + reward[2] * pw[i][2]
                if i == 0:
                    max_value = coef_reward
                else:
                    if coef_reward > max_value:
                        max_value = coef_reward
                        index = i

            if  len(Pnum_queue[index]) != Psize:
                Pnum_queue.append([model,max_value])
            else:
                for i in range(Psize):
                    if max_value > Pnum_queue[i][1]:
                        Pnum_queue[i] = [model,max_value]
        self.model_list.clear()
        for model_tuple in Pnum_queue:
            if len(model_tuple) != 0:
                model = model_tuple[0]
                self.model_list.append(model)

    def EP_process(self):

        def dominate(model1,model2):
            m1dm2 = 0
            m2dm1 = 0

            r1 = model1.r
            r2 = model2.r

            for i in range(3):
                if r1[i] > r2[i]:
                    m1dm2 += 1
                elif r2[i] > r1[i]:
                    m2dm1 += 1

            if m1dm2 == 3:
                return model2
            elif m2dm1 == 3:
                return model1
            else:
                return None


        temp_list = [[] for i in range(n_tasks)]
        for model in self.offspring_list:
            w = model.w
            for index,task in enumerate(self.w):
                if w == task:
                    temp_list[index].append(model)


        for index,model_list in enumerate(temp_list):
            if len(model_list) != 0:
                for model in model_list:
                    EP_origin = self.EP_list[index]
                    if len(EP_origin) == 0:
                        EP_origin.append(model)
                    else:
                        delete_index = []
                        domine = False
                        done = False
                        while not domine and not done:
                            for i,origin in enumerate(EP_origin):
                                dominate_model = dominate(model,origin)

                                if dominate_model == None: #not dominate continue
                                    # print("none")
                                    continue
                                elif dominate_model == model: # model is dominated
                                    # print("model")
                                    domine = True
                                    break
                                else:                                # origin model is dominated,dominated model need be deleted
                                    # print("origin")
                                    delete_index.append(i)
                            done = True

                            if len(delete_index) != 0:
                                update_EP = []
                                for i in range(len(EP_origin)):
                                    if delete_index.count(i) == 0:
                                        update_EP.append(EP_origin[i])
                                self.EP_list[index] = update_EP

                            if not domine:
                                self.EP_list[index].append(model)

    def re_allocation_task(self):
        temp_model_list = []
        for w in self.w:
            max_reward = 0
            candidate = None
            for model in self.model_list:
                r = model.r
                coef_reward = r[0] * w[0] + r[1] * w[1] + r[2] * w[2]
                if candidate == None or coef_reward > max_reward:
                    candidate = model
                    max_reward = coef_reward

            candidate = copy.deepcopy(candidate)
            candidate.w = w
            temp_model_list.append(candidate)

        self.model_list.clear()
        self.model_list = temp_model_list

    def check_paroto_space(self):
        # temp_list = [[] for i in range(n_tasks)]
        for index,model_list in enumerate(self.EP_list):
            w = self.w[index]
            print("task (%f,%f,%f) space: " % (w[0],w[1],w[2]),"current number model with task: %d" % (len(model_list)))
            for model in model_list:
                r = model.r
                reward = r[0] * w[0] + r[1] * w[1] + r[2] * w[2]
                print("pareto reward: ",reward)


    def evolution(self):
        base_station = Node() # base_station not move
        node_list = [Node() for i in range(K)]

        """
                   未解决：添加warm-up过程，初始化
                   考虑解决办法：在原有的迭代次数基础上增加迭代次数达到warm-up的效果

        """
        for iter in range(Gmax):

            print("----------------------------------------------------------")
            print("----------------------------------------------------------")
            print("evolution iter: ",iter)
            self.offspring_list.clear()  # clear offspring list
            print("MMPPO generate offspring ...")
            for index,model in enumerate(self.model_list):
                print("model_%d conduct MMPPO..." % (index+1))
                self.offspring_list.append(copy.deepcopy(model))
                model.w = self.w[index]
                conduct(model.ppo,model.w,self.offspring_list,node_list,base_station)
            print("conduct TPU to get new population ...")
            self.TPU(node_list,base_station)
            print("conduct EP to get pareto space ...")
            self.EP_process()
            self.check_paroto_space()
            print("reassign task ...")
            self.re_allocation_task()





if __name__ == '__main__':
    np.random.seed(3)
    evo = Evolutionary()
    evo.evolution()
















