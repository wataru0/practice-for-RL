# pytorchでdqnの実装2
# こっちのほうがスマート
# 2/14
# https://book.mynavi.jp/manatee/detail/id=89498 : 第12回から見る
# https://book.mynavi.jp/manatee/detail/id=89831
# https://schemer1341.hatenablog.com/entry/2019/05/04/002300

import gym 
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn # ネットワーク構築用
import torch.nn.functional as F # ネットワーク用の様々な関数
import torch.optim as optim # 最適化関数
from torch.autograd import Variable # 自動微分用
import copy

import argparse
import datetime
import random
import time
from gym import wrappers

config = {
    #'env':'CartPole-v1',
    'env':"Breakout-v0",

}

env = gym.make(config['env'])
obs_num = env.observation_space.shape[0]
act_num = env.action_space.n

def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video',default=False,action='store_true')

    return parser.parse_args()

class NN(nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(obs_num,config['HIDDEN_SIZE'])
        self.fc2 = nn.Linear(confg['HIDDEN_SIZE'],confg['HIDDEN_SIZE'])
        self.fc3 = nn.Linear(confg['HIDDEN_SIZE'],confg['HIDDEN_SIZE'])
        self.fc4 = nn.Linear(confg['HIDDEN_SIZE'],act_num)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        y = F.relu(self.fc4(x))

        return y


if __name__ == "__main__":
    args = arg_parser()
    env = gym.make(config['env'])
    print("observation space :",env.observation_space.shape[0])
    print("action space :",env.action_space.n)

    # 動画出力
    if args.video:
        env = wrappers.Monitor(env,"./videos/" + config['env'] +  "-" + datetime.datetime.now().isoformat(),force=True,video_callable=(lambda ep: ep % 1 == 0))
    