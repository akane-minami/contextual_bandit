# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# coding: shift_jis

"""実世界データを用いたシミュレーションをおこなうモジュール"""
import os
import time
from typing import List

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from policy.base_policy import BaseContextualPolicy
from bandit.base_bandit import BaseBandit

from bandit.contextual_bandit import ContextualBandit
from realworld.setup_context import ContextData

import codecs

class ContextualBanditSimulator(object):
  """指定されたアルゴリズムで文脈付きバンディットを実行する

  Args:
    context_dim(int): 特徴量の次元数
    num_actions(int): 行動数
    dataset(int, float): 行はデータ数, 列は特徴量 + 各行動の報酬 が入っている 
    algos(list of str): 使用するアルゴリズムのリスト
  """
  def __init__(self, policy_list: List[BaseContextualPolicy],
                 bandit: BaseBandit, n_sims: int, n_steps: int, n_arms: int,
                 n_features: int, data_type) -> None:
    """クラスの初期化
    Args:
      policy_list(list of str) : 方策リスト
      bandit : bandit環境
      n_sims(int) : sim数
      n_steps(int) : step数
      n_arms(int) : アーム数(行動数)
      n_features(int) : 特徴量の次元数
      data_type(str) : dataset名
    """

    self.policy_list = policy_list
    self.bandit = bandit
    self.n_sims = n_sims
    self.n_steps = n_steps
    self.n_arms = n_arms
    self.n_features = n_features
    self.data_type = data_type
    self.policy_name = []
    self.result_list = []

  def generate_feature_data(self) -> None:
    """加工したdataset・最適行動を取った時の報酬・最適行動・報酬期待値を返す"""
    dataset, opt_rewards, opt_actions, exp_rewards = ContextData.sample_data(self.data_type, self.n_steps)
    return dataset, opt_rewards, opt_actions,exp_rewards

  def processing_data(self, rewards,regrets,accuracy,times):
    """データを平均,最小値,最大値に処理して返す"""
    result_data = np.concatenate(([rewards], [regrets], [accuracy],[times]),axis=0)
    min_data = result_data.min(axis=1)
    max_data = result_data.max(axis=1)
    ave_data = np.sum(result_data, axis=1) / self.n_sims

    return ave_data[0], ave_data[1], ave_data[2],ave_data[3],min_data, max_data

  def run_sim(self):
    """方策ごとにシミュレーションを回す"""
    cmab = self.bandit
    """方策ごとに実行"""
    for policy in self.policy_list:
        print(policy.name)
        self.policy_name.append(policy.name)
        """結果表示に用いる変数の初期化"""
        rewards = np.zeros((self.n_sims, self.n_steps), dtype=float)
        regrets = np.zeros((self.n_sims, self.n_steps), dtype=float)
        accuracy = np.zeros((self.n_sims, self.n_steps), dtype=int)
        times = np.zeros((self.n_sims, self.n_steps), dtype=float)
        elapsed_time = 0.0

        start_tmp = time.time()
        """シミュレーション開始"""
        for sim in np.arange(self.n_sims):
            print('{} : '.format(sim), end='')
            dataset, opt_rewards, opt_actions,exp_rewards = self.generate_feature_data()#加工したデータセットを持ってくる
            cmab.feed_data(dataset)#1シミュレーションごとにデータの中から必要なぶんだけ取り出し(初期化も兼ねてる)
            policy.initialize()
            """初期化"""
            elapsed_time = 0.0
            sum_reward, sum_regret = 0.0, 0.0
            sum_error,rmse = 0.0, 0.0
            error = 0.0
            """step開始"""
            for step in np.arange(self.n_steps):
                start=time.time()
                x = cmab.context(step)#特徴量のみ持ってくる
                chosen_arm = policy.choose_arm(x,step)

                reward = cmab.reward(step, chosen_arm)
                regret = opt_rewards[step] - reward
                success_acc = 1 if chosen_arm == opt_actions[step] else 0#真のgreedy(Accuracy)

                policy.update(x, chosen_arm, reward)#方策アップデート

                sum_reward += reward
                sum_regret += regret
                
                rewards[sim, step] += sum_reward
                regrets[sim, step] += sum_regret
                accuracy[sim, step] += success_acc
                elapsed_time += time.time()-start
                times[sim,step] +=elapsed_time

            print('{}'.format(regrets[sim, -1]))

        elapsed_time_tmp = time.time() - start_tmp
        print("経過時間 : {}".format(elapsed_time_tmp))

        ave_rewards, ave_regrets, accuracy,ave_times,min_data,max_data = \
           self.processing_data(rewards, regrets, accuracy,times)
        data = [ave_rewards, ave_regrets, accuracy,ave_times,min_data, max_data]

        data_dic = \
            {'rewards': data[0], 'regrets': data[1], 'accuracy': data[2],'times':data[3],
             'min_rewards': data[4][0], 'min_regrets': data[4][1],'min_accuracy': data[4][2],'min_times':data[4][3],
             'max_rewards': data[5][0],'max_regrets': data[5][1], 'max_accuracy': data[5][2],'max_times':data[5][3]}

        print('rewards: {0}\nregrets: {1}\naccuracy: {2}\ntimes: {3}\n'
              'min_rewards: {4}\nmin_regrets: {5}\nmin_accuracy: {6}\nmin_times:{7}\n'
              'max_rewards: {8}\nmax_regrets: {9}\nmax_accuracy: {10}\nmax_times:{11}'
              .format(data_dic['rewards'], data_dic['regrets'],data_dic['accuracy'],data_dic['times'],
                      data_dic['min_rewards'],data_dic['min_regrets'], data_dic['min_accuracy'],data_dic['min_times'],
                      data_dic['max_rewards'], data_dic['max_regrets'],data_dic['max_accuracy'],data_dic['max_times']))

        self.result_list.append(data_dic)
        self.dy = np.gradient(data[1])
        print('傾き: ', self.dy)
    self.result_list = pd.DataFrame(self.result_list)
    

  def run(self) -> None:
        """一連のシミュレーションを実行"""
        self.run_sim()
        self.plots()

  def plots(self) -> None:
        """結果データのプロット"""
        mpl.rcParams['axes.xmargin'] = 0
        mpl.rcParams['axes.ymargin'] = 0

        for i, data_name in enumerate(['rewards', 'regrets','accuracy']):
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)

            for j, policy_name in enumerate(self.policy_name):
                cmap = plt.get_cmap("tab10")
                if data_name =='accuracy':
                    """通常ver"""
                    #ax.plot(np.linspace(1, self.n_steps, num=self.n_steps), self.result_list.at[j, data_name], label=policy_name,linewidth=1.5, alpha=0.8)
                    """移動平均ver"""
                    b = np.ones(10)/10.0
                    y3 = np.convolve(self.result_list.at[j, data_name], b, mode='same')
                    ax.plot(np.linspace(1, self.n_steps, num=self.n_steps), y3, label=policy_name+"moving_average", color=cmap(j), linewidth=1.5, alpha=0.8)
                    #ax.set_ylim([0.8, 1.2])
                else:
                    ax.plot(np.linspace(1, self.n_steps, num=self.n_steps), self.result_list.at[j, data_name], label=policy_name, linewidth=1.5, alpha=0.8)
                    #ax.fill_between(x=np.linspace(1, self.n_steps, num=self.n_steps), y1=self.result_list.at[j, 'min_'+data_name], y2=self.result_list.at[j, 'max_'+data_name], alpha=0.1)
            ax.set_xlabel('steps',fontsize = 14)
            ax.set_ylabel(data_name,fontsize = 14)
            leg = ax.legend(loc = 'upper left', fontsize = 23)
            plt.tick_params(labelsize = 10)
            ax.grid(axis = 'y')

            plt.show()
            path = os.getcwd()#現在地
            results_dir = os.path.join(path, 'png/')#保存場所
            fig.savefig(results_dir + data_name, bbox_inches='tight', pad_inches=0)

        plt.clf()
