""" dataを加工するモジュール"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from absl import app
from absl import flags
import numpy as np
import os
import tensorflow as tf
import pandas as pd

base_route = os.getcwd()
data_route = 'datasets'

def one_hot(df, cols):
  """特徴量をone_hotベクトルに直す
  Args:
    cols(pandas.core.indexes.base.Index):特徴(edible,cap-shape...)

  Returns:
    df(pandas.core.frame.DataFrame):特徴量をone-hotベクトルに変換したもの

  Example:
    >>> df = one_hot(df, df.columns)
    >>> print(df.iloc(0, ))

    edible_e               0
    edible_p               1
    cap-shape_b            0
    cap-shape_c            0
    cap-shape_f            0
    cap-shape_k            0
    cap-shape_s            0
    cap-shape_x            1
    .
    .
    .
  """
  for col in cols:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(col, axis=1)
  return df
  

def sample_mushroom_data(num_contexts,
                         r_noeat=0,
                         r_eat_safe=5,
                         r_eat_poison_bad=-35,
                         r_eat_poison_good=5,
                         prob_poison_bad=0.5):

    """mushroomデータセットの加工
    Args:
      num_contexts(int):データの数
      r_noeat(int):食べなかった時の報酬
      r_eat_safe(int):食用キノコを食べた時の報酬
      r_eat_poison_bad(int):毒キノコを食べた時の負の報酬
      r_eat_poison_good(int):毒キノコを食べた時の正の報酬
      prob_poison_bad(float):毒キノコを食べた時に負の報酬になる確率

    Returns:
      np.hstack((contexts, no_eat_reward, eat_reward)):加工後のデータセット,食べなかった・食べた時の報酬
      opt_vals(float):各データの最適な行動を選択した時の報酬
      exp_rewards(float):各データの報酬
    """

    path = os.path.join(base_route, data_route, 'mushroom.csv')

    df = pd.read_csv(path)
    df = one_hot(df, df.columns)
    ind = np.random.choice(range(df.shape[0]), num_contexts, replace=True)
    contexts = df.iloc[ind, 2:]
    exp_rewards =[[r_noeat,r_noeat],[(r_eat_poison_bad + r_eat_poison_good)*prob_poison_bad,r_eat_safe]]

    # キノコの報酬を設定する
    no_eat_reward = r_noeat * np.ones((num_contexts, 1))
    random_poison = np.random.choice(
          [r_eat_poison_bad, r_eat_poison_good],
          p=[prob_poison_bad, 1 - prob_poison_bad],
          size=num_contexts)
    eat_reward = r_eat_safe * df.iloc[ind, 0]
    eat_reward += np.multiply(random_poison, df.iloc[ind, 1])
    eat_reward = eat_reward.values.reshape((num_contexts, 1))

    # 最適な期待報酬と最適な行動を計算
    exp_eat_poison_reward = r_eat_poison_bad * prob_poison_bad
    exp_eat_poison_reward += r_eat_poison_good * (1 - prob_poison_bad) 
    opt_exp_reward = r_eat_safe * df.iloc[ind, 0] + max(
      r_noeat, exp_eat_poison_reward) * df.iloc[ind, 1]

    if r_noeat > exp_eat_poison_reward:
        # actions: no eat = 0 ; eat = 1
        opt_actions = df.iloc[ind, 0]
    else:
        opt_actions = np.ones((num_contexts, 1))
    opt_vals = (opt_exp_reward.values, opt_actions.values)

    return np.hstack((contexts, no_eat_reward, eat_reward)), opt_vals, exp_rewards