"""データのセットアップをおこなうモジュール"""
from realworld.data_sampler import sample_mushroom_data

class ContextData(object):
    """特徴のデータの基本情報をセットするクラス"""
    def sample_data(data_type, num_contexts=None):
        """データセットの最適行動、報酬期待値をセット
        Args:
            data_type(str):データセット名
            num_contexts(int):データ数
        Returns:
            dataset:加工済みデータ
            opt_rewards(int, float):最適報酬
            opt_actions(int):最適行動
            exp_rewards(int, float):報酬
        Raises:
            DATA NAME ERROR: data_typeがどれも当てはまらない場合
        """

        if data_type == 'mushroom':
            dataset, opt_mushroom, exp_rewards = sample_mushroom_data(num_contexts)
            opt_rewards, opt_actions = opt_mushroom
        else:
            print("DATA NAME ERROR.")

        return dataset, opt_rewards, opt_actions, exp_rewards

    def get_data_info(data_type):
        """dataの基本情報の取得
        Returns:
            num_actions(int):行動数
            context_dim(int):特徴量の次元
        """
        if data_type == 'mushroom':
            num_actions = 2
            context_dim = 117
        else:
            print("DATA NAME ERROR.")
        
        return num_actions, context_dim
