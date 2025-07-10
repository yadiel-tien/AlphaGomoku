import os
import pickle
import random

from config import CONFIG
from gomoku import Gomoku
from inference import make_engine
from player import AIServer
from train import get_logger


def rating1v1(a, b, score_a=0.0):
    # 玩家 A 的预期得分
    e_a = 1 / (1 + 10 ** ((b.scores - a.scores) / 400))

    # 玩家 B 的实际得分
    score_b = 1 - score_a
    # 玩家 B 的预期得分
    e_b = 1 - e_a
    # 计算新评分
    a.scores += a.k * (score_a - e_a)
    b.scores += b.k * (score_b - e_b)


class Elo:
    def __init__(self, index, initial_score=1500.0):
        self.index = index
        self.scores = initial_score
        self.games_played = 0
        self.records = {}  # {rival_index:[win,lose,draw]}

    @property
    def k(self):
        """根据对弈局数动态获取 K 值"""
        if self.games_played < 50:  # 假设前50局是新手期
            return 32
        elif self.games_played < 200:  # 50-200局是中等活跃期
            return 24
        else:  # 200局以上是稳定期
            return 16

    def defeat(self, rival):
        self.games_played += 1
        rival.games_played += 1
        if rival.index not in self.records:
            self.records[rival.index] = [0] * 3
        self.records[rival.index][0] += 1
        if self.index not in rival.records:
            rival.records[self.index] = [0] * 3
        rival.records[self.index][1] += 1
        rating1v1(self, rival, 1)

    def draw(self, rival):
        self.games_played += 1
        rival.games_played += 1
        if rival.index not in self.records:
            self.records[rival.index] = [0] * 3
        self.records[rival.index][2] += 1
        if self.index not in rival.records:
            rival.records[self.index] = [0] * 3
        rival.records[self.index][2] += 1
        rating1v1(self, rival, 0.5)

    def __str__(self):
        return f'index:{self.index},score:{self.scores:.2f},games:{self.games_played}'

    def show_records(self):
        for k, v in self.records.items():
            print(f'VS {k},win:{v[0]},loss:{v[1]},draw:{v[2]},win_rate:{v[0] / sum(v):.2%}.')


class Arena:
    def __init__(self, path):
        self.rates = {}
        self.infer1 = make_engine(0)
        self.infer2 = make_engine(0)
        self.logger = get_logger('arena')
        self.file_path = path
        self.load(path)

    def run(self, n_games, env):
        for i in range(n_games):
            model_indices = [i for i in self.rates.keys()]
            # 根据比赛次数多少随机，次数越多，选中概率越低
            weights = [1 / (self.rates[i].games_played + 1) for i in model_indices]
            chosen_index = random.choices(model_indices, weights=weights, k=1)[0]
            # 选择得分相近的对手，权重为1/abs(a-b)
            diff_weight = [1 / abs(r.scores - self.rates[chosen_index].scores) for r in self.rates.values() if
                           r != self.rates[chosen_index]]
            model_indices.remove(chosen_index)
            chosen_rival_index = random.choices(model_indices, weights=diff_weight, k=1)[0]
            index1, index2 = chosen_index, chosen_rival_index
            self.logger.info(f'{i}: {index1} VS {index2}')
            self.logger.info(f'{i}: Pre_game:{self.rates[index1]}  {self.rates[index2]}')
            self.versus(index1, index2, env)
            self.logger.info(f'{i}: Post_game:{self.rates[index1]}  {self.rates[index2]}')

    def versus(self, index1, index2, env):
        if index1 in self.rates and index2 in self.rates:
            self.infer1.update_from_index(index1)
            self.infer2.update_from_index(index2)
            env.reset()
            players = [AIServer(self.infer1), AIServer(self.infer2)]
            outcome = env.random_order_play(players, silent=True)
            e1, e2 = self.rates[index1], self.rates[index2]
            if outcome == (1, 0):
                e1.defeat(e2)
            elif outcome == (0, 1):
                e2.defeat(e1)
            elif outcome == (0, 0):
                e1.draw(e2)
            self.save()
        else:
            self.logger.info(f'{index1} or {index2} not found.')

    def add(self, index):
        if index not in self.rates:
            self.rates[index] = Elo(index)
            self.logger.info(f'index:{index} joined arena.')
        else:
            self.logger.info(f'index:{index} already existed')

    def save(self):
        with open(self.file_path, "wb") as f:
            pickle.dump(self.rates, f)

    def load(self, path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.rates = pickle.load(f)
            self.logger.info(f'loaded rates from {path}')
        else:
            self.logger.info(f'Failed to load data from "{path}",file not exist.')

    def show_rank(self):
        if not self.rates:
            self.logger.info('no rates to show')
            return
        sorted_dict = dict(sorted(self.rates.items(), key=lambda x: x[1].scores, reverse=True))
        for value in sorted_dict.values():
            print(value)

    def shutdown(self):
        self.infer1.shutdown()
        self.infer2.shutdown()


if __name__ == '__main__':
    arena = Arena()
    # for i in [120, 234, 319, 199, 209, 30, 130, 70,354,329]:
    #     arena.add(i)
    arena.run(50, Gomoku(15, 15))
    # arena.rates[319].show_records()
    arena.show_rank()
    arena.shutdown()
