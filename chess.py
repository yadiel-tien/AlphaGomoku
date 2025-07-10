from collections.abc import Callable
from typing import TypeAlias, cast, Literal

from gymnasium import spaces
import pygame
from numpy.typing import NDArray
from config import CONFIG, GameConfig
from env import BaseEnv, GameUI
from player import Human, RandomPlayer
import numpy as np

settings: GameConfig = CONFIG['ChineseChess']
ChessMove: TypeAlias = tuple[int, int, int, int]
PieceMoveFunc: TypeAlias = Callable[[NDArray, int, int], list[tuple[int, int]]]


class ChineseChess(BaseEnv):
    _move2action: dict[ChessMove, int] = {}
    _action2move: dict[int, ChessMove] = {}
    piece2id: dict[str, int] = {}
    id2piece: dict[int, str] = {}
    advisor_moves: list[ChessMove] = []
    bishop_moves: list[ChessMove] = []
    dest_func: dict[int, PieceMoveFunc] = {}

    def __init__(self):
        super().__init__()
        self.shape: tuple[int, int, int] = (10, 9, 7)
        if not ChineseChess._move2action:
            self._init_class_dicts()
        self.n_actions = 2086
        self.action_space = spaces.Discrete(2086)
        self.observation_space = spaces.Box(-1, 13, shape=self.shape, dtype=np.float32)
        self.history: list[NDArray[np.float32]] = []
        self.reset()

    def reset(self, seed=None, options=None) -> tuple[NDArray[np.float32], dict]:
        """
        重置游戏, 返回当前棋盘状态
        state.shape=(10,9,7),[0-5]层代表最近6步，数字-1代表空白，0-13代表不同棋子。[6]层代表未吃子步数，0-100归一化到0-1
        """
        self.state: NDArray[np.float32] = np.zeros(self.shape, dtype=np.float32)
        self.state[:, :, 0] = np.array([
            [7, 8, 9, 10, 11, 10, 9, 8, 7],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, 12, -1, -1, -1, -1, -1, 12, -1],
            [13, -1, 13, -1, 13, -1, 13, -1, 13],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [6, -1, 6, -1, 6, -1, 6, -1, 6],
            [-1, 5, -1, -1, -1, -1, -1, 5, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 1, 2, 3, 4, 3, 2, 1, 0],
        ])
        self.reset_status()
        return self.state, {}

    @classmethod
    def virtual_step(cls, state: NDArray[np.float32], action: int) -> NDArray[np.float32]:
        """只改变state，不计算输赢和奖励"""
        new_state = np.copy(state)
        no_capture_steps = int(state[0, 0, -1] * 100)
        r, c, to_r, to_c = cls._action2move[action]
        attacker = new_state[r, c, 0]
        target = new_state[to_r, to_c, 0]
        # 统计双方未吃子回合，用于平局
        if target == -1:
            no_capture_steps += 1
        else:
            no_capture_steps = 0

        # 历史信息
        new_state[:, :, 1:6] = new_state[:, :, 0:5]
        new_state[:, :, -1] = no_capture_steps / 100
        # 执行落子
        new_state[r, c, 0] = -1
        new_state[to_r, to_c, 0] = attacker
        return new_state

    @classmethod
    def step_fn(cls, state: NDArray[np.float32], action: int, player_to_move: int) -> tuple[
        NDArray[np.float32], int, bool, bool, dict]:
        """
       执行落子,不依赖环境
       :param state: (10,9,7)，[0-5]层最近棋局，[6]未吃子步数
       :param action: 动作编号（棋盘上的位置）
       :param player_to_move: 0红1黑
       :return: observation（新的state）, reward（奖励）, terminated（是否结束）,truncated(是否因时间限制中断）, info（额外信息）
       """
        new_state = cls.virtual_step(state, action)
        player_just_moved = player_to_move
        # 检查结果
        winner = cls.check_winner(new_state, player_just_moved, action)
        reward = winner if winner != 2 else 0
        terminated = winner != 2
        return new_state, reward, terminated, False, {}

    @classmethod
    def check_winner(cls, state: NDArray, perspective_player: int, action_just_executed: int) -> int:
        """检查胜负情况，相对于perspective_player来说
        :param action_just_executed: 刚刚做过的动作
        :param state: (10,9,6)
        :param perspective_player:相对于这个玩家来说的结果， 0红1黑
        :return: 1胜，0平，-1负, 2未分胜负"""
        # 100步未吃子判和
        if state[0, 0, -1] >= 1:
            return 0

        # 连将或连捉判负
        diff1 = np.equal(state[:, :, 0], state[:, :, 1])
        diff2 = np.equal(state[:, :, 2], state[:, :, 3])
        diff3 = np.equal(state[:, :, 4], state[:, :, 5])
        if np.array_equal(diff1, diff2) and np.array_equal(diff1, diff3):
            return -1

        board = state[:, :, 0]
        if 4 not in board:  # 红帅被杀
            return 1 if perspective_player == 1 else -1
        if 11 not in board:  # 黑帅被杀
            return 1 if perspective_player == 0 else -1

        # 未分胜负
        return 2

    def step(self, action: int) -> tuple[np.ndarray, int, bool, bool, dict]:
        """
        执行落子
        :param action: 动作编号（棋盘上的位置）
        :return: observation（新的state）, reward（奖励）, terminated（是否结束）,truncated(是否因时间限制中断）, info（额外信息）
        """
        self.state, reward, terminated, truncated, info = self.step_fn(self.state, action, self.player_to_move)

        # 处理终局。0,1代表获胜玩家，-1代表平局，2代表未决胜负
        if terminated:
            winner = self.player_to_move if reward == 1 else 1 - self.player_to_move if reward == -1 else -1
            self.set_winner(winner)

        # 记录棋谱
        self.history.append(np.copy(self.state[:, :, 0]))

        # 更改玩家
        self.player_to_move = 1 - self.player_to_move
        self.last_action = action

        # 绝杀，不用等到老将被吃
        if self.is_checkmate(self.state, self.player_to_move):
            winner = 1 - self.player_to_move
            self.set_winner(winner)

        return self.state, reward, self.terminated, self.truncated, {}

    @classmethod
    def convert_to_network(cls, state: NDArray, current_player: int) -> NDArray:
        """
        将 10x9x6 的 state 编码为 10x9x20 的 one-hot 张量供神经网络使用。

        输出 arr 的含义：
        - [0~13]：当前棋盘上的棋子类型（0~13）
        - [14~18]：最近 5 步之间的棋盘是否保持不变
        - [19]：当前下子方（0 表示红，1 表示黑）
        - [20]: 未吃子步数 steps/100 归一化

        :return: arr: shape=(10, 9, 21), dtype=np.float32

        """
        arr = np.zeros((10, 9, 20), dtype=np.float32)
        # 当前盘面
        board = state[:, :, 0]
        # 当前盘面编码，0-13代表不同棋子
        for i in range(14):
            arr[:, :, i] = np.asarray(board == i, dtype=np.float32)
        # 最近4步差分历史信息
        for i in range(1, 6):
            arr[:, :, 13 + i] = np.equal(state[i - 1], state[i]).astype(np.float32)
        # 编码当前玩家
        if current_player == 1:
            arr[:, :, 19] = 1.0
        # 编码未吃子步数，超过100判和
        arr[:, :, 20] = state[5]
        return arr

    @classmethod
    def get_valid_actions(cls, state: NDArray, player_to_move: int) -> NDArray[np.int_]:
        """获取当前局面的合法动作
        :param player_to_move: 当前玩家。0红1黑
        :param state: (10x9x6)[0-4]近5步盘面,[5]未吃子步数。
        :return: arr: 一维int类型np数组"""
        available_actions = []
        board = state[:, :, 0]
        for r in range(10):
            for c in range(9):
                piece = int(board[r, c])
                if (player_to_move == 0 and 0 <= piece <= 6) or (player_to_move == 1 and piece >= 7):
                    destinations = ChineseChess.dest_func[piece](state, r, c)
                    for to_r, to_c in destinations:
                        available_actions.append(cls._move2action[(r, c, to_r, to_c)])
        return np.array(available_actions)

    @staticmethod
    def _get_rook_dest(state: NDArray, r: int, c: int) -> list[tuple[int, int]]:
        board = state[:, :, 0]
        destinations = []
        own_side = board[r, c] < 7  # 0-6红方
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = r, c
            while True:
                nr += dr
                nc += dc
                if not (0 <= nr < 10 and 0 <= nc < 9):
                    break  # 越界
                target = board[nr, nc]

                if target == -1:  # 空位
                    destinations.append((nr, nc))
                elif (target < 7) != own_side:  # 对手棋子，可以吃
                    destinations.append((nr, nc))
                    break
                else:  # 己方棋子
                    break

        return destinations

    @staticmethod
    def _get_horse_dest(state: NDArray, r: int, c: int) -> list[tuple[int, int]]:
        board = state[:, :, 0]
        destinations = []
        horse_moves = (
            ((-2, -1), (-1, 0)),  # 上上左，蹩脚点：上
            ((-2, 1), (-1, 0)),  # 上上右
            ((-1, -2), (0, -1)),  # 上左左，蹩脚点：左
            ((-1, 2), (0, 1)),  # 上右右
            ((1, -2), (0, -1)),  # 下左左
            ((1, 2), (0, 1)),  # 下右右
            ((2, -1), (1, 0)),  # 下下左
            ((2, 1), (1, 0)),  # 下下右
        )
        own_side = board[r, c] < 7
        for (dr, dc), (br, bc) in horse_moves:
            nr, nc = r + dr, c + dc
            block_r, block_c = r + br, c + bc
            # 目标位置在棋盘上，且没有蹩脚
            if 0 <= nr < 10 and 0 <= nc < 9:
                if board[block_r, block_c] != -1:
                    continue
                # 目标位置没有棋子或有对方棋子
                target = board[nr, nc]
                if target == -1 or (target < 7) != own_side:
                    destinations.append((nr, nc))
        return destinations

    @staticmethod
    def _get_bishop_dest(state: NDArray, r: int, c: int) -> list[tuple[int, int]]:
        board = state[:, :, 0]
        destinations = []
        own_side = board[r, c] < 7
        for from_r, from_c, to_r, to_c in ChineseChess.bishop_moves:
            if from_r != r or from_c != c:
                continue
            # 象眼
            block_r, block_c = (from_r + to_r) // 2, (from_c + to_c) // 2
            if board[block_r, block_c] != -1:
                continue

            target = board[to_r, to_c]
            if target == -1 or (target < 7) != own_side:
                destinations.append((to_r, to_c))
        return destinations

    @staticmethod
    def _get_advisor_dest(state: NDArray, r: int, c: int) -> list[tuple[int, int]]:
        board = state[:, :, 0]
        destinations = []
        own_side = board[r, c] < 7
        for from_r, from_c, to_r, to_c in ChineseChess.advisor_moves:
            if from_r != r or from_c != c:
                continue
            target = board[to_r, to_c]
            if target == -1 or (target < 7) != own_side:  # 目标位置为空或为敌方棋子
                destinations.append((to_r, to_c))
        return destinations

    @staticmethod
    def _get_king_dest(state: NDArray, r: int, c: int) -> list[tuple[int, int]]:
        board = state[:, :, 0]
        destinations = []
        candidates = ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1))
        is_red_king = board[r, c] == 4
        valid_rows = range(7, 10) if is_red_king else range(0, 3)
        rival_king = 11 if is_red_king else 4
        for to_r, to_c in candidates:
            if to_c < 3 or to_c > 5:
                continue
            if to_r not in valid_rows:
                continue

            target = board[to_r, to_c]
            if target == -1 or ((target < 7) != is_red_king):
                destinations.append((to_r, to_c))

        # 两帅照面的情况
        dr = -1 if is_red_king else 1
        nr = r + dr
        while 0 <= nr <= 9:
            target = board[nr, c]
            if target == -1:
                nr += dr
            elif target == rival_king:
                destinations.append((nr, c))
                break
            else:
                break

        return destinations

    @staticmethod
    def _get_cannon_dest(state: NDArray, r: int, c: int) -> list[tuple[int, int]]:
        board = state[:, :, 0]
        destinations = []
        own_side = board[r, c] < 7
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in directions:
            found_screen = False  # 炮架
            nr, nc = r + dr, c + dc
            while 0 <= nr <= 9 and 0 <= nc <= 8:
                target = board[nr, nc]
                if found_screen:
                    if target == -1:
                        pass  # 炮架后空格不可走
                    elif (target < 7) != own_side:  # 对方棋子可吃
                        destinations.append((nr, nc))
                        break
                    else:  # 己方棋子不可吃
                        break
                else:
                    if target == -1:  # 炮架前空格可走
                        destinations.append((nr, nc))
                    else:
                        found_screen = True  # 第一个遇到的棋子是炮架
                nr += dr
                nc += dc

        return destinations

    @staticmethod
    def _get_pawn_dest(state: NDArray, r: int, c: int) -> list[tuple[int, int]]:
        board = state[:, :, 0]
        destinations = []
        is_red_pawn = board[r, c] == 6
        dr = -1 if is_red_pawn else 1
        has_crossed = (r <= 4) if is_red_pawn else (r > 4)  # 过河
        candidates = [(r + dr, c), (r, c - 1), (r, c + 1)] if has_crossed else [(r + dr, c)]
        for to_r, to_c in candidates:
            if 0 <= to_r < 10 and 0 <= to_c < 9:  # 在棋盘上
                target = board[to_r, to_c]
                if target == -1 or ((target < 7) != is_red_pawn):
                    destinations.append((to_r, to_c))

        return destinations

    @staticmethod
    def is_checkmate(state: NDArray, perspective_player: int) -> bool:
        """判断player_to_move是否已被将死，是否还有棋可走"""
        valid_actions = ChineseChess.get_valid_actions(state, perspective_player)
        for action in valid_actions:
            new_state = ChineseChess.virtual_step(state, action)
            if not ChineseChess.is_check(new_state, perspective_player):
                return False
        return True

    @staticmethod
    def is_check(state: NDArray, perspective_player: int) -> bool:
        """判断当前state局面下perspective_player是否被将军"""
        perspective_king = 4 if perspective_player == 0 else 11
        rival = 1 - perspective_player
        valid_actions = ChineseChess.get_valid_actions(state, rival)
        for action in valid_actions:
            _, _, to_r, to_c = ChineseChess.action2move(action)
            target = state[to_r, to_c, 0]
            if target == perspective_king:
                return True
        return False

    @staticmethod
    def get_action_executor(state: NDArray, action_to_execute: int) -> int:
        """根据要执行的动作获取动作执行方，红方0，黑方1"""
        r, c, to_r, to_c = ChineseChess._action2move[action_to_execute]
        piece = state[r, c, 0]
        if 0 <= piece < 7:
            return 0
        elif 6 <= piece < 14:
            return 1
        else:
            raise ValueError("Invalid last_action to execute")

    @classmethod
    def _init_class_dicts(cls) -> None:
        cls.piece2id = {
            '红车': 0, '红马': 1, '红象': 2, '红士': 3, '红帅': 4, '红炮': 5, '红兵': 6,
            '黑车': 7, '黑马': 8, '黑象': 9, '黑士': 10, '黑帅': 11, '黑炮': 12, '黑兵': 13, '一一': -1
        }
        cls.id2piece = {v: k for k, v in cls.piece2id.items()}
        a = 0
        # 垂直水平移动
        for r in range(10):
            for c in range(9):
                for to_r in range(10):
                    if to_r != r:
                        move = (r, c, to_r, c)
                        cls._move2action[move] = a
                        a += 1
                for to_c in range(9):
                    if to_c != c:
                        move = (r, c, r, to_c)
                        cls._move2action[move] = a
                        a += 1
        # 士的动作
        cls.advisor_moves = [
            (0, 3, 1, 4), (0, 5, 1, 4), (1, 4, 0, 3), (1, 4, 0, 5), (1, 4, 2, 3), (1, 4, 2, 5), (2, 3, 1, 4),
            (2, 5, 1, 4), (9, 3, 8, 4), (9, 5, 8, 4), (8, 4, 9, 3), (8, 4, 9, 5), (8, 4, 7, 3), (8, 4, 7, 5),
            (7, 3, 8, 4), (7, 5, 8, 4)
        ]
        for move in cls.advisor_moves:
            cls._move2action[move] = a
            a += 1
        # 象的动作
        cls.bishop_moves = [
            (0, 2, 2, 0), (0, 2, 2, 4), (0, 6, 2, 4), (0, 6, 2, 8), (2, 0, 0, 2), (2, 0, 4, 2), (2, 4, 0, 2),
            (2, 4, 4, 2), (2, 4, 0, 6), (2, 4, 4, 6), (2, 8, 0, 6), (2, 8, 4, 6), (4, 2, 2, 0), (4, 2, 2, 4),
            (6, 2, 2, 4), (6, 2, 2, 8)
        ]
        rival_bishop_moves = [(9 - r, c, 9 - to_r, to_c) for r, c, to_r, to_c in cls.bishop_moves]
        cls.bishop_moves.extend(rival_bishop_moves)
        for move in cls.bishop_moves:
            cls._move2action[move] = a
            a += 1
        # 马的动作
        for r in range(10):
            for c in range(9):
                for dr, dc, in ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)):
                    to_r, to_c = r + dr, c + dc
                    if 0 <= to_r < 10 and 0 <= to_c < 9:
                        move = (r, c, to_r, to_c)
                        cls._move2action[move] = a
                        a += 1
        cls._action2move = {v: k for k, v in cls._move2action.items()}

        cls.dest_func = {
            0: cls._get_rook_dest,
            1: cls._get_horse_dest,
            2: cls._get_bishop_dest,
            3: cls._get_advisor_dest,
            4: cls._get_king_dest,
            5: cls._get_cannon_dest,
            6: cls._get_pawn_dest
        }
        for i in range(7, 14):
            cls.dest_func[i] = cls.dest_func[i - 7]

    @classmethod
    def render_fn(cls, state: NDArray) -> None:
        """打印棋盘"""
        board_str = ''
        board_str += ' ' + ' '.join([f'{i:>5}' for i in range(9)]) + '\n'
        for i, row in enumerate(state[:, :, 0]):
            row_str = f'{i}'
            for piece_id in row:
                if 0 <= piece_id <= 6:
                    row_str += f' \033[91m{cls.id2piece[piece_id]:^4}\033[0m'
                else:
                    row_str += f' {cls.id2piece[piece_id]:^4}'
            board_str += row_str + '\n'

        print(board_str)

    def render(self) -> None:
        self.render_fn(self.state)

    @classmethod
    def handle_human_input(cls, state: NDArray, last_action: int, player_to_move: int) -> int:
        cls.render_fn(state)
        valid_actions = cls.get_valid_actions(state, player_to_move)
        while True:
            txt = input('输入一个4位数字，前两位代表当前棋子位置，后两位代表移动到的位置，例如红方炮7平4为7774。\n')
            if not (len(txt) == 4 and txt.isdigit()):
                print("输入格式有误！请确保是 4 位数字，例如 7774。")
                continue
            r, c, to_r, to_c = map(int, txt)
            move = r, c, to_r, to_c
            if move not in cls.move2action:
                print("该步不在合法动作表中，可能位置超出棋盘或走法无效。")
                continue

            action = cls.move2action(move)
            if action not in valid_actions:
                print("该走法不合法（可能被蹩马脚、被将军、或无该棋子）！请重新输入。")
                continue

            return action
        raise RuntimeError("handle_human_input should never reach here")

    @classmethod
    def describe_move(cls, state: NDArray, action: int) -> None:
        r, c, to_r, to_c = cls._action2move[action]
        piece = cls.id2piece[int(state[r, c, 0])]
        eat_piece = cls.id2piece[int(state[to_r, to_c, 0])]
        result = '' if eat_piece == '一一' else '吃 ' + eat_piece
        print(f'{piece} ({r},{c}) -> ({to_r}, {to_c}) {result}')

    @classmethod
    def action2move(cls, action: int) -> ChessMove:
        return cls._action2move[action]

    @classmethod
    def move2action(cls, move: ChessMove) -> int:
        """:return 如果move不存在返回 -1"""
        return cls._move2action.get(move, -1)


Mark: TypeAlias = Literal['green_dot', 'red_dot', 'blue_circle', 'green_circle']


class ChineseChessUI(GameUI):

    def __init__(self, players):
        super().__init__(ChineseChess(), players, settings['img_path'])
        self.piece_pics: dict[int, pygame.Surface] = {}
        self.mark_pics: dict[Mark, pygame.Surface] = {}
        self.init_resource()
        self.place_sound = pygame.mixer.Sound('sound/piece_down.mp3')
        self.capture_sound = pygame.mixer.Sound('sound/capture.mp3')
        self.check_sound = pygame.mixer.Sound('sound/check.mp3')
        self.checkmate_sound = pygame.mixer.Sound('sound/checkmate.mp3')
        self.image = pygame.transform.scale(self.image, (486, 540))
        self.rect = self.image.get_rect(center=self.screen.get_rect().center)
        self.env = cast(ChineseChess, self.env)
        self.selected_pos: tuple[int, int] | None = None
        self.settings = settings
        self.check_buffer = {'action': -1, 'checkmate': False, 'check': False, 'red_dot': {}}

    def init_resource(self) -> None:
        """加载棋子和标记图片"""
        for i in range(14):
            pic = pygame.image.load(f'graphics/chess/piece{i}.png')
            pic = pygame.transform.smoothscale(pic, (65, 65))
            self.piece_pics[i] = pic

        marks: tuple[Mark, ...] = ('red_dot', 'green_dot', 'blue_circle', 'green_circle')
        for mark in marks:
            pic = pygame.image.load(f'graphics/chess/{mark}.png')
            self.mark_pics[mark] = pygame.transform.smoothscale(pic, (65, 65))

    def handle_human_input(self) -> None:
        player = cast(Human, self.players[self.env.player_to_move])
        if player.selected_grid is None:
            return

        if self.selected_pos:  # 已选择棋子
            # 根据已选择棋子和目标位置生成动作
            move = self.selected_pos + player.selected_grid
            action = self.env.move2action(move)
            if action in self.env.valid_actions:
                player.pending_action = action
                self.place_sound.play()
            self.selected_pos = None
            if player.pending_action == -1:
                self.piece_sound.play()
        else:
            # 选择棋子
            chosen_piece = self.env.state[*player.selected_grid, 0]
            if chosen_piece == -1:
                return
            is_red_piece = (0 <= chosen_piece < 7)
            is_red_turn = self.env.player_to_move == 0
            if is_red_piece == is_red_turn:
                self.selected_pos = player.selected_grid
                self.piece_sound.play()

    def play_place_sound(self, action: int) -> None:
        """执行action时播放的音效"""
        if self.check_buffer['action'] != action:
            self.check_buffer['action'] = action
            self.check_buffer['checkmate'] = self.env.is_checkmate(self.env.state, self.env.player_to_move)
            self.check_buffer['check'] = self.env.is_check(self.env.state, self.env.player_to_move)

        # 绝杀，对方无论怎么走都输
        if self.check_buffer['checkmate']:
            self.checkmate_sound.play()
            return

        # 将军
        if self.check_buffer['check']:
            self.check_sound.play()
            return

        # 吃子
        _, _, to_r, to_c = self.env.action2move(action)
        target = self.env.state[to_r, to_c, 1]
        if target not in [-1, 4, 11]:
            self.capture_sound.play()

    def draw(self) -> None:
        self.screen.fill('#DDDDBB')
        self.screen.blit(self.image, self.rect)
        self.draw_pieces()
        self.draw_select_mark()
        self.draw_last_mark()
        self.draw_dot_mark()
        if self.status == 'finished':
            self.draw_victory_badge()
            self.start_btn.draw()
            self.reverse_player_btn.draw()
        elif self.status == 'new':
            self.draw_new_game_title()
            self.start_btn.draw()
            self.reverse_player_btn.draw()
        else:
            self.draw_player()

    def draw_victory_badge(self) -> None:
        winner = 'red' if self.env.winner == 0 else 'black' if self.env.winner == 1 else 'draw'
        path = f'graphics/chess/{winner}_win.png'
        self.draw_victory(path)

    def draw_pieces(self) -> None:
        """绘制棋子"""
        for row in range(10):
            for col in range(9):
                piece = int(self.env.state[row, col, 0])
                if piece != -1:
                    x, y = self._grid2pos((row, col))
                    self.screen.blit(self.piece_pics[piece], (x, y))

    def draw_select_mark(self) -> None:
        """选中的棋子周围绘制绿色圆圈"""
        if self.selected_pos:
            grid = self.selected_pos
            x, y = self._grid2pos(grid)
            self.screen.blit(self.mark_pics['green_circle'], (x, y))

    def draw_last_mark(self) -> None:
        """最后走的棋子周围绘制蓝色圆圈"""
        if self.history:
            action, _ = self.history[-1]
            _, _, to_r, to_c = self.env.action2move(action)
            grid = to_r, to_c
            x, y = self._grid2pos(grid)
            self.screen.blit(self.mark_pics['blue_circle'], (x, y))

    def draw_dot_mark(self) -> None:
        """用来指示所有可走棋步"""
        if self.selected_pos:
            is_red_dot = self.check_buffer['red_dot']
            piece = int(self.env.state[*self.selected_pos, 0])
            grids = self.env.dest_func[piece](self.env.state, *self.selected_pos)
            for grid in grids:
                x, y = self._grid2pos(grid)
                # 模拟行棋，如果导致自身被将，则标红，否则标绿
                move = self.selected_pos + grid
                action = self.env.move2action(move)
                if len(is_red_dot) == len(grids):  # 已有缓存，避免重复计算
                    if is_red_dot[action]:
                        self.screen.blit(self.mark_pics['red_dot'], (x, y))
                    else:
                        self.screen.blit(self.mark_pics['green_dot'], (x, y))
                else:  # 重建缓存
                    new_state = self.env.virtual_step(self.env.state, action)
                    is_red_dot[action] = self.env.is_check(new_state, self.env.player_to_move)
        else:
            self.check_buffer['red_dot'] = {}

    def _grid2pos(self, grid: tuple[int, int]) -> tuple[int, int]:
        """调节位置偏差"""
        x, y = super()._grid2pos(grid)
        return x - 4 - x // 200, y - y // 100


if __name__ == '__main__':
    env = ChineseChess()
    competences = [Human(), RandomPlayer()]
    env.run(competences)
    env.render()
