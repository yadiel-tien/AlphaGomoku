from typing import TypedDict, Literal

import torch

class ColorTheme(TypedDict):
    blue: list[str]
    green: list[str]
    red: list[str]
    orange: list[str]
    grey: list[str]
    black: list[str]


class GameConfig(TypedDict):
    screen_size: tuple[int, int]
    rates_path: str
    model_path_prefix: str
    buffer_path: str
    best_index_path: str
    log_dir: str
    grid_size: float
    n_filter: int
    n_cells: int
    n_res_blocks: int
    n_channels: int
    n_actions: int
    batch_size: int
    max_delay: float
    img_path: str


class AppConfig(TypedDict):
    color_themes: ColorTheme
    ChineseChess: GameConfig
    Gomoku: GameConfig
    dirichlet: float
    base_url: str
    device: torch.device
    game_name: Literal['ChineseChess', 'Gomoku']


CONFIG: AppConfig = {
    'color_themes': {
        'blue': ['#007bff', '#0056b3', '#004085', '#0056b3'],
        'green': ['#28a745', '#218838', '#1e7e34', '#218838'],
        'red': ['#dc3545', '#c82333', '#bd2130', '#c82333'],
        'orange': ['#fd7e14', '#e67e00', '#d45d02', '#e67e00'],
        'grey': ['#6c757d', '#5a6268', '#343a40', '#5a6268'],
        'black': ['#000000', '#333333', '#555555', '#333333']
    },
    'ChineseChess': {
        'screen_size': (600, 800),
        'rates_path': './data/chess/rates.pkl',
        'model_path_prefix': './data/chess/model_',
        'buffer_path': './data/chess/buffer.pkl',
        'best_index_path': './data/chess/best_index.pkl',
        'log_dir': './logs/chess/',
        'grid_size': 54,
        'n_filter': 256,
        'n_cells': 10 * 9,
        'n_res_blocks': 11,
        'n_channels': 20,
        'n_actions': 2086,
        'batch_size': 8,
        'max_delay': 1e-3,
        'img_path': 'graphics/chess/board.jpeg'
    },
    'Gomoku': {
        'screen_size': (600, 800),
        'rates_path': './data/gomoku/rates.pkl',
        'model_path_prefix': './data/gomoku/model_',
        'buffer_path': './data/gomoku/buffer.pkl',
        'best_index_path': './data/gomoku/best_index.pkl',
        'log_dir': './logs/gomoku/',
        'grid_size': 35.2857,
        'n_filter': 256,
        'n_cells': 15 * 15,
        'n_res_blocks': 7,
        'n_channels': 2,
        'n_actions': 15 * 15,
        'batch_size': 8,
        'max_delay': 1e-3,
        'img_path': 'graphics/gomoku/board.jpeg'
    },
    'dirichlet': 0.2,
    'base_url': 'http://192.168.0.126:5000/',
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'game_name': 'Gomoku'
}
