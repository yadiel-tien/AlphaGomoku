import pygame
import torch

PIECE_PLACED = pygame.event.custom_type()

CONFIG = {
    'screen_size': (600, 800),
    'color_themes': {
        'blue': ['#007bff', '#0056b3', '#004085', '#0056b3'],
        'green': ['#28a745', '#218838', '#1e7e34', '#218838'],
        'red': ['#dc3545', '#c82333', '#bd2130', '#c82333'],
        'orange': ['#fd7e14', '#e67e00', '#d45d02', '#e67e00'],
        'grey': ['#6c757d', '#5a6268', '#343a40', '#5a6268'],
        'black': ['#000000', '#333333', '#555555', '#333333']
    },
    'dirichlet': 0.2,
    'buffer_path': './data/buffer.pkl',
    'win_buffer_path': './data/win_buffer.pkl',
    'base_url': 'http://192.168.0.126:5000/',
    'MODEL_PATH': './data/model_311.pt',
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'best_index_path': './data/best_index.pkl',
    'board_shape': (15, 15),
}
