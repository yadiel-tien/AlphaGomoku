import sys
from typing import TypeAlias, Literal

import pygame

from chess import ChineseChessUI
from gomoku import GomokuUI
from config import CONFIG
from player import Human, AIClient

game_name = CONFIG['game_name']
settings = CONFIG[game_name]


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(settings['screen_size'])
        pygame.display.set_caption(game_name)
        players = [Human(game_name), AIClient(1, 354, game_name)]
        if game_name == 'Gomoku':
            self.board = GomokuUI(players)
        else:
            self.board = ChineseChessUI(players=players)

    def play(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.board.handle_input(event)

            self.board.update()
            self.board.draw()
            pygame.display.update()


if __name__ == '__main__':
    game = Game()
    game.play()
    pygame.quit()
    sys.exit()
