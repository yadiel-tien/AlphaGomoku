import sys

import pygame

from board import BoardUI
from config import CONFIG
from player import Human, AIClient
from train import read_best_index


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(CONFIG['screen_size'])
        pygame.display.set_caption("五子棋")
        h, w = CONFIG['board_shape']
        players = [AIClient(0,30), AIClient(1, 90)]

        self.board = BoardUI(h, w, players)

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
