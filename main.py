import sys

import pygame

from board import BoardUI
from config import CONFIG
from player import Human, AIClient


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(CONFIG['screen_size'])
        pygame.display.set_caption("五子棋")
        h, w = 9, 9
        players = {0:Human((h, w)), 1:AIClient()}
        self.board = BoardUI(9, 9, players)

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
