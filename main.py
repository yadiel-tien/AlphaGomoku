import sys

import pygame

from gomoku import GomokuUI
from config import SETTINGS
from player import Human, AIClient


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(SETTINGS['screen_size'])
        pygame.display.set_caption("五子棋")
        players = [Human(), AIClient(1,354)]

        self.board = GomokuUI(players=players)

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
