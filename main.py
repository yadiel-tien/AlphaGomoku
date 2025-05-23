import sys

import pygame

from board import BoardUI
from config import CONFIG


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(CONFIG['screen_size'])
        pygame.display.set_caption("五子棋")
        self.board = BoardUI(8, 8)

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

    def quit(self):
        self.board.quit()


if __name__ == '__main__':
    game = Game()
    game.play()
    game.quit()
    pygame.quit()
    sys.exit()
