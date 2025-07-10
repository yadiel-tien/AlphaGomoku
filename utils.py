from typing import Callable

import pygame


class Timer:
    def __init__(self, limit: int = 15000, func: Callable = None):
        self.limit = limit
        self.start = 0
        self.is_active = False
        self.remain = limit
        self.func = func

    def activate(self) -> None:
        self.start = pygame.time.get_ticks()
        self.is_active = True

    def update(self) -> None:
        if self.is_active:
            self.remain = self.limit - pygame.time.get_ticks() + self.start
            self.remain = max(self.remain, 0)
            if self.remain == 0:
                self.is_active = False
                self.func()

    def reset(self) -> None:
        self.is_active = False
        self.remain = self.limit
