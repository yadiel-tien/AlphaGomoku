import pygame


class Timer:
    def __init__(self, limit=15000, func=None):
        self.limit = limit
        self.start = 0
        self.is_active = False
        self.remain = limit
        self.func = func

    def activate(self):
        self.start = pygame.time.get_ticks()
        self.is_active = True

    def update(self):
        if self.is_active:
            self.remain = self.limit - pygame.time.get_ticks() + self.start
            self.remain = max(self.remain, 0)
            if self.remain == 0:
                self.func()

    def reset(self):
        self.is_active = False
        self.remain = self.limit
