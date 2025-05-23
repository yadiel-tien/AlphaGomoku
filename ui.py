import pygame

from config import CONFIG


class Button:
    def __init__(self, text, action, pos, size=(200, 50), color='blue'):
        self.text = text
        self.action = action
        self.size = size
        self.rect = pygame.Rect(pos, size)
        self.colors = CONFIG['color_themes'][color]
        self.image = pygame.Surface(size, pygame.SRCALPHA)
        self.state = 'normal'
        self.screen = pygame.display.get_surface()

    def draw(self):
        color = self.colors[0]
        if self.state == 'hover':
            color = self.colors[1]
        elif self.state == 'click':
            color = self.colors[2]
        pygame.draw.rect(self.image, color, (0, 0) + self.size, border_radius=15)
        pygame.draw.rect(self.image, self.colors[3], (0, 0) + self.size, border_radius=15, width=2)

        font = pygame.font.Font(None, 36)
        text = font.render(self.text, True, (255, 255, 255))
        rect = text.get_rect(center=(self.size[0] // 2, self.size[1] // 2))
        self.image.blit(text, rect.topleft)
        self.screen.blit(self.image, self.rect)

    def handle_input(self, event):
        if event.type == pygame.MOUSEMOTION:
            if self.rect.collidepoint(event.pos):
                self.state = 'hover'
            else:
                self.state = 'normal'
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.state = 'click'
                self.action()
