import pygame as pg
from math import sqrt


class Bird(pg.sprite.Sprite):
    def __init__(self, board, alive=False):
        pg.sprite.Sprite.__init__(self)
        self.x, self.y = board.mid
        self.board = board
        self.score = 0

        self.size = (int(26*0.1*board.board_w/17),
                     int(17*0.1*board.board_w/17))
        self.IMGS = [pg.transform.scale(pg.image.load('data/Bird.png'), self.size),
                     pg.transform.scale(pg.image.load('data/Bird_jumping.png'), self.size)]
        self.IMGS = [pg.transform.flip(i, True, False) for i in self.IMGS]
        self.image = self.IMGS[0]

        self.alive = alive
        self.jumping = False
        self.tick = 0
        self.vel = 5

        self.best_gap = (self.board.right, self.board.mid[1])
        self.reset_rect()

    def collision(self, spikes):
        coll = pg.sprite.spritecollideany(self, spikes, False)
        if coll:
            if pg.sprite.spritecollide(self, spikes, False, pg.sprite.collide_mask):
                return True
            return False

    def jump(self):
        self.tick = -20
        self.jumping = True
        self.alive = True

    def move(self):
        self.tick += 1
        parabola = -0.001*self.tick**2+0.5*self.tick
        # 60fps    -0.001*self.tick**2+0.5*self.tick
        self.y += parabola
        self.x += self.vel

        self.reset_rect()

    def collision_check(self, spikes, AI=False, fitness=None):
        # COLISION WITH RIGHT SITE
        if self.x+self.image.get_width()/2 >= self.board.right:
            self.IMGS = [pg.transform.flip(i, True, False) for i in self.IMGS]
            self.vel *= -1
            self.score += 1
            spikes.random_spikes(self.board.left)
        # COLISION WITH LEFT SITE
        elif self.x-self.image.get_width()/2 <= self.board.left:
            self.IMGS = [pg.transform.flip(i, True, False) for i in self.IMGS]
            self.vel *= -1
            self.score += 1
            spikes.random_spikes(self.board.right)
        # COLISION WITH BOTTOM SITE
        elif self.y+self.image.get_height() > self.board.bottom:
            self.reset_bird(spikes)
        # COLISION WITH TOP SITE
        elif self.y < self.board.top:
            self.reset_bird(spikes)
        # COLISION WITH SPIKES
        if self.collision(spikes.spikes):
            self.reset_bird(spikes)

        if AI:
            # self.find_best_gap(spikes)
            return self.alive

    def find_best_gap(self, spikes):
        distances = []
        for g in spikes.gaps:
            distances.append(sqrt((self.x-g[0])**2+(self.y-g[1])**2))
        if len(spikes.gaps) > 0:
            self.best_gap = spikes.gaps[distances.index(min(distances))]

    def draw(self, screen):
        self.image = self.IMGS[0]
        if self.jumping:
            self.image = self.IMGS[1]
            if self.tick > 3:
                self.jumping = False
        screen.blit(self.image, self.rect)

    def reset_bird(self, spikes):
        self.image = self.IMGS[0]
        self.jumping = False
        self.alive = False
        self.score = 0
        self.x, self.y = self.board.mid
        spikes = spikes.initial_spikes()
        self.reset_rect()

    def reset_rect(self):
        self.rect = self.image.get_rect()
        self.rect.center = (self.x, int(self.y+self.image.get_height()/2))
        self.mask = pg.mask.from_surface(self.image)
