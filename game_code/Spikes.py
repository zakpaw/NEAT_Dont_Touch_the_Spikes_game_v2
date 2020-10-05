import pygame as pg
import random



class Spike(pg.sprite.Sprite):
    def __init__(self, x, y, scale):
        pg.sprite.Sprite.__init__(self)
        self.x, self.y = x, y
        self.image = pg.transform.scale(pg.image.load('data/Spike.png'), scale)
        self.rect = self.image.get_rect()
        self.rect.center = (self.x, int(self.y+self.image.get_height()/2))
        self.mask = pg.mask.from_surface(self.image)



class Spikes(object):
    def __init__(self, board, AI=False):
        self.N_SPIKES = 11
        self.AI = AI
        self.board = board
        self.scale = (int(self.board.board_h*0.8/self.N_SPIKES), int(self.board.board_h*0.8/self.N_SPIKES))
        self.gaps = []
        self.spikes = self.initial_spikes()


    def random_spikes(self, site):
        self.gaps.clear()
        self.site = site
        dist = self.board.board_h/self.N_SPIKES
        begin = self.scale[0]/7
        one_empty = random.randint(int((self.N_SPIKES-1)*0.3), int((self.N_SPIKES-1)*0.7))
        n_active = random.randint(int(self.N_SPIKES*0.6), self.N_SPIKES-2)
        active_spikes = random.sample(range(0, self.N_SPIKES-1), n_active)
        
        spikes = []
        for i in range(self.N_SPIKES):
            if i in active_spikes and i != one_empty:
                spikes.append(Spike(site, begin+self.board.top+i*dist, self.scale))
            elif self.AI and i != 0 and i != self.N_SPIKES-1:
                spike = Spike(site, begin+self.board.top+i*dist, self.scale)
                self.gaps.append((spike.rect.center))
        self.spikes = pg.sprite.Group(spikes)


    def initial_spikes(self):
        return pg.sprite.Group([])


    def draw(self, screen):
        self.spikes.draw(screen)
