import pygame as pg
from enum import Enum
from math import sqrt
import neat
import pickle
from game_code.Bird import Bird
from game_code.Spikes import Spikes


class C(Enum):
    WHITE = (255, 255, 255)
    PINK = (220, 0, 63)
    L_BLUE = (179, 255, 255)


class Game_Board(object):
    def __init__(self, win_w, win_h, change_width=0):
        self.scale = 9/10

        self.mid = (win_w/2+change_width/2, win_h/2)

        self.board_h = self.scale*win_h
        self.board_w = self.scale*win_w

        self.left = (1-self.scale)*win_w/2+change_width/2
        self.right = self.left+self.board_w
        self.top = (1-self.scale)*win_h/2
        self.bottom = self.top + self.board_h

    def draw(self, screen):
        screen.fill(C.L_BLUE.value, (self.left, self.top,
                                     self.board_w, self.board_h))


gen = 0


class Game(object):
    def __init__(self):
        pg.font.init()
        self.RUNNING = True
        self.HEIGHT = 825
        self.WIDTH = 525
        self.size_change = 0
        self.g_board = self.spikes = self.bird = None
        self.clock = pg.time.Clock()
        self.screen = None
        icon = pg.image.load('data/Bird.png')
        pg.display.set_icon(icon)

    def run(self):
        '''Main game loop'''
        self.screen = pg.display.set_mode(
            [self.WIDTH, self.HEIGHT], pg.RESIZABLE)
        self.init_objects(self.WIDTH, self.HEIGHT)
        while self.RUNNING:
            self.clock.tick(60)
            self.draw(self.bird.score, self.g_board.mid)
            self.events_handler()

            if self.bird.alive:
                self.bird.move()
                self.bird.collision_check(self.spikes)
            elif not self.bird.alive:
                self.spikes.spikes = self.spikes.initial_spikes()

            pg.display.update()

    def watch_NEAT(self, NNet, config):
        '''Main game loop'''
        NNets = [NNet]
        self.screen = pg.display.set_mode([self.WIDTH, self.HEIGHT])
        self.init_objects(self.WIDTH, self.HEIGHT, True)
        self.bird = Bird(self.g_board, True)

        self.bird.alive = True
        fps = 60
        while self.RUNNING:
            self.clock.tick(fps)
            self.draw(self.bird.score, self.g_board.mid)

            events = pg.event.get()
            for e in events:
                if e.type == pg.QUIT:
                    self.RUNNING = False
                    pg.quit()
                    quit()
                if e.type == pg.KEYDOWN:
                    if e.key == pg.K_UP and fps != 120:
                        fps = fps*2
                    if e.key == pg.K_DOWN and fps != 60:
                        fps = fps/2
            if self.bird.alive:
                self.bird.move()

                prev_score = self.bird.score
                self.bird.collision_check(self.spikes)
                if self.bird.score > prev_score:
                    self.bird.find_best_gap(self.spikes)

                if self.bird.vel < 0:
                    site = self.g_board.left+self.bird.image.get_width()/2
                else:
                    site = self.g_board.right-self.bird.image.get_width()/2

                by = self.bird.y+self.bird.image.get_height()/2
                output = NNets[0].activate(
                    (self.bird.x, by, abs(self.bird.x-site), abs(by-self.bird.best_gap[1])))

                if output[0] > 0.5:
                    self.bird.jump()
            else:
                self.RUNNING = False

            pg.display.update()

    def train_NEAT(self, genomes, config):
        '''Main game loop for AI'''
        self.screen = pg.display.set_mode([self.WIDTH, self.HEIGHT])

        self.init_objects(self.WIDTH, self.HEIGHT, True)
        self.spikes.spikes = self.spikes.initial_spikes()
        self.RUNNING = True
        NNets, ge, self.bird = [], [], []

        for _, g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, config)
            NNets.append(net)
            self.bird.append(Bird(self.g_board, True))
            g.fitness = 0
            ge.append(g)

        prev_score, site = 0, 0
        fps = 240
        while self.RUNNING:
            # self.clock.tick(fps)

            #self.draw(self.bird[0].score, self.g_board.mid, True)

            events = pg.event.get()
            for e in events:
                if e.type == pg.QUIT:
                    self.RUNNING = False
                    pg.quit()
                    quit()
                if e.type == pg.KEYDOWN:
                    if e.key == pg.K_UP and fps != 240:
                        fps = fps*2
                    if e.key == pg.K_DOWN and fps != 60:
                        fps = fps/2

            if self.bird[0].score >= 500:
                pickle.dump(NNets[0], open("best.pickle", "wb"))

            image_height = self.bird[0].image.get_height()
            for x, brd in enumerate(self.bird):
                if brd.alive:
                    brd.move()
                    if brd.vel < 0:
                        site = self.g_board.left+brd.image.get_width()/2
                    else:
                        site = self.g_board.right-brd.image.get_width()/2

                    by = brd.y + brd.image.get_height()/2
                    output = NNets[self.bird.index(brd)].activate(
                        (brd.x, by, abs(brd.x-site), abs(by-brd.best_gap[1])))

                    pg.draw.line(self.screen, (0, 255, 0),
                                 (brd.x, by), (site, brd.best_gap[1]))

                    if output[0] > 0.5:
                        brd.jump()

                    prev_score = brd.score
                    if not brd.collision_check(self.spikes, True):
                        self.bird.pop(x)
                        ge[x].fitness -= 7
                        NNets.pop(x)
                        ge.pop(x)
                    else:
                        if brd.score > prev_score:
                            ge[x].fitness += 0.1
                            if by <= brd.best_gap[1]+25 and by >= brd.best_gap[1]-25 and brd.score > 0:
                                ge[x].fitness += 50
                                if brd.score > 20:
                                    ge[x].fitness += brd.score // 20 * 10
                            for brd in self.bird:
                                brd.find_best_gap(self.spikes)

            if len(self.bird) == 0:
                self.RUNNING = False

            # pg.display.update()

        global gen
        gen += 1
        print(f"generation: {gen}\t score: {prev_score}")

    def init_objects(self, width, height, AI=False):
        self.g_board = Game_Board(width, height, self.size_change)
        if AI:
            self.spikes = Spikes(self.g_board, True)
        else:
            self.spikes = Spikes(self.g_board)
            self.bird = Bird(self.g_board)

    def events_handler(self):
        events = pg.event.get()
        for e in events:
            if e.type == pg.QUIT:
                self.RUNNING = False
                pg.quit()
                quit()
            if e.type == pg.VIDEORESIZE:
                self.screen = pg.display.set_mode([e.w, e.h], pg.RESIZABLE)
                self.WIDTH, self.HEIGHT = e.w, e.h
                self.size_change = e.w-e.h*7/11  # 7/11 is proportion of the game window
                self.init_objects(self.HEIGHT*7/11, self.HEIGHT)
            if e.type == pg.KEYDOWN:
                if e.key == pg.K_SPACE:
                    self.bird.jump()
                elif e.key == pg.K_ESCAPE:
                    self.RUNNING = False

    def draw(self, score, mid_pt, AI=False):
        pg.display.set_caption(
            f'Spikes Touch NO!     FPS:{int(self.clock.get_fps())}')
        self.screen.fill(C.PINK.value)

        self.g_board.draw(self.screen)
        self.spikes.draw(self.screen)

        font = pg.font.SysFont('techno', int(self.HEIGHT/5))
        score_text = font.render(str(score), True, C.L_BLUE.value)

        pg.draw.circle(self.screen, C.WHITE.value, (int(
            mid_pt[0]), int(mid_pt[1]*0.7)), int(self.HEIGHT/9))
        self.screen.blit(score_text, score_text.get_rect(
            center=(int(mid_pt[0]), int(mid_pt[1] * 0.7))))

        if AI:
            global gen
            font = pg.font.SysFont('techno', int(self.HEIGHT/16))
            gen_text = font.render(f"GEN: {gen}", True, C.WHITE.value)
            self.screen.blit(gen_text, gen_text.get_rect(
                center=(int(self.g_board.mid[0]), int(self.g_board.mid[1] * 0.3))))
            for brd in self.bird:
                brd.draw(self.screen)
        else:
            self.bird.draw(self.screen)
