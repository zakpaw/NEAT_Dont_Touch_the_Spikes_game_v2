import pygame as pg  # version 2.0.0.dev8
import pygame_menu
import neat
import os
import pickle
from game_code.Game import Game


class Menu(object):
    def __init__(self):
        pg.init()
        self.HEIGHT = 825
        self.WIDTH = 525
        self.surf = pg.display.set_mode([self.WIDTH, self.HEIGHT])
        pg.display.set_caption('Spikes Touch NO!')
        pg.display.set_icon(pg.image.load('data/Bird.png'))
        self.setup()

    def setup(self):
        mytheme = pygame_menu.themes.Theme(background_color=(179, 255, 255),
                                           title_background_color=(0, 0, 0, 0),
                                           widget_font=pygame_menu.font.FONT_MUNRO,
                                           widget_font_color=(0, 0, 0),
                                           widget_selection_effect=pygame_menu.widgets.LeftArrowSelection(
                                           arrow_size=(20, 20), arrow_right_margin=5),
                                           title_font=pygame_menu.font.FONT_8BIT,
                                           title_offset=(35, 200),
                                           title_font_size=30,
                                           title_font_color=(0, 0, 0),
                                           menubar_close_button=False)
        self.menu = pygame_menu.Menu(
            self.HEIGHT*9/10, self.WIDTH*9/10, 'Spikes touch NO', theme=mytheme)
        self.menu.add_button('Play', self.play_game)
        self.menu.add_button('Train AI', self.train_AI)
        self.menu.add_button('Watch AI play', self.play_AI)
        self.menu.add_button('Quit', pygame_menu.events.EXIT)

    def run(self):
        self.surf.fill((220, 0, 63))
        self.menu.mainloop(self.surf)
        pg.display.update()

    def play_game(self):
        Game().run()

    def play_AI(self):
        NNet = pickle.load(open("best.pickle", "rb"))
        #NNet = [(1, NNet)]

        loc_dir = os.path.dirname(__file__)
        config_dir = os.path.join(loc_dir, 'data', 'NEATconfig.txt')
        config = neat.config.Config(neat.DefaultGenome,
                                    neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet,
                                    neat.DefaultStagnation,
                                    config_dir)
        Game().watch_NEAT(NNet, config)

    def train_AI(self):
        num_of_generations = 500
        loc_dir = os.path.dirname(__file__)
        config_dir = os.path.join(loc_dir, 'data', 'NEATconfig.txt')
        config = neat.config.Config(neat.DefaultGenome,
                                    neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet,
                                    neat.DefaultStagnation,
                                    config_dir)

        p = neat.Population(config)
        # p.add_reporter(neat.StdOutReporter(True))
        # p.add_reporter(neat.StatisticsReporter())
        winner = p.run(Game().train_NEAT, num_of_generations)

        print(f'\n\nBest Bird:\n{winner}')


if __name__ == '__main__':
    Menu().run()
    pg.quit()
    quit()
