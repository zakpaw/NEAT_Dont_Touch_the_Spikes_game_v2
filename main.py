import pygame as pg  # pip install pygame==2.0.0.dev8
import pygame_menu
import neat
import os
from game_code.Game import Game


def play_game():
    game = Game()
    game.run()
    pg.quit()


def watch_AI():
    num_of_generations = 5000
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
    winner = p.run(Game().run_for_NEAT, num_of_generations)

    print(f'\nBest Bird:\n{winner}')


if __name__ == '__main__':
    pg.init()
    surf = pg.display.set_mode((525, 825))
    menu = pygame_menu.Menu(
        825, 525, 'Dont touch the spikes', theme=pygame_menu.themes.THEME_DARK)
    menu.add_button('Play', play_game)
    menu.add_button('Watch AI play', watch_AI)
    menu.add_button('Quit', pygame_menu.events.EXIT)
    menu.mainloop(surf)
