import pygame as pg  # pip install pygame==2.0.0.dev8
import neat
import os
from game_code.Game import Game


if __name__ == '__main__':
    AI = 0

    if AI:
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
    else:
        game = Game()
        game.run()
        pg.quit()
