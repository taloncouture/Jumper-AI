import pygame
import random
import os
import neat
import pickle
pygame.init()

clock = pygame.time.Clock()

ground_image = pygame.transform.scale(pygame.image.load('ground.png'), (800, 96))
sky_color = (150, 150, 255)
score_font = pygame.font.SysFont('comicsans', 60)
pygame.display.set_caption('Jumper AI')

class Tree:
    def __init__(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed
        self.WIDTH = 48
        self.HEIGHT = 64
        self.image = pygame.transform.scale(pygame.image.load('tree.png'), (self.WIDTH, self.HEIGHT))
        self.rect = self.image.get_rect()

    def draw(self, window):
        window.blit(self.image, (self.x, self.y))

    def update(self):
        self.x -= self.speed
        self.rect.x = self.x
        self.rect.y = self.y

    def collision(self, bot):
        if self.rect.colliderect(bot.rect):
            return True
        return False

class Bot:
    def __init__(self, x, y, window_width, window_height):
        self.x = x
        self.y = y
        self.WIDTH = 48
        self.HEIGHT = 48
        self.gravity = 0
        self.window_width = window_width
        self.window_height = window_height
        self.image = pygame.transform.scale(pygame.image.load('tv.png'), (self.WIDTH, self.HEIGHT))
        self.rect = self.image.get_rect()

    def update(self):

        self.y += self.gravity

        if self.y <= self.window_height - 96 - 48:
            self.gravity += 1
        else:
            self.gravity = 0
            self.y = self.window_height - 96 - 48

        self.rect.x = self.x
        self.rect.y = self.y

    def draw(self, window):
        window.blit(self.image, (self.x, self.y))

    def jump(self):
        if self.y >= self.window_height - 96 - 48:
            self.gravity = -16
    


def Game(network):
    width, height = 800, 600
    window = pygame.display.set_mode((width, height))

    net = network
    bot = Bot(96, height - 96 - 48, width, height)

    tree = Tree(width, height - 96 - 64, 12)
    score = 0

    ground_x = 0

    run = True
    while run:
            
        tree.update()
        if tree.x <= -48:
            tree.x = width + random.randint(0, height)
            score += 1

        if ground_x > -width:
            ground_x -= 12
        else:
            ground_x = 0


        output = net.activate((bot.y, abs(bot.y - tree.x)))

        if output[0] > 0.5:
            bot.jump()

        bot.update()

        if bot.rect.colliderect(tree.rect):
            pygame.quit()
            quit()
            break

        window.fill(sky_color)
        bot.draw(window)
        window.blit(ground_image, (ground_x, height - 96))
        window.blit(ground_image, (ground_x + width, height - 96))
        tree.draw(window)

        score_text = score_font.render(f'{score}', False, (0, 0, 0))
        score_text_rect = score_text.get_rect(center = (width / 2, 40))
        window.blit(score_text, score_text_rect)



        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        pygame.display.update()
        clock.tick(60)


    pygame.quit()

def eval_genomes(genomes, config):

    width, height = 800, 600
    window = pygame.display.set_mode((width, height))

    networks = []
    bots = []
    genome_list = []

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        networks.append(net)
        bots.append(Bot(96, height - 96 - 48, width, height))
        genome_list.append(genome)

    tree = Tree(width, height - 96 - 64, 12)
    score = 0
    ground_x = 0

    run = True
    while run:

            
        tree.update()
        if tree.x <= -48:
            tree.x = width + random.randint(0, height)
            score += 1
            for genome in genome_list:
                genome.fitness += 0.2

        if ground_x > -width:
            ground_x -= 12
        else:
            ground_x = 0


        for index, bot in enumerate(bots):
            bot.update()

            output = networks[bots.index(bot)].activate((bot.y, abs(bot.y - tree.x)))

            if output[0] > 0.5:
                bot.jump()
                if bot.rect.x >= height - 96 - 48:
                    genome_list[bots.index(bot)].fitness -= 0.5

        for bot in bots:
            if bot.rect.colliderect(tree.rect):
                genome_list[bots.index(bot)].fitness -= 1
                networks.pop(bots.index(bot))
                genome_list.pop(bots.index(bot))
                bots.pop(bots.index(bot))

        window.fill(sky_color)
        for bot in bots:
            bot.draw(window)
        window.blit(ground_image, (ground_x, height - 96))
        window.blit(ground_image, (ground_x + width, height - 96))
        tree.draw(window)

        # score_text = score_font.render(f'{score}', False, (0, 0, 0))
        # score_text_rect = score_text.get_rect(center = (width / 2, 40))
        # window.blit(score_text, score_text_rect)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        pygame.display.update()
        #clock.tick(240)

        #The max_score is how many points the AI will reach before ending the generation and saving the current generation as the best
        max_score = 20
        if score > max_score:
            pickle.dump(networks[0], open('best.pickle', 'wb'))
            break

    pygame.quit()

def test_ai(config_path):
    with open('best.pickle', 'rb') as f:
        winner = pickle.load(f)

    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    Game(winner)

def run_neat(config):
    p = neat.Population(config)
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-2')
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 50)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    #To keep training the AI, uncomment this line and comment test_ai(config_path)
    #run_neat(config)

    #To test the AI, uncomment this line and comment run_neat(config)
    test_ai(config_path)

