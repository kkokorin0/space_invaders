import pygame
import time
import random


class Game:
    screen = None
    aliens = []
    rockets = []
    lost = False
    width = 0
    height = 0

    def __init__(self, width, height):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height), display=0)
        self.clock = pygame.time.Clock()
        self.generator = Generator(self)
        self.speed = 1
        self.rocket = None

    def run_training(self, client_socket, local_ip, local_port, n_trials,
                     trial_s):
        directions = ["LEFT", "RIGHT"] * n_trials
        random.shuffle(directions)
        print(directions)
        if directions[0] == "LEFT":
            hero = Hero(self, random.randint(self.width // 1.5, self.width - 20),
                        self.height - 20)
        else:
            hero = Hero(self, random.randint(20, self.width // 2.5), self.height - 20)

        # Start with blank
        pygame.font.init()
        pygame.font.SysFont('Arial', 150)
        self.show_msg('READY', 2000)

        # Training
        count = 0
        done = False
        for d_ind in range(len(directions)):
            direction = directions[d_ind]
            self.show_msg(direction, 2000)  # let user know the direction

            print("Starting x: ", hero.x)

            # get the start time
            st = time.time()

            # Send start signal
            bytesToSend = str.encode(direction)
            client_socket.sendto(bytesToSend, (local_ip, local_port))

            while not done:
                if direction == "LEFT":  # sipka doleva
                    if hero.x > 20:
                        hero.x -= self.speed
                else:
                    if hero.x < self.width - 20:
                        hero.x += self.speed

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                    # if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and not self.lost:
                if count % 50 == 0:
                    self.rockets.append(Rocket(self, hero.x, hero.y))
                count += 1

                pygame.display.flip()
                self.clock.tick(60)
                self.screen.fill((0, 0, 0))

                # get the end time
                et = time.time()
                # get the execution time
                elapsed_time = et - st

                if elapsed_time > trial_s:
                    print('Execution time:', elapsed_time, 'seconds')
                    try:
                        if directions[d_ind + 1] == "LEFT":
                            hero.x = random.randint(self.width // 1.5, self.width - 20)
                            print(hero.x)
                        else:
                            hero.x = random.randint(20, self.width // 2.5)
                        Generator(self)
                        pygame.display.flip()
                    except:
                        None
                    break

                for alien in self.aliens:
                    alien.draw()
                    alien.check_collision(self)
                    if alien.y > self.height:
                        self.lost = True
                        self.display_text("YOU DIED")
                        pygame.time.delay(2000)
                        done = True

                for rocket in self.rockets:
                    rocket.draw()

                if not self.lost:
                    hero.draw()
            self.show_msg('', 2000)  # blank screen after trial

    def run_online(self, client_socket, buffer_size):
        # Play Game
        count = 0
        hero = Hero(self, self.width / 2, self.height - 20)
        done = False
        while not done:
            direction, addr = client_socket.recvfrom(buffer_size)
            direction = int(direction.decode(encoding='UTF-8', errors='strict'))
            print(direction)

            if len(self.aliens) == 0:
                self.display_text("VICTORY ACHIEVED")

            if direction == 0:
                st = time.time()
                while time.time() - st < 1:
                    hero.x -= self.speed if hero.x > 20 else 0
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            done = True
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and not self.lost:
                            self.rockets.append(Rocket(self, hero.x, hero.y))
                    if count % 50 == 0:
                        self.rockets.append(Rocket(self, hero.x, hero.y))
                    count += 1

                    pygame.display.flip()
                    self.clock.tick(60)
                    self.screen.fill((0, 0, 0))

                    for alien in self.aliens:
                        alien.draw()
                        alien.check_collision(self)
                        if alien.y > self.height:
                            self.lost = True
                            self.display_text("YOU DIED")

                    for rocket in self.rockets:
                        rocket.draw()

                    if not self.lost:
                        hero.draw()

            elif direction == 1:
                st = time.time()
                while time.time() - st < 1:
                    hero.x += self.speed if hero.x < self.width - 20 else 0
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            done = True
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and not self.lost:
                            self.rockets.append(Rocket(self, hero.x, hero.y))
                    if count % 50 == 0:
                        self.rockets.append(Rocket(self, hero.x, hero.y))
                    count += 1

                    pygame.display.flip()
                    self.clock.tick(60)
                    self.screen.fill((0, 0, 0))

                    for alien in self.aliens:
                        alien.draw()
                        alien.check_collision(self)
                        if alien.y > self.height:
                            self.lost = True
                            self.display_text("YOU DIED")

                    for rocket in self.rockets:
                        rocket.draw()

                    if not self.lost:
                        hero.draw()

    def display_text(self, text):
        pygame.font.init()
        font = pygame.font.SysFont('Arial', 150)
        textsurface = font.render(text, False, (255, 255, 255))
        self.screen.blit(textsurface, (int(self.width / 2) - 120, 530))

    def show_msg(self, text, duration):
        self.screen.fill((0, 0, 0))
        self.display_text(text)
        pygame.display.flip()
        pygame.time.delay(500)
        self.screen.fill((0, 0, 0))
        self.display_text('')
        pygame.display.flip()
        pygame.time.delay(duration - 500)


class Alien:
    def __init__(self, game, x, y):
        self.x = x
        self.game = game
        self.y = y
        self.size = 60

    def draw(self):
        pygame.draw.rect(self.game.screen,
                         (81, 43, 88),
                         pygame.Rect(self.x, self.y, self.size, self.size))
        # self.y += 0.05

    def check_collision(self, game):
        for rocket in game.rockets:
            if ((rocket.x < self.x + self.size) and (rocket.x > self.x - self.size) and
                    (rocket.y < self.y + self.size) and (rocket.y > self.y - self.size)):
                game.rockets.remove(rocket)
                game.aliens.remove(self)


class Hero:
    def __init__(self, game, x, y):
        self.x = x
        self.game = game
        self.y = y

    def draw(self):
        pygame.draw.rect(self.game.screen,
                         (210, 250, 251),
                         pygame.Rect(self.x, self.y, 80, 50))


class Generator:
    def __init__(self, game):
        margin = 50
        width = 100
        for x in range(margin, game.width - margin, width):
            for y in range(margin, int(game.height / 2), width):
                game.aliens.append(Alien(game, x, y))

        # game.aliens.append(Alien(game, 280, 50))


class Rocket:
    def __init__(self, game, x, y):
        self.x = x
        self.y = y
        self.game = game

    def draw(self):
        pygame.draw.rect(self.game.screen,  # renderovací plocha
                         (254, 52, 110),  # barva objektu
                         pygame.Rect(self.x + 35, self.y, 15, 50))
        self.y -= 8  # poletí po herní ploše nahoru 2px/snímek
