import random
import time

import pygame


class Game:
    """Game class for Space Invaders game"""

    speed = 1

    def __init__(self, width, height):
        """Setup game screen

        Args:
            width (int): screen width
            height (int): screen height
        """
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode(
            (width, height), pygame.FULLSCREEN, display=1
        )
        self.clock = pygame.time.Clock()
        self.generator = Generator(self)
        self.rocket = None
        self.aliens = []
        self.rockets = []
        self.lost = False

    def run_training(self, client_socket, local_ip, local_port, n_trials, trial_s):
        """Record training data while the ship moves left or right

        Args:
            client_socket (socket): game client
            local_ip (str): IP address
            local_port (int): port
            n_trials (int): number of trials
            trial_s (float): trial duration (s)
        """
        directions = ["LEFT", "RIGHT"] * n_trials
        random.shuffle(directions)
        print(directions)
        if directions[0] == "LEFT":
            hero = Hero(
                self,
                random.randint(self.width // 1.5, self.width - 20),
                self.height - 20,
            )
        else:
            hero = Hero(self, random.randint(20, self.width // 2.5), self.height - 20)

        # Start with blank
        pygame.font.init()
        pygame.font.SysFont("Arial", 150)
        self.show_msg("READY", 2000)

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
                self.aliens = []
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
                    print("Execution time:", elapsed_time, "seconds")
                    try:
                        if directions[d_ind + 1] == "LEFT":
                            hero.x = random.randint(self.width // 1.5, self.width - 20)
                            print(hero.x)
                        else:
                            hero.x = random.randint(20, self.width // 2.5)
                        self.aliens = []
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
            self.show_msg("", 2000)  # blank screen after trial

    def run_online(self, client_socket, buffer_size):
        """Play the game using client commands

        Args:
            client_socket (socket): game client
            buffer_size (int): buffer samples
        """
        # Play Game
        count = 0
        hero = Hero(self, self.width / 2, self.height - 20)
        done = False
        while not done:
            direction, addr = client_socket.recvfrom(buffer_size)
            direction = int(direction.decode(encoding="UTF-8", errors="strict"))
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
                        if (
                            event.type == pygame.KEYDOWN
                            and event.key == pygame.K_SPACE
                            and not self.lost
                        ):
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
                        if (
                            event.type == pygame.KEYDOWN
                            and event.key == pygame.K_SPACE
                            and not self.lost
                        ):
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
        """Show text overlay in game

        Args:
            text (str): text to display
        """
        pygame.font.init()
        font = pygame.font.SysFont("Arial", 150)
        textsurface = font.render(text, False, (255, 255, 255))
        self.screen.blit(textsurface, (int(self.width / 2) - 120, 530))

    def show_msg(self, text, duration):
        """Show text over black screen

        Args:
            text (str): text to display
            duration (float): duration to show screen with text
        """
        self.screen.fill((0, 0, 0))
        self.display_text(text)
        pygame.display.flip()
        pygame.time.delay(500)
        self.screen.fill((0, 0, 0))
        self.display_text("")
        pygame.display.flip()
        pygame.time.delay(duration)


class Alien:
    """Alien class for Space Invaders game"""

    def __init__(self, game, x, y):
        """Setup alien

        Args:
            game (Game): game instance
            x (int): x position
            y (int): y position
        """
        self.x = x
        self.game = game
        self.y = y
        self.size = 60

    def draw(self):
        """Draw the alien on the screen"""
        pygame.draw.rect(
            self.game.screen,
            (81, 43, 88),
            pygame.Rect(self.x, self.y, self.size, self.size),
        )

    def check_collision(self, game):
        """Check if a rocket has hit the alien

        Args:
            game (Game): game instance
        """
        for rocket in game.rockets:
            if (
                (rocket.x < self.x + self.size)
                and (rocket.x > self.x - self.size)
                and (rocket.y < self.y + self.size)
                and (rocket.y > self.y - self.size)
            ):
                game.rockets.remove(rocket)
                game.aliens.remove(self)


class Hero:
    """Spaceship for Space Invaders game"""

    def __init__(self, game, x, y):
        """Setup spaceship

        Args:
            game (Game): game instance
            x (int): x position
            y (int): y position
        """
        self.x = x
        self.game = game
        self.y = y

    def draw(self):
        """Draw the spaceship on the screen"""
        pygame.draw.rect(
            self.game.screen, (210, 250, 251), pygame.Rect(self.x, self.y, 80, 50)
        )


class Generator:
    """Generate aliens for Space Invaders game"""

    margin = 50
    width = 100

    def __init__(self, game):
        """Update array of aliens

        Args:
            game (Game): game instance
        """
        for x in range(self.margin, game.width - self.margin, self.width):
            for y in range(self.margin, int(game.height / 2), self.width):
                game.aliens.append(Alien(game, x, y))


class Rocket:
    """Rocket class for Space Invaders game"""

    def __init__(self, game, x, y):
        """Setup rocket

        Args:
            game (Game): game instance
            x (int): x position
            y (int): y position
        """

        self.x = x
        self.y = y
        self.game = game

    def draw(self):
        """Draw rocket on the screen"""
        pygame.draw.rect(
            self.game.screen,
            (254, 52, 110),
            pygame.Rect(self.x + 35, self.y, 15, 50),
        )
        self.y -= 8
