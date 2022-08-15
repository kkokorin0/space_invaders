import pygame
import time
import random
import socket


localIP = "127.0.0.1"
localPort = 12345
bufferSize  = 1024

# Create a datagram socket
UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
print("UDP client up")


class Game:
    screen = None
    aliens = []
    rockets = []
    lost = False

    def __init__(self, width, height):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height), display=0)
        self.clock = pygame.time.Clock()
        done = False

        generator = Generator(self)
        speed = 1
        trial_s = 8
        rocket = None
        count = 0

        directions = ["Left"]
        if directions[0] == "Left":
            hero = Hero(self, random.randint(width//1.5, width - 20), height - 20)
        else:
            hero = Hero(self, random.randint(20, width//2.5), height - 20)

        # Start with blank
        self.screen.fill((0, 0, 0))

        pygame.font.init()
        font = pygame.font.SysFont('Arial', 150)
        textsurface = font.render("Waiting for 3 seconds", False, (255, 255, 255))
        self.screen.blit(textsurface, (110, 160))

        pygame.display.flip()
        pygame.time.delay(3000)
        self.screen.fill((0, 0, 0))

        # Training
        for d_ind in range(len(directions)):
            aliens = []
            direction = directions[d_ind]
            self.displayText(direction)
            pygame.display.flip()
            pygame.time.delay(1000)

            print("Starting x: ", hero.x)

            # get the start time
            st = time.time()

            # Send start signal
            bytesToSend = str.encode(str((d_ind+1)//2))
            UDPClientSocket.sendto(bytesToSend, (localIP, localPort))

            while not done:
                if direction == "Left":  # sipka doleva
                    if hero.x > 20:
                        hero.x -= speed
                else:
                    if hero.x < width - 20:
                        hero.x += speed

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
                        if directions[d_ind+1] == "Left":
                            hero.x = random.randint(width//1.5, width-20)
                            print(hero.x)
                        else:
                            hero.x = random.randint(20, width//2.5)
                        generator = Generator(self)
                        pygame.display.flip()
                    except:
                        None
                    break

                for alien in self.aliens:
                    alien.draw()
                    alien.checkCollision(self)
                    if (alien.y > height):
                        self.lost = True
                        self.displayText("YOU DIED")
                        pygame.time.delay(2000)
                        done = True

                for rocket in self.rockets:
                    rocket.draw()

                if not self.lost: hero.draw()


        bytesToSend = str.encode(str(666))
        UDPClientSocket.sendto(bytesToSend, (localIP, localPort))

        hero = Hero(self, width / 2, height - 20)

        # Play Game
        aliens = []
        while not done:
            direction, addr = UDPClientSocket.recvfrom(bufferSize)
            direction = int(direction.decode(encoding = 'UTF-8', errors = 'strict'))
            print(direction)

            if len(self.aliens) == 0:
                self.displayText("VICTORY ACHIEVED")

            if direction == 1:
                st = time.time()
                while time.time() - st < 1:
                    hero.x -= speed if hero.x > 20 else 0
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
                        alien.checkCollision(self)
                        if (alien.y > height):
                            self.lost = True
                            self.displayText("YOU DIED")

                    for rocket in self.rockets:
                        rocket.draw()

                    if not self.lost: hero.draw()

            elif direction == -1:
                st = time.time()
                while time.time() - st < 1:
                    hero.x += speed if hero.x < width - 20 else 0
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
                        alien.checkCollision(self)
                        if (alien.y > height):
                            self.lost = True
                            self.displayText("YOU DIED")

                    for rocket in self.rockets:
                        rocket.draw()

                    if not self.lost: hero.draw()



    def displayText(self, text):
        pygame.font.init()
        font = pygame.font.SysFont('Arial', 150)
        textsurface = font.render(text, False, (255, 255, 255))
        self.screen.blit(textsurface, (int(self.width/2)-120, 530))


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

    def checkCollision(self, game):
        for rocket in game.rockets:
            if (rocket.x < self.x + self.size and
                    rocket.x > self.x - self.size and
                    rocket.y < self.y + self.size and
                    rocket.y > self.y - self.size):
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
                         pygame.Rect(self.x+35, self.y, 15, 50))
        self.y -= 8  # poletí po herní ploše nahoru 2px/snímek


if __name__ == '__main__':
    game = Game(1393, 833)
