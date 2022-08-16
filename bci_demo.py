import socket
import threading
import time
import pickle
from mi_stream import get_training_data, build_classifier
from space_invaders import Game
from enum import Enum

BUFFER_SIZE = 1024
LOCAL_PORT = 12345
LOCAL_IP = "127.0.0.1"
DATA_FOLDER = r'C:\Users\kkokorin\Documents\GitHub\space_invaders\test_data'

GAME_W = 1393
GAME_H = 833

N_TRIALS = 40
TRIAL_LEN_MS = 5500
MI_START_MS = 1000
MI_STOP_MS = 3000
MI_DURATION_MS = 1000

SAMPLE_RATE_HZ = 250
N_CH = 8

RUN_TYPE = 1  # 1 for train, 2 for classify, 3 for test


class EEGThread(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        print("Starting " + self.name)
        # setup data streaming server
        eeg_server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        eeg_server_socket.bind((LOCAL_IP, LOCAL_PORT))
        eeg_server_socket.setblocking(0)

        get_training_data(TRIAL_LEN_MS, SAMPLE_RATE_HZ, N_CH, eeg_server_socket, BUFFER_SIZE,
                          DATA_FOLDER)

        print("Exiting " + self.name)


class GameThread(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        print("Starting " + self.name)
        # setup data streaming server
        game_client_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        Game(GAME_W, GAME_H, game_client_socket, LOCAL_IP, LOCAL_PORT, BUFFER_SIZE, N_TRIALS,
             TRIAL_LEN_MS//1000)

        print("Exiting " + self.name)


if __name__ == "__main__":
    # start LSL
    # cd OpenBCI_LSL
    # python openbci_lsl.py COM6 --stream
    # /start /stop /exit

    if RUN_TYPE == 1:
        print('Getting training data')
        # setup eeg stream
        eeg_thread = EEGThread(1, "EEG stream thread", 1)
        eeg_thread.start()

        # setup game
        time.sleep(2)
        game_thread = GameThread(2, "Game thread", 2)
        game_thread.start()

    elif RUN_TYPE == 2:
        print('Building classifier')
        build_classifier(DATA_FOLDER, SAMPLE_RATE_HZ, N_CH, N_TRIALS,
                         MI_START_MS//1000*SAMPLE_RATE_HZ, MI_STOP_MS//1000*SAMPLE_RATE_HZ,
                         MI_DURATION_MS//1000*SAMPLE_RATE_HZ)
    elif RUN_TYPE == 3:
        # to do
        model_file = '%s//clf.sav' % DATA_FOLDER
        loaded_model = pickle.load(open(model_file, 'rb'))
        print('Running test')
    else:
        print('Invalid mode')
