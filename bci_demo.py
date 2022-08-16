import socket
import threading
import time
import pickle
from mi_stream import get_training_data, build_classifier, process_online_data
from space_invaders import Game

BUFFER_SIZE = 1024
LOCAL_PORT = 12345
LOCAL_IP = "127.0.0.1"
DATA_FOLDER = r'C:\Users\kkokorin\Documents\GitHub\space_invaders\test_data'

GAME_W = 1393
GAME_H = 833

N_TRIALS = 5
TRIAL_LEN_MS = 5500
MI_START_MS = 1000
MI_STOP_MS = 3000
MI_DURATION_MS = 1000

SAMPLE_RATE_HZ = 250
N_CH = 8

RUN_TYPE = 1  # 1 for train, 2 for classify, 3 for test


class MyThread(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter


class EEGTrainingThread(MyThread):
    def run(self):
        print("Starting " + self.name)
        # setup data streaming server
        eeg_server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        eeg_server_socket.bind((LOCAL_IP, LOCAL_PORT))
        eeg_server_socket.setblocking(0)

        get_training_data(TRIAL_LEN_MS, SAMPLE_RATE_HZ, N_CH, N_TRIALS, eeg_server_socket,
                          BUFFER_SIZE, DATA_FOLDER)
        print("Exiting " + self.name)


class EEGOnlineThread(MyThread):
    def run(self):
        print("Starting " + self.name)
        # setup data streaming server
        eeg_server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        eeg_server_socket.bind((LOCAL_IP, LOCAL_PORT))
        eeg_server_socket.setblocking(0)

        process_online_data()
        print("Exiting " + self.name)


class GameTrainingThread(MyThread):
    def run(self):
        print("Starting " + self.name)
        # setup data streaming server
        game_client_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        space_invaders = Game(GAME_W, GAME_H)
        space_invaders.run_training(game_client_socket, LOCAL_IP, LOCAL_PORT, N_TRIALS,
                                    TRIAL_LEN_MS//1000)
        print("Exiting " + self.name)


class GamePlayThread(MyThread):
    def run(self):
        print("Starting " + self.name)
        # setup data streaming server
        game_client_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        space_invaders = Game(GAME_W, GAME_H)
        space_invaders.run_online()
        print("Exiting " + self.name)


if __name__ == "__main__":
    # start LSL (or use GUI)
    # cd OpenBCI_LSL
    # python openbci_lsl.py COM6 --stream
    # /start /stop /exit

    if RUN_TYPE == 1:
        print('Getting training data')
        # setup eeg stream
        eeg_thread = EEGTrainingThread(1, "EEG training thread", 1)
        eeg_thread.start()

        # setup game
        time.sleep(2)
        game_thread = GameTrainingThread(2, "Game training thread", 2)
        game_thread.start()

    elif RUN_TYPE == 2:
        print('Building classifier')
        build_classifier(DATA_FOLDER, SAMPLE_RATE_HZ, N_CH, N_TRIALS,
                         MI_START_MS//1000*SAMPLE_RATE_HZ, MI_STOP_MS//1000*SAMPLE_RATE_HZ,
                         MI_DURATION_MS//1000*SAMPLE_RATE_HZ)
    elif RUN_TYPE == 3:
        # to do
        print('Loading model')
        model_file = '%s//clf.sav' % DATA_FOLDER
        loaded_model = pickle.load(open(model_file, 'rb'))

        print('Running test')
        # setup eeg stream
        eeg_thread = EEGOnlineThread(1, "online EEG thread", 1)
        eeg_thread.start()

        # setup game
        time.sleep(2)
        game_thread = GamePlayThread(2, "Game play thread", 2)
        game_thread.start()
    else:
        print('Invalid mode')
