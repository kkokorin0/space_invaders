import pickle
import socket
import threading
import time

from mi_stream import (
    build_classifier,
    create_mi_filters,
    get_training_data,
    process_online_data,
)
from space_invaders import Game

# Comms
BUFFER_SIZE = 1024
LOCAL_PORT = 12345
LOCAL_IP = "127.0.0.1"
DATA_FOLDER = r""

# Game window
GAME_W = 1280
GAME_H = 720

# Trial parameters
N_TRIALS = 10
TRIAL_LEN_MS = 5500
MI_START_MS = 1000
MI_STOP_MS = 5000
MI_DURATION_MS = 1000

# Decoding parameters
SAMPLE_RATE_HZ = 250
N_CH = 3
MU_BAND = [7.5, 12.5]
BETA_BAND = [12.5, 30]
FILTER_ORDER = 10
STOPBAND_DB = 40
PSD_WINDOW = [8, 30]

# Select run type
RUN_TYPE = 2  # 1 for train, 2 for classify, 3 for test


class MyThread(threading.Thread):
    """Custom thread class

    Args:
        threading.Thread: thread
    """

    def __init__(self, threadID, name, counter):
        """Setup new thread

        Args:
            threadID (int): thread id
            name (str): thread label
            counter (int): thread count
        """
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter


class EEGTrainingThread(MyThread):
    """Thread for recording EEG data for training

    Args:
        MyThread: custom thread
    """

    def run(self):
        """Setup thread and get training data from EEG stream"""
        print("Starting " + self.name)
        # setup data streaming server
        eeg_server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        eeg_server_socket.bind((LOCAL_IP, LOCAL_PORT))
        eeg_server_socket.setblocking(0)

        get_training_data(
            TRIAL_LEN_MS,
            SAMPLE_RATE_HZ,
            N_CH,
            N_TRIALS,
            eeg_server_socket,
            BUFFER_SIZE,
            DATA_FOLDER,
        )
        print("Exiting " + self.name)


class EEGOnlineThread(MyThread):
    """Thread for online EEG data processing

    Args:
        MyThread: custom thread
    """

    def run(self):
        """Setup thread and process online EEG data for game control"""
        print("Starting " + self.name)
        print("Loading model")
        model_file = "%s//clf.sav" % DATA_FOLDER
        loaded_model = pickle.load(open(model_file, "rb"))

        # setup data streaming server
        eeg_server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        psd_params = [SAMPLE_RATE_HZ] + PSD_WINDOW
        process_online_data(
            MI_DURATION_MS * SAMPLE_RATE_HZ // 1000,
            N_CH,
            psd_params,
            loaded_model,
            eeg_server_socket,
            LOCAL_IP,
            LOCAL_PORT,
        )
        print("Exiting " + self.name)


class GameTrainingThread(MyThread):
    """Thread for training game

    Args:
        MyThread: custom thread
    """

    def run(self):
        """Run training game"""
        print("Starting " + self.name)
        # setup data streaming server
        game_client_socket = socket.socket(
            family=socket.AF_INET, type=socket.SOCK_DGRAM
        )
        space_invaders = Game(GAME_W, GAME_H)
        space_invaders.run_training(
            game_client_socket, LOCAL_IP, LOCAL_PORT, N_TRIALS, TRIAL_LEN_MS // 1000
        )
        print("Exiting " + self.name)


class GamePlayThread(MyThread):
    """Thread for playing game

    Args:
        MyThread: custom thread
    """

    def run(self):
        print("Starting " + self.name)
        # setup data streaming server
        game_client_socket = socket.socket(
            family=socket.AF_INET, type=socket.SOCK_DGRAM
        )
        game_client_socket.bind((LOCAL_IP, LOCAL_PORT))
        space_invaders = Game(GAME_W, GAME_H)
        space_invaders.run_online(game_client_socket, BUFFER_SIZE)
        print("Exiting " + self.name)


if __name__ == "__main__":
    """Run BCI demo for controlling a Space Invaders game using the OpenBCI headset.
    To set up LSL (or use the OpenBCI GUI):
        cd OpenBCI_LSL
        python openbci_lsl.py COM6 --stream
        /start /stop /exit
    """
    if RUN_TYPE == 1:
        print("Getting training data")
        # setup eeg stream
        eeg_thread = EEGTrainingThread(1, "EEG training thread", 1)
        eeg_thread.start()

        # setup game
        time.sleep(2)
        game_thread = GameTrainingThread(2, "Game training thread", 2)
        game_thread.start()

    elif RUN_TYPE == 2:
        print("Building classifier")
        filter_params = create_mi_filters(
            SAMPLE_RATE_HZ, MU_BAND, BETA_BAND, FILTER_ORDER, STOPBAND_DB
        )
        psd_params = [SAMPLE_RATE_HZ] + PSD_WINDOW
        build_classifier(
            DATA_FOLDER,
            N_TRIALS,
            MI_START_MS // 1000 * SAMPLE_RATE_HZ,
            MI_STOP_MS // 1000 * SAMPLE_RATE_HZ,
            MI_DURATION_MS // 1000 * SAMPLE_RATE_HZ,
            psd_params,
        )
    elif RUN_TYPE == 3:
        print("Running test")
        # setup eeg stream
        eeg_thread = EEGOnlineThread(1, "online EEG thread", 1)
        eeg_thread.start()

        # setup game
        time.sleep(2)
        game_thread = GamePlayThread(2, "Game play thread", 2)
        game_thread.start()
    else:
        print("Invalid mode")
