import serial
import scipy
import numpy as np
from pylsl import StreamInlet, resolve_stream
import socket
import os
import pygame

def process_eeg_block(time_data, eeg_data, title, trial_n):
    print(time_data.shape, eeg_data.shape)
    np.savetxt("MI_trials/"+title+"_"+str(trial_n)+".csv", eeg_data, delimiter=",")
    return

def read_eeg_data(window_size, Fs, Nch, trial_n):
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    #store batch data
    samples_per_window = Fs*window_size//1000

    eeg_data = np.zeros((samples_per_window, Nch))
    time_data = np.zeros(samples_per_window)

    eeg_sample, start_time = inlet.pull_sample()
    eeg_data[0, :] = eeg_sample
    time_data[0] = start_time
    n_samples = 1

    #record
    while True:
        # get a new sample to maintain synchrony?
        eeg_sample, timestamp = inlet.pull_sample()
        # print(int(np.mean(eeg_sample)))

        try: # Keep running the non-blocking udp and make sure you run the eeg
        # if UDPClientSocket.recv is not None:
            msgFromServer, addr = UDPClientSocket.recvfrom(bufferSize)
            msg = msgFromServer.decode(encoding = 'UTF-8', errors = 'strict')
            print("Direction: ", msg)

            if int(msg) != 666:
                while n_samples < samples_per_window:
                    # get a new sample
                    eeg_sample, timestamp = inlet.pull_sample()

                    #store samples
                    eeg_data[n_samples,:] = eeg_sample
                    time_data[n_samples] = timestamp
                    n_samples += 1

                    if n_samples >= samples_per_window:
                        print((timestamp-start_time)*1000)
                        time_window = time_data[:]
                        eeg_window = eeg_data[:,:]

                        process_eeg_block(time_window, eeg_window, msg, trial_n)
                        start_time = timestamp
                        trial_n += 1

                        print("Stop")
                n_samples = 0
            else:
                print("Stop training")
                break

        except socket.error: # Non-blocking EEG will return error if none
            pass

    ############################################################################
    # Read data and fit classifier
    import os
    import glob

    path = 'MI_trials'
    extension = 'csv'
    os.chdir(path)
    csvFiles = glob.glob('*.{}'.format(extension))
    # print(csvFiles)

    from numpy import genfromtxt
    slideWin = Fs * 2
    outputHz = 1
    all_trials = []
    y = []
    for csvFile in csvFiles:
        file = genfromtxt(csvFile, delimiter=',')
        trialType = int(csvFile.split('_')[0])
        # print(trialType, file.shape)

        for i in range(0, file.shape[0]-(slideWin-Fs), outputHz*Fs):
            trial = file[i:i+slideWin, :]
            var = np.var(trial, axis=0)
            log_var = np.log(var/sum(var))
            all_trials.append(log_var)
            y.append(trialType)
    X = np.array(all_trials)
    y = np.array(y)
    n_channels = X.shape[-1]
    n_trials = X.shape[0]
    print(X.shape, y.shape, n_channels)

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    model = LDA(shrinkage='auto', solver='eigen')
    model.fit(X, y)
    ############################################################################


    # msg = 1
    n_samples = 0
    while True:
        # get a new sample
        eeg_sample, timestamp = inlet.pull_sample()

        #store samples
        eeg_data[n_samples,:] = eeg_sample
        n_samples += 1

        if n_samples >= slide_win:
            n_samples = 0
            while True:
                # get a new sample
                eeg_sample, timestamp = inlet.pull_sample()
                eeg_data[:-1,:] = eeg_data[1:,:].copy()
                eeg_data[-1,:] = eeg_sample
                n_samples += 1

                if n_samples == output_hz*Fs:
                    # Put classifier here to give output
                    msg = model.predict(eeg_data)
                    msg_encode = str.encode(str(msg))
                    UDPClientSocket.sendto(msg_encode, addr)
                    print(msg)
                    n_samples = 0




WINDOW_SIZE_MS = 8000
SAMPLE_RATE_HZ = 250
N_CH = 8
trial_n = 1

serverAddressPort   = ("127.0.0.1", 12345)
bufferSize = 1024

# Create a UDP socket at client side
UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
UDPClientSocket.bind(serverAddressPort)
UDPClientSocket.setblocking(0)


if __name__ == "__main__":
    read_eeg_data(WINDOW_SIZE_MS, SAMPLE_RATE_HZ, N_CH, trial_n)
