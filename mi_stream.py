import numpy as np
import socket
import pandas as pd
import pickle
from pylsl import StreamInlet, resolve_stream
from scipy import signal
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def record_epoch(inlet, Ns, Nch, folder, msg, index):
    eeg_data = np.zeros((Ns, Nch))
    time_data = np.zeros(Ns)
    n_samples = 0

    while n_samples < Ns:
        # get a new sample
        eeg_sample, timestamp = inlet.pull_sample()
        eeg_data[n_samples, :] = eeg_sample
        time_data[n_samples] = timestamp
        n_samples += 1

    fname = '%s//%s_%d.csv' % (folder, msg, index+1)
    np.savetxt(fname, eeg_data, delimiter=",")


def build_classifier(folder, Fs, Nch, n_trials, mi_start, mi_stop, mi_len,
                     bp_order=10, stopband_db=40):
    # bp_sos_mu = signal.cheby2(N=bp_order, rs=stopband_db, Wn=[2, 13],
    #                           btype='bandpass', analog=False, output='sos', fs=Fs)
    bp_sos_beta = signal.cheby2(N=bp_order, rs=stopband_db, Wn=[2, 20],
                                btype='bandpass', analog=False, output='sos', fs=Fs)
    Nfilt = 1

    # total trials after subdivision
    n_slices = (mi_stop - mi_start) // mi_len
    n_split_trials = n_trials*n_slices
    y = np.array([0]*n_split_trials + [1]*n_split_trials)  # 0 left, 1 right
    X = np.zeros((n_split_trials*2, Nch*Nfilt))
    split_i = 0

    print('Feature extraction')
    for direction in ['LEFT', 'RIGHT']:
        for trial_i in range(1, n_trials+1):
            # read data from csv
            fname = '%s//%s_%d.csv' % (folder, direction, trial_i)
            raw_data = pd.read_csv(fname, header=None).to_numpy()

            # common average reference
            rref_data = raw_data.transpose() - np.mean(raw_data, axis=1)

            # bandpass
            # filtered_mu = signal.sosfilt(bp_sos_mu, rref_data.transpose(), axis=0)
            filtered_beta = signal.sosfilt(bp_sos_beta, rref_data.transpose(), axis=0)
            # filtered_data = np.concatenate((filtered_mu, filtered_beta), axis=1)
            filtered_data = filtered_beta

            # subdivide each trial
            imagery_data = filtered_data[mi_start:mi_stop, :]
            split_data = np.reshape(imagery_data, (n_slices, mi_len, -1))

            # extract log-variance features
            for slice_i in range(0, n_slices):
                eeg_slice = split_data[slice_i, :, :]
                var = np.var(eeg_slice, axis=0)
                X[split_i, :] = np.log(var/sum(var))
                split_i += 1

    print('Building classifier')

    # LDA classifier
    model = LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr')
    model.fit(X, y)

    # cross validation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Cross validation mean=%.2f var=%.2f' % (np.mean(scores), np.var(scores)))

    model_file = '%s//clf.sav' % folder
    pickle.dump(model, open(model_file, 'wb'))


def get_training_data(window_size_ms, Fs, Nch, n_trials, server_socket, buffer_size, folder):
    # first resolve an EEG stream on the lab network
    print('looking for an EEG stream...')
    streams = resolve_stream('type', 'EEG')
    print('EEG stream found')

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    # store batch data
    Ns = Fs*window_size_ms//1000

    # record eeg data
    left_i = 0
    right_i = 0
    while (left_i < n_trials) or (right_i < n_trials):
        # keep running the non-blocking udp and make sure you run the eeg
        try:
            msgFromServer, addr = server_socket.recvfrom(buffer_size)
            msg = msgFromServer.decode(encoding='UTF-8', errors='strict')
            print("Direction: ", msg)

            if isinstance(msg, str) and msg == 'LEFT':
                record_epoch(inlet, Ns, Nch, folder, msg, left_i)
                left_i += 1
            elif isinstance(msg, str) and msg == 'RIGHT':
                record_epoch(inlet, Ns, Nch, folder, msg, right_i)
                right_i += 1

        # non-blocking socket will return error if no msg
        except socket.error:
            pass
    print('Completed %d left and %d right trials' % (left_i, right_i))


def process_online_data():
    # msg = 1
    n_samples = 0
    # while True:
    #     # get a new sample
    #     eeg_sample, timestamp = inlet.pull_sample()
    #
    #     #store samples
    #     eeg_data[n_samples,:] = eeg_sample
    #     n_samples += 1
    #
    #     if n_samples >= slide_win:
    #         n_samples = 0
    #         while True:
    #             # get a new sample
    #             eeg_sample, timestamp = inlet.pull_sample()
    #             eeg_data[:-1,:] = eeg_data[1:,:].copy()
    #             eeg_data[-1,:] = eeg_sample
    #             n_samples += 1
    #
    #             if n_samples == output_hz*Fs:
    #                 # Put classifier here to give output
    #                 msg = model.predict(eeg_data)
    #                 msg_encode = str.encode(str(msg))
    #                 server_socket.sendto(msg_encode, addr)
    #                 print(msg)
    #                 n_samples = 0