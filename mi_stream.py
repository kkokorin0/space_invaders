import numpy as np
import socket
import pandas as pd
import pickle
from pylsl import StreamInlet, resolve_stream
from scipy import signal
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import svm
from statistics import mode


def record_epoch(inlet, Ns, Nch, folder, msg, index):
    eeg_data = np.zeros((Ns, Nch))
    time_data = np.zeros(Ns)
    n_samples = 0

    inlet.flush()
    while n_samples < Ns:
        # get a new sample
        eeg_sample, timestamp = inlet.pull_sample()
        eeg_data[n_samples, :] = eeg_sample[0:Nch]
        time_data[n_samples] = timestamp
        n_samples += 1

    fname = '%s//%s_%d.csv' % (folder, msg, index+1)
    np.savetxt(fname, eeg_data, delimiter=",")


def create_mi_filters(Fs, mu_pass_band, beta_pass_band, order, stopband_db):
    bp_sos_mu = signal.cheby2(N=order, rs=stopband_db, Wn=mu_pass_band,
                              btype='bandpass', analog=False, output='sos', fs=Fs)
    bp_sos_beta = signal.cheby2(N=order, rs=stopband_db, Wn=beta_pass_band,
                                btype='bandpass', analog=False, output='sos', fs=Fs)
    return [bp_sos_mu, bp_sos_beta]


def process_eeg_block(raw_data, filter_params):
    # common average reference
    # rref_data = raw_data.transpose() - np.mean(raw_data, axis=1)

    # Cz ref
    rref_data = raw_data.transpose() - raw_data[:, 2]
    return rref_data.transpose()
    # return raw_data

    # # bandpass
    # filter_sets = []
    # for filter_sos in filter_params:
    #     filter_sets.append(signal.sosfilt(filter_sos, rref_data.transpose(), axis=0))
    #
    # return np.concatenate(filter_sets, axis=1)


def log_var_feature(eeg_data):
    var = np.var(eeg_data, axis=0)
    return np.log(var / sum(var))


def psd_feature(eeg_data, Fs, Fmin, Fmax):
    Ns, Nch = eeg_data.shape
    psd_list = []
    for ch_i in range(0, Nch):
        channel_eeg = np.squeeze(eeg_data[:, ch_i])
        # freqs, psd = signal.periodogram(channel_eeg, fs=Fs)
        freqs, psd = signal.welch(channel_eeg, fs=Fs, nperseg=Fs)
        psd_list.append(psd[Fmin:Fmax+1])

    return np.array(psd_list).flatten()


def build_classifier(folder, Nch, n_trials, mi_start, mi_stop, mi_len, filter_params, psd_params):
    # total trials after subdivision
    n_slices = (mi_stop - mi_start) // mi_len
    n_split_trials = n_trials*n_slices
    y = np.array([0]*n_split_trials + [1]*n_split_trials)  # 0 left, 1 right
    # X = np.zeros((n_split_trials*2, Nch))
    X = []
    split_i = 0

    print('Feature extraction')
    for direction in ['LEFT', 'RIGHT']:
        print(direction)
        psd_list = []
        for trial_i in range(1, n_trials+1):
            # read data from csv
            fname = '%s//%s_%d.csv' % (folder, direction, trial_i)
            raw_data = pd.read_csv(fname, header=None).to_numpy()

            # rref and bp
            filtered_data = process_eeg_block(raw_data, filter_params)

            # subdivide each trial
            imagery_data = filtered_data[mi_start:mi_stop, :]
            split_data = np.reshape(imagery_data, (n_slices, mi_len, -1))

            # extract log-variance features
            for slice_i in range(n_slices):
                slice_data = split_data[slice_i, :, :]
                # feature_vector = log_var_feature(slice_data)
                feature_vector = (psd_feature(slice_data, psd_params[0], psd_params[1], psd_params[2]))
                X.append(feature_vector)
                # psd_list.append(feature_vector)
                # print(feature_vector)

                split_i += 1
        # print(np.mean(np.array(psd_list), axis=1))
    X = np.array(X)

    # LDA classifier
    print('Fitting classifier')
    # model = LDA(shrinkage='auto', solver='lsqr')
    model = make_pipeline(StandardScaler(), LDA(shrinkage='auto', solver='lsqr'))

    # SVM classifier
    # svm_params = {'kernel': ('linear', 'rbf'), 'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10, 100]}
    # model = svm.SVC(kernel='rbf')
    # model = GridSearchCV(svc, svm_params, cv=10, refit=True)
    # print(model)

    # cross validation
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Accuracies:', scores)
    print('Cross validation mean=%.2f var=%.3f' % (np.mean(scores), np.var(scores)))

    model.fit(X, y)
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
        # get a new sample to maintain synchrony?
        eeg_sample, timestamp = inlet.pull_sample()

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

            print('Completed %d,%d of %d' % (left_i, right_i, n_trials))

        # non-blocking socket will return error if no msg
        except socket.error:
            pass
    print('Completed %d left and %d right trials' % (left_i, right_i))


def process_online_data(Ns, Nch, filter_params, psd_params, model,
                        server_socket, local_ip, local_port):
    # first resolve an EEG stream on the lab network
    print('looking for an EEG stream...')
    streams = resolve_stream('type', 'EEG')
    print('EEG stream found')

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    n_samples = 0
    eeg_data = np.zeros((Ns, Nch))
    cmd_list = []
    while True:
        if n_samples < Ns:
            # get a new sample
            eeg_sample, timestamp = inlet.pull_sample()
            eeg_data[n_samples, :] = eeg_sample[0:Nch]
            n_samples += 1
        else:
            # process block need to use 5s of data like training
            # print('Raw:', eeg_data)
            filtered_data = process_eeg_block(eeg_data, filter_params)
            # print('Filtered:', filtered_data)

            # print(filtered_data)
            # X = log_var_feature(filtered_data)
            X = psd_feature(filtered_data, psd_params[0],
                            psd_params[1], psd_params[2])


            # send msg to game
            cmd = model.predict(X.reshape(1, -1))[0]
            if len(cmd_list) < 3:
                msg = cmd
            else:
                cmd_list.append(cmd)
                cmd_list = cmd_list[1:]
                msg = mode(cmd_list)

            msg_encode = str.encode(str(msg))
            server_socket.sendto(msg_encode, (local_ip, local_port))

            # reset vars
            n_samples = 0
            eeg_data = np.zeros((Ns, Nch))
