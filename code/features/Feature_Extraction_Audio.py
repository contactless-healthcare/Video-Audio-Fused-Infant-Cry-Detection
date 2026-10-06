###########################################################################
#   Feature extraction is based on pyAudioAnalysis
#   Ref:  https://github.com/tyiannak/pyAudioAnalysis
###########################################################################

from __future__ import print_function
import numpy as np
import sys
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from scipy.fftpack.realtransforms import dct
from tqdm import tqdm
import config
import os
import librosa
import utils


eps = sys.float_info.epsilon

def dc_normalize(sig_array):
    """Removes DC and normalizes to -1, 1 range"""
    sig_array_norm = sig_array.copy()
    sig_array_norm -= sig_array_norm.mean()
    sig_array_norm /= abs(sig_array_norm).max() + 1e-10
    return sig_array_norm


def zero_crossing_rate(frame):
    """Computes zero crossing rate of frame"""
    count = len(frame)
    count_zero = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    return np.float64(count_zero) / np.float64(count - 1.0)


def energy(frame):
    """Computes signal energy of frame"""
    return np.sum(frame ** 2) / np.float64(len(frame))


def energy_entropy(frame, n_short_blocks=10):
    """Computes entropy of energy"""
    # total frame energy
    frame_energy = np.sum(frame ** 2)
    frame_length = len(frame)
    sub_win_len = int(np.floor(frame_length / n_short_blocks))
    if frame_length != sub_win_len * n_short_blocks:
        frame = frame[0:sub_win_len * n_short_blocks]

    # sub_wins is of size [n_short_blocks x L]
    sub_wins = frame.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = np.sum(sub_wins ** 2, axis=0) / (frame_energy + eps)

    # Compute entropy of the normalized sub-frame energies:
    entropy = -np.sum(s * np.log2(s + eps))
    return entropy


""" Frequency-domain audio features """


def spectral_centroid_spread(fft_magnitude, sampling_rate):
    """Computes spectral centroid of frame (given abs(FFT))"""
    ind = (np.arange(1, len(fft_magnitude) + 1)) * \
          (sampling_rate / (2.0 * len(fft_magnitude)))

    Xt = fft_magnitude.copy()
    Xt_max = Xt.max()
    if Xt_max == 0:
        Xt = Xt / eps
    else:
        Xt = Xt / Xt_max

    NUM = np.sum(ind * Xt)
    DEN = np.sum(Xt) + eps

    # Centroid:
    centroid = (NUM / DEN)

    # Spread:
    spread = np.sqrt(np.sum(((ind - centroid) ** 2) * Xt) / DEN)

    # Normalize:
    centroid = centroid / (sampling_rate / 2.0)
    spread = spread / (sampling_rate / 2.0)

    return centroid, spread


def spectral_entropy(signal, n_short_blocks=10):
    """Computes the spectral entropy"""
    # number of frame samples
    num_frames = len(signal)

    # total spectral energy
    total_energy = np.sum(signal ** 2)

    # length of sub-frame
    sub_win_len = int(np.floor(num_frames / n_short_blocks))
    if num_frames != sub_win_len * n_short_blocks:
        signal = signal[0:sub_win_len * n_short_blocks]

    # define sub-frames (using matrix reshape)
    sub_wins = signal.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # compute spectral sub-energies
    s = np.sum(sub_wins ** 2, axis=0) / (total_energy + eps)

    # compute spectral entropy
    entropy = -np.sum(s * np.log2(s + eps))

    return entropy


def spectral_flux(fft_magnitude, previous_fft_magnitude):
    """
    Computes the spectral flux feature of the current frame
    ARGUMENTS:
        fft_magnitude:            the abs(fft) of the current frame
        previous_fft_magnitude:        the abs(fft) of the previous frame
    """
    # compute the spectral flux as the sum of square distances:
    fft_sum = np.sum(fft_magnitude + eps)
    previous_fft_sum = np.sum(previous_fft_magnitude + eps)
    sp_flux = np.sum(
        (fft_magnitude / fft_sum - previous_fft_magnitude /
         previous_fft_sum) ** 2)

    return sp_flux


def spectral_rolloff(signal, c):
    """Computes spectral roll-off"""
    energy = np.sum(signal ** 2)
    fft_length = len(signal)
    threshold = c * energy
    # Find the spectral rolloff as the frequency position
    # where the respective spectral energy is equal to c*totalEnergy
    cumulative_sum = np.cumsum(signal ** 2) + eps
    a = np.nonzero(cumulative_sum > threshold)[0]
    if len(a) > 0:
        sp_rolloff = np.float64(a[0]) / (float(fft_length))
    else:
        sp_rolloff = 0.0
    return sp_rolloff


def mfcc_filter_banks(sampling_rate, num_fft, lowfreq=133.33, linc=200 / 3,
                      logsc=1.0711703, num_lin_filt=13, num_log_filt=27):
    """
    Computes the triangular filterbank for MFCC computation
    (used in the stFeatureExtraction function before the stMFCC function call)
    This function is taken from the scikits.talkbox library (MIT Licence):
    https://pypi.python.org/pypi/scikits.talkbox
    """

    if sampling_rate < 8000:
        nlogfil = 5

    # Total number of filters
    num_filt_total = num_lin_filt + num_log_filt

    # Compute frequency points of the triangle:
    frequencies = np.zeros(num_filt_total + 2)
    frequencies[:num_lin_filt] = lowfreq + np.arange(num_lin_filt) * linc
    frequencies[num_lin_filt:] = frequencies[num_lin_filt - 1] * logsc ** \
                                 np.arange(1, num_log_filt + 3)
    heights = 2. / (frequencies[2:] - frequencies[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((num_filt_total, num_fft))
    nfreqs = np.arange(num_fft) / (1. * num_fft) * sampling_rate

    for i in range(num_filt_total):
        low_freqs = frequencies[i]
        cent_freqs = frequencies[i + 1]
        high_freqs = frequencies[i + 2]

        lid = np.arange(np.floor(low_freqs * num_fft / sampling_rate) + 1,
                        np.floor(cent_freqs * num_fft / sampling_rate) + 1,
                        dtype=int)
        lslope = heights[i] / (cent_freqs - low_freqs)
        rid = np.arange(np.floor(cent_freqs * num_fft / sampling_rate) + 1,
                        np.floor(high_freqs * num_fft / sampling_rate) + 1,
                        dtype=int)
        rslope = heights[i] / (high_freqs - cent_freqs)
        fbank[i][lid] = lslope * (nfreqs[lid] - low_freqs)
        fbank[i][rid] = rslope * (high_freqs - nfreqs[rid])

    return fbank, frequencies


def mfcc(fft_magnitude, fbank, num_mfcc_feats):
    """
    Computes the MFCCs of a frame, given the fft mag
    ARGUMENTS:
        fft_magnitude:  fft magnitude abs(FFT)
        fbank:          filter bank (see mfccInitFilterBanks)
    RETURN
        ceps:           MFCCs (13 element vector)
    Note:    MFCC calculation is, in general, taken from the
             scikits.talkbox library (MIT Licence),
             with small modifications for compactness and use in pyAudioAnalysis.
    """

    mspec = np.log10(np.dot(fft_magnitude, fbank.T) + eps)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:num_mfcc_feats]
    return ceps


def chroma_features_init(num_fft, sampling_rate):
    """
    This function initializes the chroma matrices used in the calculation
    of the chroma features
    """
    freqs = np.array([((f + 1) * sampling_rate) /
                      (2 * num_fft) for f in range(num_fft)])
    cp = 27.50
    num_chroma = np.round(12.0 * np.log2(freqs / cp)).astype(int)

    num_freqs_per_chroma = np.zeros((num_chroma.shape[0],))

    unique_chroma = np.unique(num_chroma)
    for u in unique_chroma:
        idx = np.nonzero(num_chroma == u)
        num_freqs_per_chroma[idx] = idx[0].shape

    return num_chroma, num_freqs_per_chroma


def chroma_features(signal, sampling_rate, num_fft):
    # Verified settings: 16 kHz, 256-sample FFT window (num_fft=128).
    # Other FFT sizes are not validated: small sizes can fail assignment;
    # large sizes can produce negative chroma indices that wrap in NumPy.

    num_chroma, num_freqs_per_chroma = \
        chroma_features_init(num_fft, sampling_rate)
    chroma_names = ['A', 'A#', 'B', 'C', 'C#', 'D',
                    'D#', 'E', 'F', 'F#', 'G', 'G#']
    spec = signal ** 2
    if num_chroma.max() < num_chroma.shape[0]:
        C = np.zeros((num_chroma.shape[0],))
        C[num_chroma] = spec
        C /= num_freqs_per_chroma[num_chroma]
    else:
        I = np.nonzero(num_chroma > num_chroma.shape[0])[0][0]
        C = np.zeros((num_chroma.shape[0],))
        C[num_chroma[0:I - 1]] = spec
        C /= num_freqs_per_chroma
    final_matrix = np.zeros((12, 1))
    newD = int(np.ceil(C.shape[0] / 12.0) * 12)
    C2 = np.zeros((newD,))
    C2[0:C.shape[0]] = C
    C2 = C2.reshape(int(C2.shape[0] / 12), 12)
    final_matrix = np.sum(C2, axis=0).reshape(1, -1).T

    spec_sum = spec.sum()
    if spec_sum == 0:
        final_matrix /= eps
    else:
        final_matrix /= spec_sum


    return chroma_names, final_matrix


def spectrogram(signal, sampling_rate, window, step, plot=False,
                show_progress=False):
    """
    Short-term FFT mag for spectogram estimation:
    Returns:
        a np array (numOfShortTermWindows x num_fft)
    ARGUMENTS:
        signal:         the input signal samples
        sampling_rate:  the sampling freq (in Hz)
        window:         the short-term window size (in samples)
        step:           the short-term window step (in samples)
        plot:           flag, 1 if results are to be ploted
        show_progress flag for showing progress using tqdm
    RETURNS:
    """
    window = int(window)
    step = int(step)
    signal = np.double(signal)
    signal = signal / (2.0 ** 15)
    signal = dc_normalize(signal)

    num_samples = len(signal)  # total number of signals
    count_fr = 0
    num_fft = int(window / 2)
    specgram = np.zeros((int((num_samples-window) / step) + 1, num_fft),
                        dtype=np.float64)
    for cur_p in tqdm(range(window, num_samples - window + 1, step),
                      disable=not show_progress):
        count_fr += 1
        x = signal[cur_p:cur_p + window]
        X = abs(fft(x))
        X = X[0:num_fft]
        X = X / len(X)
        specgram[count_fr-1, :] = X

    freq_axis = [float((f + 1) * sampling_rate) / (2 * num_fft)
                 for f in range(specgram.shape[1])]
    time_axis = [float(t * step) / sampling_rate
                 for t in range(specgram.shape[0])]

    if plot:
        fig, ax = plt.subplots()
        imgplot = plt.imshow(specgram.transpose()[::-1, :])
        fstep = int(num_fft / 5.0)
        frequency_ticks = range(0, int(num_fft) + fstep, fstep)
        frequency_tick_labels = \
            [str(sampling_rate / 2 -
                 int((f * sampling_rate) / (2 * num_fft)))
             for f in frequency_ticks]
        ax.set_yticks(frequency_ticks)
        ax.set_yticklabels(frequency_tick_labels)
        t_step = int(count_fr / 3)
        time_ticks = range(0, count_fr, t_step)
        time_ticks_labels = \
            ['%.2f' % (float(t * step) / sampling_rate) for t in time_ticks]
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(time_ticks_labels)
        ax.set_xlabel('time (secs)')
        ax.set_ylabel('freq (Hz)')
        imgplot.set_cmap('jet')
        plt.colorbar()
        plt.show()
    return specgram, time_axis, freq_axis




""" Windowing and feature extraction """


def feature_extraction(signal, sampling_rate, window, step, deltas=True):
    """
    This function implements the shor-term windowing process.
    For each short-term window a set of features is extracted.
    This results to a sequence of feature vectors, stored in a np matrix.
    ARGUMENTS
        signal:         the input signal samples
        sampling_rate:  the sampling freq (in Hz)
        window:         the short-term window size (in samples)
        step:           the short-term window step (in samples)
        deltas:         (opt) True/False if delta features are to be
                        computed
    RETURNS
        features (numpy.ndarray):        contains features
                                         (n_feats x numOfShortTermWindows)
        feature_names (python list):     contains feature names
                                         (n_feats x numOfShortTermWindows)
    """

    window = int(window)
    step = int(step)

    # signal normalization

    signal = dc_normalize(signal)

    number_of_samples = len(signal)  # total number of samples
    current_position = 0
    count_fr = 0
    num_fft = int(window / 2)

    # compute the triangular filter banks used in the mfcc calculation
    fbank, freqs = mfcc_filter_banks(sampling_rate, num_fft)

    n_time_spectral_feats = 8
    n_harmonic_feats = 0
    n_mfcc_feats = 13
    n_chroma_feats = 13
    n_total_feats = n_time_spectral_feats + n_mfcc_feats + n_harmonic_feats + \
                    n_chroma_feats

    # define list of feature names
    feature_names = ["zcr", "energy", "energy_entropy"]
    feature_names += ["spectral_centroid", "spectral_spread"]
    feature_names.append("spectral_entropy")
    feature_names.append("spectral_flux")
    feature_names.append("spectral_rolloff")
    feature_names += ["mfcc_{0:d}".format(mfcc_i)
                      for mfcc_i in range(1, n_mfcc_feats + 1)]
    feature_names += ["chroma_{0:d}".format(chroma_i)
                      for chroma_i in range(1, n_chroma_feats)]
    feature_names.append("chroma_std")

    # add names for delta features:
    if deltas:
        feature_names_2 = feature_names + ["delta " + f for f in feature_names]
        feature_names = feature_names_2

    features = []
    # for each short-term window to end of signal
    while current_position + window - 1 < number_of_samples:
        count_fr += 1
        # get current window
        x = signal[current_position:current_position + window]

        # update window position
        current_position = current_position + step

        # get fft magnitude
        fft_magnitude = abs(fft(x))

        # normalize fft
        fft_magnitude = fft_magnitude[0:num_fft]
        fft_magnitude = fft_magnitude / len(fft_magnitude)

        # keep previous fft mag (used in spectral flux)
        if count_fr == 1:
            fft_magnitude_previous = fft_magnitude.copy()
        feature_vector = np.zeros((n_total_feats, 1))

        # zero crossing rate
        feature_vector[0] = zero_crossing_rate(x)

        # short-term energy
        feature_vector[1] = energy(x)

        # short-term entropy of energy
        feature_vector[2] = energy_entropy(x)

        # sp centroid/spread
        [feature_vector[3], feature_vector[4]] = \
            spectral_centroid_spread(fft_magnitude,
                                     sampling_rate)

        # spectral entropy
        feature_vector[5] = \
            spectral_entropy(fft_magnitude)

        # spectral flux
        feature_vector[6] = \
            spectral_flux(fft_magnitude,
                          fft_magnitude_previous)

        # spectral rolloff
        feature_vector[7] = \
            spectral_rolloff(fft_magnitude, 0.90)

        # MFCCs
        mffc_feats_end = n_time_spectral_feats + n_mfcc_feats
        feature_vector[n_time_spectral_feats:mffc_feats_end, 0] = \
            mfcc(fft_magnitude, fbank, n_mfcc_feats).copy()

        # chroma features
        chroma_names, chroma_feature_matrix = \
            chroma_features(fft_magnitude, sampling_rate, num_fft)
        chroma_features_end = n_time_spectral_feats + n_mfcc_feats + \
                              n_chroma_feats - 1
        feature_vector[mffc_feats_end:chroma_features_end] = \
            chroma_feature_matrix
        feature_vector[chroma_features_end] = chroma_feature_matrix.std()
        if not deltas:
            features.append(feature_vector)
        else:
            # delta features
            if count_fr > 1:
                delta = feature_vector - feature_vector_prev
                feature_vector_2 = np.concatenate((feature_vector, delta))
            else:
                feature_vector_2 = np.concatenate((feature_vector,
                                                   np.zeros(feature_vector.
                                                            shape)))
            feature_vector_prev = feature_vector
            features.append(feature_vector_2)

        fft_magnitude_previous = fft_magnitude.copy()

    features = np.concatenate(features, 1)
    return features, feature_names


def NICUWav2Segments(dataDir, labelDir, sceneDir):
    raw_audio, sr = librosa.load(dataDir, sr=config.audioSampleRate)
    min_t, max_t = utils.get_valid_time_range(labelDir)
    raw_audio = utils.crop_data_by_time(raw_audio, sr, min_t, max_t)

    # Process cry segments
    cry_ranges = []
    if os.path.exists(labelDir):
        with open(labelDir) as f:
            for line in filter(None, map(str.strip, f)):
                parts = line.replace('\t', ' ').split()
                if len(parts) == 3 and int(parts[2]) == 1:
                    start, end = (int((float(t)-min_t)*sr) for t in parts[:2])
                    cry_ranges.append((start, end))

    # Generate windows and labels
    win_size, step_size = utils.calculate_window_params(sr, config.slidingWindows, config.step)
    valid_indices = np.arange(0, len(raw_audio) - win_size + 1, step_size)
    windows = np.stack([raw_audio[i:i+win_size] for i in valid_indices])
    
    cry_mask = np.zeros(len(raw_audio), dtype=bool)
    for start, end in cry_ranges:
        cry_mask[start:end] = True
    labels = np.array([cry_mask[i:i+win_size].any() for i in valid_indices], dtype=int)

    # Process scene labels
    scene_priority = {3:0, 1:1, 4:2, 2:3, 8:4, 7:5, 6:6, 5:7}
    scene_ranges = {stype: [] for stype in scene_priority}
    
    if sceneDir is not None and os.path.exists(sceneDir):
        with open(sceneDir) as f:
            for line in filter(None, map(str.strip, f)):
                parts = line.replace('\t', ' ').split()
                if len(parts) == 3 and (scene_type := int(parts[2])) in scene_priority:
                    start, end = (int((float(t)-min_t)*sr) for t in parts[:2])
                    scene_ranges[scene_type].append((start, end))

    scene_mask = np.full(len(raw_audio), -1, dtype=int)
    for scene_type in sorted(scene_priority, key=scene_priority.get):
        for start, end in scene_ranges[scene_type]:
            scene_mask[start:end] = scene_type

    scenes = np.zeros(len(valid_indices), dtype=int)
    for i, idx in enumerate(valid_indices):
        window_scenes = scene_mask[idx:idx+win_size]
        valid_scenes = window_scenes[window_scenes != -1]
        if len(valid_scenes) > 0:
            scenes[i] = min(valid_scenes, key=lambda x: scene_priority[x])
        else:
            scenes[i] = 5

    return {"data": windows, "label": labels, "scene": scenes}


def acoustic_features_and_spectrogram(audioData):
    # Feature extraction for acoustic features
    feature_List = []
    feature_names = []
    for i in range(audioData.shape[0]):
        features, base_feature_names = feature_extraction(signal=audioData[i], sampling_rate=config.audioSampleRate,
                                                     window=config.FFTwindow, step=config.FFTOverlap, deltas=False)

        features_mean = np.mean(features, axis=1)
        features_median = np.median(features, axis=1)
        features_std = np.std(features, axis=1)

        features = np.concatenate((features_mean, features_median, features_std))
        feature_List.append(features)

        if i == 0:
            for name in base_feature_names:
                feature_names.append(f"{name}_mean")
            for name in base_feature_names:
                feature_names.append(f"{name}_median")
            for name in base_feature_names:
                feature_names.append(f"{name}_std")

    acoustic_feature = np.array(feature_List)
    return acoustic_feature, feature_names