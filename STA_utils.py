import os
import functools

import h5py
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import seaborn as sns
import scipy.signal as signal
from matplotlib.path import Path
import matplotlib.patches as patches

def smooth(x, axis=0, wid=5):
    # this is way faster than convolve
    if wid < 2:
        return x
    cumsum_vec = np.cumsum(np.insert(x, 0, 0, axis=axis), axis=axis)
    ma_vec = (cumsum_vec[wid:] - cumsum_vec[:-wid]) / wid
    y = x.copy()
    start_ind = int(np.floor((wid-1)/2))
    end_ind = wid-1-start_ind
    y[start_ind:-end_ind] = ma_vec
    return y

def coords2path(coords):
    codes = [Path.MOVETO]
    for i, _ in enumerate(coords):
        if i > 0:
            codes.append(Path.LINETO)
    coords = np.append(coords, coords[0][None], axis=0)
    codes.append(Path.CLOSEPOLY)

    path = Path(coords, codes)
    return path

def path2mask(path, brain_shape):
    pixX = np.arange(brain_shape[1])
    pixY = np.arange(brain_shape[0])
    xv, yv = np.meshgrid(pixX, pixY)
    roi_pix = np.vstack((xv.flatten(), yv.flatten())).T

    mask = np.zeros(shape=xv.shape)

    xy_indices = np.reshape(path.contains_points(roi_pix, radius=0.5), xv.shape)
    mask[xy_indices] = 1

    mask = mask == 1
    return mask

def brain2vec(brain, mask):
    signal = []
    for i in range(brain.shape[0]):
        tmp = brain[i, :, :]
        signal.append(tmp[mask].mean())
    return np.array(signal)


### DO NOT USE interp1d FUNCTION, MESSED THINGS UP
def interp(tp, p, tq, wid=10):
    '''
    interp uses nearby values for generating a smooth function of time (axis=-1)
    '''
    output_shape = list(p.shape)
    output_shape[-1] = tq.shape[0]
    q = np.zeros(output_shape)
    for i, t in enumerate(tq):
        idx = np.argpartition(np.abs(tp-t), wid)[:wid]
        q[..., i] = np.mean(p[..., idx], axis=-1)
    return q


def find_series(name, obj, series_number):
    """return hdf5 group object if it corresponds to indicated series_number."""
    target_group_name = 'series_{}'.format(str(series_number).zfill(3))
    if target_group_name in name:
        return obj
    return None


def encode_stim(keys, epoch):
    coding_str = ''
    for key in keys:
        coding_str = coding_str + key + '_' + str(epoch[key]).replace('.', 'p') + '#'
    return coding_str


def encode_stim_value(keys, values):
    coding_str = ''
    for key, value in zip(keys, values):
        coding_str = coding_str + key + '_' + str(value).replace('.', 'p') + '#'
    return coding_str


def simple_hour2sec(hour_string):
    hour = [float(num) for num in hour_string.split(':')]
    return hour[0]*3600 + hour[1]*60 + hour[2]


class ImagingData():
    __slots__ = ["file_path", "series_number",  "colors", "quiet"]
    def __init__(self, file_path, series_number, quiet=False):
        self.file_path = file_path
        self.series_number = series_number
        self.quiet = quiet

        self.colors = sns.color_palette("Set2", n_colors=20)

        # check to see if hdf5 file exists
        if not os.path.exists(self.file_path):
            raise Exception('No hdf5 file found at {}, check your filepath'.format(self.file_path))

        # check to see if series exists in this file:
        with h5py.File(self.file_path, 'r') as experiment_file:
            find_partial = functools.partial(find_series, series_number=self.series_number)
            epoch_run_group = experiment_file.visititems(find_partial)
            if epoch_run_group is None:
                raise Exception('No series {} found in {}'.format(self.series_number, self.file_path))

    def getRunParameters(self):
        """Return epoch run parameters as dict."""
        with h5py.File(self.file_path, 'r') as experiment_file:
            find_partial = functools.partial(find_series, series_number=self.series_number)
            epoch_run_group = experiment_file.visititems(find_partial)
            run_parameters = {}
            for attr_key in epoch_run_group.attrs:
                run_parameters[attr_key] = epoch_run_group.attrs[attr_key]
        return run_parameters

    def getEpochParameters(self):
        """Return list of epoch parameters, one dict for each trial."""
        with h5py.File(self.file_path, 'r') as experiment_file:
            find_partial = functools.partial(find_series, series_number=self.series_number)
            epoch_run_group = experiment_file.visititems(find_partial)
            epoch_parameters = []
            for epoch in epoch_run_group['epochs'].values():
                new_params = {}
                for attr_key in epoch.attrs:
                    new_params[attr_key] = epoch.attrs[attr_key]
                epoch_parameters.append(new_params)
        return epoch_parameters
    
    def showEpochKeys(self):
        epoch_parameters = self.getEpochParameters()
        print(epoch_parameters[0].keys())
        
    def getStimulusTypes(self, para_key=['color']):
        stims = []
        epoch_parameters = self.getEpochParameters()
        for epoch in epoch_parameters:
            stims.append(encode_stim(para_key, epoch))
        stim_types = list(set(stims))
        type2ind = {tp:ind for ind, tp in enumerate(stim_types)}
        return stims, type2ind
    
    def getStimulusTimingH5(self):
        with h5py.File(self.file_path, 'r') as experiment_file:
            find_partial = functools.partial(find_series, series_number=self.series_number)
            epoch_run_group = experiment_file.visititems(find_partial)
            start_time = simple_hour2sec(epoch_run_group.attrs['run_start_time'])
            stim_time = float(epoch_run_group.attrs['stim_time'])
            stimulus_start_times = []
            stimulus_end_times = []
            for epoch in epoch_run_group['epochs'].values():
                t = simple_hour2sec(epoch.attrs['epoch_time'])
                stimulus_start_times.append(t-start_time)
                stimulus_end_times.append(t-start_time+stim_time)

            stimulus_timing = {'stimulus_end_times': stimulus_end_times,
                       'stimulus_start_times': stimulus_start_times,
                       'dropped_frame_times': [],
                       'frame_rate': 120}
        return stimulus_timing
        
    def getStimulusTiming(self, frame_monitor_channels, time_vector, sample_rate,
                          threshold=0.8,
                          frame_slop=20,  # datapoints +/- ideal frame duration
                          command_frame_rate=120):
        """
        Returns stimulus timing information based on photodiode voltage trace from frame tracker signal.
        """
        frame_monitor_channels[frame_monitor_channels < 0] = 0  # This is to cut the initial ramping of the diode
        run_parameters = self.getRunParameters()
        epoch_parameters = self.getEpochParameters()

        if len(frame_monitor_channels.shape) == 1:
            frame_monitor_channels = frame_monitor_channels[np.newaxis, :]

        minimum_epoch_separation = 0.9 * (run_parameters['pre_time'] + run_parameters['tail_time']) * sample_rate

        num_channels = frame_monitor_channels.shape[0]
        for ch in range(num_channels):
            frame_monitor = frame_monitor_channels[ch, :]

            # Low-pass filter frame_monitor trace
            b, a = signal.butter(4, 10*command_frame_rate, btype='low', fs=sample_rate)
            frame_monitor = signal.filtfilt(b, a, frame_monitor)

            # shift & normalize so frame monitor trace lives on [0 1]
            frame_monitor = frame_monitor - np.min(frame_monitor)
            frame_monitor = frame_monitor / np.max(frame_monitor)

            # find frame flip times
            V_orig = frame_monitor[0:-2]
            V_shift = frame_monitor[1:-1]
            ups = np.where(np.logical_and(V_orig < threshold, V_shift >= threshold))[0] + 1
            downs = np.where(np.logical_and(V_orig >= threshold, V_shift < threshold))[0] + 1
            frame_times = np.sort(np.append(ups, downs))

            # Use frame flip times to find stimulus start times
            stimulus_start_frames = np.append(0, np.where(np.diff(frame_times) > minimum_epoch_separation)[0] + 1)
            stimulus_end_frames = np.append(np.where(np.diff(frame_times) > minimum_epoch_separation)[0], len(frame_times)-1)
            stimulus_start_times = frame_times[stimulus_start_frames] / sample_rate  # datapoints -> sec
            stimulus_end_times = frame_times[stimulus_end_frames] / sample_rate  # datapoints -> sec

            stim_durations = stimulus_end_times - stimulus_start_times # sec

            ideal_frame_len = 1 / command_frame_rate * sample_rate  # datapoints
            frame_durations = []
            dropped_frame_times = []
            for s_ind, ss in enumerate(stimulus_start_frames):
                frame_len = np.diff(frame_times[stimulus_start_frames[s_ind]:stimulus_end_frames[s_ind]+1])
                dropped_frame_inds = np.where(np.abs(frame_len - ideal_frame_len)>frame_slop)[0] + 1 # +1 b/c diff
                if len(dropped_frame_inds) > 0:
                    dropped_frame_times.append(frame_times[ss]+dropped_frame_inds * ideal_frame_len) # time when dropped frames should have flipped
                    # print('Warning! Ch. {} Dropped {} frames in epoch {}'.format(ch, len(dropped_frame_inds), s_ind))
                good_frame_inds = np.where(np.abs(frame_len - ideal_frame_len) <= frame_slop)[0]
                frame_durations.append(frame_len[good_frame_inds]) # only include non-dropped frames in frame rate calc

            if len(dropped_frame_times) > 0:
                dropped_frame_times = np.hstack(dropped_frame_times) # datapoints
            else:
                dropped_frame_times = np.array(dropped_frame_times)

            frame_durations = np.hstack(frame_durations) # datapoints
            measured_frame_len = np.mean(frame_durations)  # datapoints
            frame_rate = 1 / (measured_frame_len / sample_rate)  # Hz

            # Print timing summary
            print('===================TIMING: Channel {}======================'.format(ch))
            print('{} Stims presented (of {} parameterized)'.format(len(stim_durations), len(epoch_parameters)))
            inter_stim_starts = np.diff(stimulus_start_times)
            print('Stim start to start: [min={:.3f}, median={:.3f}, max={:.3f}] / parameterized = {:.3f} sec'.format(inter_stim_starts.min(),
                                                                                                                         np.median(inter_stim_starts),
                                                                                                                         inter_stim_starts.max(),
                                                                                                                run_parameters['stim_time'] + run_parameters['pre_time'] + run_parameters['tail_time']))
            print('Stim duration: [min={:.3f}, median={:.3f}, max={:.3f}] / parameterized = {:.3f} sec'.format(stim_durations.min(), np.median(stim_durations), stim_durations.max(), run_parameters['stim_time']))
            total_frames = len(frame_times)
            dropped_frames = len(dropped_frame_times)
            print('Dropped {} / {} frames ({:.2f}%)'.format(dropped_frames, total_frames, 100*dropped_frames/total_frames))
            print('==========================================================')

        # examine if there is epoch that was not presented in the detected events
        num_of_epochs = len(epoch_parameters)
        num_of_detected_stims = len(stimulus_start_times)
        if num_of_detected_stims < num_of_epochs:
            print('Warning! Detected {} stims out of {} epochs'.format(num_of_detected_stims, num_of_epochs))
            # make a list of which epochs were detected
            run_start_time = run_parameters['run_start_time']
            run_start_time = simple_hour2sec(run_start_time)   
            # for each epoch, get the epoch time in seconds
            epoch_time = [simple_hour2sec(ep['epoch_time']) for ep in epoch_parameters]
            epoch_time = np.array(epoch_time) - run_start_time
            # in the stimulus_start_times, find the closest epoch time, and assign the epoch index to the stimulus
            # this assumes the offset is smaller than half of the epoch duration!!!
            epoch_index = np.zeros(len(stimulus_start_times), dtype=int)
            for i, stim_time in enumerate(stimulus_start_times):
                epoch_index[i] = np.argmin(np.abs(epoch_time - stim_time))
        else:
            # make a list from 0 to num_of_epochs
            epoch_index = list(range(num_of_epochs))

        # for stimulus_timing just use one of the channels, both *should* be in sync
        stimulus_timing = {'stimulus_end_times': stimulus_end_times,
                           'stimulus_start_times': stimulus_start_times,
                           'dropped_frame_times': dropped_frame_times,
                           'epoch_index': epoch_index,
                           'frame_rate': frame_rate}
        return stimulus_timing
    
    def get_responses_per_condition(self, timestamps, trace, stimulus_timing, para_key=['color'], relative_time=None):
        '''
        WARNING: This method should be deprecated, use get_ensemble_per_condition instead
        The reason is that the returned ensemble is not a dictionary, but a list of lists, so it is hard to register them to the stimulus type consistently
        '''
        epoch_parameters = self.getEpochParameters()
        run_parameters = self.getRunParameters()
        stims, type2ind = self.getStimulusTypes(para_key)
        
        epoch_start_times = stimulus_timing['stimulus_start_times'] - run_parameters['pre_time']
        epoch_end_times = stimulus_timing['stimulus_end_times'] + run_parameters['tail_time']
        which_epoch = stimulus_timing['epoch_index']

        epoch_duration = 1.00*(run_parameters['pre_time'] +
                   run_parameters['stim_time'] +
                   run_parameters['tail_time']) # sec

        ensemble = [[] for _ in type2ind.keys()]
        # relative time should be from -run_parameters['pre_time'], and its duration should be epoch_duration
        # the relative time should be the same sample rate as timestamps
        if relative_time is None:
            timestamps_fs = 1.0/np.gradient(timestamps).mean()
            relative_time = np.linspace(-run_parameters['pre_time'], -run_parameters['pre_time']+epoch_duration, 
                                        int(np.round(epoch_duration*timestamps_fs))+1)

        cut_inds = np.empty(0, dtype=int)
        for idx, val in enumerate(epoch_start_times):
            stack_inds = np.where(np.logical_and(timestamps < val + epoch_duration+0.01,
                                  timestamps >= val-0.01))[0]
            which_type = type2ind[stims[which_epoch[idx]]]
            segmented_trace = trace[stack_inds]
            stim_time = stimulus_timing['stimulus_start_times'][idx]
            segment_timing = timestamps[stack_inds]-stim_time
            segmented_trace_resampled = np.interp(relative_time, segment_timing, segmented_trace)
            ensemble[which_type].append(segmented_trace_resampled)
        
        return ensemble, relative_time
    
    def get_ensemble_per_condition(self, timestamps, trace, stimulus_timing, para_key=['color'], relative_time=None):
        '''
        Return a dictionary of ensemble traces for each stimulus type
        '''
        epoch_parameters = self.getEpochParameters()
        run_parameters = self.getRunParameters()
        stims, type2ind = self.getStimulusTypes(para_key)
        
        epoch_start_times = stimulus_timing['stimulus_start_times'] - run_parameters['pre_time']
        epoch_end_times = stimulus_timing['stimulus_end_times'] + run_parameters['tail_time']
        which_epoch = stimulus_timing['epoch_index']

        epoch_duration = 1.00*(run_parameters['pre_time'] +
                   run_parameters['stim_time'] +
                   run_parameters['tail_time']) # sec

        ensemble = {k:[] for k in type2ind.keys()}
        # relative time should be from -run_parameters['pre_time'], and its duration should be epoch_duration
        # the relative time should be the same sample rate as timestamps
        if relative_time is None:
            timestamps_fs = 1.0/np.gradient(timestamps).mean()
            relative_time = np.linspace(-run_parameters['pre_time'], -run_parameters['pre_time']+epoch_duration, 
                                        int(np.round(epoch_duration*timestamps_fs))+1)

        for idx, val in enumerate(epoch_start_times):
            stack_inds = np.where(np.logical_and(timestamps < val + epoch_duration+0.01,
                                  timestamps >= val-0.01))[0]
            which_type = stims[which_epoch[idx]]
            segmented_trace = trace[stack_inds]
            stim_time = stimulus_timing['stimulus_start_times'][idx]
            segment_timing = timestamps[stack_inds]-stim_time
            segmented_trace_resampled = np.interp(relative_time, segment_timing, segmented_trace)
            ensemble[which_type].append(segmented_trace_resampled)
        
        return ensemble, relative_time

    
        