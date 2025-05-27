import os
import glob
import h5py
import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nptdms import TdmsFile
from scipy import signal
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray

from STA_utils import ImagingData, encode_stim_value, encode_stim


def low_pass_filter(ts, fs, cutoff=1):
    nyquist = fs / 2
    cutoff = cutoff / nyquist
    b, a = butter(2, cutoff, btype='low', analog=False)
    return filtfilt(b, a, ts)

def high_pass_filter(ts, fs, cutoff=1):
    nyquist = fs / 2
    cutoff = cutoff / nyquist
    b, a = butter(2, cutoff, btype='high', analog=False)
    return filtfilt(b, a, ts)

def show_poi(ax, poi_coords, snapshot):
    ax.imshow(snapshot, cmap='gray')
    ax.set_aspect('equal')
    ax.set_axis_off()
    for i, coord in enumerate(poi_coords):
        ax.scatter(coord[0], coord[1], s=20, facecolors='none', edgecolors='r')
        ax.annotate('poi {}'.format(i), xy=coord, c='w', size='xx-small') 

def flatten(x, wid):
    t = np.arange(x.shape[0])
    n_wid = np.ceil(x.shape[0]/wid).astype('int')
    xq = np.zeros(n_wid)
    tq = np.zeros(n_wid)
    for i in range(n_wid):
        tmp = x[(i*wid):((i+1)*wid-1)]
        lo = np.quantile(tmp, 0.3)
        hi = np.quantile(tmp, 0.8)
        tmp = tmp[tmp>lo]
        tmp = tmp[tmp<hi]
        xq[i] = tmp.mean()
        tq[i] = t[(i*wid):((i+1)*wid-1)].mean()
    y = np.interp(t, tq, xq)
    return y


class PointScanData:
    def __init__(self, point_scan_file, ID:ImagingData, point_groups=[]):
        self.point_scan_file = point_scan_file
        self.point_scan_file_id = point_scan_file.split('/')[-1]
        self.snap_path = os.path.join(self.point_scan_file, self.point_scan_file_id + "_pmt1.jpeg")
        self.ID = ID
        self.channel_time = None
        self.acquisision_rate = None
        self.channel_dic = None
        self.channel_dic_filtered = None
        self.dff_trace_dic = None
        self.poi_response_dic = None
        self._extract_point_scan_data(point_groups=point_groups)
        self._extract_alignment_sigal()
    
    def _extract_point_scan_data(self, point_groups=[]):
        # extract the time series data from the PMT
        # group the points according to point_groups
        # if point_groups is empty, the the channel_dict stores the data for each point
        pmt_data = os.path.join(self.point_scan_file, self.point_scan_file_id + "_pmt1.tdms")
        channel_dic_raw = {}
        channel_names = []
        with TdmsFile.open(pmt_data) as tdms_file:
            green_group = tdms_file['PMT1']
            for channel in green_group.channels():
                channel_name = channel.name
                if 'time' in channel_name:
                    channel_time = channel[:]
                else:
                    channel_dic_raw[channel_name] = channel[:]
                    channel_names.append(channel_name)
            parameter = tdms_file['parameter']
            parameters = parameter.channels()[0][:]
            values = parameter.channels()[1][:]
        parameter_dic = {}
        poi_coord = []
        for i, p in enumerate(parameters):
            if p.isnumeric():
                poi_coord.append((int(p), int(values[i])))
            else:
                parameter_dic[p] = values[i]

        channel_dic = {}
        if len(point_groups) > 0:
            for pg in point_groups:
                if len(pg) > 0:
                    group_name = 'group_' + '_'.join([str(i) for i in pg])
                    point_names = ['POI {} '.format(i) for i in pg]
                    channel_dic[group_name] = np.zeros_like(channel_dic_raw[point_names[0]])
                    for pn in point_names:
                        channel_dic[group_name] += channel_dic_raw[pn]
                else:
                    group_name = 'whole_cell'
                    channel_dic[group_name] = np.zeros_like(channel_dic_raw[point_names[0]])
                    for pn in channel_names:
                        channel_dic[group_name] += channel_dic_raw[pn]
        else:
            channel_dic = channel_dic_raw
        acquisision_rate = 1000/np.gradient(channel_time).mean()
        channel_time = np.arange(0, len(list(channel_dic_raw.values())[0]))/acquisision_rate
        self.channel_time = channel_time
        self.acquisision_rate = acquisision_rate
        self.channel_dic = channel_dic
        self.channel_names = channel_names
        self.acquisition_parameters = parameter_dic
        self.poi_coords = poi_coord
    
    def _extract_alignment_sigal(self, sync_sr=10000):
        # extract the stimuli timing from the analogIN 
        flicker_data = os.path.join(self.point_scan_file, self.point_scan_file_id + "-AnalogIN.tdms")
        with TdmsFile.open(flicker_data) as sync_file:
            group = sync_file['external analogIN']
            sync_ = group['AnalogGPIOBoard/ai0']
            sync = sync_[:]
        sync_time = np.arange(0, len(sync))
        self.sync = sync
        self.sync_time = sync_time / sync_sr

    def resample_data(self, resample_rate=1000):
        # first calculate the ratio of original sample rate and the desired sample rate
        # for ratio > 2, first bin the data to the closest integer multiple of the desired sample rate
        reduction_ratio = self.acquisision_rate / resample_rate
        if reduction_ratio > 2:
            print('resampling data with binning')
            bin_size = int(np.round(reduction_ratio))
            number_of_bins = len(self.channel_time) // bin_size
            # take the start of the bin as the time point, and the sum of the bin as the value
            new_channel_time = np.zeros(number_of_bins)
            new_channel_dic = {}
            for k, v in self.channel_dic.items():
                new_channel_dic[k] = np.zeros(number_of_bins)
            for i in range(number_of_bins):
                new_channel_time[i] = self.channel_time[i*bin_size]
                for k, v in self.channel_dic.items():
                    new_channel_dic[k][i] = np.sum(v[i*bin_size:(i+1)*bin_size])
            self.channel_time = new_channel_time
            self.channel_dic = new_channel_dic
        # resample the data to the desired rate, this is to make sure the data length is consistent across different recordings
        number_of_samples = int(np.round(self.channel_time[-1] * resample_rate)) + 1
        new_channel_time = np.linspace(0, self.channel_time[-1], number_of_samples)
        resampled_channel_dic = {}
        for k, v in self.channel_dic.items():
            resampled_channel_dic[k] = np.interp(new_channel_time, self.channel_time, v)
        self.channel_time = new_channel_time
        self.channel_dic = resampled_channel_dic
        self.acquisision_rate = resample_rate
    
    def filter_data(self, notch_freq=120, low_pass_fc=200, Q_factor=10):
        # norch filter the data to remove visual stimulus artifacts
        b, a = signal.iirnotch(notch_freq, Q_factor, self.acquisision_rate)
        # 240-Hz low pass filter the data to remove high frequency noise (mainly the multiplier noise)
        b2, a2 = butter(3, low_pass_fc/(self.acquisision_rate/2), btype='low', analog=False)
        self.channel_dic_filtered = {}
        for i, poi in enumerate(self.channel_dic.keys()):
            notch_filtered = signal.filtfilt(b, a, self.channel_dic[poi])
            low_pass_filtered = signal.filtfilt(b2, a2, notch_filtered)
            self.channel_dic_filtered[poi] = low_pass_filtered

    def calculate_dff(self, updown=1, flatten_window=5):
        '''
        calculate the dF/F for each channel
        updown = 1 for down, -1 for up
        flatten_window = 5 for 5 seconds
        '''
        dff_groups = {}
        for key, value in self.channel_dic.items():
            x = value * updown
            baseline = flatten(x, 5000)
            dff = x/baseline-1
            dff = -updown * dff * 100
            dff_groups[key] = dff
        self.dff_trace_dic = dff_groups
    
    def high_pass_filter_data(self, high_pass_fc=5):
        # high-pass filter the data for AP detection
        if not self.dff_trace_dic:
            print('High-pass filter can only be applied to dFF data!')
        b2, a2 = butter(3, high_pass_fc/(self.acquisision_rate/2), btype='high', analog=False)
        for i, poi in enumerate(self.dff_trace_dic.keys()):
            high_pass_filtered = signal.filtfilt(b2, a2, self.dff_trace_dic[poi])
            self.dff_trace_dic[poi] = high_pass_filtered
    
    def smooth_data(self, sigma=5):
        # smooth the data using gaussian filter
        for k, v in self.channel_dic_filtered.items():
            self.channel_dic_filtered[k] = gaussian_filter(v, sigma=sigma)
    
    def segment_signal_by_stimuli(self, para_key=['current_color'], bg_index=-1, relative_time=None):
        if not self.channel_dic_filtered:
            self.channel_dic_filtered = self.channel_dic
            print('Data not filtered, use unfiltered data!')
        if bg_index < -1:
            print('not going to use background subtraction')
            self.bg = 0
            bg_poi = ''
        else:
            bg_poi = self.channel_names[bg_index] + '_green'  ## TODO: need to work on the background subtraction for both green and red channel
            print(f'using {bg_poi} for background subtraction')
            self.bg = self.channel_dic_filtered[bg_poi]  
            self.channel_dic_filtered.pop(bg_poi)
        
        frame_monitor = self.sync.reshape(1, -1)
        time_vector = self.sync_time
        sample_rate = (1.0/np.gradient(time_vector)).mean()
        stimulus_timing = self.ID.getStimulusTiming(frame_monitor, time_vector, sample_rate)
        stim_types, type2ind = self.ID.getStimulusTypes(para_key=para_key)
        try:
            discard_index = np.where(stimulus_timing['stimulus_end_times'] > self.channel_time[-1]-0.2)[0][0]
            print(discard_index)
            stimulus_timing['stimulus_end_times'] = stimulus_timing['stimulus_end_times'][:discard_index]
            stimulus_timing['stimulus_start_times'] = stimulus_timing['stimulus_start_times'][:discard_index]
        except:
            pass
        poi_response_dic = {}
        all_poi_trace = 0.0
        for k, v in self.channel_dic_filtered.items():
            trace = v - self.bg
            all_poi_trace += trace
            ensemble, relative_time = self.ID.get_ensemble_per_condition(self.channel_time, trace, stimulus_timing, para_key=para_key, relative_time=relative_time)
            response_dic = {}
            for (stim_type, stim_ind) in type2ind.items():
                responses = ensemble[stim_type]
                response_dic[stim_type] = responses
                response_dic[stim_type+'time'] = relative_time
            poi_response_dic[k] = response_dic
        self.poi_response_dic = poi_response_dic
        # now calculate the average response across all pois
        all_poi_response_dict = {}
        ensemble, relative_time = self.ID.get_ensemble_per_condition(self.channel_time, all_poi_trace, stimulus_timing, para_key=para_key, relative_time=relative_time)
        for (stim_type, stim_ind) in type2ind.items():
            responses = ensemble[stim_type]
            all_poi_response_dict[stim_type] = responses
            all_poi_response_dict[stim_type+'time'] = relative_time
        self.poi_response_dic['All_POI_sum'] = all_poi_response_dict  # include the sum of all pois in the response dictionary, instead of making a seperate entry

    def summary_figure(self, save_dir=None, figure_size=(5, 5), ylim=None, condition=None, run_para_key=None, bg=-999):
        if ylim is None:
            ylim = [-0.3, 0.3]
        run_parameters = self.ID.getRunParameters()
        epoch_parameters = self.ID.getEpochParameters()
        if condition is None:
            condition = [k for k in epoch_parameters[0].keys() if 'current' in k]
        self.segment_signal_by_stimuli(para_key=condition, bg_index=bg)
        stim_types, type2ind = self.ID.getStimulusTypes(para_key=condition)
        experiment_date = self.ID.file_path.split('/')[-1].split('.')[0]
        para_set = list(sorted(type2ind.keys()))
        roi_set_names = list(sorted(self.poi_response_dic.keys()))

        figure_title = run_parameters['protocol_ID'] + '_' + experiment_date + '_trial_' + str(self.ID.series_number)
        big_figure_size = (figure_size[0]*(len(roi_set_names)+1), figure_size[1]*len(para_set))
        fh, ax = plt.subplots(len(para_set), len(roi_set_names)+1, figsize=big_figure_size, tight_layout=True, facecolor='snow')
        fh.suptitle(figure_title)
        if len(para_set)>1:
            [x.set_ylim(ylim) for x in ax[:, :-1].ravel()]
        else:
            [x.set_ylim(ylim) for x in ax[:-1].ravel()]

        for point_ind, (poi, response_dict) in enumerate(self.poi_response_dic.items()):
            for para_ind, stim_type in enumerate(para_set):
                responses = response_dict[stim_type]
                stim_time = response_dict[stim_type + 'time']
                mean_response = np.mean(responses, axis=0)
                baseline = np.mean(mean_response[stim_time < 0])  # sometimes there is stimulus in the pre-stim period
                dff = mean_response/baseline-1.0
                if len(para_set)>1:
                    ax[para_ind, point_ind].plot(stim_time, dff, color=self.ID.colors[point_ind], lw=2)
                    if para_ind == 0:
                        ax[para_ind, point_ind].set_title(poi)
                    if point_ind == 0:
                        ax[para_ind, point_ind].set_ylabel(stim_type)
                else:
                    ax[point_ind].plot(stim_time, dff, color=self.ID.colors[point_ind], lw=2)
                    ax[point_ind].set_title(poi)
                    if point_ind == 0:
                        ax[point_ind].set_ylabel(stim_type)
        snap_shot = plt.imread(self.snap_path)
        snap_shot = rgb2gray(snap_shot)
        if len(para_set)>1:
            show_poi(ax[0, len(roi_set_names)], self.poi_coords, snap_shot)
        else:
            show_poi(ax[len(roi_set_names)], self.poi_coords, snap_shot)
        # save the figure
        if save_dir:
            fh.savefig(os.path.join(save_dir, figure_title + '.png'), dpi=300)
            plt.close(fh)
    
    def plot_response_poi(self, save_dir=None):
        for poi, response_dic in self.poi_response_dic.items():
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            for stim_type, responses in response_dic.items():
                if 'time' in stim_type:
                    continue
                stim_time = response_dic[stim_type + 'time']
                mean_response = np.mean(responses, axis=0)
                baseline = np.mean(mean_response[stim_time < 0])
                ax.plot(stim_time, mean_response/baseline-1.0, label=stim_type)
            ax.legend()
            ax.set_title(poi)
            if save_dir:
                fig.savefig(os.path.join(save_dir, self.point_scan_file_id + '_' + poi + '.png'))
                plt.close(fig)
            plt.show()
    
    # def plot_response_poi_QC(self, save_dir=None):
    #     for poi, response_dic in self.poi_response_dic.items():
    #         num_stim = len(response_dic) // 2
    #         # make a grid plot for each poi
    #         fig, ax = plt.subplots(num_stim, 1, figsize=(10, 10*num_stim))
    #         i = 0
    #         for stim_type, responses in response_dic.items():
    #             if 'time' in stim_type:
    #                 continue
    #             stim_time = response_dic[stim_type + 'time']
    #             for res in responses:
    #                 baseline = np.mean(res[stim_time < 0])
    #                 ax[i].plot(stim_time, res/baseline-1.0, alpha=0.3)
    #             ax[i].set_title(stim_type)
    #             i += 1
    #         if save_dir:
    #             fig.savefig(os.path.join(save_dir, self.point_scan_file_id + '_' + poi + '_QC.png'))
    #             plt.close(fig)
    #         plt.show()
    
    def show_data_with_trigger(self, point_list=[], gaussian_sigma=3, zoom_in_range=[0,10]):
        # average the data across pois in the point_list, if point_list is empty, average all pois
        if len(point_list) == 0:  # average traces for all pois
            F = np.vstack(list(self.channel_dic.values())).sum(0)
        if len(point_list) > 0:
            poi_keys = ['POI {} '.format(k) for k in point_list]
            trace_stack = []
            for k in poi_keys:
                try:
                    trace_stack.append(self.channel_dic[k])
                except:
                    print('point {} not exist'.format(k))
            F = np.vstack(trace_stack).sum(0)
        # smooth the data
        F_smooth = gaussian_filter(F, sigma=gaussian_sigma)
        # plot the data with the trigger signal
        fig, ax = plt.subplots(2, 1, figsize=(15, 5))
        ax[0].plot(self.channel_time, F_smooth)
        ax[0].set_title('Smoothed data')
        ax[0].set_xlim(zoom_in_range)
        ax[1].plot(self.sync_time, self.sync)
        ax[1].set_title('Trigger signal')
        ax[1].set_xlim(zoom_in_range)
        plt.show()
    
    def show_dff_data(self, zoom_in_range=[0,10], filter='savgol', save_dir=None):
        # average the data across pois in the point_list, if point_list is empty, average all pois
        # make a figure of dff_groups, with each group in a subplot
        dff_groups = self.dff_trace_dic
        t = self.channel_time
        if filter == 'savgol':
            dff_groups = {k: savgol_filter(v, 9, 3) for k, v in dff_groups.items()}
        if filter == 'gaussian':
            dff_groups = {k: gaussian_filter(v, sigma=2) for k, v in dff_groups.items()}
        if len(dff_groups) == 1:
            value = list(dff_groups.values())[0]
            fig, axs = plt.subplots(1, 1, figsize=(15, 5))
            axs.plot(t, value + 5)
            axs.set_title(list(dff_groups.keys())[0])
            axs.set_xlabel('Time (s)')
            axs.set_ylabel('dF/F (%)')
            axs.set_xlim(zoom_in_range)
        else:
            fig, axs = plt.subplots(len(dff_groups), 1, figsize=(15, 5*len(dff_groups)))
            for i, (key, value) in enumerate(dff_groups.items()):
                axs[i].plot(t,value)
                axs[i].set_title(key)
                axs[i].set_xlabel('Time (s)')
                axs[i].set_ylabel('dF/F (%)')
                axs[i].set_xlim(zoom_in_range)
                axs[i].set_ylim([-40, 40])
            plt.tight_layout() 
        plt.suptitle('scanning_rate{}, wavelength{}'.format(self.acquisision_rate, self.acquisition_parameters['wavelength']))
        if save_dir:
            fig.savefig(os.path.join(save_dir, self.point_scan_file_id + '_dff.png'), dpi=300)
            plt.close(fig)
        else:
            plt.show()
    
    def save_segmented_data(self, save_dir):
        if not os.path.exists(save_dir):
            print('save folder does NOT exist')
            return
        # save the poi_response_dic in pickle format
        save_file = os.path.join(save_dir, self.point_scan_file_id + '_segmented_data.pkl')
        if not self.poi_response_dic:
            epoch_parameters = self.ID.getEpochParameters()
            condition = [k for k in epoch_parameters[0].keys() if 'current' in k]
            self.segment_signal_by_stimuli(para_key=condition, bg_index=-999)
        run_parameters = self.ID.getRunParameters()
        with open(save_file, 'wb') as f:
            pickle.dump(self.poi_response_dic, f)
            pickle.dump(run_parameters, f)
            pickle.dump(self.channel_dic_filtered, f)
    
    def save_dff_data(self, save_dir):
        if not os.path.exists(save_dir):
            print('save folder does NOT exist')
            return
        # save the dff_trace_dic in pickle format
        save_file = os.path.join(save_dir, self.point_scan_file_id + '_dff_data.pkl')
        with open(save_file, 'wb') as f:
            pickle.dump(self.dff_trace_dic, f)
            pickle.dump(self.channel_dic, f)



def compare_AOC_two_dendrite(aoc_poi1, poi1, aoc_poi2, poi2, ax=None):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    max_val = 0
    for k, v in aoc_poi1.items():
        if len(v) == 0 or len(aoc_poi2[k]) == 0:
            continue
        ax.scatter(v, aoc_poi2[k], label=k)
        max_val = max(max_val, max(np.abs(v)), max(np.abs(aoc_poi2[k])))
    max_val += 1
    # add vertical and horizontal lines at 0
    ax.axvline(0, color='k', lw=0.5)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel(poi1)
    ax.set_ylabel(poi2)
    ax.set_aspect('equal')
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    # if number of conditions is less than 5, add a legend
    if len(aoc_poi1) < 5:
        ax.legend()
    ci_index = compartment_index(aoc_poi1, aoc_poi2)
    ax.set_title('compartment index: {:.2f}'.format(ci_index))

def compartment_index_(x_list, y_list, thr=0.5):
    CI = []
    for x, y in zip(x_list, y_list):
        if abs(x) + abs(y) < thr:
            continue
        CI.append(abs(x-y)/abs(x+y))
    return np.median(CI)

def compartment_index(aoc_poi1, aoc_poi2):
    aoc_poi1_list = []
    aoc_poi2_list = []
    for k, v in aoc_poi1.items():
        aoc_poi1_list += v
        aoc_poi2_list += aoc_poi2[k]
    return compartment_index_(aoc_poi1_list, aoc_poi2_list)

def compare_dendrites(aoc_dic, save_prefix=None):
    dendrite_list = list(aoc_dic.keys())
    # remove the all poi sum from the list
    dendrite_list.remove('All_POI_sum')
    CI_dic = {}
    N = len(dendrite_list)
    num_pairs = N*(N-1)//2
    # make a figure with num_pairs subplots in a row
    fig, ax = plt.subplots(1, num_pairs, figsize=(5*num_pairs, 5))
    figure_index = 0
    for i in range(len(dendrite_list)):
        for j in range(i+1, len(dendrite_list)):
            if num_pairs > 1:
                compare_AOC_two_dendrite(aoc_dic[dendrite_list[i]], dendrite_list[i], aoc_dic[dendrite_list[j]], dendrite_list[j], ax=ax[figure_index])
            else:
                compare_AOC_two_dendrite(aoc_dic[dendrite_list[i]], dendrite_list[i], aoc_dic[dendrite_list[j]], dendrite_list[j], ax=ax)
            CI_dic[(dendrite_list[i], dendrite_list[j])] = compartment_index(aoc_dic[dendrite_list[i]], aoc_dic[dendrite_list[j]])
            figure_index += 1
    if save_prefix:
        fig.savefig(save_prefix + '_pairwise_scatter.png')
        # close the figure
        plt.close(fig)
    else:
        plt.show()
    return CI_dic


from matplotlib.patches import Ellipse
from fitting_utils import fitgaussian

def summary_offcenter_grid_result(psd:PointScanData, qc_figure_path=None, adding_fitted_center=True):
    '''
    This method is to plot the tuning for all pois
    It will generate 2*n subplots, n for tuning map, n for the response trace with the maximum response
    '''
    epoch_parameters = psd.ID.getEpochParameters()
    run_parameters = psd.ID.getRunParameters()
    para_key = ['current_theta', 'current_phi']
    psd.segment_signal_by_stimuli(para_key=para_key, bg_index=-999)

    phi_list = [ele['current_phi'] for ele in epoch_parameters]
    phi_list = list(set(phi_list))
    phi_list.sort()  # sort for the mesh
    theta_list = [ele['current_theta'] for ele in epoch_parameters]
    theta_list = list(set(theta_list))
    theta_list.sort()
    PHI, THETA = np.meshgrid(phi_list,theta_list)
    peak_response_dict = {poi: np.zeros(PHI.shape) for poi in psd.poi_response_dic.keys()}
    fit_parameter_dict = {poi: None for poi in psd.poi_response_dic.keys()}
    for poi, response_dic in psd.poi_response_dic.items():
        for j, theta in enumerate(theta_list):
            for i, phi in enumerate(phi_list):
                stim_type = encode_stim_value(para_key, [theta, phi])
                responses = response_dic[stim_type]
                time_vec = response_dic[stim_type+'time']
                stim_window = np.where(np.logical_and(time_vec>0.0, time_vec<0.15))[0]
                mean_response = np.mean(responses, axis=0)
                baseline = np.mean(mean_response[time_vec < 0])  # sometimes there is stimulus in the pre-stim period
                dff = mean_response/baseline-1.0
                dff = -dff
                peak_response_dict[poi][j, i] = np.abs(dff[stim_window].mean())  # note that this is j, i for x,y display
        try:
            fit_params = fitgaussian(peak_response_dict[poi], PHI, THETA)
        except:
            fit_params = None
        fit_parameter_dict[poi] = fit_params
    
    # now make 2*n subplots, n for tuning map, n for the response trace with the maximum response
    num_poi = len(psd.poi_response_dic.keys())
    fig, ax = plt.subplots(2, num_poi+1, figsize=(5*(num_poi+1), 10), tight_layout=True)
    for i, (poi, peak_response) in enumerate(peak_response_dict.items()):
        ax[0, i].pcolor(PHI, THETA, peak_response, vmin=0.0, vmax=0.2)
        ax[0, i].set_xlabel('phi')
        ax[0, i].set_ylabel('theta')
        ax[0, i].set_title(poi)
        if adding_fitted_center:
            height, x, y, width_x, width_y = fit_parameter_dict[poi]
            ellipse = Ellipse(xy=(x, y), width=width_x, height=width_y, edgecolor='r', fc='None', lw=2)
            ax[0, i].add_patch(ellipse)

        max_response_i, max_response_j = np.unravel_index(peak_response.argmax(), peak_response.shape)
        max_response_theta = theta_list[max_response_i]
        max_response_phi = phi_list[max_response_j]
        stim_type = encode_stim_value(para_key, [max_response_theta, max_response_phi])
        responses = np.array(psd.poi_response_dic[poi][stim_type])
        time_vec = psd.poi_response_dic[poi][stim_type+'time']
        mean_response = np.mean(responses, axis=0)
        #baseline = np.mean(mean_response[time_vec > run_parameters['stim_time']+0.1])
        baseline = np.mean(mean_response[time_vec < 0.0])
        dff = mean_response/baseline-1.0
        dff = -dff
        ax[1, i].plot(time_vec, dff, color='k')
        ax[1, i].set_xlabel('time')
        ax[1, i].set_ylabel('response')
        ax[1, i].set_ylim(-0.2, 0.2)
        ax[1, i].set_title(poi)
    snap_shot = plt.imread(psd.snap_path)
    snap_shot = rgb2gray(snap_shot)
    show_poi(ax[0, -1], psd.poi_coords, snap_shot)
    
    if qc_figure_path is not None:
        fig.savefig(qc_figure_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()
    
    return THETA, PHI, peak_response_dict, fit_parameter_dict