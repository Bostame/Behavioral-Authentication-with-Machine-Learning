"""
@author: Miguel Alba
BAwML data prep
"""
import copy
import glob
import os
import re

import numpy as np
from python_speech_features import logfbank
from tqdm import tqdm

from utils import util
from utils.read_config import ReadConfig


class SpectrogramGeneration():

    def create_spectrograms_ex(self, ex, window_size, nfilt):
        """
        :param ex: series, should be a matrix with the coordinates of the different sensors
        :param window_size: the window to cut the bank transformation
        :param nfilt: number of filters used in the transformation more than 50 increase the height of the spectrogram
        :return: list with all the spectrogram generated for that series example
        """
        examples_list = []
        seq_meta = {
            'label': ex['label'],
            'person': ex['person'],
            'sensor': ex['sensor'],
            'timestamps': ex['timestamps']
        }
        series = ex['series']
        parts = [(k * window_size) for k in range(series.shape[0]) if
                 (k * window_size) < len(series)]
        for w, p in enumerate(parts):
            window = series[p:(p + window_size), :]
            for cor in range(window.shape[1]):
                coordinate = window[:, cor]
                if coordinate.shape[0] < window_size:
                    continue
                # skip sensor coordinates with 0 values
                if np.all(coordinate == 0):
                    continue
                # apply bank to the series
                bank = logfbank(np.array(coordinate), samplerate=400,
                                nfilt=nfilt, nfft=512).T
                coord_meta = copy.deepcopy(seq_meta)
                coord_meta['image'] = bank
                coord_meta['coordinate'] = cor
                coord_meta['window'] = w
                examples_list.append(coord_meta)
        return examples_list

    def preprocess_one_person(self, fn, save_path, filtered_labels,
                              filtered_sensor, **kwargs):
        """
        :param fn: filename of the person
        :param save_path: path to save the dicts of spectrograms
        :param filtered_labels: Labels not considered
        :param filtered_sensor: Sensors not considered
        :return: sequence of all spectrograms generated for one person
        """
        data = util.load1(fn)
        for idx, ex in enumerate(data):
            if ex['sensor'] in filtered_sensor:
                continue
            if ex['label'] in filtered_labels:
                continue
            specs = self.create_spectrograms_ex(ex, **kwargs)
            suffix = "{}_{}-sensor_{}.pz".format(idx, ex['label'], ex['sensor'])
            out_fn = os.path.join(save_path, '{}_ex_{}'.format(
                os.path.basename(fn).split('.')[0], suffix))
            util.save1(out_fn, specs)
        return None

    def generate_example_sets(self, path, outdir, **kwargs):
        """
        :param path: Series path
        :param name: preprocessing name
        :param outdir: output directory
        :return: None
        """
        if str(path).endswith('/'):
            path = path[:-1]
        #
        trash_labels = ['handbag_hand_left_activity_step',
                        'handbag_hand_right_activity_step',
                        'handbag_shoulder_left_activity_step',
                        'handbag_shoulder_right_activity_step',
                        'satchel_diagonal_right_activity_step',
                        'satchel_shoulder_right_activity_step',
                        'sample_task_activity_step']
        trash_sensors = [5, 6, 8, 18, 19, 22, 27, 65536]
        #
        util.mdir(outdir)
        #
        filenames = glob.glob('{}/*'.format(path))
        for fn in tqdm(filenames):
            self.preprocess_one_person(fn, outdir,
                                       filtered_labels=trash_labels,
                                       filtered_sensor=trash_sensors, **kwargs)
        print(
            'Spectrograms for the series have been generated for {} persons'.format(
                len(filenames)))
        return None

    def generate_3_channel_examples(self, trackings, out_dir, sensor_list=[1],
                                    reduce_labels=False,
                                    spectrogram_dim=(50, 50)):
        """
        :param trackings: spectrogram tracks
        :param sensor_list: list of sensors desired (4) gyroscope and (1) accelerometer
        :type reduce_labels: if True, it selects 10 labels instead of 18
        :param spectrogram_dim: original dimension of the spectrograms in trackings
        :return: None
        """
        #
        if str(trackings).endswith('/'):
            trackings = trackings[:-1]

        if str(out_dir).endswith('/'):
            out_dir = out_dir[:-1]
        #
        # trackings_path = os.path.dirname(
        #     trackings)  # to be saved in the same directory as
        # _3channel_path = os.path.join(trackings_path, '3CTrackings')
        util.mdir(out_dir)
        #
        filenames = [fn for fn in glob.glob('{}/*'.format(trackings)) if int(
            re.split('_|.pz', os.path.basename(fn))[-2]) in sensor_list]
        #
        if reduce_labels:
            important_labels = [
                'backpack_activity_step',
                'holding_right_hand_activity_step',
                'jacket_outer_left_pocket_activity_step',
                'landscape_texting_activity_step',
                'listening_voice_message_activity_step',
                'reading_scrolling_activity_step',
                'reading_watching_activity_step',
                'recording_voice_message_activity_step',
                'telephoning_activity_step',
                'texting_activity_step'
            ]
            filenames = [fn for fn in filenames if
                         re.split(r"(^[^_]+_[^_]+_[^_]+)_|-|.pz",
                                  os.path.basename(fn))[2] in important_labels]

        counter = 0
        for idx, fn in enumerate(filenames):
            track = util.load1(fn)
            windows = list(set([ex['window'] for ex in track]))
            track_meta = {
                'label': list(set([ex['label'] for ex in track]))[0],
                'coordinates': list(set([ex['coordinate'] for ex in track])),
                'person': list(set([ex['person'] for ex in track]))[0],
                'sensor': int(re.split('_|.pz', os.path.basename(fn))[-2])
            }
            examples_list = []
            for w in windows:
                window_image_meta = copy.deepcopy(track_meta)
                data_one_window = [ex['image'] for ex in track if
                                   ex['window'] == w]
                reshaping = [(lambda x: x.reshape(spectrogram_dim[0],
                                                  spectrogram_dim[1], 1))(e) for
                             e in data_one_window]
                new_img = np.concatenate(
                    (reshaping[0], reshaping[1], reshaping[2]),
                    axis=2)
                window_image_meta['image'] = new_img
                examples_list.append(window_image_meta)
            counter += len(examples_list)  # just for info purposes
            suffix = "{}_{}-sensor_{}.pz".format(idx, track_meta['label'],
                                                 track_meta['sensor'])
            out_fn = os.path.join(out_dir,
                                  '{}_ex_{}'.format(track_meta['person'],
                                                    suffix))
            util.save1(out_fn, examples_list)
        print('Number of generated spectrograms is : {}'.format(counter))
        return None

    def trackings_generator(self, series_path, spectrogram_path):

        self.generate_example_sets(series_path, outdir=spectrogram_path,
                                   window_size=406, nfilt=50)
    #TODO: Need to fix the problem
    def generate_3channel_images(self, spectrogram_dir,
                                 threeChannelSpectrogram):

        # Accelerometer and gyroscope
        self.generate_3_channel_examples(spectrogram_dir,
                                         threeChannelSpectrogram,
                                         sensor_list=[1, 4], reduce_labels=True)

    def generate_3channel_longer(self, spectrogram_dir,
                                 threeChannelSpectrogram):
        """
        Put here the trackings generated "single spectrograms forlder"
        """
        # Accelerometer and gyroscope
        self.generate_3_channel_examples(spectrogram_dir,
                                         threeChannelSpectrogram,
                                         sensor_list=[1, 4],
                                         reduce_labels=True,
                                         spectrogram_dim=(50, 100))


if __name__ == '__main__':
    read_config = ReadConfig()
    file_path = read_config.parse_file_path()

    sp_gen = SpectrogramGeneration()
    sp_gen.trackings_generator(series_path=file_path.series_dir,
                               spectrogram_path=file_path.spectrogram_dir)

    # _trackings_generator()
    # _20190711_generate_3channel_images()
    pass
