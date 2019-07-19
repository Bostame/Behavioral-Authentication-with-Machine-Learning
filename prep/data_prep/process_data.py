"""
@author: Miguel Alba
BAwML data processing
"""

from utils import util
import os
import glob
import re
import numpy as np
from tqdm import tqdm

from utils.read_config import ReadConfig


def detect_activities_person(data_path):
    if str(data_path).endswith('/'):
        data_path = data_path[:-1]
    #
    filenames = glob.glob('{}/*'.format(data_path))
    file_acivities = []
    for fn in filenames:
        if util.is_file_in_filtered_criteria(fn):
            continue

        activity = re.split('__|([0-9]+)_|walking_normal_',
                            os.path.basename(fn).split('_', 1)[-1])[2]
        file_acivities.append(activity)
    return list(set(file_acivities))


def process_sensor_data(meta, sensorEventBatch, sensor_type):
    sensor_data = \
        [ex for ex in sensorEventBatch if ex["sensorType"] == sensor_type][0]
    records = sensor_data['sensorEventRecords']
    series = np.empty((0, len(records[0]['values'])))
    for ex in records:
        series = np.vstack((series, np.array(ex['values'])))
    meta['series'] = series
    return meta


def activity_cache(path, activity, print_activity=False):
    if str(path).endswith('/'):
        path = path[:-1]
    #
    filenames = glob.glob('{}/*{}*'.format(path, activity))
    user = os.path.basename(path).split('_')[0]
    #
    single_register = []
    for fn in filenames:
        meta = {}
        # meta = {} --- This is just a comment (maybe in the future we want to keep info of the data)
        if util.is_file_in_filtered_criteria(fn):
            continue

        if not str(activity) in os.path.basename(fn):
            continue
        #
        label = activity
        #
        meta['person'] = user
        meta['label'] = label
        meta['timestamp'] = re.split('__|\.\s*', os.path.basename(fn))[1]
        #
        raw_data = util.slurpjson(fn)
        inner_dict = [values for keys, values in raw_data[0].items()]
        # These are the records
        sensorEventBatch = inner_dict[2]
        # all sensor types
        all_sensors = [ex['sensorType'] for ex in sensorEventBatch]
        #
        sensors_stacks = []
        for sensor in all_sensors:
            meta['sensor'] = sensor
            sensor_dict = process_sensor_data(meta, sensorEventBatch, sensor)
            sensors_stacks.append(sensor_dict.copy())
        #
        single_register.extend(sensors_stacks)
    if print_activity:
        print('cached info {} for {} was generated'.format(activity,
                                                           os.path.basename(
                                                               path)))
    return single_register, all_sensors


def process_person_activity_stack(path, activity):
    stacked_register, sensor_list = activity_cache(path, activity)
    ordered_register = sorted(stacked_register, key=lambda k: k['timestamp'])
    full_register = []
    for sensor in sensor_list:
        sequences = [ex for ex in ordered_register if ex['sensor'] == sensor]
        times = [ex['timestamp'] for ex in sequences]
        stacked_series = np.vstack([ex['series'] for ex in sequences])
        meta = {
            'label': str(activity),
            'timestamps': times,
            'person': stacked_register[0]['person'],
            'sensor': sensor,
            'series': stacked_series
        }
        full_register.append(meta)
    return full_register


def save_activities_one_person(person_path, out_dir=None):
    if str(person_path).endswith('/'):
        person_path = person_path[:-1]
    # detect all acivities in one folder
    activities = detect_activities_person(person_path)
    full_data = []
    for activity in tqdm(activities):
        person_activity_process = process_person_activity_stack(person_path,
                                                            str(activity))
        full_data.extend(person_activity_process)
#
    fn = '{}.pz'.format(os.path.basename(person_path).split('_')[0])
    # save_path = os.path.join(out_dir, 'series')
    util.mdir(out_dir)
    util.save1(os.path.join(out_dir, fn), full_data)
    print('Number of activities for this person: {}'.format(len(activities)))
    if out_dir is None:
        return full_data
    else:
        return None


def create_series(raw_path, out_dir):
    if str(raw_path).endswith('/'):
        raw_path = raw_path[:-1]
    #
    all_folders = util.get_all_folder(raw_path)
    for folder in all_folders:
        save_activities_one_person(folder, out_dir)
        print(folder)
    print('Series successfully processed')
    return None


if __name__ == '__main__':
    rc = ReadConfig()
    file_path = rc.parse_file_path()
    create_series(file_path.raw_data_dir, file_path.series_dir)
