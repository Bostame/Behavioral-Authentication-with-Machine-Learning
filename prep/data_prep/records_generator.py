import glob
import math
import os
import sys

import tensorflow as tf
from tqdm import tqdm

from utils import util
from utils.read_config import ReadConfig


class RecordsGenerator():

    def tfrecord_debugger(self, fn):
        """
        :param fn: tf record
        :return: printing of shape of the tf record file elements
        """

        def tfrec_data_input_fn(filenames, batch_size=64, shuffle=False):
            def _input_fn():
                def _parse_record(tf_record):
                    features = {
                        'image_raw': tf.FixedLenFeature([], dtype=tf.string),
                        'label': tf.FixedLenFeature([], dtype=tf.int64),
                        'height': tf.FixedLenFeature([], dtype=tf.int64),
                        'width': tf.FixedLenFeature([], dtype=tf.int64),
                        'depth': tf.FixedLenFeature([], dtype=tf.int64)
                    }
                    record = tf.parse_single_example(tf_record, features)

                    image_raw = tf.decode_raw(record['image_raw'], tf.float64)
                    height = tf.cast(record['height'], tf.int32)
                    width = tf.cast(record['width'], tf.int32)
                    depth = tf.cast(record['depth'], tf.int32)
                    image_raw = tf.reshape(image_raw,
                                           shape=(height, width, depth))

                    label = tf.cast(record['label'], tf.int32)
                    label = tf.one_hot(label, depth=18)

                    return image_raw, label

                dataset = tf.data.TFRecordDataset(filenames)
                dataset = dataset.map(_parse_record)
                if shuffle:
                    dataset = dataset.shuffle(buffer_size=10000)
                dataset = dataset.repeat(3)
                dataset = dataset.batch(batch_size)
                iterator = dataset.make_one_shot_iterator()
                features, labels = iterator.get_next()
                # X = {'image': features}
                # y = labels
                # return X, y
                return features, labels

            return _input_fn

        #
        tfrec_dev_input_fn = tfrec_data_input_fn([fn], batch_size=64)
        features, labels = tfrec_dev_input_fn()
        with tf.Session() as sess:
            img, label = sess.run([features, labels])
            print(
                "are these ones the correct dimensions? {}-{}".format(img.shape,
                                                                      label.shape))
        sess.close()
        return None

    def check_inside_tfrecord(self, fn):
        """
        :param fn: file of the tf record
        :return: lenght of the tfrecord and a single example to see how they are encoded
        """
        results = []
        for example in tf.python_io.tf_record_iterator(fn):
            result = tf.train.Example.FromString(example)
            results.append(dict(result.features.feature))
        size = len(results)
        sanity_check = results[0]
        return size, sanity_check

    def _encode_labels(self, label):
        if label == 'back_right_pocket_activity_step':
            return 0
        if label == 'holding_left_hand_activity_step':
            return 1
        if label == 'front_right_pocket_activity_step':
            return 2
        if label == 'recording_voice_message_activity_step':
            return 3
        if label == 'jacket_inner_pocket_activity_step':
            return 4
        if label == 'jacket_outer_right_pocket_activity_step':
            return 5
        if label == 'jacket_outer_left_pocket_activity_step':
            return 6
        if label == 'back_left_pocket_activity_step':
            return 7
        if label == 'jacket_breast_pocket_activity_step':
            return 8
        if label == 'reading_watching_activity_step':
            return 9
        if label == 'texting_activity_step':
            return 10
        if label == 'reading_scrolling_activity_step':
            return 11
        if label == 'backpack_activity_step':
            return 12
        if label == 'holding_right_hand_activity_step':
            return 13
        if label == 'telephoning_activity_step':
            return 14
        if label == 'landscape_texting_activity_step':
            return 15
        if label == 'front_left_pocket_activity_step':
            return 16
        if label == 'listening_voice_message_activity_step':
            return 17

    def _10_class_encoding(self, label):
        if label == 'backpack_activity_step':
            return 0
        if label == 'holding_right_hand_activity_step':
            return 1
        if label == 'jacket_outer_left_pocket_activity_step':
            return 2
        if label == 'landscape_texting_activity_step':
            return 3
        if label == 'listening_voice_message_activity_step':
            return 4
        if label == 'reading_scrolling_activity_step':
            return 5
        if label == 'reading_watching_activity_step':
            return 6
        if label == 'recording_voice_message_activity_step':
            return 7
        if label == 'telephoning_activity_step':
            return 8
        if label == 'texting_activity_step':
            return 9

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _process_examples(self, example_data, filename: str, channels=1):
        # depending on the amount of classes change the encoding function (_encode_labels)  or (_10_class_encoding)
        print(f'Processing {filename} data')
        dataset_length = len(example_data)
        with tf.python_io.TFRecordWriter(filename) as writer:
            for index, ex in enumerate(example_data):
                sys.stdout.write(
                    f"\rProcessing sample {index + 1} of {dataset_length}")
                sys.stdout.flush()
                image_raw = ex['image'].flatten()
                image_raw = image_raw.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': self._int64_feature(ex['image'].shape[0]),
                    'width': self._int64_feature(ex['image'].shape[1]),
                    'depth': self._int64_feature(channels),
                    'label': self._int64_feature(
                        int(self._10_class_encoding(ex['label']))),
                    # _encode_labels
                    'image_raw': self._bytes_feature(image_raw)
                }))
                writer.write(example.SerializeToString())
            print()
        return None

    def stack_data_chunks(self, chunk):
        """
        :param chunk: filenames of the data that is going to be extend together (ex 50 compressed files)
        :return: Data that contain several compressed files together
        """
        full_data = []
        for fn in chunk:
            data = util.load1(fn)
            full_data.extend(data)
        return full_data

    def transform_preprocess_data(self, path, out_dir, name: str, channels=None,
                                  test_persons=None, eval_persons=None,
                                  chunk_len=50, num_tfrecords=10):
        """
        :param path: path of the trackings
        :param out_dir: path to save the tf records
        :param name: name of the records
        :param channels: number of channels None means 1 by default
        :param test_persons: Persons that are going to be included in test/validation
        :param eval_persons: Persons that are going to be included in evaluation
        :param chunk_len: Number of compressed files for one chunk data
        :param num_tfrecords: Number of records in which the chunk data is going to be saved
        :return: None
        """
        if str(path).endswith('/'):
            path = path[:-1]
        #
        if channels is None:
            channels = 1
        #
        filenames = glob.glob('{}/*'.format(path))
        data_files = list()
        if name is 'test' and test_persons is not None:
            data_files = [fn for fn in filenames if
                          os.path.basename(fn).split('_')[0] in test_persons]
        if name is 'evaluation' and eval_persons is not None:
            data_files = [fn for fn in filenames if
                          os.path.basename(fn).split('_')[0] in eval_persons]
        if name is 'train' and eval_persons is not None and test_persons is not None:
            persons = eval_persons
            persons.extend(test_persons)
            data_files = [fn for fn in filenames if
                          os.path.basename(fn).split('_')[0] not in persons]
        #
        chunks = [data_files[k:k + chunk_len] for k in
                  range(0, len(data_files), chunk_len)]
        # len_chunks = len(chunks)
        for idx, c in enumerate(tqdm(chunks)):
            chunk_data = self.stack_data_chunks(c)
            samples_per_tf_record = len(chunk_data) // num_tfrecords
            tf_parts = [(k * samples_per_tf_record) for k in
                        range(len(chunk_data)) if
                        (k * samples_per_tf_record) < len(chunk_data)]
            for i, j in enumerate(tf_parts):
                out_fn = os.path.join(out_dir,
                                      '{}_{:03d}_{:03d}-{:03d}.tfrecord'.format(
                                          name, i + 1, idx + 1, num_tfrecords))
                self._process_examples(
                    chunk_data[j:(j + samples_per_tf_record)],
                    out_fn, channels=channels)
        return None

    def generate_records(self, trackings, out_dir):
        util.mdir(out_dir)
        evaluation = ['0fc8854f-c', '27089a2b-2', '0b7cf78c-1']
        test = ['d847614e-4', '2f3d5c91-2']
        self.transform_preprocess_data(trackings, out_dir, 'train')
        self.transform_preprocess_data(trackings, out_dir, 'test',
                                       test_persons=test)
        self.transform_preprocess_data(trackings, out_dir, 'evaluation',
                                       eval_persons=evaluation)
        return None

    def _20190710_generate_records(self, trackings, out_dir):
        """
        :return: REMEMBER TO ELIMINATE FROM THE FOLDER OF THE RECORDS, THE FILES WITH SMALL SIZE, THEY ARE NOT USEFUL
        ### THEY HAVE FEW EXAMPLES
        """
        util.mdir(out_dir)
        evaluation = ['0fc8854f-c', '27089a2b-2', '0b7cf78c-1']
        test = ['d847614e-4', '2f3d5c91-2']
        self.transform_preprocess_data(trackings, out_dir, 'train', channels=3,
                                       num_tfrecords=5)
        self.transform_preprocess_data(trackings, out_dir, 'test',
                                       test_persons=test, channels=3,
                                       num_tfrecords=5)
        self.transform_preprocess_data(trackings, out_dir, 'evaluation',
                                       eval_persons=evaluation, channels=3,
                                       num_tfrecords=5)
        return None

    def train_test_split_preprocess_data(self, path, out_dir, name: str = None,
                                         channels=None, eval_persons=None,
                                         chunk_len=50, num_tfrecords=10):
        """
        :param path: path of the trackings
        :param out_dir: path to save the tf records
        :param name: name of the records
        :param channels: number of channels None means 1 by default
        :param test_persons: Persons that are going to be included in test/validation
        :param eval_persons: Persons that are going to be included in evaluation
        :param chunk_len: Number of compressed files for one chunk data
        :param num_tfrecords: Number of records in which the chunk data is going to be saved
        :return: None
        """
        if str(path).endswith('/'):
            path = path[:-1]
        #
        if channels is None:
            channels = 1
        #
        filenames = glob.glob('{}/*'.format(path))
        data_files = list()
        if name is 'evaluation' and eval_persons is not None:
            data_files = [fn for fn in filenames if
                          os.path.basename(fn).split('_')[0] in eval_persons]
            self.generate_examples(None, data_files, out_dir, channels, name,
                                   chunk_len,
                                   num_tfrecords)
        else:
            data_files = [fn for fn in filenames if
                          os.path.basename(fn).split('_')[
                              0] not in eval_persons]

            for activity in self.important_labels:
                activity_wise_data_files = [i for i in data_files if
                                            activity in i]

                seventy_percent = math.ceil(len(activity_wise_data_files) * 0.7)

                self.generate_examples(activity, activity_wise_data_files[
                                                 :seventy_percent],
                                       out_dir, channels, 'train', chunk_len,
                                       num_tfrecords)
                self.generate_examples(activity, activity_wise_data_files[
                                                 seventy_percent:],
                                       out_dir, channels, 'test', chunk_len,
                                       num_tfrecords)

        return None

    def generate_examples(self, activity, data_files, out_dir, channels, name,
                          chunk_len=50,
                          num_tfrecords=5):
        chunks = [data_files[k:k + chunk_len] for k in
                  range(0, len(data_files), chunk_len)]
        for idx, c in enumerate(tqdm(chunks)):
            chunk_data = self.stack_data_chunks(c)
            samples_per_tf_record = len(chunk_data) // num_tfrecords
            tf_parts = [(k * samples_per_tf_record) for k in
                        range(len(chunk_data))
                        if (k * samples_per_tf_record) < len(chunk_data)]
            for i, j in enumerate(tf_parts):
                if activity is None:
                    file_name = '{}_{:03d}_{:03d}-{:03d}.tfrecord'.format(
                        name, i + 1, idx + 1, num_tfrecords)
                else:
                    file_name = '{}_{:03d}_{:03d}_{}-{:03d}.tfrecord'.format(
                        name, i + 1, idx + 1, activity,
                        num_tfrecords)
                out_fn = os.path.join(out_dir, file_name)
                self._process_examples(chunk_data[j:(j +
                                                     samples_per_tf_record)],
                                       out_fn, channels=channels)

    def _20190712_generate_records_10classes_train_test_split(self,
                                                              threeChannelTrackings,
                                                              out_dir):
        """
        # ACCE AND GYRO DATA
        :return: REMEMBER TO ELIMINATE FROM THE FOLDER OF THE RECORDS,
        THE FILES WITH SMALL SIZE, THEY ARE NOT USEFUL
        ### THEY HAVE FEW EXAMPLES
        """
        util.mdir(out_dir)
        evaluation = ['0fc8854f-c', '27089a2b-2', '0b7cf78c-1']
        self.train_test_split_preprocess_data(threeChannelTrackings, out_dir,
                                              eval_persons=evaluation,
                                              channels=3, num_tfrecords=5)
        self.train_test_split_preprocess_data(threeChannelTrackings, out_dir,
                                              name='evaluation',
                                              eval_persons=evaluation,
                                              channels=3, num_tfrecords=5)
        return None

    def _20190714_generate_records_10classes_acgy_longer(self,
                                                         threeChannelTrackings,
                                                         out_dir):
        """
        # ACCE AND GYRO DATA
        :return: REMEMBER TO ELIMINATE FROM THE FOLDER OF THE RECORDS, THE FILES WITH SMALL SIZE, THEY ARE NOT USEFUL
        ### THEY HAVE FEW EXAMPLES
        """
        util.mdir(out_dir)
        evaluation = ['0fc8854f-c', '27089a2b-2', '0b7cf78c-1']
        test = ['d847614e-4', '2f3d5c91-2']
        self.transform_preprocess_data(threeChannelTrackings, out_dir,
                                       'train', test_persons=test,
                                       eval_persons=evaluation, channels=3,
                                       num_tfrecords=5)
        self.transform_preprocess_data(threeChannelTrackings, out_dir, 'test',
                                       test_persons=test, channels=3,
                                       num_tfrecords=5)
        self.transform_preprocess_data(threeChannelTrackings, out_dir,
                                       'evaluation', eval_persons=evaluation,
                                       channels=3, num_tfrecords=5)
        return None

    def _20190712_generate_records_10classes_acgy(self, threeChannelTrackings,
                                                  out_dir):
        """
        # ACCE AND GYRO DATA
        :return: REMEMBER TO ELIMINATE FROM THE FOLDER OF THE RECORDS, THE FILES WITH SMALL SIZE, THEY ARE NOT USEFUL
        ### THEY HAVE FEW EXAMPLES
        """
        util.mdir(out_dir)
        evaluation = ['0fc8854f-c', '27089a2b-2', '0b7cf78c-1']
        test = ['d847614e-4', '2f3d5c91-2']
        self.transform_preprocess_data(threeChannelTrackings, out_dir,
                                       'train', test_persons=test,
                                       eval_persons=evaluation, channels=3,
                                       num_tfrecords=5)
        self.transform_preprocess_data(threeChannelTrackings, out_dir, 'test',
                                       test_persons=test, channels=3,
                                       num_tfrecords=5)
        self.transform_preprocess_data(threeChannelTrackings, out_dir,
                                       'evaluation', eval_persons=evaluation,
                                       channels=3, num_tfrecords=5)
        return None

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


if __name__ == '__main__':
    rc = ReadConfig()
    file_path = rc.parse_file_path()
    rg = RecordsGenerator()
    rg._20190714_generate_records_10classes_acgy_longer(
        file_path.threeChannelSpectrogram, file_path.training_dir)  # put in
    #  this
    # last
    # function the 3 channels spectrograms directory and the output directory for training as the last function shows
    pass
