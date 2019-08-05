import glob

import tensorflow as tf


class DataLoader(object):
    def __init__(self, config, training_mode, shuffle):
        """
        :param config:
        :param training_mode:
        :param shuffle:
        """
        self.config = config
        self.training_mode = training_mode
        self.shuffle = shuffle


class _CNN_Data_Loader(DataLoader):
    def __init__(self, config, training_mode, shuffle):
        super().__init__(config, training_mode, shuffle)
        self.data_dir = self.config.data_dir
        self.image_width = self.config.image_width
        self.image_height = self.config.image_height
        self.image_depth = self.config.image_depth
        self.data_dir = self.config.data_dir
        self.batch_size = self.config.batch_size
        self.n_classes = self.config.n_classes
        self.normalize = self.config.normalize

    def _normalization(self, tensor):
        tensor = tf.div(tf.subtract(tensor, tf.reduce_min(tensor)), tf.subtract(tf.reduce_max(tensor), tf.reduce_min(tensor)))
        return tensor

    def _parse_record(self, tf_record):
        features = {
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        }
        record = tf.parse_single_example(tf_record, features)
        image_raw = tf.decode_raw(record['image_raw'], tf.float64)
        height = tf.cast(record['height'], tf.int32)
        width = tf.cast(record['width'], tf.int32)
        depth = tf.cast(record['depth'], tf.int32)
        image_raw = tf.reshape(image_raw, shape=(height, width, depth))
        #
        label = tf.cast(record['label'], tf.int32)
        label = tf.one_hot(label, depth=self.n_classes)
        if self.normalize:
            image_raw = self._normalization(image_raw)
        return image_raw, label

    def _generate_batch(self):
        if self.training_mode is True:
            data_filenames = glob.glob(
                '{}/train_*.tfrecord'.format(self.data_dir))
        else:
            data_filenames = glob.glob(
                '{}/test_*.tfrecord'.format(self.data_dir))
        #
        dataset = tf.data.TFRecordDataset(data_filenames)
        dataset = dataset.map(self._parse_record)
        #
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=1000 + 3 * self.batch_size)
        # unlimited repetitions as we work with all iterations that we want, 32000/128 = 250 iter to finish 1 epoch
        dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        #
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels

    def _generate_evaluaton_batch(self):
        data_filenames = glob.glob(
            '{}/evaluation_*.tfrecord'.format(self.data_dir))
        #
        dataset = tf.data.TFRecordDataset(data_filenames)
        dataset = dataset.map(self._parse_record)
        #
        dataset = dataset.repeat(1)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)

        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels


# if __name__ == '__main__':
#   pass
    # conf_path = ''
    # config_reader = ConfigReader(conf_path)
    # data_config = DataConfig(config_reader.get_train_config())
    # test_loader = _CNN_Data_Loader(data_config, training_mode=False, shuffle=False)
    # test_image_batch, test_labels_batch = test_loader._generate_batch()
