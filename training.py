import os

import numpy as np
import tensorflow as tf

from lib.cnn.small_sence_v4 import _sence_net
from lib.data_loader.data_loader import _CNN_Data_Loader
from lib.network_util.config import ConfigReader, TrainNetConfig, DataConfig
from utils.read_config import ReadConfig


def train(conf_path, out_dir):
    conf_path = conf_path
    config_reader = ConfigReader(conf_path)
    train_config = TrainNetConfig(config_reader.get_train_config())
    data_config = DataConfig(config_reader.get_train_config())

    out_dir = os.path.join(out_dir, train_config.name)
    train_log_dir = '{}/logs/train/'.format(out_dir)
    test_log_dir = '{}/logs/test/'.format(out_dir)

    if not os.path.exists(train_log_dir):
        os.makedirs(train_log_dir)
    if not os.path.exists(test_log_dir):
        os.makedirs(test_log_dir)
    #
    # net = Vanilla_net(train_config)
    net = _sence_net(train_config)
    #
    with tf.name_scope('input'):
        train_loader = _CNN_Data_Loader(data_config, training_mode=True,
                                        shuffle=True)
        train_image_batch, train_labels_batch = train_loader._generate_batch()
        test_loader = _CNN_Data_Loader(data_config, training_mode=False,
                                       shuffle=True)  # default false
        test_image_batch, test_labels_batch = test_loader._generate_batch()

    train_op = net.build_model()
    summaries = net.get_summary()

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge(summaries)

    init = tf.global_variables_initializer()

    # config = tf.ConfigProto(log_device_placement=True)
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    #
    # if tf.test.is_gpu_available(
    #         cuda_only=False,
    #         min_cuda_compute_capability=None
    # ):
    #     device_name = '/gpu:0'
    # else:
    #     device_name = '/cpu:0'
    #
    # with tf.device(device_name):
    # sess = tf.Session(config=config)
    sess = tf.Session()
    sess.run(init)

    train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(test_log_dir, sess.graph)

    try:
        for step in np.arange(train_config.max_step):
            train_image, train_label = sess.run(
                [train_image_batch, train_labels_batch])
            _, train_loss, train_acc = sess.run(
                [train_op, net.loss, net.accuracy],
                feed_dict={net.x: train_image, net.y: train_label})
            if step % 50 == 0 or step + 1 == train_config.max_step:
                print('===TRAIN===: Step: %d, loss: %.4f, accuracy: %.4f%%' % (
                step, train_loss, train_acc))
                summary_str = sess.run(summary_op,
                                       feed_dict={net.x: train_image,
                                                  net.y: train_label})
                train_summary_writer.add_summary(summary_str, step)
            if step % 100 == 0 or step + 1 == train_config.max_step:
                val_image, val_label = sess.run(
                    [test_image_batch, test_labels_batch])
                plot_images = tf.summary.image(
                    'val_images_{}'.format(step % 200), val_image, 10)
                val_loss, val_acc, plot_summary = sess.run(
                    [net.loss, net.accuracy, plot_images],
                    feed_dict={net.x: val_image, net.y: val_label})
                print(
                    '====VAL====: Step %d, val loss = %.4f, val accuracy = %.4f%%' % (
                    step, val_loss, val_acc))
                summary_str = sess.run(summary_op, feed_dict={net.x: val_image,
                                                              net.y: val_label})
                val_summary_writer.add_summary(summary_str, step)
                val_summary_writer.add_summary(plot_summary, step)
            if step % 2000 == 0 or step + 1 == train_config.max_step:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
    except tf.errors.OutOfRangeError:
        sess.close()
        print(
            '===INFO====: Training completed, reaching the maximum number of steps')
    sess.close()


def _train20190622_debug():
    rc = ReadConfig()
    path = rc.read_data_path()
    conf_path = rc.read_experiment_1_path()
    out_dir = path['common_path'] + path['model_out_dir']
    train(conf_path, out_dir)
    return None


def _train20190711_3channel():
    rc = ReadConfig()
    path = rc.read_data_path()
    conf_path = rc.read_experiment_2_path()
    out_dir = path['common_path'] + path['model_out_dir']
    train(conf_path, out_dir)
    return None


def _train20190712_3channel_acgy_10_classes():
    rc = ReadConfig()
    path = rc.read_data_path()
    conf_path = rc.read_experiment_2_path()
    out_dir = path['common_path'] + path['model_out_dir']
    train(conf_path, out_dir)
    return None


def main():
    # _train20190711_3channel()
    _train20190712_3channel_acgy_10_classes()


if __name__ == '__main__':
    main()
