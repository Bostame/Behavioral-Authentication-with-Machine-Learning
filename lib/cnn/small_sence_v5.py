import tensorflow as tf
import numpy as np
from lib.networks.base_networks import Net


class _sence_net(Net):
    def __init__(self, cfg_):
        super().__init__(cfg_)
        self.x = tf.placeholder(tf.float32, name='x', shape=[self.config.batch_size,
                                                             self.config.image_width,
                                                             self.config.image_height,
                                                             self.config.image_depth], )
        self.y = tf.placeholder(tf.int16, name='y', shape=[self.config.batch_size,
                                                           self.config.n_classes])
        self.loss = None
        self.accuracy = None
        self.summary = []

    def init_saver(self):
        pass

    def get_summary(self):
        return self.summary

    def conv(self, layer_name, bottom, out_channels, kernel_size=[3, 3], stride=[1, 1, 1, 1]):
        in_channels = bottom.get_shape()[-1]
        with tf.variable_scope(layer_name):
            w = tf.get_variable(name='weights',
                                trainable=self.config.is_pretrain,
                                shape=[kernel_size[0], kernel_size[1],
                                       in_channels, out_channels],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='biases',
                                trainable=self.config.is_pretrain,
                                shape=[out_channels],
                                initializer=tf.constant_initializer(0.0))
            bottom = tf.nn.conv2d(bottom, w, stride, padding='SAME', name='conv')
            bottom = tf.nn.bias_add(bottom, b, name='bias_add')
            bottom = tf.nn.relu(bottom, name='relu')
            return bottom

    def pool(self, layer_name, bottom, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True):
        with tf.name_scope(layer_name):
            if is_max_pool:
                bottom = tf.nn.max_pool(bottom, kernel, stride, padding='SAME', name=layer_name)
            else:
                bottom = tf.nn.avg_pool(bottom, kernel, stride, padding='SAME', name=layer_name)
            return bottom

    def fc(self, layer_name, bottom, out_nodes, ReLu=True):
        shape = bottom.get_shape()
        if len(shape) == 4:  # x is 4D tensor
            size = shape[1].value * shape[2].value * shape[3].value
        else:  # x has already flattened
            size = shape[-1].value
        with tf.variable_scope(layer_name):
            w = tf.get_variable('weights',
                                shape=[size, out_nodes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('biases',
                                shape=[out_nodes],
                                initializer=tf.constant_initializer(0.0))
            flat_x = tf.reshape(bottom, [-1, size])
            bottom = tf.nn.bias_add(tf.matmul(flat_x, w), b)
            if ReLu:
                return tf.nn.relu(bottom)
            else:
                return bottom

    def dropout(self, x, keep_prob):
        return tf.nn.dropout(x, keep_prob)

    def batch_normalization(self, layer_name, bottom, training=True):
        with tf.name_scope(layer_name):
            epsilon = 1e-3
            bottom = tf.layers.batch_normalization(bottom, epsilon=epsilon, training=training)
            return bottom

    def concat(self, layer_name, inputs):
        with tf.name_scope(layer_name):
            one_by_one = inputs[0]
            three_by_three = inputs[1]
            five_by_five = inputs[2]
            pooling = inputs[3]
            return tf.concat([one_by_one, three_by_three, five_by_five, pooling], axis=3)

    def cal_loss(self, logits, labels):
        with tf.name_scope('loss') as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name='cross-entropy')
            self.loss = tf.reduce_mean(cross_entropy, name='loss')
            loss_summary = tf.summary.scalar(scope, self.loss)
            self.summary.append(loss_summary)

    def cal_accuracy(self, logits, labels):
        with tf.name_scope('accuracy') as scope:
            self.pred_prob = tf.nn.softmax(logits, name='probabilities')
            correct = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(labels, 1))
            correct = tf.cast(correct, tf.float32)
            self.accuracy = tf.reduce_mean(correct) * 100.0
            accuracy_summary = tf.summary.scalar(scope, self.accuracy)
            self.summary.append(accuracy_summary)

    def optimize(self):
        with tf.name_scope('optimizer'):
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
            train_op = optimizer.minimize(self.loss, global_step=self.global_step_tensor)
            return train_op

    def build_model(self):
        self.conv1_1 = self.conv('conv1_1', self.x, 16, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        self.conv1_2 = self.conv('conv1_2', self.conv1_1, 16, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        self.pool1 = self.pool('pool1', self.conv1_2, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        self.conv2_1 = self.conv('conv2_1', self.pool1, 16, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        self.conv2_2 = self.conv('conv2_2', self.conv2_1, 16, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        self.pool2 = self.pool('pool2', self.conv2_2, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        self.conv3_1 = self.conv('conv3_1', self.pool2, 16, kernel_size=[5, 5], stride=[1, 1, 1, 1])
        self.conv3_2 = self.conv('conv3_2', self.conv3_1, 16, kernel_size=[5, 5], stride=[1, 1, 1, 1])
        self.pool3 = self.pool('pool3', self.conv3_2, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        self.conv4_1 = self.conv('conv4_1', self.pool3, 48, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        self.conv4_2 = self.conv('conv4_2', self.conv4_1, 48, kernel_size=[3, 3], stride=[1, 1, 1, 1])
        self.pool4 = self.pool('pool4', self.conv4_2, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        self.fc1 = self.fc('fc1', self.pool4, out_nodes=128)
        self.dropout1 = self.dropout(self.fc1, 0.25)

        self.fc2 = self.fc('fc2', self.dropout1, out_nodes=256)
        self.dropout2 = self.dropout(self.fc2, 0.25)

        self.fc3 = self.fc('fc3', self.dropout2, out_nodes=512)
        self.dropout3 = self.dropout(self.fc3, 0.25)
        self.logits = self.fc('fc4', self.dropout3, out_nodes=self.config.n_classes, ReLu=False)

        self.cal_loss(self.logits, self.y)
        self.cal_accuracy(self.logits, self.y)
        train_op = self.optimize()
        return train_op