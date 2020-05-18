# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'nihxray', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '/media/whi-gpu01/fd2bf426-d2ee-4129-8a76-35c8216ac3a3/NIH-Xray/output/', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', 'nihxray', 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=14,
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    #[image, label] = provider.get(['image', 'label'])
    #label -= FLAGS.labels_offset
    [image, label1, label2, label3, label4, label5, label6, label7,
     label8, label9, label10, label11, label12, label13, label14] = \
        provider.get(['image', 'label1', 'label2', 'label3', 'label4', 'label5',
                      'label6', 'label7', 'label8', 'label9', 'label10',
                      'label11', 'label12', 'label13', 'label14'])
    print(image.shape)

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = 'nihxray'
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    #images, labels = tf.train.batch(
    #    [image, label],
    #    batch_size=FLAGS.batch_size,
    #    num_threads=FLAGS.num_preprocessing_threads,
    #    capacity=5 * FLAGS.batch_size)
    images, labels1, labels2, labels3, labels4, labels5, labels6, labels7, \
    labels8, labels9, labels10, labels11, labels12, labels13, labels14 \
        = tf.train.batch(
        [image, label1, label2, label3, label4, label5, label6, label7,
         label8, label9, label10, label11, label12, label13, label14],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)
    labels1 = tf.expand_dims(labels1, 1); labels2 = tf.expand_dims(labels2, 1)
    labels3 = tf.expand_dims(labels3, 1); labels4 = tf.expand_dims(labels4, 1)
    labels5 = tf.expand_dims(labels5, 1); labels6 = tf.expand_dims(labels6, 1)
    labels7 = tf.expand_dims(labels7, 1); labels8 = tf.expand_dims(labels8, 1)
    labels9 = tf.expand_dims(labels9, 1); labels10 = tf.expand_dims(labels10, 1)
    labels11 = tf.expand_dims(labels11, 1); labels12 = tf.expand_dims(labels12, 1)
    labels13 = tf.expand_dims(labels13, 1); labels14 = tf.expand_dims(labels14, 1)
    labels = tf.concat([labels1, labels2, labels3, labels4, labels5, labels6, labels7,
                        labels8, labels9, labels10, labels11, labels12, labels13, labels14], 1)
    print(labels.shape)

    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(images)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    #predictions = tf.argmax(logits, 1)
    #labels = tf.squeeze(labels)
    predictions = logits
    pred1 = predictions[:,0]; pred2 = predictions[:,1]; pred3 = predictions[:,2]
    pred4 = predictions[:, 3]; pred5 = predictions[:, 4]; pred6 = predictions[:, 5]
    pred7 = predictions[:, 6]; pred8 = predictions[:, 7]; pred9 = predictions[:, 8]
    pred10 = predictions[:, 9]; pred11 = predictions[:, 10]; pred12 = predictions[:, 11]
    pred13 = predictions[:, 12]; pred14 = predictions[:, 13]

    pred1 = tf.div(tf.subtract(pred1,tf.reduce_min(pred1)),
                   tf.subtract(tf.reduce_max(pred1), tf.reduce_min(pred1)))
    pred2 = tf.div(tf.subtract(pred2, tf.reduce_min(pred2)),
                   tf.subtract(tf.reduce_max(pred2), tf.reduce_min(pred2)))
    pred3 = tf.div(tf.subtract(pred3, tf.reduce_min(pred3)),
                   tf.subtract(tf.reduce_max(pred3), tf.reduce_min(pred3)))
    pred4 = tf.div(tf.subtract(pred4, tf.reduce_min(pred4)),
                   tf.subtract(tf.reduce_max(pred4), tf.reduce_min(pred4)))
    pred5 = tf.div(tf.subtract(pred5, tf.reduce_min(pred5)),
                   tf.subtract(tf.reduce_max(pred5), tf.reduce_min(pred5)))
    pred6 = tf.div(tf.subtract(pred6, tf.reduce_min(pred6)),
                   tf.subtract(tf.reduce_max(pred6), tf.reduce_min(pred6)))
    pred7 = tf.div(tf.subtract(pred7, tf.reduce_min(pred7)),
                   tf.subtract(tf.reduce_max(pred7), tf.reduce_min(pred7)))
    pred8 = tf.div(tf.subtract(pred8, tf.reduce_min(pred8)),
                   tf.subtract(tf.reduce_max(pred8), tf.reduce_min(pred8)))
    pred9 = tf.div(tf.subtract(pred9, tf.reduce_min(pred9)),
                   tf.subtract(tf.reduce_max(pred9), tf.reduce_min(pred9)))
    pred10 = tf.div(tf.subtract(pred10, tf.reduce_min(pred10)),
                    tf.subtract(tf.reduce_max(pred10), tf.reduce_min(pred10)))
    pred11 = tf.div(tf.subtract(pred11, tf.reduce_min(pred11)),
                    tf.subtract(tf.reduce_max(pred11), tf.reduce_min(pred11)))
    pred12 = tf.div(tf.subtract(pred12, tf.reduce_min(pred12)),
                    tf.subtract(tf.reduce_max(pred12), tf.reduce_min(pred12)))
    pred13 = tf.div(tf.subtract(pred13, tf.reduce_min(pred13)),
                    tf.subtract(tf.reduce_max(pred13), tf.reduce_min(pred13)))
    pred14 = tf.div(tf.subtract(pred14, tf.reduce_min(pred14)),
                    tf.subtract(tf.reduce_max(pred14), tf.reduce_min(pred14)))
    labels1 = labels[:,0]; labels2 = labels[:,1]; labels3 = labels[:,2]; labels4 = labels[:,3]
    labels5 = labels[:,4]; labels6 = labels[:,5]; labels7 = labels[:,6]; labels8 = labels[:,7]
    labels9 = labels[:, 8]; labels10 = labels[:, 9]; labels11 = labels[:, 10];
    labels12 = labels[:, 11]; labels13 = labels[:, 12]; labels14 = labels[:, 13]
    print(pred1.shape)
    print(labels1.shape)

    # Define the metrics:
    #names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    #    'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
    #    'Recall_5': slim.metrics.streaming_recall_at_k(
    #        logits, labels, 5),
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'AUC1': slim.metrics.streaming_auc(pred1, labels1),
        'AUC2': slim.metrics.streaming_auc(pred2, labels2),
        'AUC3': slim.metrics.streaming_auc(pred3, labels3),
        'AUC4': slim.metrics.streaming_auc(pred4, labels4),
        'AUC5': slim.metrics.streaming_auc(pred5, labels5),
        'AUC6': slim.metrics.streaming_auc(pred6, labels6),
        'AUC7': slim.metrics.streaming_auc(pred7, labels7),
        'AUC8': slim.metrics.streaming_auc(pred8, labels8),
        'AUC9': slim.metrics.streaming_auc(pred9, labels9),
        'AUC10': slim.metrics.streaming_auc(pred10, labels10),
        'AUC11': slim.metrics.streaming_auc(pred11, labels11),
        'AUC12': slim.metrics.streaming_auc(pred12, labels12),
        'AUC13': slim.metrics.streaming_auc(pred13, labels13),
        'AUC14': slim.metrics.streaming_auc(pred14, labels14),
    })

    # Print the summaries to screen.
    for name, value in names_to_values.items():
      summary_name = 'eval/%s' % name
      op = tf.summary.scalar(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    slim.evaluation.evaluate_once(
        master=FLAGS.master,
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=list(names_to_updates.values()),
        variables_to_restore=variables_to_restore)


if __name__ == '__main__':
  tf.app.run()
