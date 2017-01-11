from __future__ import division, print_function, absolute_import

import os

import tensorflow as tf
import dataset
import network
import util
slim = tf.contrib.slim

print('Setting up run')
nc, dc, rc = util.parse_arguments()
run_name = util.run_name(nc,dc,rc)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_dir', 'checkpoint/' + run_name + '/','output directory for model checkpoints')
flags.DEFINE_string('summary_dir', 'logs/' + run_name, 'output directory for training summaries')
flags.DEFINE_float('gamma',0.5,'learning rate change per step')
flags.DEFINE_float('learning_rate',0.03,'learning rate change per step')

inception = tf.contrib.slim.nets.inception

# Fine-tuning from there
inception_v3_ckpt = '/u/vul-d1/scratch/wso226/deepcalibration_data/inception_v3.ckpt'

if not tf.gfile.Exists(FLAGS.checkpoint_dir):
    print('Making checkpoint dir')
    os.makedirs(FLAGS.checkpoint_dir)

if not tf.gfile.Exists(FLAGS.summary_dir):
    print('Making summary dir')
    os.makedirs(FLAGS.summary_dir)

#
# Define graph 
#

with tf.variable_scope('Input'):
    print('Defining input pipeline')

    feat, label, recname = dataset.records_train_all(**dc)

with tf.variable_scope('Predictor'):
    print('Defining prediction network')

    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(
                                             feat,
                                             num_classes=2,
                                             is_training=True)

with tf.variable_scope('Loss'):
    print('Defining loss functions')

    loss_reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits,
            label)

    prediction = tf.cast(tf.argmax(logits,1),dtype=tf.int32)

    loss_class = 10*tf.reduce_mean(loss_class)

    loss = loss_class + loss_reg 

with tf.variable_scope('Train'):
    print('Defining training methods')

    global_step = tf.Variable(0,name='global_step',trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,global_step,40000,FLAGS.gamma,staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate,epsilon=.1)
    train_op = optimizer.minimize(loss,global_step=global_step)

    acc = tf.contrib.metrics.accuracy(prediction,label)

with tf.variable_scope('Summaries'):
    print('Defining summaries')

    tf.summary.scalar('loss_class', loss_class)
    tf.summary.scalar('loss_reg', loss_reg)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('accuracy', acc)
    # tf.audio_summary('audio', feat, 44100.0)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:

    update_ops = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    summary_writer = tf.summary.FileWriter(FLAGS.summary_dir, 
                                           sess.graph,
                                           flush_secs=5)
    summary = tf.summary.merge_all()
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    #ckpt = tf.train.get_checkpoint_state('checkpoint/v5_relu_1.00_1.00_yes/')

    if ckpt and ckpt.model_checkpoint_path: 
        print('Restoring checkpoint')
        saver.restore(sess, ckpt.model_checkpoint_path)

    #sess.run(global_step.assign(0))
    _i = sess.run(global_step)

    print('Starting training')
    while _i < 300000:

        _,_,_i, \
        _loss,_loss_reg,_loss_class,_acc, \
        _summary \
        = sess.run([
            train_op,
            update_ops,
            global_step,
            loss,
            loss_reg,
            loss_class,
            acc,
            summary
            ])

        print(str(_i) +' : lc ' + str(_loss_class) +' : lr ' +
                str(_loss_reg) + ' : l ' + str(_loss) + ' : a ' + str(_acc))

        summary_writer.add_summary(_summary, _i)
        summary_writer.flush()

	if _i % 100 == 0:
	    print("saving total checkpoint")
	    saver.save(sess, FLAGS.checkpoint_dir + 'model.ckpt', global_step=_i)

    print('Cleaning up')
    coord.request_stop()
    coord.join(threads)
    print('Done')

