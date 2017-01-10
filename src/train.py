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

dataset_names = ['freefield1010', 'warblr']

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

    s_feat, s_label, s_recname, s_dataset_label = dataset.records_train_all(
                                    dataset_names=['freefield1010'],
                                    batch_size=50,
                                    **dc)

    t_feat, t_label, t_recname, t_dataset_label = dataset.records_train_all(
                                    dataset_names=['warblr'],
                                    batch_size=14,
                                    **dc)

    feat = tf.concat(0, [s_feat, t_feat])
    dataset_label = tf.concat(0, [s_dataset_label, t_dataset_label])

with tf.variable_scope('Predictor'):
    print('Defining prediction network')

    logits, dataset_logits = network.network(feat, is_training=True,**nc)

with tf.variable_scope('Loss'):
    print('Defining loss functions')

    # Only source logits for classification loss
    s_logits = tf.slice(logits, [0,0], [50,-1])

    loss_reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(
            s_logits,
            s_label)

    loss_discrim = tf.nn.sparse_softmax_cross_entropy_with_logits(
            dataset_logits,
            dataset_label)

    loss_class = 10*tf.reduce_mean(loss_class)

    loss_discrim = tf.reduce_mean(loss_discrim)

    loss_conf = 1 / loss_discrim

    prediction = tf.cast(tf.argmax(logits,1),dtype=tf.int32)

    loss = loss_class + loss_conf + loss_reg 

with tf.variable_scope('Train'):
    print('Defining training methods')

    global_step = tf.Variable(0,name='global_step',trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,global_step,40000,FLAGS.gamma,staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate,epsilon=.1)

    model_vars = slim.get_model_variables()

    g_exclude = ['Conv_9', 'Conv_10', 'Conv_11']

    d_var_list = [var for var in model_vars if 'Conv_9' in var.op.name]
    g_var_list = []

    for var in model_vars:
        if var.op.name.split('/')[1] not in g_exclude:
            g_var_list.append(var)

    all_train_op = optimizer.minimize(loss,global_step=global_step)

    d_train_op = optimizer.minimize(loss_discrim, 
                                    global_step=global_step,
                                    var_list=d_var_list)

    g_train_op = optimizer.minimize(loss_conf,
                                    global_step=global_step,
                                    var_list=g_var_list)

    train_op = tf.group(all_train_op,d_train_op, g_train_op)

    acc = tf.contrib.metrics.accuracy(prediction,label)

with tf.variable_scope('Summaries'):
    print('Defining summaries')

    tf.summary.scalar('loss_class', loss_class)
    tf.summary.scalar('loss_reg', loss_reg)
    tf.summary.scalar('loss_discrim', loss_discrim)
    tf.summary.scalar('loss_conf', loss_conf)
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
        _loss,_loss_reg,_loss_class,_loss_discrim,_loss_conf,_acc, \
        _summary \
        = sess.run([
            train_op,
            update_ops,
            global_step,
            loss,
            loss_reg,
            loss_class,
            loss_discrim,
            loss_conf,
            acc,
            summary
            ])

        print(str(_i) +': lc ' + str(_loss_class) 
                + ': ld ' + str(_loss_discrim) 
                + ': lg ' + str(_loss_conf) 
                + ': lr ' + str(_loss_reg) 
                + ': l ' + str(_loss) 
                + ': a ' + str(_acc))

        summary_writer.add_summary(_summary, _i)
        summary_writer.flush()

	if _i % 100 == 0:
	    print("saving total checkpoint")
	    saver.save(sess, FLAGS.checkpoint_dir + 'model.ckpt', global_step=_i)

    print('Cleaning up')
    coord.request_stop()
    coord.join(threads)
    print('Done')

