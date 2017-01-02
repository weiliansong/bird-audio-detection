"""
    This script helps to find the location and dimension of the adaptation
    layer. For more information please look up "Deep Domain Confusion" paper.
"""

import util
import network
import dataset
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

nc,dc = util.parse_arguments()
run_name = util.run_name(nc,dc)

checkpoint_dir = 'checkpoint/' + run_name + '/'

feat = tf.placeholder(tf.float32)

def compute_MMD(X_s, X_t, norm=False):
    sum_source = np.sum(X_s, axis=0)
    sum_target = np.sum(X_t, axis=0)

    MMD = (sum_source - sum_target) / len(sum_source)

    if norm:
        return np.linalg.norm(MMD)
    else:
        return MMD

with tf.variable_scope('Input'):
    print('Defining input pipeline')
    
    target_feat, target_label, target_recname = dataset.records_challenge()
    source_feat, source_label, source_recname = dataset.records_train_all()

with tf.variable_scope('Predictor'):
    print('Defining prediction network')

    logits = network.network(feat, is_training=False, **nc)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print('Restoring checkpoint')
        saver.restore(sess, ckpt.model_checkpoint_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Get all network activations
    end_points = tf.get_collection(tf.GraphKeys.ACTIVATIONS, scope='Predictor')
    # Skip last layer on all networks
    end_points = end_points[:-1]

    for x in range(10):
        print('-----------------------------')

        T_f, T_l, T_r, S_f, S_l, S_r, = sess.run([
                            target_feat, target_label, target_recname,
                            source_feat, source_label, source_recname])

        T_logits, T_activations = sess.run([logits, end_points], 
                                           feed_dict={ feat : T_f })

        S_logits, S_activations = sess.run([logits, end_points], 
                                           feed_dict={ feat : S_f })

        MMDs = []

        for idx, layer in enumerate(end_points):
            MMD = compute_MMD(S_activations[idx],T_activations[idx],norm=True)
            MMDs.append((layer.op.name, MMD))
        
        for x in MMDs:
            print x
