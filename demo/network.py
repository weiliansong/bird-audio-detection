# MAYBE explicitly name layers

import tensorflow as tf
import ops
slim = tf.contrib.slim
import numpy as np

def network_arg_scope(
        weight_decay=0.0004,
        is_training=True,
        batch_norm_var_collection='moving_vars',
        activation_fn=tf.nn.relu,
        keep_prob=.8):
    ''' Sets default parameters for network layers.'''

    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.999,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # collection containing update_ops.
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        # collection containing the moving mean and moving variance.
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        }
    }

    # for activation functions that are not "mostly linear" we should
    # have a scale parameter
    if activation_fn.func_name in ['elu']: 
        batch_norm_params['scale'] = True

    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        weights_initializer=slim.variance_scaling_initializer(),
        padding='VALID',
        outputs_collections=tf.GraphKeys.ACTIVATIONS,
        activation_fn=activation_fn,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.max_pool2d], 
            stride=(2,1),
            outputs_collections=tf.GraphKeys.ACTIVATIONS):
            with slim.arg_scope([slim.batch_norm], is_training=is_training) as sc:
                with slim.arg_scope(
                        [slim.dropout], 
                        is_training=is_training, keep_prob=keep_prob) as sc:

                    return sc

def network(net, is_training=True, activation_fn=tf.nn.relu, capacity=1.0, capacity2=1.0, network='v2.1'):

    print('Using network ' + network + '.')
    return networks[network](net, 
        is_training=is_training, 
        activation_fn=activation_fn, 
        capacity=capacity)

def network_v5(net, is_training=True, activation_fn=tf.nn.relu,
        capacity=1.0, capacity2=1.0):

    net = tf.reshape(net,(-1,100000,4,1))

    with slim.arg_scope(network_arg_scope(is_training=is_training,
        activation_fn=activation_fn)):

        # extract window features
        net = slim.conv2d(net,np.rint(capacity*32),[3,4],stride=(2,1))
        net = slim.conv2d(net,np.rint(capacity*32),[9,1],stride=(5,1))
        net = slim.conv2d(net,np.rint(capacity*64),[9,1],stride=(5,1))
        #net = slim.dropout(net)
        net = slim.conv2d(net,np.rint(capacity*128),[9,1],stride=(5,1))
        #net = slim.dropout(net)
        net_early = tf.reduce_mean(net,[1],keep_dims=True)
        net = slim.conv2d(net,np.rint(capacity*128),[9,1],stride=(5,1))
        #net = slim.dropout(net)
        net = slim.conv2d(net,np.rint(capacity*256),[9,1],stride=(5,1))
        #net = slim.dropout(net)
        net_late = tf.reduce_mean(net,[1],keep_dims=True)
        net = slim.conv2d(net,np.rint(capacity*256),[3,1],stride=(2,1))
        #net = slim.dropout(net)
        net = slim.conv2d(net,np.rint(capacity*512),[3,1],stride=(2,1))
        #net = slim.dropout(net)
        print(net)
        net = tf.reduce_mean(net,[1],keep_dims=True)
        print(net)
        net = tf.concat((net,net_late,net_early),axis=3)
        print(net)
        net = slim.conv2d(net,512,[1,1], stride=(1,1))
        #net = slim.dropout(net)
        net = slim.conv2d(net,512,[1,1], stride=(1,1))
        # net = slim.dropout(net)
        net = slim.conv2d(net,2,[1,1], stride=(1,1),
                normalizer_fn=None,activation_fn=None)

        print(net)

        net = slim.flatten(net,[1])

        net = tf.squeeze(net)

        return net 

networks = {
        'v5':network_v5, 
        }

