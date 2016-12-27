import os
import wave
import util
import dataset
import network
import shutil
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import spectrogram
from scipy.cluster.hierarchy import dendrogram, linkage

slim = tf.contrib.slim

data_base = '../data/'

nc,dc = util.parse_arguments()
run_name = util.run_name(nc,dc)

checkpoint_dir = 'checkpoint/' + run_name + '/'

def plot_images(plt_title, activations):
    plt.close('all')
    fig = plt.figure(1)
    num_subplots = len(activations)
    num_cols = 2
    num_rows = np.ceil(num_subplots / num_cols)
    keys = activations.keys()
    keys.sort()
    counter = 1
    for key in keys:
        activ_map = activations[key]
        ax = plt.subplot(3, 2, counter)
        ax.set_title('%s %s' % (key, activ_map.shape))

        # If it's a 1D array
        if activ_map.shape[0] == 1:
            activ_map = activ_map.reshape((activ_map.shape[-1],))
            plt.plot(range(activ_map.size), activ_map)
            x1, x2, y1, y2 = 0, activ_map.size, -2.0, 2.0
            plt.axis((x1, x2, y1, y2))
        
        # If it's a 2D array
        else:
            map_shape = (50, 100)
            image = imresize(activations[key], map_shape)
            image = gaussian_filter(image, 1)
            plt.axis('off')
            plt.imshow(image)

        counter += 1

    fig.suptitle(plt_title, size=16)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)

with tf.variable_scope('Input'):
    print('Defining input pipeline')
    feat, label, recname = dataset.records_test_fold()

with tf.variable_scope('Predictor'):
    print('Defining prediction network')

    logits, end_points = network.network(feat,
                                         is_training=False,
                                         activation_fn=tf.nn.elu,
                                         network='v5')

    probs = tf.nn.softmax(logits)
    prediction = tf.cast(tf.argmax(logits,1),dtype=tf.int32)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    save_dir = './visualizations/%s/' % run_name
    if not os.path.exists(save_dir):
        save_dir = os.mkdir(save_dir)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path: 
        print('Restoring checkpoint')
        saver.restore(sess, ckpt.model_checkpoint_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    _feat, _label, _recname, _prediction, _end_points = sess.run([feat,
                                                                  label, 
                                                                  recname, 
                                                                  prediction, 
                                                                  end_points])
    
    for idx in range(len(_recname)):
        activations = {}
        plt_title = '%s,%d,%d' % (_recname[idx], _label[idx], _prediction[idx])
        keys = _end_points.keys() 
        keys.sort()
        for key in keys:
            activ_map = _end_points[key][idx]
            activ_map = activ_map.reshape(-1, activ_map.shape[-1])
            
            if activ_map.shape[0] > activ_map.shape[1]:
                activ_map = activ_map.T
            
            activations[key] = activ_map

        plot_images(plt_title, activations)
        
        shutil.copy(data_base + _recname[idx] + '.wav', 
                    '%s/%d.wav' % (save_dir, idx))

        plt.savefig('%s/%d.png' % (save_dir, idx), bbox_inches='tight')
