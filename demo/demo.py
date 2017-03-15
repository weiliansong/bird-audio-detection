import network
import numpy as np
import audio_tools
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.misc import imresize
from scipy.ndimage.filters import gaussian_filter

# Figure out what this needs to be
feat = tf.placeholder(tf.float32, shape=(1,400000))

DEBUG = True

# Plot everything into multiple columns, so it's most squared
def plot_images(plt_title, activations, subplot_titles=True,
                series_axis=True, series_range=(-2.0, 2.0)):
    plt.close('all')
    fig = plt.figure(1)
    num_subplots = len(activations)
    num_cols = np.floor(np.sqrt(num_subplots))
    num_rows = np.ceil(num_subplots / num_cols)
    keys = activations.keys()
    keys.sort()
    counter = 1
    for key in keys:
        activ_map = activations[key]
        ax = plt.subplot(num_rows, num_cols, counter)
        if subplot_titles:
            ax.set_title('%s %s' % (key, activ_map.shape))

        # If it's a 1D array
        if activ_map.shape[0] == 1:
            activ_map = activ_map.reshape((activ_map.shape[-1],))
            plt.plot(range(activ_map.size), activ_map)
            x1, x2, (y1, y2) = 0, activ_map.size, series_range
            plt.axis((x1, x2, y1, y2))

            if not series_axis:
                plt.axis('off')
        
        # If it's a 2D array
        else:
            map_shape = (50, 100)
            image = imresize(activations[key], map_shape)
            # image = gaussian_filter(image, 1)
            plt.axis('off')
            plt.imshow(image)

        counter += 1

    fig.suptitle(plt_title, size=16)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)

with tf.variable_scope('Predictor'):
    logits = network.network(feat, 
                             is_training=False,
                             activation_fn=tf.nn.relu,
                             capacity=0.8,
                             capacity2=0.2,
                             network='v5')

    probs = tf.nn.softmax(logits)
    prediction = tf.cast(tf.argmax(logits,0), dtype=tf.int32)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    
    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state('./checkpoint')
    print('Restoring checkpoint')
    saver.restore(sess, ckpt.model_checkpoint_path)

    # Network end_points
    end_points = tf.get_collection(tf.GraphKeys.ACTIVATIONS, scope='Predictor')

    quit = False

    while True:
        print('1. Noisy audio with bird, dog and airplane noise')
        print('2. Kid singing, no bird')
        print('3. Human breathing and background noise, no bird')
        print('4. Whistling, no bird')
        print('5. Very very very... faint bird sound')
        print('6. Record your own!')

        choice = int(raw_input('Please enter your choice!: '))

        if choice == -1:
            break
        
        if choice > 6:
            print('Invalid choice, please enter a number b/t 1 and 6')

        if choice == 6:
            raw_input('Hit enter to start recording')
            audio_tools.record()

        if not DEBUG:
            audio_tools.play('./demo_audio/%d.wav' % choice)

        audio_feat = audio_tools.process_wav('./demo_audio/%d.wav' % choice)

        _feat, _probs, _prediction, _end_points = sess.run(
            [feat, probs, prediction, end_points], feed_dict={feat:audio_feat})
        
        if _prediction:
            print('\n-----I am %.0f%% confident that there is a bird!-----\n' 
                    % (_probs[1]*100))
        else:
            print('\n-----I am %.0f%% confident that there is not a bird!-----\n' 
                    % (_probs[0]*100))

        activations = {}
        
        for idx, activ_map in enumerate(_end_points[1:-1]):
            activ_map = activ_map.reshape(-1, activ_map.shape[-1])
            
            if activ_map.shape[0] > activ_map.shape[1]:
                activ_map = activ_map.T
            
            activations['Conv_%d' % idx] = activ_map

        plot_images('Visualizations', activations)
        plt.savefig('./visualization.png')
