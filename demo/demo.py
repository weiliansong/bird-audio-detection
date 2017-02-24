import tensorflow as tf
import numpy as np
import recorder
import audio_loader
import network

# Figure out what this needs to be
feat = tf.placeholder(tf.float32, shape=(1,400000))

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
            recorder.record()

        audio_feat = audio_loader.process_wav('./demo_audio/%d.wav' % choice)

        _feat, _probs, _prediction = sess.run([feat, probs, prediction],
                                              feed_dict={feat:audio_feat})
        
        if _prediction:
            print('\n-----I am %.0f%% confident that there is a bird!-----\n' 
                    % (_probs[1]*100))
        else:
            print('\n-----I am %.0f%% confident that there is not a bird!-----\n' 
                    % (_probs[0]*100))
