import tensorflow as tf
import numpy as np
import recorder
import audio_loader
import network

with tf.variable_scope('Input'):
    feat = audio_loader.read_wav()

with tf.variable_scope('Predictor'):
    logits = network.network(feat, is_training=False)

    prediction = tf.cast(tf.argmax(logits,1), dtype=tf.int32)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())

    ckpt = tf.train.get_ckeckpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path: 
        print('Restoring checkpoint')
        saver.restore(sess, ckpt.model_checkpoint_path)

    quit = False

    while not quit:
        choice = raw_input('Record an audio clip? (y/n): ')

        if choice == 'n':
            quit = True

        else:
            recorder.record()

            _prediction = sess.run([prediction])
