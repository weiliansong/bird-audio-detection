import tensorflow as tf
import wave
import numpy as np

def process_wav(f):
    try:
        fid = wave.open('./output.wav', "rb")
        raw = fid.readframes(fid.getnframes())
        y = np.fromstring(raw,dtype=np.int16).astype(np.float32)

        # pad if necessary 
        amount_short = 400000-y.size
        if 0 < amount_short:
            y = np.pad(y, 
                    (0,amount_short),
                    'wrap') 

        y = y / 32768.
        #y = y / np.sqrt(1e-8 + np.mean(y**2))
        #y = y / 100.

        return y
    except Exception,e:
        print(e)

y = tf.py_func(read_wav, [recname], [tf.float32])
y = tf.reshape(y,(-1,1,1))
y = tf.random_crop(y,(d,1,1)) 
y = tf.squeeze(y)

return y 
