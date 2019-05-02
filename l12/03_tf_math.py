# Solution is available in the "solution.ipynb" 
import tensorflow as tf

x = tf.constant(10)
y = tf.constant(2)
z = tf.divide(tf.cast(x, tf.float32), tf.cast(y, tf.float32)) - tf.cast(tf.constant(1), tf.float32)

with tf.Session() as sess:
    output = sess.run(z)
    print(output)
