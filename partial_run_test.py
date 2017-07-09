import tensorflow as tf

a = tf.random_normal([2,2])

b = tf.identity(a)

sess = tf.Session()

h = sess.partial_run_setup([a,b], [])

print(sess.partial_run(h,a))

print(sess.partial_run(h,b))
