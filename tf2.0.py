import tensorflow as tf


# 这里 加不加@tf.function 都可以计算梯度 autograh
@tf.function
def add(a, b):
    return tf.add(a, b)


a, b = tf.Variable(2.0), tf.constant(([2.]))
with tf.GradientTape() as tape:
    c = add(a, b)
# 只能计算variables变量
gradients = tape.gradient(c, [a])
print(c)
print(gradients)


print('%04d-%s' % (1, 2))


def call(name, *args, **kwargs):
    print(name)
    print(*args)
    print(kwargs)


call('lx', 'haha', 'dd', aa='2', bb=3)
