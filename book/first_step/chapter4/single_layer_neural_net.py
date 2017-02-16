import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MINIST_data", one_hot=True)

IMAGE_SIZE = 28 * 28
NUM_CLASSES = 10
batch_size = 100


def print_mnist_info():
    print(tf.convert_to_tensor(mnist.train.images).get_shape())
    print(tf.convert_to_tensor(mnist.train.labels).get_shape())


def placeholder_inputs():
    input_images = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE))
    input_labels = tf.placeholder(tf.float32, shape=(None, 10))

    return input_images, input_labels


def define_model(input_images):
    W = tf.Variable(tf.zeros([IMAGE_SIZE, NUM_CLASSES]))
    b = tf.Variable(tf.zeros([NUM_CLASSES]))

    y = tf.nn.softmax(tf.matmul(input_images, W) + b)
    return y


def evaluate_loss(input_labels, y):
    cross_entropy = -tf.reduce_sum(input_labels * tf.log(y))
    return cross_entropy


def run_training(epoch=1000):
    input_images, input_labels = placeholder_inputs()

    y = define_model(input_images)
    loss = evaluate_loss(input_labels, y)

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss=loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(epoch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size=batch_size)
            sess.run(train_step, feed_dict={input_images: batch_xs, input_labels: batch_ys})

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(input_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        test_images = mnist.test.images
        test_labels = mnist.test.labels
        print(sess.run(accuracy, feed_dict={input_images: test_images, input_labels: test_labels}))


if __name__ == "__main__":
    run_training(epoch=1000)
