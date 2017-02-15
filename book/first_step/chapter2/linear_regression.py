import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt


def generate_simple_data():
    """Generate 2-D vectors that hold y = 0.1*x + 0.1 using Gaussian distributions"""
    num_points = 1000
    vector_set = []

    for i in range(num_points):
        x1 = np.random.normal(0.0, 0.55)
        y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
        vector_set.append([x1, y1])

    x_data = [v[0] for v in vector_set]
    y_data = [v[1] for v in vector_set]

    return x_data, y_data


def plot_data(x_data, y_data):
    """Plot 2-D vectors"""
    plt.plot(x_data, y_data, 'ro')
    plt.legend()
    plt.show()


def define_model():
    x_data, y_data = generate_simple_data()

    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    y = W * x_data + b

    loss = tf.reduce_mean(tf.square(y - y_data))

    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    return W, b, train


def training_model(W, b, train, *, epoch_size):
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for step in range(epoch_size):
            sess.run(train)

        result_W = sess.run(W)
        result_b = sess.run(b)

        return result_W, result_b


def plot_result(weights, bias, data):
    x_data, y_data = data

    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, weights * x_data + bias)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == "__main__":
    generated_data = generate_simple_data()
    weights, bias, train = define_model()

    result_weights, result_bias = training_model(weights, bias, train, epoch_size=10)

    print(result_weights, result_bias)
    plot_result(result_weights, result_bias, generated_data)

