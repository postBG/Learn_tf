import tensorflow as tf

import numpy as np

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def generate_random_points():
    num_points = 2000
    vectors_set = []

    for i in range(num_points):
        if np.random.random() > 0.5:
            vectors_set.append([np.random.normal(0.0, 0.9),
                                np.random.normal(0.0, 0.9)])
        else:
            vectors_set.append([np.random.normal(3.0, 0.5),
                                np.random.normal(1.0, 0.5)])

    return vectors_set


def plot_points(vectors_set):
    df = pd.DataFrame({"x": [v[0] for v in vectors_set],
                       "y": [v[1] for v in vectors_set]})
    sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
    plt.show()


def run_training(vectors_set, k=4, epoch=100):
    vectors = tf.constant(vectors_set)
    centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0, 0], [k, -1]))

    expanded_vectors = tf.expand_dims(vectors, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)

    distances = tf.reduce_sum(tf.square(tf.sub(expanded_vectors, expanded_centroids)), 2)  # shape => k * num_points
    assignments = tf.argmin(distances, 0)  # shape => num_points

    means = tf.concat(0, [tf.reduce_mean(tf.gather(vectors,
                                                   tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])),
                                         reduction_indices=[1]) for c in range(k)])

    update_centroids = tf.assign(centroids, means)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        for step in range(epoch):
            _, centroid_values, assignment_values = sess.run([update_centroids, centroids, assignments])

        return centroid_values, assignment_values


def plot_results(vectors_set, assignments):
    data = {"x": [], "y": [], "cluster": []}

    for i in range(len(assignments)):
        data["x"].append(vectors_set[i][0])
        data["y"].append(vectors_set[i][1])
        data["cluster"].append(assignments[i])

    df = pd.DataFrame(data)
    sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
    plt.show()


if __name__ == "__main__":
    vectors_set = generate_random_points()
    centroids, assignments = run_training(vectors_set, k=2)
    plot_results(vectors_set, assignments)
