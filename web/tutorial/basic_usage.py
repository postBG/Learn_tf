import tensorflow as tf


def computation_graph():
    # Create a Constant op that produces a 1x2 matrix.  The op is
    # added as a node to the default graph.
    #
    # The value returned by the constructor represents the output
    # of the Constant op.
    matrix1 = tf.constant([[3., 3.]])

    # Create another Constant that produces a 2x1 matrix.
    matrix2 = tf.constant([[2.], [2.]])

    # Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
    # The returned value, 'product', represents the result of the matrix
    # multiplication.
    product = tf.matmul(matrix1, matrix2)

    # To run the matmul op we call the session 'run()' method, passing 'product'
    # which represents the output of the matmul op.  This indicates to the call
    # that we want to get the output of the matmul op back.
    #
    # All inputs needed by the op are run automatically by the session.  They
    # typically are run in parallel.
    #
    # The call 'run(product)' thus causes the execution of three ops in the
    # graph: the two constants and matmul.
    #
    # The output of the matmul is returned in 'result' as a numpy `ndarray` object.
    with tf.Session() as sess:
        result = sess.run(product)
        print(result)
        # ==> [[ 12.]]


def variables():
    # Create a Variable, that will be initialized to the scalar value 0.
    state = tf.Variable(0, name="counter")

    # Create an Op to add one to `state`.

    one = tf.constant(1)
    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)

    # Variables must be initialized by running an `init` Op after having
    # launched the graph.  We first have to add the `init` Op to the graph.
    init_op = tf.global_variables_initializer()

    # Launch the graph and run the ops.
    with tf.Session() as sess:
        # Run the 'init' op
        sess.run(init_op)
        # Print the initial value of 'state'
        print("start:", sess.run(state))
        # Run the op that updates 'state' and print 'state'.
        for _ in range(3):
            sess.run(update)
            print(sess.run(state))

            # output:

            # 0
            # 1
            # 2
            # 3


def fetches():
    """모든 텐서들은 한번만 계산된다. request 받을 때마다 계산되는 것이 아님."""
    input1 = tf.constant([3.0])
    input2 = tf.constant([2.0])
    input3 = tf.constant([5.0])
    intermed = tf.add(input2, input3)
    mul = tf.mul(input1, intermed)

    with tf.Session() as sess:
        result = sess.run([mul, intermed])
        print(result)

        # output:
        # [array([ 21.], dtype=float32), array([ 7.], dtype=float32)]


def feeds():
    """feed back 줄 때 사용되는 매커니즘"""
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.mul(input1, input2)

    with tf.Session() as sess:
        print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))

        # output:
        # [array([ 14.], dtype=float32)]


def main():
    computation_graph()
    variables()
    fetches()


if __name__ == "__main__":
    main()
