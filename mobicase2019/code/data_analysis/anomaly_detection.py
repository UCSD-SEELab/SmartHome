import sys
sys.path.append("../")

import numpy as np
import statsmodels.api as sm
import tensorflow as tf

from lib.hierarchical_neural_networks import LocalSensorNetwork, CloudNetwork


def main():
    X, y = generate_data(1000)
    S1X = X[:,:3].astype(np.float32)
    S2X = X[:,3:6].astype(np.float32)
    S3X = X[:,6:9].astype(np.float32)

    s1, s2, s3 = train(S1X, S2X, S3X, y)
    VV = np.zeros((len(s1), 2))
    VV[:,0] = s2
    VV[:,1] = s3


def arma_based(VV, s1):
    # add constant
    VV = np.concatenate((np.ones((VV.shape[0],1)), VV), axis=0)
    model = sm.OLS(s1, VV)


def bayes_based(VV, s1):
    pass

def generate_data(N, K=9):
    A = np.random.rand(K, K)
    COV = A.dot(A.T)
    mu = np.random.rand(K,1).ravel()

    X = np.random.multivariate_normal(mu, COV, size=N)
    V = np.zeros((N, K+3))
    V[:,:K] = X
    V[:,K] = X[:,0]*X[:,3]
    V[:,K+1] = X[:,1]*X[:,7]
    V[:,K+2] = X[:,4]*X[:,8]

    w_actual = np.random.rand(V.shape[1],1)
    y = V.dot(w_actual) + np.random.normal(0, 1, size=(N,1))

    return X, y


def train(S1X, S2X, S3X, y, l2=0.0,
          keep_prob=1.0,
          step_size_init=1e-3,
          batch_size=256, tol=1e-3, max_epochs=1000):
    K = S1X.shape[1]
    _y = tf.placeholder(tf.float32, (None, 1))
    _S1X = tf.placeholder(tf.float32, (None, K))
    _S2X = tf.placeholder(tf.float32, (None, K))
    _S3X = tf.placeholder(tf.float32, (None, K))

    S1 = LocalSensorNetwork(
        "S1", _S1X, (32, 16, 1), keep_prob=keep_prob)
    S2 = LocalSensorNetwork(
        "S2", _S2X, (32, 16, 1), keep_prob=keep_prob)
    S3 = LocalSensorNetwork(
        "S3", _S3X, (32, 16, 1), keep_prob=keep_prob)
    A = CloudNetwork("AGG", (32,16,1), keep_prob=keep_prob)
    out = A.connect([S1, S2, S3])

    eps = tf.losses.mean_squared_error(_y, out)
    l2_penalty = sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
        if not ("noreg" in tf_var.name or "bias" in tf_var.name))
    total_cost = eps + l2 * l2_penalty

    global_step = tf.Variable(0, trainable=False)
    step_size = tf.train.exponential_decay(
        step_size_init, global_step, 1000, 0.9, staircase=True)

    optim = tf.train.AdamOptimizer(step_size).minimize(total_cost, global_step=global_step)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        prev_cost = np.inf
        epoch = 0
        converged = False
        while epoch < max_epochs:
            if (epoch % 10) == 0:
                print("EPOCH: {} => {}".format(epoch, prev_cost))
            batcherator = batch_iterator(S1X, S2X, S3X, y, batch_size)
            for s1x, s2x, s3x, yy in batcherator:
                sess.run(optim, feed_dict={_S1X: s1x,
                                           _S2X: s2x,
                                           _S3X: s3x,
                                           _y: yy})
                cost = sess.run(total_cost, feed_dict={_S1X: s1x,
                                                       _S2X: s2x,
                                                       _S3X: s3x,
                                                       _y: yy})
                if np.abs(cost - prev_cost) < tol:
                    converged = True
                    break
                prev_cost = cost
            epoch += 1

            if converged:
                break

        print("Converged! Cost: {}".format(prev_cost))

        # now get all the values of the local outputs

        s1 = []
        s2 = []
        s3 = []
        batcherator = batch_iterator(S1X, S2X, S3X, y, 1)
        for s1x, s2x, s3x, yy in batcherator:
            s1v, s2v, s3v = sess.run([S1.output, S2.output, S3.output],
                                     feed_dict={_S1X: s1x, _S2X: s2x,
                                                _S3X: s3x, _y: yy})
            s1.append(s1v.ravel()[0])
            s2.append(s2v.ravel()[0])
            s3.append(s3v.ravel()[0])

        return s1, s2, s3

def batch_iterator(S1X, S2X, S3X, y, batch_size):
    lb = 0
    while lb < S1X.shape[0]:
        s1xn = S1X[lb:lb+batch_size,:]
        s2xn = S2X[lb:lb+batch_size,:]
        s3xn = S3X[lb:lb+batch_size,:]
        yn = y[lb:lb+batch_size]
        lb += batch_size

        yield (s1xn, s2xn, s3xn, yn)


if __name__=="__main__":
    main()
