import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from algorithms import SVGD
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import logging
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ToyExample(object):
    def __init__(self, n_init_points=1000, n_target_points=1000, step_size=0.2, iters=(0, 50, 75, 100, 150, 500)):
        self.n_init_points = n_init_points
        self.n_target_points = n_target_points
        self.step_size = step_size
        self.iters = iters

    def dlnp_fun(self, x):
        norm = lambda x, m, ss: np.exp(-(x-m)**2/2/ss)/np.sqrt(2*np.pi*ss)
        dnorm = lambda x, m, ss: (m-x)/ss*norm(x, m, ss)
        res = (1/3 * dnorm(x, -2, 1) + 2/3 * dnorm(x, 2, 1)) / (1/3 * norm(x, -2, 1) + 2/3 * norm(x, 2, 1))
        return res

    def process(self):
        np.random.seed(42)
        random.seed(42)
        init_points =  np.random.normal(-10, 1, (self.n_init_points, 1))
        target_points = np.zeros((self.n_target_points, 1))
        for i in range(self.n_target_points):
            if random.random() < 1 / 3:
                target_points[i] = np.random.normal(-2, 1, 1)
            else:
                target_points[i] = np.random.normal(2, 1, 1)
        svgd = SVGD(0, 0, self.dlnp_fun)

        x = init_points
        for i in range(len(self.iters)):
            n_iters = self.iters[0] if i == 0 else self.iters[i]-self.iters[i-1]
            x = svgd.update(x, n_iters, self.step_size)
            self.plot(x, target_points, '{0}th iteration'.format(self.iters[i]))

    def plot(self, x, target, title):
        sns.distplot(x, hist=False, color='g')
        sns.distplot(target, hist=False, color='r')
        plt.title(title)
        plt.axis([-12, 12, 0, 0.4])
        plt.show()


class BNNExperiment(object):
    def __init__(self, step_size=1e-3, batch_size=100, hidden_dim=50, a0=1.0, b0=0.1, n_points=20):
        self.train_x, self.valid_x, self.test_x, self.train_y, self.valid_y, self.test_y = self.get_data()
        self.step_size = step_size
        self.batch_size = batch_size
        self.batch_x = None
        self.batch_y = None

        # save
        self.input_dim = self.train_x.shape[1]
        self.hidden_dim = hidden_dim
        self.n_points = n_points
        self.points = np.zeros((self.n_points, (self.input_dim + 2) * self.hidden_dim + 3))
        self.a0 = a0
        self.b0 = b0
        self.epoch = 0
        self.h = 0
        self.iter = 0
        self.train_x_mean = np.mean(self.train_x, axis=0)
        self.train_x_std = np.std(self.train_x, axis=0)

        # init params
        np.random.seed(42)
        for i in range(self.n_points):
            w1 = np.random.normal(scale=1/np.sqrt(self.input_dim), size=(self.input_dim, self.hidden_dim))
            b1 = np.zeros((1, self.hidden_dim))
            w2 = np.random.normal(scale=1/np.sqrt(self.hidden_dim), size=(self.hidden_dim, 1))
            b2 = 0
            ln_gamma = np.log(np.random.gamma(self.a0, self.b0))
            ln_lambda = np.log(np.random.gamma(self.a0, self.b0))
            self.points[i, :] = self.params_to_vector(w1, b1, w2, b2, ln_gamma, ln_lambda)

        # logging
        self.logger = logging.getLogger('solver')
        self.logger.setLevel(logging.INFO)
        stream_hander = logging.StreamHandler()
        file_handler = logging.FileHandler('train.log')
        self.logger.addHandler(stream_hander)
        self.logger.addHandler(file_handler)

        self.generate_dlnp_graph()

    def save(self):
        stat_dict = {
            'epoch': self.epoch,
            'points': self.points,
            'a0': self.a0,
            'b0': self.b0,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'n_points': self.points,
            'h': self.h,
            'iter': self.iter,
            'train_x_mean': self.train_x_mean,
            'train_x_std': self.train_x_std
        }
        with open('models/model_{0}.pkl'.format(self.epoch), 'wb') as f:
            pickle.dump(stat_dict, f)
        with open('models/model_last.pkl', 'wb') as f:
            pickle.dump(stat_dict, f)

    def load(self, check_point='last'):
        with open('models/model_{0}.pkl'.format(check_point), 'rb') as f:
            state_dict = pickle.load(f)
        for k, v in state_dict.items():
            setattr(self, k, v)

    def get_data(self, test_size=0.1, valid_size=0.1):
        data = pd.read_csv('UCI CBM Dataset/data.txt', sep='   ', header=None, names=[
            'Lever position',
            'Ship speed',
            'Gas Turbine shaft torque',
            'Gas Turbine rate of revolutions',
            'Gas Generator rate of revolutions',
            'Starboard Propeller Torque',
            'Port Propeller Torque',
            'HP Turbine exit temperature',
            'GT Compressor inlet air temperature',
            'GT Compressor outlet air temperature',
            'HP Turbine exit pressure',
            'GT Compressor inlet air pressure',
            'GT Compressor outlet air pressure',
            'Gas Turbine exhaust gas pressure',
            'Turbine Injecton Control',
            'Fuel flow',
            'GT Compressor decay state coefficient',
            'GT Turbine decay state coefficient'
        ])
        data = data.values
        data = data[~np.isnan(data).any(axis=1)]
        x, y = data[:, :16], data[:, 16]
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size, random_state=42)
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=valid_size, random_state=42)
        return train_x, valid_x, test_x, train_y, valid_y, test_y

    def generate_dlnp_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32)
            self.y = tf.placeholder(tf.float32)
            self.w1 = tf.placeholder(tf.float32, shape=(self.input_dim, self.hidden_dim))
            self.b1 = tf.placeholder(tf.float32, shape=(1, self.hidden_dim))
            self.w2 = tf.placeholder(tf.float32, shape=(self.hidden_dim, 1))
            self.b2 = tf.placeholder(tf.float32)
            self.ln_gamma = tf.placeholder(tf.float32)
            self.ln_lambda = tf.placeholder(tf.float32)

            a0 = tf.constant(self.a0, dtype=tf.float32)
            b0 = tf.constant(self.b0, dtype=tf.float32)

            n_data = tf.cast(tf.shape(self.x)[0], tf.float32)
            n_params = (self.input_dim + 2) * self.hidden_dim + 1

            # neural network
            self.f = tf.matmul(tf.nn.relu(tf.matmul(self.x, self.w1) + self.b1), self.w2) + self.b2

            # posterior
            self.ln_likelihood = -n_data / 2 * (tf.math.log(2 * np.pi) - self.ln_gamma) - tf.exp(self.ln_gamma) / 2 * tf.reduce_sum(tf.pow(self.y - self.f, 2))
            ln_p_gamma = a0 * self.ln_gamma - b0 * tf.exp(self.ln_gamma)
            ln_p_lambda = a0 * self.ln_lambda - b0 * tf.exp(self.ln_lambda)
            ln_prior = -n_params / 2 * (tf.math.log(2 * np.pi) - self.ln_lambda) - tf.exp(self.ln_lambda) / 2 * (
                    tf.reduce_sum(tf.pow(self.w1, 2)) +
                    tf.reduce_sum(tf.pow(self.b1, 2)) +
                    tf.reduce_sum(tf.pow(self.w2, 2)) +
                    tf.pow(self.b2, 2)) + ln_p_gamma + ln_p_lambda
            ln_posterior = self.ln_likelihood + ln_prior

            # gradients
            self.result = tf.gradients(ln_posterior, [self.w1, self.b1, self.w2, self.b2, self.ln_gamma, self.ln_lambda])

    def dlnp_fun(self, points):
        grad = np.zeros(points.shape)
        for i in range(points.shape[0]):
            w1, b1, w2, b2, ln_gamma, ln_lambda = self.vector_to_params(points[i])
            with self.graph.as_default():
                with tf.Session() as sess:
                    result = sess.run(self.result, feed_dict={
                        self.x: (self.batch_x-self.train_x_mean)/self.train_x_std,
                        self.y: self.batch_y,
                        self.w1: w1,
                        self.b1: b1,
                        self.w2: w2,
                        self.b2: b2,
                        self.ln_gamma: ln_gamma,
                        self.ln_lambda: ln_lambda,
                    })
            dw1, db1, dw2, db2, dln_gamma, dln_lambda = result
            grad[i, :] = self.params_to_vector(dw1, db1, dw2, db2, dln_gamma, dln_lambda)
        return grad

    def get_rmse(self, x, points):
        losses = []
        for i in range(points.shape[0]):
            w1, b1, w2, b2, _, _ = self.vector_to_params(points[i])
            with self.graph.as_default():
                with tf.Session() as sess:
                    result = sess.run(self.f, feed_dict={
                        self.x: (x-self.train_x_mean)/self.train_x_std,
                        self.w1: w1,
                        self.b1: b1,
                        self.w2: w2,
                        self.b2: b2
                    })
            losses.append(np.sqrt(((result - self.test_y) ** 2).mean()))
        return min(losses)

    def get_ll(self, x, y, points):
        lls = []
        for i in range(points.shape[0]):
            w1, b1, w2, b2, ln_gamma, _ = self.vector_to_params(points[i])
            with self.graph.as_default():
                with tf.Session() as sess:
                    result = sess.run(self.ln_likelihood, feed_dict={
                        self.x: (x-self.train_x_mean)/self.train_x_std,
                        self.y: y,
                        self.w1: w1,
                        self.b1: b1,
                        self.w2: w2,
                        self.b2: b2,
                        self.ln_gamma: ln_gamma
                    })
            lls.append(result.mean())
        return max(lls)

    def params_to_vector(self, w1, b1, w2, b2, ln_gamma, ln_lambda):
        return np.concatenate([w1.flatten(), b1.flatten(), w2.flatten(), [b2], [ln_gamma], [ln_lambda]])

    def vector_to_params(self, v):
        w1 = v[: self.input_dim*self.hidden_dim].reshape((self.input_dim, self.hidden_dim))
        b1 = v[self.input_dim*self.hidden_dim: (self.input_dim+1)*self.hidden_dim].reshape((1, self.hidden_dim))
        w2 = v[(self.input_dim+1)*self.hidden_dim: (self.input_dim+2)*self.hidden_dim].reshape((self.hidden_dim, 1))
        b2 = v[-3]
        ln_gamma = v[-2]
        ln_lambda = v[-1]
        return w1, b1, w2, b2, ln_gamma, ln_lambda

    def train(self, target_metric, patience=3, init_h=False):
        svgd = SVGD(self.h, self.iter, self.dlnp_fun)
        count_inc = 0

        # early stopping
        prev_loss = None
        while True:
            self.logger.info('{0}th epoch......'.format(self.epoch))
            if init_h:
                self.h=0
            def unison_shuffled_copies(a, b):
                assert len(a) == len(b)
                p = np.random.permutation(len(a))
                return a[p], b[p]
            train_x, train_y = unison_shuffled_copies(self.train_x, self.train_y)
            for step, i in enumerate(range(0, train_x.shape[0], self.batch_size)):
                self.batch_x, self.batch_y = train_x[i:i+self.batch_size], train_y[i:i+self.batch_size]
                self.points = svgd.update(self.points, 1, self.step_size)
                if step%10 == 0:
                    self.logger.info('{0}th step - {1}'.format(step, self.report()))
            self.logger.info('{0}th epoch finish - {1}'.format(self.epoch, self.report()))

            self.h, self.iter = svgd.adagrad.h, svgd.adagrad.iter
            self.epoch += 1
            self.save()

            current_loss = self.get_rmse(self.valid_x, self.points) if target_metric=='RMSE' else self.get_ll(self.valid_x, self.valid_y, self.points)
            if prev_loss is not None and prev_loss < current_loss:
                count_inc += 1
                if count_inc > patience:
                    return
            else:
                count_inc = 0
                prev_loss = current_loss

    def report(self):
        return 'valid_RMSE: {0}, valid_LL: {1}, test_RMSE: {2}, test_LL: {3}'.format(
            self.get_rmse(self.valid_x, self.points),
            self.get_ll(self.valid_x, self.valid_y, self.points),
            self.get_rmse(self.test_x, self.points),
            self.get_ll(self.test_x, self.test_y, self.points)
        )

