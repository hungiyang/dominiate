from dominion import *
from rl_agent import *
import numpy as np
import tensorflow as tf

class DQLagent():
    def __init__(self, epochs=10, length=129):
        self.epochs=epochs
        self.target_iterations=5
        self.predict_iterations=200
        self.length = length
        # number of samples drawn every time
        self.mtrain = 1000
        self.gamma = 0.99
        self.epsilon = 0.1
        self.create_model()
        self.data = []
        self.replaybuffer = 1000000

    def create_model(self):
        model = tf.keras.models.Sequential([
          tf.keras.layers.Dense(self.length, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(self.length, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(30, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])
        self.model_predict = model
        model = tf.keras.models.Sequential([
          tf.keras.layers.Dense(self.length, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(self.length, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(30, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])
        self.model_target = model
        return

    def add_data(self, data):
        if self.data == []:
            self.data = data
        else:
            self.data = tuple([np.concatenate([d_this, data[i]]) for i,d_this in enumerate(self.data)])
        # truncate data down to replay buffer size
        if self.data[0].shape[0] > self.replaybuffer:
            print('truncate {:d} samples'.format(self.data[0].shape[0] - self.replaybuffer))
            self.data = tuple([d_this[-self.replaybuffer:,:] for d_this in self.data])
        return
        

    def compute_target(self, data):
        """
        compute_target use the target network to predict the Q value
        n is the next state
        with a Q(s,a) model
        compute r + gamma*max_a' Q(s',a')
        It outputs the target that the deep neural network wants to fit for.
        shape of action is 18
        """
        s,a,r,n = data
        allact = np.identity(a.shape[1])
        # array of (18, number of samples, state action pair)
        tmp = np.array([self.model_target.predict(np.concatenate([n, np.tile(act, [n.shape[0],1])],axis=1))\
                        for act in allact])
        # maximum q(s',a')
        qn = np.max(tmp, axis=0)
        # output expected Q(s,a) from the target network evaluation
        return self.gamma*qn + r

    def fit_target(self, data):
        """
        fit_target_network
        computes the target network prediction and fit for it with prediction network
        """
        # state, action, reward, next state
        s,a,r,n = data
        sa = np.concatenate([s,a],axis=1)
        target = self.compute_target(data)
        self.model_predict.fit(sa, target, epochs=self.epochs, verbose = 0)
        return

    def draw_sample(self):
        """
        draw random samples from the full dataset generated
        """
        m = self.data[1].shape[0]
        select = np.random.choice(m,self.mtrain,replace=False)
        return tuple([d[select,:] for d in self.data])

    def do_target_iteration(self):
        for j in range(self.target_iterations):
            print('start target model iteration {:d}'.format(j))
            # set the weights of the target model to predict model
            self.model_target.set_weights(self.model_predict.get_weights()) 
            for i in range(self.predict_iterations):
                self.fit_target(self.draw_sample())

    def save_model(self, fname='test'):
        self.model_predict.save_weights(fname + '_predict.h5')
        self.model_target.save_weights(fname + '_target.h5')
        return

    def load_model(self, fname='test'):
        self.model_predict.load_weights(fname + '_predict.h5')
        self.model_target.load_weights(fname + '_target.h5')
        return

    def generate_data(self, ngames=50, fname=''):
        """
        generate a new batch of data with the latest prediction model self.model_predict
        rl vs. random bot
        """
        vf = lambda x: self.model_predict.predict(x)
        p1 = RLPlayer(vf)
        p1.epsilon = self.epsilon
        p1.record_history = 1
        p2 = RandomPlayer()
        p2.record_history = 0
        d_this = record_game(ngames, [p1,p2],fname)
        self.add_data(d_this)
        return d_this


    def generate_data_smithy(self, ngames=50, fname=''):
        """
        generate a new batch of data with the latest prediction model self.model_predict
        rl vs. smithy bot
        """
        vf = lambda x: self.model_predict.predict(x)
        p1 = RLPlayer(vf)
        p1.epsilon = self.epsilon
        p1.record_history = 1
        p2 = SmithyBot()
        p2.record_history = 0
        d_this = record_game(ngames, [p1,p2],fname)
        self.add_data(d_this)
        return d_this


    def generate_data_rl(self, ngames=50, fname=''):
        """
        generate a new batch of data with the latest prediction model self.model_predict
        rl vs. smithy bot
        """
        vf = lambda x: self.model_predict.predict(x)
        p1 = RLPlayer(vf)
        p1.epsilon = self.epsilon
        p1.record_history = 1
        p2 = RLPlayer(vf)
        p2.epsilon = self.epsilon
        p2.record_history = 1
        d_this = record_game(ngames, [p1,p2],fname)
        self.add_data(d_this)
        return d_this



