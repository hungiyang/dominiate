from dominion import *
from rl_agent import *
import numpy as np
import tensorflow as tf

class DQLagent():
    def __init__(self, epochs=10):
        self.epochs=epochs
        self.target_iterations=4
        self.predict_iterations=4
        # number of samples drawn every time
        self.mtrain = 100
        self.gamma = 0.99
        self.create_model()

    def create_model(self):
        model = tf.keras.models.Sequential([
          tf.keras.layers.Dense(129, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(129, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(30, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])
        self.model_predict = model
        self.model_target = model
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

    def draw_sample(self,data):
        """
        draw random samples from the full dataset generated
        """
        m = data[1].shape[0]
        select = np.random.choice(m,self.mtrain,replace=False)
        return tuple([d[select,:] for d in data])

    def do_target_iteration(self,data):
        for j in range(self.target_iterations):
            print('start target model iteration {:d}'.format(j))
            self.model_target = self.model_predict
            for i in range(self.predict_iterations):
                model_predict = self.fit_target(self.draw_sample(data))

    def save_model(self, fname='test'):
        self.model_predict.save(fname.join('_predict.h5'))
        self.model_target.save(fname.join('_target.h5'))
        return

    def generate_data(self, ngames=50):
        """
        generate a new batch of data with the latest prediction model self.model_predict
        """
        vf = lambda x: self.model_predict.predict(x)
        p1 = RLPlayer(vf)
        p1.record_history = 1
        p2 = RandomPlayer()
        p2.record_history = 0
        return record_game(ngames, [p1,p2])






