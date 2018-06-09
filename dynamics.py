import tensorflow as tf
import numpy as np
import tqdm
import math

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        
        self.env = env
        self.n_layers = n_layers
        self.size = size
        self.activation = activation
        self.output_activation = output_activation
        self.batch_size = batch_size
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.sess = sess
        self.mean_obs, self.std_obs, self.mean_deltas, self.std_deltas, self.mean_actions, self.std_actions = normalization
        
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        self.st_at_placeholder = tf.placeholder(shape = [None, self.state_dim + self.action_dim], 
                                                name = "input_st_at", 
                                                dtype = tf.float32)
        self.deltas_placeholder = tf.placeholder(shape = [None, self.state_dim], name = "input_delta", dtype = tf.float32)
        
        self.deltas_predict = build_mlp(self.st_at_placeholder, 
                                 self.state_dim, 
                                 scope = "MPC", 
                                 n_layers = self.n_layers, 
                                 size = self.size,
                                 activation = self.activation,
                                 output_activation = self.output_activation)
        
        self.loss = tf.reduce_mean(tf.square(self.deltas_predict - self.deltas_placeholder))
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
        
        """ Be careful about normalization """

    def fit(self, data):
        """
        take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """
        
        obs = np.concatenate([item['observations'] for item in data])
        next_obs = np.concatenate([item['next_observations'] for item in data])
        acts = np.concatenate([item['actions'] for item in data])
        
        obs_norm = (obs - self.mean_obs) / (self.std_obs + 1e-7)
        deltas_obs_norm = ((next_obs - obs) - self.mean_deltas) / (self.std_deltas + 1e-7)
        acts_norm = (acts - self.mean_actions) / (self.std_actions + 1e-7)
        
        obs_act_norm = np.concatenate((obs_norm, acts_norm), axis = 1)
        
        train_indice = np.arange(obs.shape[0])
        
        for i in range(self.iterations): #tqdm.tqdm(range(self.iterations)):
            np.random.shuffle(train_indice)

            for j in range((obs.shape[0] // self.batch_size) + 1): #tqdm.tqdm(range(obs.shape[0] // self.batch_size)):
                start_indice = j*self.batch_size
                indice_shuffled = train_indice[start_indice:start_indice + self.batch_size]
                
                input_batch = obs_act_norm[indice_shuffled, :]
                label_batch = deltas_obs_norm[indice_shuffled, :]
                
                self.sess.run([self.train_op], feed_dict = {self.st_at_placeholder: input_batch, self.deltas_placeholder: label_batch})
            

    def predict(self, states, actions):
        """ take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """

        
        obs_norm = (states - self.mean_obs) / (self.std_obs + 1e-7)
        act_norm = (actions - self.mean_actions) / (self.std_actions + 1e-7)
        obs_act_norm = np.concatenate((obs_norm, act_norm), axis = 1 )
        
        deltas = self.sess.run(self.deltas_predict, feed_dict = {self.st_at_placeholder: obs_act_norm})
        
        return deltas * self.std_deltas + self.mean_deltas + states
