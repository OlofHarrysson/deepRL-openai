import tensorflow as tf
import numpy as np
from collections import deque
import random
from exploration_noise import Ornstein_uhlenbeck_noise
from datetime import datetime
import pathlib

class Actor():
  def __init__(self, sess, env_helper, batch_size, lr=0.0001, tau=0.001):
    self.sess = sess
    self.state_dim = env_helper.get_state_dim()
    self.action_dim = env_helper.get_action_dim()
    self.action_bound = env_helper.get_action_bound()
    self.env_helper = env_helper

    self.batch_size = batch_size
    self.tau = tau
    self.noise_gen = Ornstein_uhlenbeck_noise(mu = np.zeros(self.action_dim))

    self.input, self.output, self.weights = self._build_net('actor_net')
    self.target_input, self.target_output, self.target_weights = self._build_net('target_actor_net')

    with tf.variable_scope('actor'):
      with tf.variable_scope('gradients'):
        self.gradients = tf.placeholder(tf.float32, [None, self.action_dim], name='input')
        unnormalized_gradients = tf.gradients(self.output, self.weights, -self.gradients)
        
        gradients = list(map(lambda x: tf.divide(x, self.batch_size),
                             unnormalized_gradients))
      
      self.optimizer = tf.train.AdamOptimizer(lr).apply_gradients(zip(gradients, self.weights))

      # target_weights = tau * weights + (1-tau) * target_weights
      update = lambda w, t_w: t_w.assign(tf.scalar_mul(self.tau, w) + tf.scalar_mul(1. - self.tau, t_w))

      self.update_target = list(map(update, self.weights, self.target_weights))


  def _build_net(self, scope):
    with tf.variable_scope(scope):
      with tf.variable_scope('input'):
        state = tf.placeholder(tf.float32, [None, self.state_dim])

      l1 = tf.layers.dense(state, units=400, activation=tf.nn.relu)
      l2 = tf.layers.dense(l1, units=300, activation=tf.nn.relu)
      l3 = tf.layers.dense(l2, units=self.action_dim, activation=tf.nn.tanh)
      output = tf.multiply(self.action_bound, l3, name='output')

      return state, output, tf.trainable_variables(scope)


  def predict(self, state):
    return self.sess.run(self.output, {self.input: state})


  def target_predict(self, state):
    return self.sess.run(self.target_output, {self.target_input: state})


  def act(self, state):
    action = self.predict(state) + self.noise_gen()
    return np.clip(action, -self.action_bound, self.action_bound)


  def train(self, state, actor_gradients):
    self.sess.run(self.optimizer, {
        self.input: state,
        self.gradients: actor_gradients})


  def train_target(self):
    self.sess.run(self.update_target)


class Critic():
  def __init__(self, sess, env_helper, lr=0.001, tau=0.001):
    self.sess = sess
    self.state_dim = env_helper.get_state_dim()
    self.action_dim = env_helper.get_action_dim()

    self.tau = tau

    self.state_input, self.action_input, self.output, self.weights = self._build_net('critic_net')
    self.target_state_input, self.target_action_input, self.target_output, self.target_weights = self._build_net('target_critic_net')

    with tf.variable_scope('critic'):
      self.q_val = tf.placeholder(tf.float32, [None, 1])
      loss = tf.reduce_mean(tf.square(tf.subtract(self.q_val, self.output), name='loss'))
      self.optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

      self.actor_gradients = tf.gradients(self.output, self.action_input)

      # target_weights = tau * weights + (1-tau) * target_weights
      update = lambda w, t_w: t_w.assign(tf.scalar_mul(self.tau, w) + tf.scalar_mul(1. - self.tau, t_w))

      self.update_target = list(map(update, self.weights, self.target_weights))

  
  def _build_net(self, scope):
    with tf.variable_scope(scope):
      with tf.variable_scope('input'):
        state = tf.placeholder(tf.float32, [None, self.state_dim])
        action = tf.placeholder(tf.float32, [None, self.action_dim])

      with tf.variable_scope('state'):
        s_l1 = tf.layers.dense(state, units=400, activation=tf.nn.relu)
        s_l2 = tf.layers.dense(s_l1, units=300, activation=tf.nn.relu)

      with tf.variable_scope('action'):
        a_l1 = tf.layers.dense(action, units=300, activation=tf.nn.relu)
      
      with tf.variable_scope('merge'):
        merge = tf.concat([s_l2, a_l1], axis=-1)
        merge_h1 = tf.layers.dense(merge, units=300, activation=tf.nn.relu)
      
      output = tf.layers.dense(merge_h1, units=1, activation=None, name="output")

      return state, action, output, tf.trainable_variables(scope)


  def predict(self, state, action):
    return self.sess.run(self.output, {self.state_input: state, self.action_input: action})

  def target_predict(self, state, action):
    return self.sess.run(self.target_output, {
                  self.target_state_input: state,
                  self.target_action_input: action})


  def train(self, state, action, q_val):
    self.sess.run(self.optimizer, {
                  self.state_input: state,
                  self.action_input: action,
                  self.q_val: q_val})


  def train_target(self):
    self.sess.run(self.update_target)


  def calc_actor_gradients(self, state, action):
    return self.sess.run(self.actor_gradients, {
                         self.state_input: state,
                         self.action_input: action})


class DDPG_agent():
  def __init__(self, env_helper, gamma = 0.99):
    # TODO: Save, load model
    self.sess = tf.Session()
    self.state_dim = env_helper.get_state_dim() # TODO:
    self.memory = deque(maxlen=5000)
    self.batch_size = 64

    self.gamma = gamma

    self.actor = Actor(self.sess, env_helper, self.batch_size)
    self.critic = Critic(self.sess, env_helper)

    self.sess.run(tf.global_variables_initializer())


  def get_state_dim(self):
    return self.state_dim


  def act(self, state, training=False):
    return self.actor.act(state) if training else self.actor.predict(state)


  def train(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
    self._replay_memory()


  def _replay_memory(self):
    if len(self.memory) > self.batch_size:
      minibatch = random.sample(self.memory, self.batch_size)
      to_np_array = lambda x: np.reshape(list(x), (len(minibatch),-1))
      state, action, reward, next_state, done = map(to_np_array, zip(*minibatch))

      # Update critic
      next_a = self.actor.target_predict(next_state)
      next_value = self.critic.target_predict(next_state, next_a)
      q_val = reward + (1 - done) * self.gamma * next_value
      self.critic.train(state, action, q_val)

      # Update actor
      actor_gradients = self.critic.calc_actor_gradients(state, action)
      self.actor.train(state, actor_gradients[0])

      # Update target networks
      self.actor.train_target()
      self.critic.train_target()


  def load(self, path):
    self.model.load_weights(path)

  def save(self, name):
    # TODO: Save parameters, score, model, envname
    if isinstance(name, str): 
      dir_name = "./saves/last_run/"
    else:
      class_name = self.__class__.__name__
      time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
      dir_name = "./saves/{} {}/".format(class_name, time)

    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
    self.model.save_weights("{}model.h5".format(dir_name))