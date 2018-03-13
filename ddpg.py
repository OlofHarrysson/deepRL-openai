import tensorflow as tf
import numpy as np
from collections import deque
import random
from noise import OrnsteinUhlenbeckActionNoise

class Actor():
  def __init__(self, sess, state_dim, action_dim, action_bound, lr=0.0001):
    self.sess = sess
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.action_bound = action_bound
    self.tau = 0.001 # TODO
    self.noise_gen = OrnsteinUhlenbeckActionNoise(mu = np.zeros(action_dim)) # TODO

    self.input, self.output, self.weights = self._build_net('actor_net')
    self.target_input, self.target_output, self.target_weights = self._build_net('target_actor_net')

    with tf.variable_scope('actor'):
      self.actor_gradients = tf.placeholder(tf.float32, [None, action_dim])
      unnormalized_actor_gradients = tf.gradients(self.output, self.weights,
                                                       -self.actor_gradients)

      grads = list(map(lambda x: tf.divide(x, 64) if x != None else x,
                                      unnormalized_actor_gradients)) # TODO: batchsize and remove check for None?
      
      self.optimizer = tf.train.AdamOptimizer(lr).apply_gradients(zip(grads, self.weights))

      self.update_target = [self.target_weights[i].assign(tf.multiply(self.weights[i], self.tau) + tf.multiply(self.target_weights[i], 1. - self.tau)) for i in range(len(self.target_weights))]


  def predict(self, state):
    return self.sess.run(self.output, {self.input: state})


  def target_predict(self, state):
    return self.sess.run(self.target_output, {self.target_input: state})


  def act(self, state):
    return self.predict(state) + self.noise_gen()


  def train(self, state, actor_gradients):
    self.sess.run(self.optimizer, {
        self.input: state,
        self.actor_gradients: actor_gradients[0]}) # TODO


  def train_target(self):
    self.sess.run(self.update_target)


  def _build_net(self, scope):
    with tf.variable_scope(scope):
      with tf.variable_scope('input'):
        state = tf.placeholder(tf.float32, [None, self.state_dim])

      l1 = tf.layers.dense(state, units=400, activation=tf.nn.relu)
      l2 = tf.layers.dense(l1, units=300, activation=tf.nn.relu)
      l3 = tf.layers.dense(l2, units=self.action_dim, activation=tf.nn.tanh)
      output = tf.multiply(self.action_bound, l3, name='output')

      return state, output, tf.trainable_variables(scope)


class Critic():
  def __init__(self, sess, state_dim, action_dim, lr=0.001):
    self.sess = sess
    self.state_dim = state_dim
    self.action_dim = action_dim

    self.tau = 0.001 # TODO

    self.state_input, self.action_input, self.output, self.weights = self._build_net('critic_net')
    self.target_state_input, self.target_action_input, self.target_output, self.target_weights = self._build_net('target_critic_net')

    with tf.variable_scope('critic'):
      self.q_val = tf.placeholder(tf.float32, [None, 1])
      loss = tf.reduce_mean(tf.square(tf.subtract(self.q_val, self.output), name='loss'))
      self.optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

      self.actor_gradients = tf.gradients(self.output, self.action_input)

      self.update_target = [self.target_weights[i].assign(tf.multiply(self.weights[i], self.tau) + tf.multiply(self.target_weights[i], 1. - self.tau)) for i in range(len(self.target_weights))]

  
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


  def actor_gradients2(self, state, action):
    return self.sess.run(self.actor_gradients, {
                         self.state_input: state,
                         self.action_input: action})


  def train_target(self):
    self.sess.run(self.update_target)


class DDPG_agent():
  def __init__(self, state_dim, action_dim):
    self.sess = tf.Session()
    self.state_dim = state_dim
    self.action_dim = action_dim

    self.action_bound = 2.0 # TODO
    self.gamma = 0.9 # TODO

    self.actor = Actor(self.sess, state_dim, action_dim, self.action_bound)
    self.critic = Critic(self.sess, state_dim, action_dim)

    self.memory = deque(maxlen=5000)
    self.batch_size = 64

    self.sess.run(tf.global_variables_initializer())


  def get_state_dim(self):
    return self.state_dim


  def add_memory(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))


  def act(self, state, training=False):
    return self.actor.act(state) if training else self.actor.predict(state)


  def replay_memory(self):
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
      actor_gradients = self.critic.actor_gradients2(state, action) # TODO
      self.actor.train(state, actor_gradients)

      # Update target networks
      self.actor.train_target()
      self.critic.train_target()