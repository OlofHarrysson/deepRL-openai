import tensorflow as tf
import numpy as np
from collections import deque
import random
from agent_helpers.exploration_noise import Ornstein_uhlenbeck_noise
import pathlib
import json

class Actor():
  def __init__(self, sess, env_helper, batch_size, lr = 0.0001, tau = 0.001,
               lr_decay = 100):
    self.sess = sess
    self.state_dim = env_helper.get_state_dim()
    self.action_dim = env_helper.get_action_dim()
    self.action_bound = env_helper.get_action_bound()
    self.env_helper = env_helper

    self.global_step = tf.Variable(0, trainable=False)

    self.lr_initial = lr
    self.lr_decay_initial = lr_decay

    self.lr = tf.train.exponential_decay(lr, self.global_step, lr_decay, 0.99, staircase=True)

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
      
      self.optimizer = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(gradients, self.weights), global_step=self.global_step)

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
  def __init__(self, sess, env_helper, lr = 0.001, tau = 0.001, lr_decay = 100):
    self.sess = sess
    self.state_dim = env_helper.get_state_dim()
    self.action_dim = env_helper.get_action_dim()

    self.global_step = tf.Variable(0, trainable=False)

    self.lr_initial = lr
    self.lr_decay_initial = lr_decay

    self.lr = tf.train.exponential_decay(lr, self.global_step, lr_decay, 0.99, staircase=True)

    self.tau = tau

    self.state_input, self.action_input, self.output, self.weights = self._build_net('critic_net')
    self.target_state_input, self.target_action_input, self.target_output, self.target_weights = self._build_net('target_critic_net')

    with tf.variable_scope('critic'):
      self.q_val = tf.placeholder(tf.float32, [None, 1])
      self.loss = tf.reduce_mean(tf.square(tf.subtract(self.q_val, self.output), name='loss'))
      self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

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
    loss, _ = self.sess.run([self.loss, self.optimizer], {
                  self.state_input: state,
                  self.action_input: action,
                  self.q_val: q_val})
    return loss


  def train_target(self):
    self.sess.run(self.update_target)


  def calc_actor_gradients(self, state, action):
    return self.sess.run(self.actor_gradients, {
                         self.state_input: state,
                         self.action_input: action})


class DDPG_agent():
  def __init__(self, env_helper, actor_parameters = {}, critic_parameters = {},
               gamma = 0.99):
    # TODO: Save, load model
    self.sess = tf.Session()
    self.global_step = tf.Variable(0, trainable=False)

    self.state_dim = env_helper.get_state_dim()
    self.memory = deque(maxlen=5000)
    self.batch_size = 64

    self.gamma = gamma

    self.actor = Actor(self.sess, env_helper, self.batch_size, **actor_parameters)
    self.critic = Critic(self.sess, env_helper, **critic_parameters)

    self.sess.run(tf.global_variables_initializer())


  def get_state_dim(self):
    return self.state_dim


  def act(self, state, training=False):
    return self.actor.act(state) if training else self.actor.predict(state)


  def add_memory(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))


  def train(self, logger):
    critic_loss, critic_q_values, actor_gradients = self._replay_memory()

    # Update target networks
    self.actor.train_target()
    self.critic.train_target()

    actor_g_step = tf.train.global_step(self.sess, self.actor.global_step)
    critic_g_step = tf.train.global_step(self.sess, self.critic.global_step)

    if actor_g_step % 10 == 0:
      # Files were getting to big
      logger.add_agent_specifics(critic_loss, critic_q_values, actor_gradients, actor_g_step, critic_g_step)


  def _replay_memory(self):
    if len(self.memory) > self.batch_size:
      minibatch = random.sample(self.memory, self.batch_size)
      to_np_array = lambda x: np.reshape(list(x), (len(minibatch),-1))
      state, action, reward, next_state, done = map(to_np_array, zip(*minibatch))

      # Update critic
      next_a = self.actor.target_predict(next_state)
      next_value = self.critic.target_predict(next_state, next_a)
      q_val = reward + (1 - done) * self.gamma * next_value
      critic_loss = self.critic.train(state, action, q_val)

      # Update actor
      actor_gradients = self.critic.calc_actor_gradients(state, action)
      self.actor.train(state, actor_gradients[0])

      return critic_loss, np.reshape(q_val, -1), np.reshape(actor_gradients, -1)



  def create_noise_generator(self, nbr_episodes):
    return Ornstein_uhlenbeck_noise(mu = np.zeros(self.actor.action_dim)) # TODO: Other mean for envs with actions not centered around 0


  def load(self, dir_name):
    dir_name = "./saves/{}/".format(dir_name)

    with open('{}parameters.txt'.format(dir_name), 'r') as file:
      agent_params = json.load(file)

    # TODO: Other parameters as well?
    self.batch_size = agent_params['batch_size']
    self.gamma = agent_params['gamma']
    self.memory = deque(maxlen=agent_params['memory_size'])

    saver = tf.train.Saver()
    saver.restore(self.sess, "{}model.ckpt".format(dir_name))
    print("Model succesfully loaded")


  def save(self, name, n_train_episodes, episode_length, env_type, score, run_id):
    # TODO: Move parts of save / load to other class that can handle all agents
    if isinstance(name, str): 
      dir_name = "./saves/last_run/"
    else:
      dir_name = "./saves/{}-{}/".format(run_id, str(self))

    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
    with open('{}parameters.txt'.format(dir_name), 'w') as file:
      agent_params = {}
      agent_params['actor_learning_rate'] = self.actor.lr_initial
      agent_params['actor_learning_rate_decay'] = self.actor.lr_decay_initial
      agent_params['critic_learning_rate'] = self.critic.lr_initial
      agent_params['critic_learning_rate_decay'] = self.critic.lr_decay_initial
      agent_params['batch_size'] = self.batch_size
      agent_params['actor_tau'] = self.actor.tau
      agent_params['critic_tau'] = self.critic.tau
      agent_params['gamma'] = self.gamma
      agent_params['memory_size'] = self.memory.maxlen
      file.write(json.dumps(agent_params))

    with open('{}parameter_info.txt'.format(dir_name), 'w') as file:
      data = {}
      data['n_train_episodes'] = n_train_episodes
      data['episode_length'] = episode_length
      data['env_type'] = env_type
      data['score'] = score
      file.write(json.dumps(data))

    saver = tf.train.Saver()
    with self.sess as sess:
      save_path = saver.save(sess, "{}model.ckpt".format(dir_name))
      print("Model saved in path: %s" % save_path)


  def __str__(self):
    return "DDPG-agent"