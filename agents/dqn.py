import tensorflow as tf
import numpy as np
from collections import deque
import random
from agent_helpers.exploration_noise import Epsilon_greedy
import pathlib
import json

class DQN_agent:
  def __init__(self, env_helper, lr = 0.001, batch_size = 64,
               gamma = 0.99, tau = 0.01, memory_size = 50000, lr_decay = 100):

    self.sess = tf.Session()

    self.state_dim = env_helper.get_state_dim()
    self.action_dim = env_helper.get_action_dim()
    self.lr_inital = lr
    self.lr_decay_inital = lr_decay
    self.global_step = tf.Variable(0, trainable=False)
    self.lr = tf.train.exponential_decay(lr, self.global_step, lr_decay, 0.99, staircase=True)

    self.batch_size = batch_size
    self.memory = deque(maxlen = memory_size)

    self.gamma = gamma
    self.tau = tau
    self.input, self.output, self.weights = self._build_model('net')
    self.target_input, self.target_output, self.target_weights = self._build_model('target_net')

    self.y = tf.placeholder(tf.float32, [None, self.action_dim])
    self.loss = tf.reduce_mean(tf.square(tf.subtract(self.y, self.output), name='loss'))
    self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

    # target_weights = tau * weights + (1-tau) * target_weights
    update = lambda w, t_w: t_w.assign(tf.scalar_mul(self.tau, w) + tf.scalar_mul(1. - self.tau, t_w))

    self.update_target = list(map(update, self.weights, self.target_weights))

    self.sess.run(tf.global_variables_initializer())


  def get_state_dim(self):
    return self.state_dim


  def _build_model(self, scope):
    with tf.variable_scope(scope):
      state = tf.placeholder(tf.float32, [None, self.state_dim])
      h1 = tf.layers.dense(state, units=16, activation=tf.nn.relu)
      h2 = tf.layers.dense(h1, units=16, activation=tf.nn.relu)
      h3 = tf.layers.dense(h2, units=16, activation=tf.nn.relu)
      output = tf.layers.dense(h3, units=self.action_dim, activation=None)

      return state, output, tf.trainable_variables(scope)


  def target_predict(self, state):
    return self.sess.run(self.target_output, {self.target_input: state})


  def predict(self, state):
    return self.sess.run(self.output, {self.input: state})


  def act(self, state, epsilon):
    if np.random.rand() <= epsilon:
      return [np.random.randint(self.action_dim)]
    action_values = self.predict(state)
    return np.argmax(action_values, axis=1)


  def train(self, logger):
    loss, max_q = self._replay_memory()
    self._train_target()

    g_step = tf.train.global_step(self.sess, self.global_step)
    if g_step % 10 == 0:
      # Files were getting to big
      logger.add_agent_specifics(loss, max_q, g_step)


  def train_net(self, state, y):
    loss, _ = self.sess.run([self.loss, self.optimizer], {
        self.input: state,
        self.y:y})
    return loss


  def _train_target(self):
    self.sess.run(self.update_target)


  def add_memory(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))


  def _replay_memory(self):
    if len(self.memory) >= self.batch_size:
      minibatch = random.sample(self.memory, self.batch_size)

      to_np_array = lambda x: np.reshape(list(x), (len(minibatch),-1))
      state, action, reward, next_state, done = map(to_np_array, zip(*minibatch))

      target_next_q = self.target_predict(next_state)

      # Q(s,a) = r + γ * max Q(s',a)
      q = reward + (1 - done) * self.gamma * np.amax(target_next_q,
                                                     axis=1, keepdims=True)
      q = q.astype(np.float32)

      y = self.predict(state)

      mask = np.zeros((self.batch_size, self.action_dim))
      mask[np.arange(self.batch_size), action.flatten()] = 1

      # Replace the y values with q, but only for taken actions.
      # Non replaced output nodes/weights wont change during update
      np.place(y, mask, q.flatten())

      loss = self.train_net(state, y)

      np.amax(target_next_q, axis=1, keepdims=True)

      return loss, np.amax(target_next_q, axis=1, keepdims=False)


  def create_noise_generator(self, nbr_episodes):
    return Epsilon_greedy(nbr_episodes)


  def load(self, dir_name):
  # TODO: Make it nicer regarding that every parameter has a line to save/load.
    dir_name = "./saves/{}/".format(dir_name)

    with open('{}parameters.txt'.format(dir_name), 'r') as file:
      agent_params = json.load(file)

    self.lr = agent_params['learning_rate']
    self.batch_size = agent_params['batch_size']
    self.tau = agent_params['tau']
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
      agent_params['learning_rate'] = self.lr_inital
      agent_params['learning_rate_decay'] = self.lr_decay_inital
      agent_params['batch_size'] = self.batch_size
      agent_params['tau'] = self.tau
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
    return "DQN-agent"