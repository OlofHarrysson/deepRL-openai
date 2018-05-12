import tensorflow as tf
import numpy as np
from collections import deque
import random
from agent_helpers.exploration_noise import Epsilon_greedy

class DQN_agent:
  def __init__(self, env_helper, lr=0.001, batch_size=64,
               gamma=0.99, tau=0.01):

    self.sess = tf.Session()

    self.state_dim = env_helper.get_state_dim()
    self.action_dim = env_helper.get_action_dim()
    self.lr = lr

    self.batch_size = batch_size
    self.memory = deque(maxlen=50000)

    self.gamma = gamma
    self.tau = tau
    self.input, self.output, self.weights = self._build_model('net')
    self.target_input, self.target_output, self.target_weights = self._build_model('target_net')

    self.y = tf.placeholder(tf.float32, [None, self.action_dim])
    self.loss = tf.reduce_mean(tf.square(tf.subtract(self.y, self.output), name='loss'))
    self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss)

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

  def train_net(self, state, y):
    loss, _ = self.sess.run([self.loss, self.optimizer], {
        self.input: state,
        self.y:y})
    return loss


  def act(self, state, epsilon, training=False):
    if np.random.rand() <= epsilon and training:
      return [np.random.randint(self.action_dim)]
    action_values = self.predict(state)
    return np.argmax(action_values, axis=1)


  def train(self):
    loss, max_q = self.replay_memory()
    self._train_target()
    return loss, max_q


  def add_memory(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))


  def replay_memory(self):
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

      return loss, np.amax(target_next_q, axis=1, keepdims=True)


  def _train_target(self):
    self.sess.run(self.update_target)


  def create_noise_generator(self, nbr_episodes):
    return Epsilon_greedy(nbr_episodes)


  def load(self, path):
    self.model.load_weights(path)

  def save(self, name):
    # TODO: Save parameters, weights, env, nbr_episodes, policy used, target net
    # TODO: Return all data and create dir+file in main?
    pass
    # if isinstance(name, str): 
    #   dir_name = "../saves/last_run/"
    # else:
    #   class_name = self.__class__.__name__
    #   time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    #   dir_name = "../saves/{} {}/".format(class_name, time)

    # pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
    # self.model.save_weights("{}model.h5".format(dir_name))