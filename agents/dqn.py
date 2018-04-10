from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import regularizers
import numpy as np
from collections import deque
import random
from datetime import datetime
import pathlib
from agent_helpers.exploration_noise import Epsilon_greedy

class DQN_agent:
  def __init__(self, env_helper, learning_rate=0.001, batch_size=64,
               gamma=0.99, tau=0.01):

    self.state_dim = env_helper.get_state_dim()
    self.action_dim = env_helper.get_action_dim()
    self.learning_rate = learning_rate

    self.batch_size = batch_size
    self.memory = deque(maxlen=50000)

    self.gamma = gamma
    self.tau = tau
    self.model = self._build_model()
    self.target_model = self._build_model()


  def get_state_dim(self):
    return self.state_dim


  def _build_model(self):
    model = Sequential()
    model.add(Dense(units=16, activation='relu', input_dim=self.state_dim,
              kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(units=16, activation='relu',
              kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(units=16, activation='relu',
              kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(units=self.action_dim, activation='linear',
              kernel_regularizer=regularizers.l2(0.01)))

    model.compile(optimizer=Adam(lr=self.learning_rate),
                  loss='mean_squared_error')
    return model


  def act(self, state, epsilon, training=False):
    if np.random.rand() <= epsilon and training:
      return [np.random.randint(self.action_dim)]
    action_values = self.model.predict(state)
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

      target_next_q = self.target_model.predict_on_batch(next_state)

      # Q(s,a) = r + γ * max Q(s',a)
      q = reward + (1 - done) * self.gamma * np.amax(target_next_q,
                                                     axis=1, keepdims=True)
      q = q.astype(np.float32) # Depends on your keras options

      y = self.model.predict_on_batch(state)

      mask = np.zeros((self.batch_size, self.action_dim))
      mask[np.arange(self.batch_size), action.flatten()] = 1

      # Replace the y values with q, but only for taken actions.
      # Non replaced output nodes/weights wont change during update
      np.place(y, mask, q.flatten())

      loss = self.model.train_on_batch(state, y)

      return loss, np.amax(target_next_q, axis=1, keepdims=True)


  def _train_target(self):
    model_w = self.model.get_weights()
    target_w = self.target_model.get_weights()
    tau = self.tau

    update = lambda w, t_w: tau * w + (1-tau) * t_w
    new_w = list(map(update, model_w, target_w))
    self.target_model.set_weights(new_w)


  def create_noise_generator(self, nbr_episodes):
    return Epsilon_greedy(nbr_episodes)


  def load(self, path):
    self.model.load_weights(path)

  def save(self, name):
    # TODO: Save parameters, weights, env, nbr_episodes, policy used, target net
    # TODO: Return all data and create dir+file in main?
    if isinstance(name, str): 
      dir_name = "../saves/last_run/"
    else:
      class_name = self.__class__.__name__
      time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
      dir_name = "../saves/{} {}/".format(class_name, time)

    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
    self.model.save_weights("{}model.h5".format(dir_name))