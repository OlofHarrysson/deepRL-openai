from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from collections import deque
import random
from datetime import datetime
import pathlib

class DQAgent:
  def __init__(self, state_dim, action_dim, learning_rate=0.001,
               batch_size=64, gamma=0.95,
               epsilon=1.0, epsilon_min=0.01, epsilon_decay = 0.99):

    self.state_dim = state_dim
    self.action_dim = action_dim
    self.learning_rate = learning_rate

    self.batch_size = batch_size
    self.memory = deque(maxlen=4000)

    self.gamma = gamma
    self.epsilon = epsilon
    self.epsilon_min = epsilon_min
    self.epsilon_decay = epsilon_decay
    self.model = self._build_model()

  def get_state_dim(self):
    return self.state_dim


  def _build_model(self):
    model = Sequential()
    model.add(Dense(units=24, activation='relu', input_dim=self.state_dim))
    model.add(Dense(units=24, activation='relu'))
    model.add(Dense(units=self.action_dim, activation='linear'))

    model.compile(optimizer=Adam(lr=self.learning_rate),
                  loss='mean_squared_error')
    return model


  def add_memory(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))


  def act(self, state, training=False):
    if np.random.rand() <= self.epsilon and training:
      return np.random.randint(self.action_dim)
    action_values = self.model.predict(state)[0]
    return np.argmax(action_values)


  def replay_memory(self):
    if len(self.memory) > self.batch_size:
      minibatch = random.sample(self.memory, self.batch_size)

      to_np_array = lambda x: np.reshape(list(x), (len(minibatch),-1))
      state, action, reward, next_state, done = map(to_np_array, zip(*minibatch))

      target = reward + (1 - done) * self.gamma * np.amax(self.model.predict(next_state), axis=1, keepdims=True)
      target = target.astype(np.float32)

      target_f = self.model.predict(state)


      mask = np.zeros((self.batch_size, self.action_dim))
      mask[np.arange(self.batch_size), action.flatten()] = 1

      np.place(target_f, mask, target.flatten())

      self.model.fit(state, target_f, epochs=1, verbose=0)


    if self.epsilon > self.epsilon_min: # TODO: Move to act?
      self.epsilon *= self.epsilon_decay

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