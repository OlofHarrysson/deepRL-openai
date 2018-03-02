from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from collections import deque
import random
from datetime import datetime
import pathlib

class DQAgent:
  def __init__(self, state_size, action_size, learning_rate=0.001,
               memory_batch_size=32, gamma=0.95,
               epsilon=1.0, epsilon_min=0.01, epsilon_decay = 0.99):

    self.state_size = state_size
    self.action_size = action_size
    self.learning_rate = learning_rate

    self.memory_batch_size = memory_batch_size
    self.memory = deque(maxlen=2000)

    self.gamma = gamma
    self.epsilon = epsilon
    self.epsilon_min = epsilon_min
    self.epsilon_decay = epsilon_decay
    self.model = self._build_model()

  def get_state_size(self):
    return self.state_size

  def get_action_size(self):
    return self.action_size

  def _build_model(self):
    model = Sequential()
    model.add(Dense(units=24, activation='relu', input_dim=self.state_size))
    model.add(Dense(units=24, activation='relu'))
    model.add(Dense(units=self.action_size, activation='linear'))

    model.compile(optimizer=Adam(lr=self.learning_rate),
                  loss='mean_squared_error')
    return model


  def add_memory(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))


  def act(self, state, training=False):
    if np.random.rand() <= self.epsilon and training:
      return np.random.randint(self.action_size)
    action_values = self.model.predict(state)[0]
    return np.argmax(action_values)


  def replay_memory(self):
    if len(self.memory) > self.memory_batch_size:
      minibatch = random.sample(self.memory, self.memory_batch_size)
      for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
          target = (reward + self.gamma * 
                    np.amax(self.model.predict(next_state)[0]))
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

  def load(self, path):
    self.model.load_weights(path)

  def save(self, name):
    # TODO: Save parameters, score, model
    if isinstance(name, str): 
      dir_name = "./saves/last_run/"
    else:
      class_name = self.__class__.__name__
      time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
      dir_name = "./saves/{} {}/".format(class_name, time)

    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True)
    self.model.save_weights("{}test.h5".format(dir_name))