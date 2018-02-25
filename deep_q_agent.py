from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from collections import deque
import random

class DQAgent:
  def __init__(self, state_size, action_size):
    self.action_size = action_size
    self.memory = deque(maxlen=2000)
    self.gamma = 0.95
    self.epsilon = 1.0
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.997
    self.learning_rate = 0.001
    self.model = self._build_model(state_size, action_size)
    


  def _build_model(self, state_size, action_size):
    model = Sequential()
    model.add(Dense(units=24, activation='relu', input_dim=state_size))
    model.add(Dense(units=24, activation='relu'))
    model.add(Dense(units=action_size, activation='linear'))

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


  def replay_memory(self, batch_size):
    minibatch = random.sample(self.memory, batch_size)
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

  def load(self, name):
    self.model.load_weights(name)

  def save(self, name):
    self.model.save_weights(name)