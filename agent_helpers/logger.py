import tensorflow as tf
import numpy as np
from datetime import datetime


class Logger():
  def __init__(self, agent_name, run_id):
    self.sess = tf.Session()

    time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    self.max_writer = tf.summary.FileWriter("./runs/{}-{}-{}-max".format(run_id, agent_name, time))
    self.min_writer = tf.summary.FileWriter("./runs/{}-{}-{}-min".format(run_id, agent_name, time))
    self.avg_writer = tf.summary.FileWriter("./runs/{}-{}-{}-avg".format(run_id, agent_name, time))
    self.val_writer = tf.summary.FileWriter("./runs/{}-{}-{}-value".format(run_id, agent_name, time))

    self.input = tf.placeholder(tf.float32)

    # Scalar summaries
    self.add_loss = tf.summary.scalar("Loss", self.input)
    self.add_q = tf.summary.scalar("Q-val", self.input)
    self.add_score = tf.summary.scalar("Score", self.input)
    self.add_test_score = tf.summary.scalar("Test score", self.input)
    self.add_noise = tf.summary.scalar("Noise", self.input)
    self.add_gradients = tf.summary.scalar("Gradients", self.input) # TODO, get this

    # Histogram summaries
    # self.add_action = tf.summary.histogram("Action", self.input) # TODO?
    
    self.sess.run(tf.global_variables_initializer())


  def _add_scalars(self, operation, values, writers, episode_number):
    ''' Run the operation for every v in values.
    Write the summaries generated with corresponding writer.
    Values and writers needs to be lists with the same length and in the same order '''

    create_summaries = lambda val: self.sess.run(operation, {self.input: val})
    loss_summaries = map(create_summaries, values)

    for summary, writer in zip(loss_summaries, writers):
      writer.add_summary(summary, episode_number)


  def _add_scalar(self, operation, value, writer, episode_number):
    summary = self.sess.run(operation, {self.input: value})
    writer.add_summary(summary, episode_number)


  def _add_histogram(self, operation, value, writer, episode_number): # TODO?
    summary = self.sess.run(operation, {self.input: value})
    writer.add_summary(summary, episode_number)


  def add(self, episode_number, score, noise, max_qs, losses):
    mma_loss = [max(losses), min(losses), sum(losses) / len(losses)]
    writers = [self.max_writer, self.min_writer, self.avg_writer]
    self._add_scalars(self.add_loss, mma_loss, writers, episode_number)

    mma_q = [max(max_qs), min(max_qs), sum(max_qs) / len(max_qs)]
    self._add_scalars(self.add_q, mma_q, writers, episode_number)

    self._add_scalar(self.add_score, score, self.val_writer, episode_number)
    self._add_scalar(self.add_noise, noise, self.val_writer, episode_number)


  def add_test(self, episode_number, score):
    self._add_scalar(self.add_test_score, score, self.val_writer, episode_number)