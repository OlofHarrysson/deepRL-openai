import tensorflow as tf
import numpy as np
from datetime import datetime

class Logger():
  def __init__(self):
    self.sess = tf.Session()

    time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    self.writer_1 = tf.summary.FileWriter("./runs/max-{}".format(time))
    self.writer_2 = tf.summary.FileWriter("./runs/min-{}".format(time))

    self.q = tf.placeholder(tf.float32)
    self.loss = tf.placeholder(tf.float32)

    # TODO: hej is name of graph. filewriter argument is line name
    # TODO, can scalar get a name input from sess.run?
    tf.summary.scalar("Q-val", self.q)
    tf.summary.scalar("Loss", self.loss)

    self.write_op = tf.summary.merge_all()

    self.sess.run(tf.global_variables_initializer())



  def add2(self, data, i):

    maxq = data['max_q']
    minq = data['min_q']

    maxl = data['max_loss']
    minl = data['min_loss']

    # for writer 1,2
    summary = self.sess.run(self.write_op, {self.q: maxq, self.loss: maxl})
    self.writer_1.add_summary(summary, i)
    self.writer_1.flush()

    summary = self.sess.run(self.write_op, {self.q: minq, self.loss: minl})
    self.writer_2.add_summary(summary, i)
    self.writer_2.flush()

    # # for writer 1,2
    # summary = self.sess.run(self.write_op, {self.loss: maxl})
    # self.writer_1.add_summary(summary, i)
    # self.writer_1.flush()

    # summary = self.sess.run(self.write_op, {self.loss: minl})
    # self.writer_2.add_summary(summary, i)
    # self.writer_2.flush()


  def add(self, data, episode_number):

    for name, value in data.items():
      self.writer.add_scalar(name, value, episode_number)
