import tensorflow as tf
from datetime import datetime

def logger_creator(agent_type, agent_name, run_id):
  picker = {
    'ddpg': DDPG_logger,
    'dqn': DQN_logger
  }

  logger = picker.get(agent_type)

  if not logger:
    available_agents = list(picker.keys())
    raise Exception("{} is not a supported agent. Choose between {}".format(agent_type, ", ".join(available_agents)))


  return logger(agent_name, run_id)

# ~~~~~~~~~~~~~~  Superclass ~~~~~~~~~~~~~~
class Logger():
  def __init__(self, agent_name, run_id):
    self.sess = tf.Session()

    time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    self.max_writer = tf.summary.FileWriter("./runs/{}-{}-{}-max".format(run_id, agent_name, time))
    self.min_writer = tf.summary.FileWriter("./runs/{}-{}-{}-min".format(run_id, agent_name, time))
    self.avg_writer = tf.summary.FileWriter("./runs/{}-{}-{}-avg".format(run_id, agent_name, time))
    self.val_writer = tf.summary.FileWriter("./runs/{}-{}-{}-value".format(run_id, agent_name, time))

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


  def add_test(self, episode_number, score):
    self._add_scalar(self.add_test_score, score, self.val_writer, episode_number)


class DDPG_logger(Logger):
  def __init__(self, agent_name, run_id):
    super().__init__(agent_name, run_id)


    self.input = tf.placeholder(tf.float32)
    # Scalar summaries
    self.add_actor_loss = tf.summary.scalar("Actor Loss", self.input)
    self.add_actor_q = tf.summary.scalar("Actor Q-val", self.input)
    self.add_critic_loss = tf.summary.scalar("Critic Loss", self.input)
    self.add_critic_q = tf.summary.scalar("Critic Q-val", self.input)
    self.add_score = tf.summary.scalar("Score", self.input)
    self.add_test_score = tf.summary.scalar("Test score", self.input)
    self.add_noise = tf.summary.scalar("Noise", self.input)



  def add_agent_specifics(self, critic_loss, critic_q_values,
                          actor_gradients, actor_g_step, critic_g_step):
    self._add_scalar(self.add_critic_loss, critic_loss, self.val_writer, critic_g_step)

    writers = [self.max_writer, self.min_writer, self.avg_writer]
    mma_q = [max(critic_q_values), min(critic_q_values), sum(critic_q_values) / len(critic_q_values)]
    self._add_scalars(self.add_critic_q, mma_q, writers, critic_g_step)


    mma_q = [max(actor_gradients), min(actor_gradients), sum(actor_gradients) / len(actor_gradients)]
    self._add_scalars(self.add_actor_q, mma_q, writers, actor_g_step)


  def add(self, episode_number, score, noise):
    self._add_scalar(self.add_score, score, self.val_writer, episode_number)
    self._add_scalar(self.add_noise, noise[0], self.val_writer, episode_number)


class DQN_logger(Logger):
  def __init__(self, agent_name, run_id):
    super().__init__(agent_name, run_id)

    self.input = tf.placeholder(tf.float32)
    # Scalar summaries
    self.add_loss = tf.summary.scalar("Loss", self.input)
    self.add_q = tf.summary.scalar("Q-val", self.input)
    self.add_score = tf.summary.scalar("Score", self.input)
    self.add_test_score = tf.summary.scalar("Test score", self.input)
    self.add_noise = tf.summary.scalar("Noise", self.input)


  def add_agent_specifics(self, loss, max_qs, global_step):
    self._add_scalar(self.add_loss, loss, self.val_writer, global_step)

    writers = [self.max_writer, self.min_writer, self.avg_writer]
    mma_q = [max(max_qs), min(max_qs), sum(max_qs) / len(max_qs)]
    self._add_scalars(self.add_q, mma_q, writers, global_step)


  def add(self, episode_number, score, noise):
    self._add_scalar(self.add_score, score, self.val_writer, episode_number)
    self._add_scalar(self.add_noise, noise, self.val_writer, episode_number)



# class Logger1():
#   def __init__(self, agent_name, run_id):
#     self.sess = tf.Session()

#     time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
#     self.max_writer = tf.summary.FileWriter("./runs/{}-{}-{}-max".format(run_id, agent_name, time))
#     self.min_writer = tf.summary.FileWriter("./runs/{}-{}-{}-min".format(run_id, agent_name, time))
#     self.avg_writer = tf.summary.FileWriter("./runs/{}-{}-{}-avg".format(run_id, agent_name, time))
#     self.val_writer = tf.summary.FileWriter("./runs/{}-{}-{}-value".format(run_id, agent_name, time))

#     self.input = tf.placeholder(tf.float32)

#     # Scalar summaries
#     self.add_loss = tf.summary.scalar("Loss", self.input)
#     self.add_q = tf.summary.scalar("Q-val", self.input)
#     self.add_score = tf.summary.scalar("Score", self.input)
#     self.add_test_score = tf.summary.scalar("Test score", self.input)
#     self.add_noise = tf.summary.scalar("Noise", self.input)
#     self.add_gradients = tf.summary.scalar("Gradients", self.input) # TODO, get this

#     # Histogram summaries
#     # self.add_action = tf.summary.histogram("Action", self.input) # TODO?
    
#     self.sess.run(tf.global_variables_initializer())


#   def _add_scalars(self, operation, values, writers, episode_number):
#     ''' Run the operation for every v in values.
#     Write the summaries generated with corresponding writer.
#     Values and writers needs to be lists with the same length and in the same order '''

#     create_summaries = lambda val: self.sess.run(operation, {self.input: val})
#     loss_summaries = map(create_summaries, values)

#     for summary, writer in zip(loss_summaries, writers):
#       writer.add_summary(summary, episode_number)


#   def _add_scalar(self, operation, value, writer, episode_number):
#     summary = self.sess.run(operation, {self.input: value})
#     writer.add_summary(summary, episode_number)


#   def _add_histogram(self, operation, value, writer, episode_number): # TODO?
#     summary = self.sess.run(operation, {self.input: value})
#     writer.add_summary(summary, episode_number)


#   def add(self, episode_number, score, noise, max_qs, losses):
#     mma_loss = [max(losses), min(losses), sum(losses) / len(losses)]
#     writers = [self.max_writer, self.min_writer, self.avg_writer]
#     self._add_scalars(self.add_loss, mma_loss, writers, episode_number)

#     mma_q = [max(max_qs), min(max_qs), sum(max_qs) / len(max_qs)]
#     self._add_scalars(self.add_q, mma_q, writers, episode_number)

#     self._add_scalar(self.add_score, score, self.val_writer, episode_number)
#     self._add_scalar(self.add_noise, noise, self.val_writer, episode_number)


#   def add_test(self, episode_number, score):
#     self._add_scalar(self.add_test_score, score, self.val_writer, episode_number)