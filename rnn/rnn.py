import numpy as np
from collections import namedtuple
import json
import tensorflow as tf


# hyperparameters for our model. I was using an older tf version, when HParams was not available ...

# controls whether we concatenate (z, c, h), etc for features used for car.
MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3 # extra hidden later
MODE_ZH = 4


def get_pi_idx(x, pdf):
  # samples from a categorial distribution
  N = pdf.size
  accumulate = 0
  for i in range(0, N):
    accumulate += pdf[i]
    if (accumulate >= x):
      return i
  print('error with sampling ensemble')
  return -1

def tf_lognormal(y, mean, logstd, logSqrtTwoPI):
      return -0.5 * ((y - mean) / tf.exp(logstd)) ** 2 - logstd - logSqrtTwoPI

def get_lossfunc(logmix, mean, logstd, y, logSqrtTwoPI):
      v = logmix + tf_lognormal(y, mean, logstd, logSqrtTwoPI)
      v = tf.reduce_logsumexp(v, 1, keep_dims=True)
      return -tf.reduce_mean(v)

def get_mdn_coef(output):
      logmix, mean, logstd = tf.split(output, 3, 1)
      logmix = logmix - tf.reduce_logsumexp(logmix, 1, keep_dims=True)
      return logmix, mean, logstd

HyperParams = namedtuple('HyperParams', ['num_steps',
                                         'max_seq_len',
                                         'input_seq_width',
                                         'output_seq_width',
                                         'rnn_size',
                                         'batch_size',
                                         'grad_clip',
                                         'num_mixture',
                                         'learning_rate',
                                         'decay_rate',
                                         'min_learning_rate',
                                         'use_layer_norm',
                                         'use_recurrent_dropout',
                                         'recurrent_dropout_prob',
                                         'use_input_dropout',
                                         'input_dropout_prob',
                                         'use_output_dropout',
                                         'output_dropout_prob',
                                         'is_training',
                                         'obs_size',
                                        ])

def default_hps():
  return HyperParams(num_steps=2000, # train model for 2000 steps.
                     max_seq_len=1000, # train on sequences of 100
                     input_seq_width=34,    # width of our data (32 + 2 actions + 2* num_oppo actions)
                     output_seq_width=32,    # width of our data is 32
                     rnn_size=256,    # number of rnn cells
                     batch_size=10,   # minibatch sizes
                     grad_clip=1.0,
                     num_mixture=5,   # number of mixtures in MDN
                     learning_rate=0.001,
                     decay_rate=1.0,
                     min_learning_rate=0.00001,
                     use_layer_norm=0, # set this to 1 to get more stable results (less chance of NaN), but slower
                     use_recurrent_dropout=0,
                     recurrent_dropout_prob=0.90,
                     use_input_dropout=0,
                     input_dropout_prob=0.90,
                     use_output_dropout=0,
                     output_dropout_prob=0.90,
                     is_training=1,
                     obs_size = 6)

hps_model = default_hps()

# MDN-RNN model
class MDNRNN():
  def __init__(self, hps, gpu_mode=False, reuse=False):
    self.hps = hps
    with tf.variable_scope('mdn_rnn', reuse=reuse):
      if not gpu_mode:
        with tf.device("/cpu:0"):
          print("model using cpu")
          self.g = tf.Graph()
          with self.g.as_default():
            self.build_model(hps)
      else:
        print("model using gpu")
        self.g = tf.Graph()
        with self.g.as_default():
          self.build_model(hps)
    self.init_session()
  def build_model(self, hps):
    
    if self.hps.is_training:
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

    cell_fn = tf.contrib.rnn.LayerNormBasicLSTMCell # use LayerNormLSTM

    use_recurrent_dropout = False if self.hps.use_recurrent_dropout == 0 else True
    use_input_dropout = False if self.hps.use_input_dropout == 0 else True
    use_output_dropout = False if self.hps.use_output_dropout == 0 else True
    is_training = False if self.hps.is_training == 0 else True
    use_layer_norm = False if self.hps.use_layer_norm == 0 else True

    if use_recurrent_dropout:
      cell = cell_fn(self.hps.rnn_size, layer_norm=use_layer_norm, dropout_keep_prob=self.hps.recurrent_dropout_prob)
    else:
      cell = cell_fn(self.hps.rnn_size, layer_norm=use_layer_norm)

    # multi-layer, and dropout:
    print("input dropout mode =", use_input_dropout)
    print("output dropout mode =", use_output_dropout)
    print("recurrent dropout mode =", use_recurrent_dropout)
    if use_input_dropout:
      print("applying dropout to input with keep_prob =", self.hps.input_dropout_prob)
      cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.hps.input_dropout_prob)
    if use_output_dropout:
      print("applying dropout to output with keep_prob =", self.hps.output_dropout_prob)
      cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.hps.output_dropout_prob)
    self.cell = cell

    # self.sequence_lengths = self.hps.max_seq_len # assume every sample has same length.
    self.input_x = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len, self.hps.input_seq_width])
   
    self.output_x = tf.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len, self.hps.output_seq_width])

    # actual_input_x = self.input_x
    self.initial_state = cell.zero_state(batch_size=hps.batch_size, dtype=tf.float32) 

    self.num_output = self.hps.output_seq_width * self.hps.num_mixture * 3

    with tf.variable_scope('RNN'):
      output_w = tf.get_variable("output_w", [self.hps.rnn_size, self.num_output])
      output_b = tf.get_variable("output_b", [self.num_output])

    output, last_state = tf.nn.dynamic_rnn(cell, self.input_x, initial_state=self.initial_state,
                                           time_major=False, swap_memory=True, dtype=tf.float32, scope="RNN")

    output = tf.reshape(output, [-1, self.hps.rnn_size])
    output = tf.nn.xw_plus_b(output, output_w, output_b)
    output = tf.reshape(output, [-1, self.hps.num_mixture * 3])
    self.final_state = last_state    

    logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
    
    self.out_logmix, self.out_mean, self.out_logstd = get_mdn_coef(output)

    # reshape target data so that it is compatible with prediction shape
    flat_target_data = tf.reshape(self.output_x,[-1, 1])

    lossfunc = get_lossfunc(self.out_logmix, self.out_mean, self.out_logstd, flat_target_data, logSqrtTwoPI)

    self.cost = tf.reduce_mean(lossfunc)

    if self.hps.is_training == 1:
      self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
      optimizer = tf.train.AdamOptimizer(self.lr)

      gvs = optimizer.compute_gradients(self.cost)
      capped_gvs = [(tf.clip_by_value(grad, -self.hps.grad_clip, self.hps.grad_clip), var) for grad, var in gvs]
      self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step, name='train_step')

    # initialize vars
    self.init = tf.global_variables_initializer()
  def init_session(self):
    """Launch TensorFlow session and initialize variables"""
    self.sess = tf.Session(graph=self.g)
    self.sess.run(self.init)
  def close_sess(self):
    """ Close TensorFlow session """
    self.sess.close()
  def get_model_params(self):
    # get trainable params.
    model_names = []
    model_params = []
    model_shapes = []
    with self.g.as_default():
      t_vars = tf.trainable_variables()
      for var in t_vars:
        param_name = var.name
        p = self.sess.run(var)
        model_names.append(param_name)
        params = np.round(p*10000).astype(np.int).tolist()
        model_params.append(params)
        model_shapes.append(p.shape)
    return model_params, model_shapes, model_names
  def get_random_model_params(self, stdev=0.5):
    # get random params.
    _, mshape, _ = self.get_model_params()
    rparam = []
    for s in mshape:
      #rparam.append(np.random.randn(*s)*stdev)
      rparam.append(np.random.standard_cauchy(s)*stdev) # spice things up
    return rparam
  def set_random_params(self, stdev=0.5):
    rparam = self.get_random_model_params(stdev)
    self.set_model_params(rparam)
  def set_model_params(self, params):
    with self.g.as_default():
      t_vars = tf.trainable_variables()
      idx = 0
      for var in t_vars:
        pshape = self.sess.run(var).shape
        p = np.array(params[idx])
        assert pshape == p.shape, "inconsistent shape"
        assign_op = var.assign(p.astype(np.float)/10000.)
        self.sess.run(assign_op)
        idx += 1
  def load_json(self, jsonfile='rnn.json'):
    with open(jsonfile, 'r') as f:
      params = json.load(f)
    self.set_model_params(params)
  def save_json(self, jsonfile='rnn.json'):
    model_params, model_shapes, model_names = self.get_model_params()
    qparams = []
    for p in model_params:
      qparams.append(p)
    with open(jsonfile, 'wt') as outfile:
      json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))

  def rnn_init_state(self):
    return self.sess.run(self.initial_state)

  def rnn_next_state(self, obs, a, act_traj, prev_state):

    input_x = np.concatenate((np.array(obs).reshape((1, 1, np.array(obs).size)), np.array(a).reshape((1, 1, np.array(a).size)), np.array(act_traj).reshape((1,1, np.array(act_traj).size ))), axis=2)
    feed = {self.input_x: input_x, self.initial_state:prev_state}
    return self.sess.run(self.final_state, feed)
  def rnn_output(self, state, obs, act_traj, mode):
  # reshape act_traject, observation will be agent_num x 2 , act_traj be agent_num -1 x timestep x action_space
    obs = np.array(obs)
    act_traj = np.array(act_traj)
    obs = obs.reshape((obs.size,))
    act_traj = act_traj.reshape((act_traj.size))
    if mode == MODE_ZCH:
      return np.concatenate([obs, act_traj,np.concatenate((state.c,state.h), axis=1)[0]])
    if mode == MODE_ZC:
      return np.concatenate([obs,act_traj, state.c[0]])
    if mode == MODE_ZH:
      # print(z.shape, state.h[0].shape)
      return np.concatenate([obs,act_traj, state.h[0]])
    return np.concatenate([obs,act_traj]) # MODE_Z or MODE_Z_HIDDEN

  def rnn_output_size(self, mode):
    if mode == MODE_ZCH:
      return (self.hps.obs_size+256+256  )
    if (mode == MODE_ZC) or (mode == MODE_ZH):
      return (self.hps.obs_size+256 )
    return self.hps.obs_size # MODE_Z or MODE_Z_HIDDEN  
  
  def sample_sequence(self, init_z, actions, temperature=1.0, seq_len=1000):
    # generates a random sequence using the trained model

    prev_x = np.zeros((1, 1, self.hps.output_seq_width))
    prev_x[0][0] = init_z

    prev_state = sess.run(s_model.initial_state)

    strokes = np.zeros((seq_len, self.hps.output_seq_width), dtype=np.float32)

    for i in range(seq_len):
      input_x = np.concatenate((prev_x, actions[i].reshape((1, 1, 3))), axis=2)
      feed = {s_model.input_x: input_x, s_model.initial_state:prev_state}
      [logmix, mean, logstd, next_state] = sess.run([s_model.out_logmix, s_model.out_mean, s_model.out_logstd, s_model.final_state], feed)

      # adjust temperatures
      logmix2 = np.copy(logmix)/temperature
      logmix2 -= logmix2.max()
      logmix2 = np.exp(logmix2)
      logmix2 /= logmix2.sum(axis=1).reshape(self.hps.output_seq_width, 1)

      mixture_idx = np.zeros(self.hps.output_seq_width)
      chosen_mean = np.zeros(self.hps.output_seq_width)
      chosen_logstd = np.zeros(self.hps.output_seq_width)
      for j in range(self.hps.output_seq_width):
        idx = get_pi_idx(np.random.rand(), logmix2[j])
        mixture_idx[j] = idx
        chosen_mean[j] = mean[j][idx]
        chosen_logstd[j] = logstd[j][idx]

      rand_gaussian = np.random.randn(self.hps.output_seq_width)*np.sqrt(temperature)
      next_x = chosen_mean+np.exp(chosen_logstd)*rand_gaussian

      strokes[i,:] = next_x

      prev_x[0][0] = next_x
      prev_state = next_state

    return strokes

