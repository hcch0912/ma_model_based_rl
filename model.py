import numpy as np
import random

import json
import sys

from two_player.pong import PongGame
from prey_predator.env import PreyPredatorEnv
import time
import argparse
from rnn.rnn import hps_model, MDNRNN
from util.pong_simulate import *
from util.pp_simulate import * 
# from oppo_model import Oppo_Model
import collections
import util.tf_util as U 

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for stochastic games")
    parser.add_argument("--agent", type =str, default = 'DDPG', help = "agent type")
    parser.add_argument("--game", type=str, default="Pong-2p-v0", help="name of the  env")
    parser.add_argument("--timestep", type= int, default= 2, help ="the time step of act_trajectory")
    parser.add_argument("--iteration", type=int, default=60000, help="number of episodes")
    parser.add_argument("--seed", type = int, default = 10, help = "random seed")
    parser.add_argument("--batch_size", type = int, default = 1000, help = "set the batch_size")
    parser.add_argument("--max_episode_len", type = int, default = 50, help = "max episode length")
    parser.add_argument("--lr", type = float, default = 0.0001, help = "learning rate")
    parser.add_argument("--gamma", type = float, default = 0.99, help = "discount rate")
    parser.add_argument("--kl_tolerance", type = float, default = 0.5, help = "dk divergence tolerance")
    parser.add_argument("--data_dir", type = str,default = "./record")
    parser.add_argument("--model_save_path", type = str,default = "./tf_vae", help= "model save path")
    parser.add_argument("--render_mode", type = bool, default = False, help = "render mode")
    parser.add_argument("--use_model", type = bool, default = False, help = "use model")
    parser.add_argument("--model_path", type = str, default = "", help = "load model path")
    parser.add_argument("--recording_mode", type = bool, default = True, help = "training model")
    parser.add_argument("--competitive", type = bool, default = False, help  = "competitive or cooperative")
    parser.add_argument("--train_mode", type =  bool, default = False, help = "train mode")
    parser.add_argument("--simu_episode", type = int, default = 5, help = "simulate how many episodes")
    parser.add_argument("--screen_size", type = int, default = 200, help = "game screen size")
    parser.add_argument("--view_size", type = int, default = 100, help = "POMDP, partial obs size")
    parser.add_argument("--rnn_model_dir",type = str, default = 'rnn/rnn.json', help  = "load the pretrained rnn model")
    parser.add_argument("--agent_num", type = int, default = 2, help = "total number of agent")
    parser.add_argument("--action_space", type = int, default = 2, help = "action space for each agent")
    parser.add_argument("--supervise", type = bool, default = False, help = "supervise oppo_modelling ")
    parser.add_argument("--inference", type = bool, default = False, help = "use inferred intent in policy or not")
    parser.add_argument("--exp_mode", type = int, default = 4, help = "control the feature concat")
    parser.add_argument("--obs_size", type = int, default = 6, help = "observation size")

    return parser.parse_args()


# controls whether we concatenate (z, c, h), etc for features used for car.
MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3 # extra hidden later
MODE_ZH = 4


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return np.maximum(x, 0)

def clip(x, lo=-2, hi=2):
  return np.minimum(np.maximum(x, lo), hi)

def passthru(x):
  return x

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)

def sample(p):
  return np.argmax(np.random.multinomial(1, p))

class Model:
  def __init__(self, arglist,action_space, scope, load_model=True):
    self.action_space = action_space
    self.arglist = arglist
    # self.vae = ConvVAE(batch_size=1, gpu_mode=False, is_training=False, reuse=True)

    hps_sample = hps_model._replace(batch_size=1,
                input_seq_width = arglist.obs_size+ arglist.action_space + (arglist.agent_num-1) * arglist.action_space * arglist.timestep,   
                max_seq_len=1, use_recurrent_dropout=0, is_training=0,
                obs_size = arglist.obs_size )

    self.rnn = MDNRNN(hps_sample, gpu_mode=False, reuse=True)

    if load_model:
      # self.vae.load_json(arglist.vae_model_dir)
      self.rnn.load_json(arglist.rnn_model_dir)

    self.state = self.rnn.rnn_init_state()
    self.rnn_mode = True
    print(arglist.inference)
    if arglist.inference == True: 
      self.input_size = self.rnn.rnn_output_size(arglist.exp_mode) +(arglist.agent_num-1)  * arglist.action_space
    elif arglist.inference == False:
      self.input_size = self.rnn.rnn_output_size(arglist.exp_mode) + (arglist.timestep) *( arglist.agent_num-1) * arglist.action_space  
    # self.z_size = 32
    # action trajectories recording 
    self.act_traj = [collections.deque(np.zeros((arglist.timestep, arglist.action_space)), maxlen = arglist.timestep)] *(arglist.agent_num -1)
    # self.oppo_model = Oppo_Model(arglist.agent_num, arglist.timestep, arglist.action_space,arglist.action_space, "oppo_model_{}".format(scope) )
    self.inference = arglist.inference

    if arglist.exp_mode == MODE_Z_HIDDEN: # one hidden layer
      self.hidden_size = 40
      self.weight_hidden = np.random.randn(self.input_size, self.hidden_size)
      self.bias_hidden = np.random.randn(self.hidden_size)
      self.weight_output = np.random.randn(self.hidden_size, self.action_space)
      self.bias_output = np.random.randn(self.action_space)
      self.param_count = ((self.input_size+1)*self.hidden_size) + (self.hidden_size*self.action_space+self.action_space)
    else:
      self.weight = np.random.randn(self.input_size, self.action_space)
      self.bias = np.random.randn(self.action_space)
      self.param_count = (self.input_size)*self.action_space+self.action_space

  def reset(self):
    self.state = self.rnn.rnn_init_state()
    # self.oppo_state = lstm_init_state(self.oppo_model)

  # def encode_obs(self, obs):
  #   # convert raw obs to z, mu, logvar
  #   result = np.copy(obs).astype(np.float)/255.0
  #   result = result.reshape(1, 64, 64, 3)
  #   mu, logvar = self.vae.encode_mu_logvar(result)
  #   mu = mu[0]
  #   logvar = logvar[0]
  #   s = logvar.shape
  #   z = mu + np.exp(logvar/2.0) * np.random.randn(*s)
  #   return z, mu, logvar

  def get_action(self, obs, act_traj):
    h = self.rnn.rnn_output(self.state, obs, act_traj, self.arglist.exp_mode)

    if self.arglist.inference:
      oppo_intents = []
      for i in range(self.arglist.agent_num - 1):
        act_traj = self.act_traj[i]
        # intent = self.oppo_model .get_inference(act_traj)
        intent = [0,0]
        oppo_intents.append(intent)
      oppo_intents = np.reshape(oppo_intents, ((self.arglist.agent_num-1 )* self.arglist.action_space))
      #Oppo intent shape (batch_size, agent_num, action_space)
      # reshape oppo_intent  agent_num * batch_size * action_space
     
      controller_input = np.concatenate((h, oppo_intents))
     
    else:
      controller_input =  h
    
    if self.arglist.exp_mode == MODE_Z_HIDDEN: # one hidden layer
      x = np.tanh(np.dot(controller_input, self.weight_hidden) + self.bias_hidden)
      action = np.tanh(np.dot(x, self.weight_output) + self.bias_output)
    else:
      action = np.tanh(np.dot(controller_input, self.weight) + self.bias)
    for i in range(self.action_space):
      action[i] = clip(action[i])
    
    self.state = self.rnn.rnn_next_state(obs, action, self.act_traj, self.state)
    # self.oppo_state = oppo_next_state(self.oppo_model, action, self.act_traj, self.oppo_state)

    # epsilon exploration
    if np.random.uniform(0,1) < 0.2:
      action = [np.random.uniform(-3,3)] * len(action)
    return action

  def set_model_params(self, model_params):
    if self.arglist.exp_mode == MODE_Z_HIDDEN: # one hidden layer
      params = np.array(model_params)
      cut_off = (self.input_size+1)*self.hidden_size
      params_1 = params[:cut_off]
      params_2 = params[cut_off:]
      self.bias_hidden = params_1[:self.hidden_size]
      self.weight_hidden = params_1[self.hidden_size:].reshape(self.input_size, self.hidden_size)
      self.bias_output = params_2[:self.action_space]
      self.weight_output = params_2[self.action_space:].reshape(self.hidden_size, self.action_space)
    else:
      self.bias = np.array(model_params[:self.action_space])
      self.weight = np.array(model_params[self.action_space:]).reshape(self.input_size, self.action_space)

  def load_model(self, filename):
    with open(filename) as f:    
      data = json.load(f)
    print('loading file %s' % (filename))
    # self.data = data
    model_params = np.array(data[0]) # assuming other stuff is in data
    self.set_model_params(model_params)

  def get_random_model_params(self, stdev=0.1):
    #return np.random.randn(self.param_count)*stdev
    return np.random.standard_cauchy(self.param_count)*stdev # spice things up

  def init_random_model_params(self, stdev=0.1):
    params = self.get_random_model_params(stdev=stdev)
    self.set_model_params(params)
    # vae_params = self.vae.get_random_model_params(stdev=stdev)
    # self.vae.set_model_params(vae_params)
    rnn_params = self.rnn.get_random_model_params(stdev=stdev)
    self.rnn.set_model_params(rnn_params)

def make_model(arglist, action_space, scope,  model_path = None, load_model=False):
  # can be extended in the future.
  model = Model( arglist, action_space, scope, load_model=load_model)
  if load_model:
     model.load_model(model_path)
  else:
     model.init_random_model_params(stdev=np.random.rand()*0.01)
  return model


def make_env(arglist, seed=-1, render_mode=False, full_episode=False):
    if arglist.game == "Pong-2p-v0":
      return PongGame(competitive = arglist.competitive)
    if arglist.game == "prey_predator":
      return PreyPredatorEnv( view_size = arglist.view_size )  

# for train model use 
def main():
  arglist = parse_args()
  N_episode = arglist.episodes
  if arglist.render_mode:
      N_episode = 1
  reward_list = []
  prey_reward_list = []
  predator_reward_list = []
  env = make_env(arglist)
  if arglist.game == "Pong-2p-v0":
    with U.single_threaded_session(): 
      model = make_model(arglist, action_space = 1, scope = "pong", model_path =arglist.model_path,load_model = arglist.use_model)  
      print('model size', model.param_count)
      
      U.initialize()
      for i in range(N_episode):
        reward, steps_taken = pong_simulate(model, env, arglist)
        if arglist.render_mode:
          print("terminal reward", reward, "average steps taken", np.mean(steps_taken)+1)
        else:
          pass
        reward_list.append(reward[0])
      if not arglist.render_mode:
        print("seed", arglist.seed, "average_reward", np.mean(reward_list), "stdev", np.std(reward_list))

  if arglist.game =="prey_predator":
    with U.single_threaded_session(): 
      prey_model = make_model(arglist, action_space = 2,  scope = "prey",model_path =arglist.model_path ,load_model = arglist.use_model)  
      predator_model = make_model(arglist, action_space = 2,scope = "predator",  model_path = arglist.model_path, load_model = arglist.use_model)
      print('model size', prey_model.param_count, predator_model.param_count )
      U.initialize()
      for i in range(N_episode):
      	prey_reward, predator_reward, steps_taken = pp_simulate([prey_model, predator_model], env, arglist)
      	if arglist.render_mode:
      		print("terminal reward", rewards, "average steps taken", np.mean(steps_taken)+1)
      	else:
      		pass
      	prey_reward_list.append(prey_reward)
      	predator_reward_list.append(predator_reward)
      if not arglist.render_mode:
      	print("Seed", arglist.seed, 
      		"prey average_reward", np.mean(prey_reward_list),
      	 	"predator average_reward", np.mean(predator_reward_list))			

if __name__ == "__main__":
  main()