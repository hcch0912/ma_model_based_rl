import numpy as np
import random
import os
import gym
from model import make_model, make_env
import argparse
import util.tf_util as U 
import collections
from copy import deepcopy
def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for stochastic games")
    parser.add_argument("--agent", type =str, default = 'DDPG', help = "agent type")
    parser.add_argument("--adv_agent", type = str, default = "script", help = "adv agent type")
    parser.add_argument("--game", type=str, default="Pong-2p-v0", help="name of the  env")
    parser.add_argument("--timestep", type= int, default= 2, help ="the time step of act_trajectory")
    parser.add_argument("--seed", type = int, default = 10, help = "random seed")
    parser.add_argument("--batch_size", type = int, default = 64, help = "set the batch_size")
    parser.add_argument("--max_episode_len", type = int, default = 50, help = "max episode length")
    parser.add_argument("--data_dir", type = str,default = "./record")
    parser.add_argument("--model_save_path", type = str,default = "./tf_rnn/", help= "model save path")
    parser.add_argument("--render_mode",type = bool, default = False, help = "render or not")
    parser.add_argument("--max_frames", type = int, default = 1000, help = "max frames to store")
    parser.add_argument("--max_trials", type = int, default = 200, help = "use this to extract one trial")
    parser.add_argument("--model_path", type = str, default = "", help = "path to load the model")
    parser.add_argument("--use_model", type = bool, default = False, help = "use model")
    parser.add_argument("--competitive", type = bool, default = False, help  = "competitive or cooperative")
    parser.add_argument("--agent_num", type = int, default = 2, help = "total number of agent")
    parser.add_argument("--action_space", type = int, default = 2, help = "action space for each agent")
    parser.add_argument("--inference", type = bool, default = False, help = "use inference in policy or not")
    parser.add_argument("--obs_size", type = int, default = 6, help = "observation size")

    parser.add_argument("--exp_mode", type = int, default = 4, help = "control the feature concat")

    return parser.parse_args()


def get_act_traj(act_traj_list, i):
    oppo = []
    for j in range(5):
        if j == i : continue
        else:
            oppo.append(deepcopy(act_traj[j]))

    return oppo


if __name__ == '__main__':


    arglist = parse_args()
    if not os.path.exists(arglist.data_dir):
        os.makedirs(arglist.data_dir)
    total_frames = 0
    env = make_env(arglist)

    if arglist.game ==  "Pong-2p-v0":
        model = make_model(arglist, action_space= 1, scope = "pong", model_path =arglist.model_path,load_model = arglist.use_model)  
        # U.initialize()
        for trial in range(arglist.max_trials): # 200 trials per worker
          try:
            random_generated_int = random.randint(0, 2**31-1)
            filename = arglist.data_dir+"/"+str(random_generated_int)+".npz"          
            recording_obs = []
            recording_action = []
            recording_act_traj = []

            np.random.seed(random_generated_int)
            env.seed(random_generated_int)
            # random policy
            model.init_random_model_params(stdev=np.random.rand()*0.01)
            model.reset()
            obs = env.reset() # pixels
            act_traj0 = collections.deque(np.zeros((arglist.timestep, arglist.action_space)), maxlen = arglist.timestep) 
            act_traj1 = collections.deque(np.zeros((arglist.timestep, arglist.action_space)), maxlen = arglist.timestep) 
            for frame in range(arglist.max_frames):
              
              if arglist.render_mode:
                env.render("human")
              else:
                env.render("rgb_array")

              recording_obs.append(np.array(obs))

              action0 = model.get_action(obs[0], deepcopy(act_traj0))
              
              action1 = model.get_action(obs[1], deepcopy(act_traj1))
              
              recording_act_traj.append([deepcopy(act_traj0),deepcopy(act_traj1) ])
              act_traj0.append(action1)
              act_traj1.append(action0)
              recording_action.append([action0, action1])
              

              obs, reward, goals, win = env.step([action0, action1])
              if win:
                break
            total_frames += (frame+1)
            print("dead at", frame+1, "total recorded frames for this worker", total_frames)
            recording_obs = np.array(recording_obs, dtype=np.uint8)
            recording_action = np.array(recording_action, dtype=np.float16)     
            recording_act_traj = np.array(recording_act_traj, dtype = np.float16)
            np.savez_compressed(filename, obs = recording_obs, action = recording_action, act_traj = recording_act_traj)
          except gym.error.Error:
            print("stupid gym error, life goes on")
            env.close()
            make_env(arglist,  render_mode=arglist.render_mode)
            continue
        env.close()
    if arglist.game == "prey_predator":
        prey_model = make_model(arglist, action_space = 2, scope = "prey",model_path =arglist.model_path ,load_model = arglist.use_model)  
        predator_model = make_model(arglist, action_space = 2, scope = "predator", model_path = arglist.model_path, load_model = arglist.use_model)
        # U.initialize()
        for trial in range(arglist.max_trials):
            try:
                random_generated_int = random.randint(0, 2**31-1)
                prey_filename = os.path.join(arglist.data_dir, "prey",str(random_generated_int)+".npz")
                predator_filename = os.path.join(arglist.data_dir, "predator", str(random_generated_int)+".npz") 
                recording_obs = [[]] * 5
                recording_action = [[]] * 5
                recording_act_traj = [[]] * 5

                np.random.seed(random_generated_int)
                env.seed(random_generated_int)
                prey_model.init_random_model_params(stdev=np.random.rand()*0.01)
                predator_model.init_random_model_params(stdev=np.random.rand()*0.01)
                prey_model.reset()
                predator_model.reset()
                obs = env.reset() #
                act_traj = []
                for i in range(arglist.agent_num):
                    act_traj.append(collections.deque(np.zeros((arglist.timestep, arglist.action_space)), maxlen = arglist.timestep))

                for frame in range(arglist.max_frames):

                    if arglist.render_mode:
                        env.render("human")
                    else:
                        env.render("rgb_array")
                    
                    action_episode = [] 
                    for i in range(5):
                        recording_obs[i].append(obs[i])
                    action0 = prey_model.get_action(obs[0], get_act_traj(deepcopy(act_traj), 0))
                    action_episode.append(action0)
                    
                    for i in range(1,5):
                        action1 = predator_model.get_action(obs[i], get_act_traj(deepcopy(act_traj), i))
                        action_episode.append(action1)

                    for i in range(5):    
                        recording_action[i].append(action_episode) 
                        recording_act_traj[i].append(get_act_traj(deepcopy(act_traj), i))
                        act_traj[i].append(action_episode[i])        

                    obs_,rewards, done, _ = env.step(action_episode)
                    assert np.array(obs).size == np.array(obs_).size, "obs size not match"
                    obs = obs_
                    if done: break
                total_frames += (frame+1)  
                print("dead at", frame+1, "total recorded frames for this worker", total_frames)
                recording_obs = np.array(recording_obs,dtype= np.float16)
                recording_action = np.array(recording_action, dtype=np.float16)  
                recording_act_traj = np.array(deepcopy(recording_act_traj), dtype = np.float16)
                np.savez_compressed(prey_filename, obs = recording_obs[0], action = recording_action[0],act_traj = deepcopy(recording_act_traj[0]))  
                for i in range(1,5):
                #     oppo = []
                #     oppo.append(deepcopy(recording_act_traj[0]))
                #     for j in range(1,5):
                #         if i ==j : continue
                #     else:
                #         oppo.append(deepcopy(recording_act_traj[j]))
                    np.savez_compressed(predator_filename, obs = recording_obs[i], action = recording_action[i],act_traj =recording_act_traj)     
            except gym.error.Error:
              print("stupid gym error, life goes on")
              env.close()
              make_env(arglist, render_mode=arglist.render_mode)
              continue      
        env.close()          



            

