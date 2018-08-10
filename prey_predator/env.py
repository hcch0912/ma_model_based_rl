import numpy.random as random
import pygame, sys
from pygame.locals import *
import pygame.surfarray as sarray
import numpy as np
from gym import Env, spaces

pygame.init()
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
YELLOW = (0,0,255)
BLACK = (0,0,0)

#globals
CUBE_SIZE = 200	   
ACTOR_RADIUS = 10

class Actor:
	def __init__(self, role, pos,vel):
		self.role = role
		self.pos = pos
		self.vel = vel
	def accelerate(self, acceleration):
		v = self.vel 
		for i in range(2):
			sign_a = np.sign(acceleration[i])
			if np.sign(v[i]) == sign_a:
				self.vel[i] = 0.8*(v[i]+acceleration[i])
			else:
				self.vel[i] = v[i]+acceleration[i]
			

class PreyPredatorEnv():
	def __init__(self,screen_size = 200,   prey_num = 1, predator_num = 4 ):
		self.viewer = None
		self.canvas = pygame.Surface((CUBE_SIZE, CUBE_SIZE))
		self.screen = sarray.array3d(self.canvas)
		self.screen_size = 200
		self.prey_num = prey_num
		self.predator_num = predator_num
		self.prey = []
		self.predator = []
		self.reward = 0 
		self.prey_reward = []
		self.predator_reward = []
		self.rng = np.random.RandomState()
		self.screen_size = screen_size
		for i in range(prey_num):
			self.prey.append(Actor(role = "prey_{}".format(i), pos = [np.random.randint(0, self.screen_size), np.random.randint(0, self.screen_size)], vel = [0,0]))
		for i in range(predator_num):
			self.predator.append(Actor(role = "predator_{}".format(i), pos = [np.random.randint(0, self.screen_size), np.random.randint(0, self.screen_size)], vel =  [0,0]))	

	def seed(self, seed=None):
		self.rng.seed(seed)
	def reset(self):
		for i in range(self.prey_num):
			self.prey.append(Actor("prey_{}".format(i), [np.random.randint(0, self.screen_size), np.random.randint(0, self.screen_size)], [0,0]))
		for i in range(self.predator_num):
			self.prey.append(Actor("predator_{}".format(i), [np.random.randint(0, self.screen_size), np.random.randint(0, self.screen_size)], [0,0]))
		obs = []
		for i in range(self.prey_num):
			obs.append(self.get_obs(self.prey[i]))
		for i in range(self.predator_num):
			obs.append(self.get_obs(self.predator[i]))
		
		return obs
	def step(self,actions):
		# follow the order of prey -> predators
		for i in range(self.prey_num):
			self.prey[i].accelerate(actions[0])
		for i in range(self.predator_num):
			self.predator[i].accelerate(actions[1+i])

		self.canvas.fill(BLACK)
		for i in range(self.prey_num):
			pygame.draw.circle(self.canvas, GREEN, [*map(int,self.prey[i].pos)], ACTOR_RADIUS, 0)		
		for i in range(self.predator_num):
			pygame.draw.circle(self.canvas, RED, [*map(int,self.predator[i].pos)], ACTOR_RADIUS, 0)		
			
		#update positions
		for i in range(self.prey_num):
			self.prey[i].pos[0]+= int(self.prey[i].vel[0])
			self.prey[i].pos[1]+= int(self.prey[i].vel[1])
			if self.prey[i].pos[0] > CUBE_SIZE :
				self.prey[i].pos[0] =  2 * CUBE_SIZE - self.prey[i].pos[0] 
			if self.prey[i].pos[0] < 0:
				self.prey[i].pos[0] = -self.prey[i].pos[0]
			if self.prey[i].pos[1] > CUBE_SIZE :
				self.prey[i].pos[1] =  2 * CUBE_SIZE - self.prey[i].pos[1] 
			if self.prey[i].pos[1] < 0:
				self.prey[i].pos[1] = -self.prey[i].pos[1]
				
		for i in range(self.predator_num):
			self.predator[i].pos[0] += int(self.predator[i].vel[0])
			self.predator[i].pos[1] += int(self.predator[i].vel[1])
			if self.predator[i].pos[0] > CUBE_SIZE :
				self.predator[i].pos[0] =  2 * CUBE_SIZE - self.predator[i].pos[0] 
			if self.predator[i].pos[0] < 0:
				self.predator[i].pos[0] = -self.predator[i].pos[0]
			if self.predator[i].pos[1] > CUBE_SIZE :
				self.predator[i].pos[1] =  2 * CUBE_SIZE - self.predator[i].pos[1] 
			if self.predator[i].pos[1] < 0:
				self.predator[i].pos[1] = -self.predator[i].pos[1]
		self.screen = sarray.array3d(self.canvas)

		# if prey is circled by predators reward -1 for prey reward +1 for predators 
		if self.prey[0].pos[0] > min([ a.pos[0] for a in self.predator]) \
			 and  self.prey[0].pos[0]  <  max([a.pos[0] for a in self.predator]) \
			 and self.prey[0].pos[1] > min([a.pos[1] for a in self.predator])\
			 and self.prey[0].pos[1] < max([a.pos[1] for a in self.predator]):
			self.reward = [-1,1,1,1,1]
			self.done = False
		else:
			self.reward = [1, -1,-1,-1,-1]
			self.done = False
		# if prey is blocked by predators, done
		if self.prey[0].pos[0]  < max([a.pos[0] for a in self.predator]) \
			and   self.prey[0].pos[0]  >  min([ a.pos[0] for a in self.predator]) \
			and self.prey[0].pos[1] > min([a.pos[1] for a in self.predator]) \
			and self.prey[0].pos[1] < max([a.pos[1] for a in self.predator]) \
			and  max([a.pos[0] for a in self.predator]) -  min([ a.pos[0] for a in self.predator]) < 30 \
			and max([a.pos[1] for a in self.predator]) - min([a.pos[1] for a in self.predator]) < 30:
			self.reward = [-1,1,1,1,1]
			self.done = True

		

		# for each actor, return a 100 x 100 cube observation, POMAP
		# obs 5 x 2 
		obs = []
		for i in range(self.prey_num):
			obs.append(self.get_obs(self.prey[i]))
		for i in range(self.predator_num):
			obs.append(self.get_obs(self.predator[i]))

		return obs, self.reward, self.done, ""

	def get_obs(self, actor):
		# observation (my pos  + other's pos)
		obs = []
		obs.append(actor.pos)
		if actor.role == self.prey[0].role:
			for i in range(4):
				obs.append(self.predator[i].pos)
			return obs
		else:
			for i in range(4):
				if actor.role == self.predator[i].role:
					continue
				else:	
					obs.append(self.predator[i].pos)
			obs.append(self.prey[0].pos)	
			return obs				

	def render(self,mode='human',close=False, image = None):
		if close:
			if self.viewer is not None:
				self.viewer.close()
				self.viewer = None
			return
		if mode == 'rgb_array':
			return self.screen
		elif mode == 'human':
			from gym.envs.classic_control import rendering
			if self.viewer is None:
				self.viewer = rendering.SimpleImageViewer()
			if  image :
				self.viewer.imshow(image)
			else:		
				self.viewer.imshow(self.screen)			

