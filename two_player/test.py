import numpy as np
from pong import PongGame
if __name__ == '__main__':
	env = PongGame(competitive = False)
	obs = env.reset()
	while 1:
		env.render(mode='human')
		obs, reward, dones, win = env.step([np.random.uniform(-1,1),np.random.uniform(-1,1)])
