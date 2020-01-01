# Hyperparameters used for the training
# memory_capacity = 100000
# num_episodes    = 100000
# batch_size      = 32
# target_update   = 10000
# start_learning  = 50000
# freq_learning   = 4
# learning_rate   = 0.00025
# gamma           = 0.99
# epsilon_decay   = 100000
# epsilon_start   = 1
# epsilon_end     = 0.1


class Config():

	def __init__(self,
				memory_capacity=150000,
				num_episodes=100000,
				batch_size=32,
				target_update=10000, 
				start_learning=50000,
				freq_learning=4,
				learning_rate=0.00025,
				gamma=0.99,
				epsilon_decay=100000,
				epsilon_start=1, 
				epsilon_end=0.1):
			   
		self._memory_capacity = memory_capacity
		self._num_episodes = num_episodes
		self._batch_size = batch_size
		self._target_update = target_update
		self._start_learning = start_learning
		self._freq_learning = freq_learning
		self._learning_rate = learning_rate
		self._gamma = gamma
		self._epsilon_decay = epsilon_decay
		self._epsilon_start = epsilon_start
		self._epsilon_end = epsilon_end


	@property
	def memory_capacity(self):
		return self._memory_capacity


	@property
	def num_episodes(self):
		return self._num_episodes


	@property
	def batch_size(self):
		return self._batch_size


	@property
	def target_update(self):
		return self._target_update

	
	@property
	def start_learning(self):
		return self._start_learning


	@property
	def freq_learning(self):
		return self._freq_learning


	@property
	def learning_rate(self):
		return self._learning_rate


	@property
	def gamma(self):
		return self._gamma


	@property
	def epsilon_decay(self):
		return self._epsilon_decay


	@property
	def epsilon_start(self):
		return self._epsilon_start


	@property
	def epsilon_end(self):
		return self._epsilon_end


	def save_config(self, path, env):
		with open(path, 'a') as f:
			f.write('environment     = ' + env + '\n')
			f.write('memory_capacity = ' + str(self.memory_capacity) + '\n')
			f.write('num_episodes    = ' + str(self.num_episodes) + '\n')
			f.write('batch_size      = ' + str(self.batch_size) + '\n')
			f.write('target_update   = ' + str(self.target_update) + '\n')
			f.write('start_learning  = ' + str(self.start_learning) + '\n')
			f.write('freq_learning   = ' + str(self.freq_learning) + '\n')
			f.write('learning_rate   = ' + str(self.learning_rate) + '\n')
			f.write('gamma           = ' + str(self.gamma) + '\n')
			f.write('epsilon_decay   = ' + str(self.epsilon_decay) + '\n')
			f.write('epsilon_start   = ' + str(self.epsilon_start) + '\n')
			f.write('epsilon_end     = ' + str(self.epsilon_end) + '\n')



