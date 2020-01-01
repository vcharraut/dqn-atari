# Hyperparameters used for the training
# memory_capacity = 100000
# num_episodes    = 100000
# batch_size      = 32
# target_update   = 10000
# start_learning  = 50000
# learning_rate   = 0.00025
# gamma           = 0.99
# epsilon_decay   = 100000
# epsilon_start   = 1
# epsilon_end     = 0.1


class Config():

	def __init__(self,
				memory_capacity=100000,
				num_episodes=100000,
				batch_size=32,
				target_update=10000, 
				start_learning=1000,
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
