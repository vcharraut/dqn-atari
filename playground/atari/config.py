# Hyperparameters used for the training DQN 
# memory_capacity = 150000
# num_steps       = 4000000
# batch_size      = 32
# target_update   = 10000
# start_learning  = 5000
# freq_learning   = 2
# learning_rate   = 0.00025
# gamma           = 0.99
# epsilon_decay   = 200000
# epsilon_start   = 1
# epsilon_end     = 0.1


class Config():

	def __init__(self,
				memory_capacity=1000000,
				num_steps=4000000,
				batch_size=32,
				target_update=10000, 
				start_learning=50000,
				freq_learning=4,
				learning_rate=0.00025,
				gamma=0.99,
				epsilon_decay=1000000,
				epsilon_start=1, 
				epsilon_end=0.05):
			   
		self.memory_capacity = memory_capacity
		self.num_steps = num_steps
		self.batch_size = batch_size
		self.target_update = target_update
		self.start_learning = start_learning
		self.freq_learning = freq_learning
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.epsilon_decay = epsilon_decay
		self.epsilon_start = epsilon_start
		self.epsilon_end = epsilon_end


	def save_config(self, path, env):
		with open(path, 'a') as f:
			f.write('environment     = ' + env + '\n')
			f.write('memory_capacity = ' + str(self.memory_capacity) + '\n')
			f.write('num_steps       = ' + str(self.num_steps) + '\n')
			f.write('batch_size      = ' + str(self.batch_size) + '\n')
			f.write('target_update   = ' + str(self.target_update) + '\n')
			f.write('start_learning  = ' + str(self.start_learning) + '\n')
			f.write('freq_learning   = ' + str(self.freq_learning) + '\n')
			f.write('learning_rate   = ' + str(self.learning_rate) + '\n')
			f.write('gamma           = ' + str(self.gamma) + '\n')
			f.write('epsilon_decay   = ' + str(self.epsilon_decay) + '\n')
			f.write('epsilon_start   = ' + str(self.epsilon_start) + '\n')
			f.write('epsilon_end     = ' + str(self.epsilon_end) + '\n')


# Hyperparameters used for the training Rainbow 
# memory_capacity        = 150000
# num_steps              = 100000
# batch_size             = 32
# target_update          = 32000
# start_learning         = 50000
# freq_learning          = 1
# learning_rate          = 0.0000625
# adam exploration       = 0.00015
# gamma                  = 0.99
# epsilon_decay          = 200000
# epsilon_start          = 1
# epsilon_end            = 0.1
# noisy nets params      = 0.5
# priortization exponant = 0.5
# priortization sampling = 0.4 -> 1.0
# multi-step returns     = 3
# distributional atoms   = 51
# distributional vmin    = -10
# distributional vmax    = 10
# architecture           = canonical


class ConfigRainbow():


	def __init__(self,
				memory_capacity=1000000,
				num_steps=int(4e6),
				batch_size=32,
				target_update=32000, 
				start_learning=50000,
				freq_learning=4,
				learning_rate=0.0000625,
				adam_exp=0.00015,
				gamma=0.99,
				epsilon_decay=100000,
				epsilon_start=1, 
				epsilon_end=0.001,
				noisy_nets=0.5,
				prior_expo=0.5,
				prior_samp=0.4,
				multi_step=3,
				atoms=51,
				vmin=-10,
				vmax=10, 
				architecture='canonical'):
			   
		self.memory_capacity = memory_capacity
		self.num_steps = num_steps
		self.batch_size = batch_size
		self.target_update = target_update
		self.start_learning = start_learning
		self.freq_learning = freq_learning
		self.learning_rate = learning_rate
		self.adam_exp = adam_exp
		self.gamma = gamma
		self.epsilon_decay = epsilon_decay
		self.epsilon_start = epsilon_start
		self.epsilon_end = epsilon_end
		self.noisy_nets = noisy_nets
		self.prior_expo = prior_expo
		self.prior_samp = prior_samp
		self.multi_step = multi_step
		self.atoms = atoms
		self.vmin = vmin
		self.vmax = vmax
		self.architecture = architecture


	def save_config(self, path, env):
		with open(path, 'a') as f:
			f.write('environment     = ' + env + '\n')
			f.write('memory_capacity = ' + str(self.memory_capacity) + '\n')
			f.write('num_steps       = ' + str(self.num_steps) + '\n')
			f.write('batch_size      = ' + str(self.batch_size) + '\n')
			f.write('target_update   = ' + str(self.target_update) + '\n')
			f.write('start_learning  = ' + str(self.start_learning) + '\n')
			f.write('freq_learning   = ' + str(self.freq_learning) + '\n')
			f.write('learning_rate   = ' + str(self.learning_rate) + '\n')
			f.write('adam_exp        = ' + str(self.adam_exp) + '\n')
			f.write('gamma           = ' + str(self.gamma) + '\n')
			f.write('epsilon_decay   = ' + str(self.epsilon_decay) + '\n')
			f.write('epsilon_start   = ' + str(self.epsilon_start) + '\n')
			f.write('epsilon_end     = ' + str(self.epsilon_end) + '\n')
			f.write('noisy_nets      = ' + str(self.noisy_nets) + '\n')
			f.write('prior_expo      = ' + str(self.prior_expo) + '\n')
			f.write('prior_samp      = ' + str(self.prior_samp) + '\n')
			f.write('multi_step      = ' + str(self.multi_step) + '\n')
			f.write('atoms           = ' + str(self.atoms) + '\n')
			f.write('vmin            = ' + str(self.vmin) + '\n')
			f.write('vmax            = ' + str(self.vmax) + '\n')
			f.write('architecture    = ' + self.architecture + '\n')

