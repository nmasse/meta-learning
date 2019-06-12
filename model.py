### Authors: Nicolas Y. Masse, Gregory D. Grant

# Required packages
import tensorflow as tf
import numpy as np
import pickle
import os, sys, time

# Plotting suite
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model modules
from parameters import par
import stimulus
import time

# Match GPU IDs to nvidia-smi command
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'



class Model:

	def __init__(self, stimulus, reward_data, reward_matrix, mask):

		print('Defining graph...')

		self.stimulus_data	= stimulus
		self.reward_data	= reward_data
		self.reward_matrix	= reward_matrix
		self.time_mask		= mask

		self.var_dict = {}
		self.prev_weights = pickle.load(open(par['saved_weights_fn'],'rb')) if par['load_weights'] else None

		self.run_model()
		self.run_A_model()
		self.optimize()

		print('Graph successfully defined.')


	def multiply(self, x, n_out, name, scope = 'network', train=True):

		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			if not par['load_weights']:
				#self.var_dict[name] = tf.get_variable(name, shape = [x.shape[1], n_out], \
				#	initializer = tf.variance_scaling_initializer(scale = 2.), trainable=train)
				self.var_dict[name] = tf.get_variable(name, shape = [x.shape[1], n_out], \
					initializer = tf.random_uniform_initializer(-0.02, 0.02), trainable=train)
			else:
				self.var_dict[name] = tf.get_variable(name, initializer = self.prev_weights[name], trainable=train)
		return x @ self.var_dict[name]

	def add_bias(self, n_out, name, scope = 'network', train=True):

		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			if not par['load_weights']:
				if name == 'bf' and False:
					self.var_dict[name] = tf.get_variable(name, initializer = 3.*np.ones((1,n_out),dtype=np.float32), trainable=train)
				else:
					self.var_dict[name] = tf.get_variable(name, shape = [1, n_out], initializer = tf.zeros_initializer(), trainable=train)
			else:
				self.var_dict[name] = tf.get_variable(name, initializer = self.prev_weights[name])
		return self.var_dict[name]


	def run_model(self):

		self.h = []
		self.y = []
		self.h_write = []
		self.c = []
		self.stim = []
		self.stim_hat = []
		self.pol_out = []
		self.val_out = []
		self.action  = []
		self.reward  = []
		self.reward_full = []
		self.mask    = []

		# lstm output and internal state
		h = tf.zeros([par['batch_size'], par['n_hidden']], dtype = tf.float32)
		c = tf.zeros([par['batch_size'], par['n_hidden']], dtype = tf.float32)

		# lstm output that will project into striatum
		lstm_output = tf.zeros([par['batch_size'], par['n_lstm_out']], dtype = tf.float32)

		h_read = tf.zeros([par['batch_size'], par['n_hidden']], dtype = tf.float32)
		h_write = tf.zeros([par['batch_size'], par['n_hidden']], dtype = tf.float32)

		reward = tf.zeros([par['batch_size'], par['n_val']])
		action = tf.zeros((par['batch_size'], par['n_output']))
		reward_matrix 	= tf.zeros([par['batch_size'], par['num_reward_types']])

		for i in range(par['trials_per_seq']):
			mask   = tf.ones([par['batch_size'], 1])

			for j in range(par['num_time_steps']):

				t = i*par['num_time_steps'] + j
				stim = mask * self.stimulus_data[t]

				if par['use_striatum']:
					striatal_input  = tf.concat([stim, lstm_output], axis = 1)
					striatal_output = self.read_striatum(striatal_input)
					striatal_output = tf.stop_gradient(striatal_output)
				else:
					striatal_output = tf.zeros([par['batch_size'], 1])


				lstm_input = tf.concat([stim, striatal_output, mask*action, mask*reward_matrix], axis = 1)
				h, c = self.run_lstm(lstm_input, h, c)
				h = tf.layers.dropout(h, rate = par['drop_rate'], training = True)

				lstm_output = tf.nn.relu(self.multiply(h, par['n_lstm_out'], 'Wl') + \
					self.add_bias(par['n_lstm_out'], 'bl'))

				pol_out = self.multiply(h, par['n_output'], 'W_pol') + self.add_bias(par['n_output'], 'b_pol')
				val_out = self.multiply(h, 1, 'W_val') + self.add_bias(1, 'b_val')

				# Compute outputs for action and policy loss
				action_index	= tf.multinomial(pol_out, 1)
				action 			= tf.one_hot(tf.squeeze(action_index), par['n_output'])
				pol_out			= tf.nn.softmax(pol_out, -1) # Note softmax for entropy calculation

				# Check for trial continuation
				# ends if previous reward is non-zero, restarts every new trial in sequence
				if j == 0:
					continue_trial = tf.ones([par['batch_size'], 1])
				else:
					continue_trial= tf.cast(tf.equal(reward, 0.), tf.float32)
				mask 		   *= continue_trial
				reward			= tf.reduce_sum(action*self.reward_data[t,...], axis=-1, keep_dims=True) \
									* mask * self.time_mask[t,:,tf.newaxis]
				reward_matrix	= tf.reduce_sum(action[...,tf.newaxis]*self.reward_matrix[t,...], axis=-2, keep_dims=False) \
									* mask * self.time_mask[t,:,tf.newaxis]

				if par['use_striatum']:
					self.write_fast_weights(mask*y, action, reward_matrix)

				self.reward_full.append(reward)

				# Record outputs
				if i >= par['dead_trials']:
					self.h.append(h)
					self.pol_out.append(pol_out)
					self.val_out.append(val_out)
					self.action.append(action)
					self.reward.append(reward)
					self.mask.append(mask * self.time_mask[t,:,tf.newaxis])

		self.h = tf.stack(self.h, axis=0)
		self.stim = tf.stack(self.stim, axis=0)
		self.pol_out = tf.stack(self.pol_out, axis=0)
		self.val_out = tf.stack(self.val_out, axis=0)
		self.action = tf.stack(self.action, axis=0)
		self.reward = tf.stack(self.reward, axis=0)
		self.reward_full = tf.stack(self.reward_full, axis=0)
		self.mask = tf.stack(self.mask, axis=0)


	def run_lstm(self, x, h, h_read, c):
		""" Compute LSTM state from inputs and vars...
				f : forgetting gate
				i : input gate
				c : cell state
				o : output gate
			...and generate an action from that state. """

		# Iterate LSTM
		N = par['n_hidden']
		f = tf.sigmoid(self.multiply(x, N, 'Wf') + self.multiply(h, N, 'Uf') + self.add_bias(N,'bf'))
		i = tf.sigmoid(self.multiply(x, N, 'Wi') + self.multiply(h, N, 'Ui') + self.add_bias(N,'bi'))
		o = tf.sigmoid(self.multiply(x, N, 'Wo') + self.multiply(h, N, 'Uo')  + self.add_bias(N,'bo'))
		cn = tf.tanh(self.multiply(x, N, 'Wc') + self.multiply(h, N, 'Uc') + self.add_bias(N,'bc'))

		c  = f * c + i * cn
		h  = o * tf.tanh(c)

		return h, c

	def read_fast_weights(self, h):

		# can we think of h as a probability over states?
		value_probe = tf.einsum('ijkm,ij->ikm', self.A, h)
		return tf.sign(tf.reshape(value_probe, [par['batch_size'], par['n_output']*par['num_reward_types']]))

		#value_probe = tf.reshape(value_probe, [par['batch_size'], par['n_output']*par['num_reward_types']])
		#value_probe /= (1e-6 + tf.reduce_sum(value_probe,axis = 1, keepdims=True))
		#return value_probe


	def write_fast_weights(self, h, a, r):

		self.A = self.A + tf.einsum('im,ijk->ijkm', r, tf.einsum('ij,ik->ijk', h, a))


	def read_hopfield_weights(self, H, h):
		# currently not in use

		h_hat = tf.zeros_like(h)
		alpha = 0.5
		for n in range(6):
			h_hat = alpha*h_hat + (1 - alpha) * tf.einsum('ij, jk->ik', h, H) + h
			h_hat = tf.sign(h_hat)

		pred_action = tf.einsum('ij, ijk->ik', h_hat, H_act)

		return pred_action


	def write_hopfield_weights(self, H, h):
		# currently not in use

		hh = tf.einsum('ij,ik->ijk',h,h)

		if par['covariance_method']:
			H += hh/par['n_hopf_stim']
		else:
			h1 = tf.einsum('ij, jk->ijk', h, self.H_mask)
			c = tf.einsum('ijk,ikm->ijm', h1, H)
			H += (hh - c - tf.transpose(c,[0,2,1]))/par['n_hopf_stim']
			H = tf.einsum('ijk,jk->ijk', H, self.H_mask)

		return H


	def optimize(self):
		""" Calculate losses and apply corrections to model """

		# Set up optimizer and required constants
		epsilon = 1e-6
		network_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network')
		opt = tf.train.AdamOptimizer(learning_rate=par['learning_rate'])

		# Get the value outputs of the network, and pad the last time step
		val_out = tf.concat([self.val_out, tf.zeros([1,par['batch_size'],par['n_val']])], axis=0)

		# Determine terminal state of the network
		terminal_state = tf.cast(tf.logical_not(tf.equal(self.reward, tf.constant(0.))), tf.float32)

		# Compute predicted value and the advantage for plugging into the policy loss
		pred_val = self.reward + par['discount_rate']*val_out[1:,:,:]*(1-terminal_state)
		advantage = pred_val - val_out[:-1,:,:]

		# Stop gradients back through action, advantage, and mask
		action_static    = tf.stop_gradient(self.action)
		advantage_static = tf.stop_gradient(advantage)
		mask_static      = tf.stop_gradient(self.mask)
		pred_val_static  = tf.stop_gradient(pred_val)

		# Policy loss
		self.pol_loss = -tf.reduce_mean(mask_static*advantage_static*action_static*tf.log(epsilon + self.pol_out))

		# Value loss
		self.val_loss = 0.5*par['val_cost']*tf.reduce_mean(mask_static*tf.square(val_out[:-1,:,:]-pred_val_static))

		# Entropy loss
		self.ent_loss = -par['entropy_cost']*tf.reduce_mean(tf.reduce_sum(mask_static*self.pol_out*tf.log(epsilon + self.pol_out), axis=2))

		# Collect RL losses
		loss = self.pol_loss + self.val_loss - self.ent_loss

		RL_grads_vars = opt.compute_gradients(loss, var_list = network_vars)
		capped_gvs = []
		for grad, var in RL_grads_vars:
			capped_gvs.append((tf.clip_by_value(grad, -par['grad_clip_val'], par['grad_clip_val']), var))

		self.train = opt.apply_gradients(capped_gvs)


def main(gpu_id=None):

	if gpu_id is not None:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

	print_key_params()

	tf.reset_default_graph()
	x = tf.placeholder(tf.float32, [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['n_input']], 'stim')
	r = tf.placeholder(tf.float32, [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['n_output']], 'reward')
	rm = tf.placeholder(tf.float32, [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['n_output'], par['num_reward_types']], 'reward_matrix')
	m = tf.placeholder(tf.float32, [par['num_time_steps']*par['trials_per_seq'], par['batch_size']], 'mask')

	stim = stimulus_sequence.Stimulus()

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)# if gpu_id == '0' else tf.GPUOptions()

	results_dict = {'reward_list': [], 'novel_reward_list':[]}
	t0 = time.time()

	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			model = Model(x, r, rm, m)

		sess.run(tf.global_variables_initializer())
		reward_list = []

		for i in range(par['n_iters']):

			name, trial_info = stim.generate_trial(novel_tasks = False)

			_, reward, pol_loss, action, h, mask  = \
				sess.run([model.train, model.reward_full, model.pol_loss, model.action, model.h, model.mask], \
				feed_dict={x:trial_info['neural_input'], r:trial_info['reward_data'],\
				rm:trial_info['reward_matrix'], m:trial_info['train_mask']})

			rw = np.reshape(reward, (par['num_time_steps'], par['trials_per_seq'], par['batch_size']),order='F')
			rw = np.sum(rw, axis = 0)
			results_dict['reward_list'].append(rw)

			if i%100 == 0:

				name, trial_info = stim.generate_trial(novel_tasks = True)
				reward_novel = sess.run(model.reward_full, feed_dict={x:trial_info['neural_input'], \
					r:trial_info['reward_data'],rm:trial_info['reward_matrix'], m:trial_info['train_mask']})
				rw0 = np.reshape(reward_novel, (par['num_time_steps'], par['trials_per_seq'], par['batch_size']),order='F')
				rw0 = np.sum(rw0, axis = 0)
				results_dict['novel_reward_list'].append(rw0)

				print('Iter {:>4} | Reward: {:6.3f} | Reward novel: {:6.3f} | Pol. Loss: {:6.3f}'.format(\
					i, np.mean(np.sum(reward, axis=0)), np.mean(np.sum(reward_novel, axis=0)),pol_loss))

				weights = sess.run(model.var_dict)
				results = {'weights': weights, 'results_dict': results_dict}
				pickle.dump(results, open(par['save_fn'],'wb'))


def print_key_params():

	key_params = ['n_hidden', 'l2l', 'feed_sparse', 'learning_rate', 'discount_rate', 'drop_rate',\
		'entropy_cost', 'val_cost', 'trials_per_seq', 'dead_trials','batch_size','task_list',\
		'load_encoder', 'train_encoder', 'grad_clip_val']

	print('Key Parameters:\n'+'-'*60)
	for k in key_params:
		print(k.ljust(30), ':', par[k])
	print('-'*60 + '\n')


if __name__ == '__main__':
	try:
		if len(sys.argv) > 1:
			main(sys.argv[1])
		else:
			main()
	except KeyboardInterrupt:
		quit('Quit by KeyboardInterrupt.')
