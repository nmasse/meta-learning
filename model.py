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
		with tf.variable_scope('network', reuse=tf.AUTO_REUSE):
			self.var_dict['h_init'] = tf.get_variable('h_init', shape = [1, par['n_hidden']], \
				initializer = tf.random_uniform_initializer(-0.02, 0.02))

		#self.prev_weights = None
		self.prev_weights = pickle.load(open('./saved_encoder_weights_80tasks.pkl','rb'))
		self.run_model()
		self.run_A_model()
		self.optimize()

		print('Graph successfully defined.')



	def multiply(self, x, n_out, name, scope = 'network', train=True, load_prev=False):

		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			if not load_prev:
				#self.var_dict[name] = tf.get_variable(name, shape = [x.shape[1], n_out], \
				#	initializer = tf.variance_scaling_initializer(scale = 2.), trainable=train)
				self.var_dict[name] = tf.get_variable(name, shape = [x.shape[1], n_out], \
					initializer = tf.random_uniform_initializer(-0.02, 0.02), trainable=train)
			else:
				self.var_dict[name] = tf.get_variable(name, initializer = self.prev_weights[name], trainable=train)
		return x @ self.var_dict[name]

	def add_bias(self, n_out, name, scope = 'network', train=True, load_prev=False):

		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			if not load_prev:
				if name == 'bf' and False:
					self.var_dict[name] = tf.get_variable(name, initializer = 3.*np.ones((1,n_out),dtype=np.float32), trainable=train)
				else:
					self.var_dict[name] = tf.get_variable(name, shape = [1, n_out], initializer = tf.zeros_initializer(), trainable=train)
			else:
				self.var_dict[name] = tf.get_variable(name, initializer = self.prev_weights[name])
		return self.var_dict[name]


	def run_A_model(self):

		#self.A = tf.zeros([par['batch_size'], par['n_latent'], par['n_output'], par['num_reward_types']], dtype = tf.float32)

		#self.task_latent = tf.stop_gradient(tf.reshape(tf.sign(self.A_final), [par['batch_size'], -1]))
		self.task_latent = tf.stop_gradient(tf.reshape(tf.sign(self.A_final), [par['batch_size'], -1]))

		self.task_hidden = tf.nn.relu(self.multiply(self.task_latent, par['n_task_latent'], 'Wt0', scope='task_ff')# \
			+ self.add_bias(par['n_task_latent'], 'bt0', scope='task_ff'))
		#self.task_latent_hat = self.multiply(self.task_hidden, par['n_latent']*par['n_output']*par['num_reward_types'],'Wt1', scope='task_ff')# \
		#	+ self.add_bias(par['n_latent']*par['n_output']*par['num_reward_types'], 'bt1', scope='task_ff')

		#self.task_hidden = tf.nn.relu(self.multiply(self.task_latent, par['n_task_latent'], 'Wt0', scope='task_ff'))
		self.task_latent_hat = tf.nn.relu(self.multiply(self.task_hidden, par['n_latent']*par['n_output']*par['num_reward_types'],'Wt1', scope='task_ff'))
		#self.task_latent_hat = tf.nn.relu(self.multiply(self.task_hidden, par['n_latent']*par['n_output']*1,'Wt1', scope='task_ff'))

		print(self.task_latent, self.task_latent_hat)


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

		#h = tf.zeros([par['batch_size'], par['n_hidden']], dtype = tf.float32)
		h = self.var_dict['h_init']
		h_read = tf.zeros([par['batch_size'], par['n_hidden']], dtype = tf.float32)
		h_write = tf.zeros([par['batch_size'], par['n_hidden']], dtype = tf.float32)
		c = tf.zeros([par['batch_size'], par['n_hidden']], dtype = tf.float32)
		#A = tf.zeros([par['batch_size'], par['n_hidden'], par['n_hidden']], dtype = tf.float32)
		self.A = tf.zeros([par['batch_size'], par['n_latent'], par['n_output'], par['num_reward_types']], dtype = tf.float32)
		self.A_final = tf.Variable(np.zeros([par['batch_size'], par['n_latent'], par['n_output'], par['num_reward_types']], dtype = np.float32), \
			name = 'A', trainable = False)
		reward = tf.zeros([par['batch_size'], par['n_val']])
		action = tf.zeros((par['batch_size'], par['n_output']))
		reward_matrix 	= tf.zeros([par['batch_size'], par['num_reward_types']])


		for i in range(par['trials_per_seq']):
			mask   = tf.ones([par['batch_size'], 1])


			for j in range(par['num_time_steps']):

				# Make two possible actions and values for the network to pursue
				# by way of the LSTM-based cortex module and the associative
				# network hippocampus module
				t = i*par['num_time_steps'] + j
				#stim = tf.cast(self.stimulus_data[t] > par['tuning_height']-0.01, tf.float32)
				stim = self.stimulus_data[t]
				y = tf.nn.relu(self.multiply(stim, par['n_latent'], 'W0', scope='ff', train=par['train_encoder'],load_prev=par['load_encoder']) \
						+ self.add_bias(par['n_latent'], 'b0', scope='ff', train=par['train_encoder'],load_prev=par['load_encoder']))

				#top_k, _ = tf.nn.top_k(y, k = 11)
				#top_cond = y >= top_k[:,-2:-1]
				#y = tf.where(top_cond, tf.ones(y.shape), tf.zeros(y.shape))

				stim_hat = self.multiply(y,par['n_input'],'W1', scope='ff', train=par['train_encoder'],load_prev=par['load_encoder'])

				if par['l2l']:
					h_read = tf.zeros([par['batch_size'], 1])
				else:
					h_read = self.read_fast_weights(mask*y)
					h_read = tf.stop_gradient(h_read)

				#h, c = self.cortex_lstm(mask*self.stimulus_data[t], h, h_read, c)
				#lstm_input = tf.concat([h_read, action, reward_matrix, self.cue[t,:,:]], axis = 1)
				lstm_input = tf.concat([h_read, mask*action, mask*reward_matrix], axis = 1)
				#lstm_input = tf.concat([action, reward_matrix], axis = 1)
				if par['feed_sparse']:
					h, c = self.cortex_lstm(mask*y, h, lstm_input, c)
				else:
					h, c = self.cortex_lstm(mask*stim, h, lstm_input, c)

				h = tf.layers.dropout(h, rate = par['drop_rate'], training = True)
				#h += tf.random.normal(h.shape, 0, par['noise_rnn'])

				#h_read, h_write, A = self.fast_weights(h, A)
				#h_read *= 0
				load_prev = False
				pol_out = self.multiply(h, par['n_output'], 'W_pol',load_prev=load_prev) + self.add_bias(par['n_output'], 'b_pol',load_prev=load_prev)
				val_out = self.multiply(h, 1, 'W_val',load_prev=load_prev) + self.add_bias(1, 'b_val',load_prev=load_prev)

				# Compute outputs for action and policy loss
				action_index	= tf.multinomial(pol_out, 1)
				action 			= tf.one_hot(tf.squeeze(action_index), par['n_output'])
				pol_out			= tf.nn.softmax(pol_out, -1) # Note softmax for entropy calculation

				# Check for trial continuation (ends if previous reward is non-zero)
				if j == 0:
					continue_trial = tf.ones([par['batch_size'], 1])
				else:
					continue_trial= tf.cast(tf.equal(reward, 0.), tf.float32)
				mask 		   *= continue_trial
				reward			= tf.reduce_sum(action*self.reward_data[t,...], axis=-1, keep_dims=True) \
									* mask * self.time_mask[t,:,tf.newaxis]
				reward_matrix	= tf.reduce_sum(action[...,tf.newaxis]*self.reward_matrix[t,...], axis=-2, keep_dims=False) \
									* mask * self.time_mask[t,:,tf.newaxis]

				if not par['l2l']:
					self.write_fast_weights(mask*y, action, reward_matrix)

				self.reward_full.append(reward)

				# Record outputs
				if i >= par['dead_trials']:
					self.h.append(h)
					self.y.append(y)
					self.stim.append(stim)
					self.stim_hat.append(stim_hat)
					#self.h_write.append(h_write)
					#self.c.append(c)
					self.pol_out.append(pol_out)
					self.val_out.append(val_out)
					self.action.append(action)
					self.reward.append(reward)
					#self.target.append(self.target_out[t, ...])
					self.mask.append(mask * self.time_mask[t,:,tf.newaxis])

		self.h = tf.stack(self.h, axis=0)
		self.y = tf.stack(self.y, axis=0)
		self.stim = tf.stack(self.stim, axis=0)
		self.stim_hat = tf.stack(self.stim_hat, axis=0)
		#self.h_write = tf.stack(self.h_write, axis=0)
		#self.c = tf.stack(self.c, axis=0)
		self.pol_out = tf.stack(self.pol_out, axis=0)
		self.val_out = tf.stack(self.val_out, axis=0)
		self.action = tf.stack(self.action, axis=0)
		self.reward = tf.stack(self.reward, axis=0)
		self.reward_full = tf.stack(self.reward_full, axis=0)
		self.mask = tf.stack(self.mask, axis=0)


	def cortex_lstm(self, x, h, h_read, c):
		""" Compute LSTM state from inputs and vars...
				f : forgetting gate
				i : input gate
				c : cell state
				o : output gate
			...and generate an action from that state. """

		# Iterate LSTM
		N = par['n_hidden']
		load_prev = False
		f = tf.sigmoid(self.multiply(x, N, 'Wf',load_prev=load_prev) + self.multiply(h, N, 'Uf',load_prev=load_prev) \
			+ self.multiply(h_read, N, 'Vf',load_prev=load_prev) + self.add_bias(N,'bf',load_prev=load_prev))

		i = tf.sigmoid(self.multiply(x, N, 'Wi',load_prev=load_prev) + self.multiply(h, N, 'Ui',load_prev=load_prev) \
			+ self.multiply(h_read, N, 'Vi',load_prev=load_prev) + self.add_bias(N,'bi',load_prev=load_prev))

		o = tf.sigmoid(self.multiply(x, N, 'Wo',load_prev=load_prev) + self.multiply(h, N, 'Uo',load_prev=load_prev) \
			+ self.multiply(h_read, N, 'Vo',load_prev=load_prev) + self.add_bias(N,'bo',load_prev=load_prev))

		cn = tf.tanh(self.multiply(x, N, 'Wc',load_prev=load_prev) + self.multiply(h, N, 'Uc',load_prev=load_prev) \
			+ self.multiply(h_read, N, 'Vc',load_prev=load_prev) + self.add_bias(N,'bc',load_prev=load_prev))

		#f  = tf.sigmoid(x @ self.var_dict['Wf'] + h @ self.var_dict['Uf'] + h_read @ self.var_dict['Vf'] + self.var_dict['bf'])
		#i  = tf.sigmoid(x @ self.var_dict['Wi'] + h @ self.var_dict['Ui'] + h_read @ self.var_dict['Vi'] + self.var_dict['bi'])
		#o  = tf.sigmoid(x @ self.var_dict['Wo'] + h @ self.var_dict['Uo'] + h_read @ self.var_dict['Vo'] + self.var_dict['bo'])
		#cn = tf.tanh(x @ self.var_dict['Wc'] + h @ self.var_dict['Uc'] + h_read @ self.var_dict['Vc'] + self.var_dict['bc'])
		c  = f * c + i * cn
		h  = o * tf.tanh(c)

		# Return action, hidden state, and cell state
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

	def write_fast_weights_v2(self, h, a, r):

		s = tf.concat([h, par['action_multiplier']*a], axis = 1)

		s0 = self.multiply(s, par['n_output'], 'W_striatum',load_prev=load_prev) # + self.add_bias(par['n_output'], 'b_pol',load_prev=load_prev)


	def read_H_weights(self, H, h):

		h_hat = tf.zeros_like(h)
		alpha = 0.5
		for n in range(6):
			h_hat = alpha*h_hat + (1 - alpha) * tf.einsum('ij, jk->ik', h, H) + h
			h_hat = tf.sign(h_hat)

		pred_action = tf.einsum('ij, ijk->ik', h_hat, H_act)

		return pred_action


	def write_H_weights(self, H, h):

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
		ff_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ff')
		task_ff_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='task_ff')
		opt = tf.train.AdamOptimizer(learning_rate=par['learning_rate'])

		# Correct time mask shape
		#self.time_mask = self.time_mask[...,tf.newaxis]

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

		# Multiply masks together
		full_mask        = mask_static#*self.time_mask#*par['sequence_mask']

		# Policy loss
		self.pol_loss = -tf.reduce_mean(mask_static*advantage_static*action_static*tf.log(epsilon + self.pol_out))

		# Value loss
		self.val_loss = 0.5*par['val_cost']*tf.reduce_mean(mask_static*tf.square(val_out[:-1,:,:]-pred_val_static))

		# Entropy loss
		self.ent_loss = -par['entropy_cost']*tf.reduce_mean(tf.reduce_sum(mask_static*self.pol_out*tf.log(epsilon + self.pol_out), axis=2))

		# Collect RL losses
		loss = self.pol_loss + self.val_loss - self.ent_loss


		train_ops = []

		RL_grads_vars = opt.compute_gradients(loss, var_list = network_vars)
		capped_gvs = []
		for grad, var in RL_grads_vars:
			capped_gvs.append((tf.clip_by_value(grad, -par['grad_clip_val'], par['grad_clip_val']), var))

		train_ops.append(opt.apply_gradients(capped_gvs))

		#train_ops.append(opt.minimize(loss, var_list = network_vars))

		y = tf.reshape(self.y, [-1, par['n_latent']])
		self.reconstruction_loss = tf.reduce_mean(tf.square(self.stim - self.stim_hat))
		self.weight_loss = tf.reduce_mean(tf.abs(self.var_dict['W0'])) + tf.reduce_mean(tf.abs(self.var_dict['W1']))
		self.sparsity_loss = tf.reduce_mean(tf.transpose(y) @ y)
		ff_loss = self.reconstruction_loss + par['sparsity_cost']*self.sparsity_loss \
			+ par['weight_cost']*self.weight_loss

		if par['train_encoder']:
			train_ops.append(opt.minimize(ff_loss, var_list = ff_vars))

		train_ops.append(tf.assign(self.A_final, self.A))

		self.train = tf.group(*train_ops)



		print('self.task_hidden', self.task_hidden)
		y = tf.reshape(self.task_hidden, [-1, par['n_task_latent']])
		self.task_reconstruction_loss = tf.reduce_mean(tf.square(self.task_latent - self.task_latent_hat))
		self.task_weight_loss = tf.reduce_mean(tf.abs(self.var_dict['Wt0'])) + tf.reduce_mean(tf.abs(self.var_dict['Wt1']))
		mask = np.ones((par['n_task_latent'],par['n_task_latent']),dtype=np.float32) - np.eye(par['n_task_latent'],dtype=np.float32)
		M = tf.constant(mask)
		#self.task_sparsity_loss = tf.reduce_mean(M*(tf.transpose(y) @ y))
		self.task_sparsity_loss = tf.reduce_mean((tf.transpose(y) @ y))
		task_loss = self.task_reconstruction_loss + par['sparsity_cost']*self.task_sparsity_loss \
			+ par['weight_cost']*self.task_weight_loss
		self.train_task_latent = opt.minimize(task_loss, var_list = task_ff_vars)





def main(gpu_id=None):

	if gpu_id is not None:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

	print_key_params()

	tf.reset_default_graph()
	x = tf.placeholder(tf.float32, [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['n_input']], 'stim')
	r = tf.placeholder(tf.float32, [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['n_output']], 'reward')
	rm = tf.placeholder(tf.float32, [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], par['n_output'], par['num_reward_types']], 'reward_matrix')
	m = tf.placeholder(tf.float32, [par['num_time_steps']*par['trials_per_seq'], par['batch_size']], 'mask')
	#c = tf.placeholder(tf.float32, [par['num_time_steps']*par['trials_per_seq'], par['batch_size'], 2], 'cue')

	#cue = np.zeros((par['num_time_steps']*par['trials_per_seq'], par['batch_size'], 2), dtype=np.float32)
	#cue[:par['num_time_steps']*par['dead_trials'], :, 0] = 1.
	#cue[par['num_time_steps']*par['dead_trials']:, :, 1] = 1.

	stim = stimulus_sequence.Stimulus()

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)# if gpu_id == '0' else tf.GPUOptions()

	results_dict = {'A_hist': [], 'reward_list': [], 'task_latent':[], 'task_latent_hat':[], 'task_hidden':[], 'novel_reward_list':[]}
	t0 = time.time()

	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			model = Model(x, r, rm, m)

		sess.run(tf.global_variables_initializer())
		reward_list = []

		print('\nGate value of 0 indicates using hippocampus (associative network).')
		print('Gate value of 1 indicates using cortex (LSTM).\n')

		for i in range(par['n_iters']):

			if i%100 == 0:
				name, trial_info = stim.generate_trial(fixed_task_num=None,task_sequence = True)
			else:
				name, trial_info = stim.generate_trial(fixed_task_num=None,task_sequence = False)
			#trial_info['train_mask'][:par['dead_trials']*par['num_time_steps'], :] = 0.


			_, reward, pol_loss, action, h, mask, y  = \
				sess.run([model.train, model.reward_full, model.pol_loss, model.action, model.h, model.mask, model.y], \
				feed_dict={x:trial_info['neural_input'], r:trial_info['reward_data'],\
				rm:trial_info['reward_matrix'], m:trial_info['train_mask']})


			""""
			A = sess.run(model.A_final)
			for _ in range(80000):
				_, t_recon, t_sparsity, task_latent, task_latent_hat, task_hidden  = \
					sess.run([model.train_task_latent, model.task_reconstruction_loss, model.task_sparsity_loss, model.task_latent, \
					model.task_latent_hat, model.task_hidden])
			"""

			rw = np.reshape(reward, (par['num_time_steps'], par['trials_per_seq'], par['batch_size']),order='F')
			rw = np.sum(rw, axis = 0)
			results_dict['reward_list'].append(rw)

			if i%100 == 0:
				"""
				results_dict['A_hist'].append(A)
				results_dict['task_latent'].append(task_latent)
				results_dict['task_latent_hat'].append(task_latent_hat)
				results_dict['task_hidden'].append(task_hidden)

				if len(results_dict['A_hist']) > 30:
					results_dict['A_hist'] = results_dict['A_hist'][1:]
					results_dict['task_latent'] = results_dict['task_latent'][1:]
					results_dict['task_latent_hat'] = results_dict['task_latent_hat'][1:]
					results_dict['task_hidden'] = results_dict['task_hidden'][1:]
				"""


				"""
				#print('Training Iter {:>4} | Reward: {:6.3f} | Pol. Loss: {:6.3f} | Mean h: {:6.3f} | y>0: {:6.3f} | task rl: {:6.3f} | task spars: {:6.3f}  '.format(\
				#	i, np.mean(np.sum(reward, axis=0)), pol_loss, np.mean(h), np.mean(y>0), 1000.*t_recon, 1000.*t_sparsity ))
				"""
				name, trial_info = stim.generate_trial(fixed_task_num=0)
				reward0 = sess.run(model.reward_full, feed_dict={x:trial_info['neural_input'], \
					r:trial_info['reward_data'],rm:trial_info['reward_matrix'], m:trial_info['train_mask']})
				rw0 = np.reshape(reward0, (par['num_time_steps'], par['trials_per_seq'], par['batch_size']),order='F')
				rw0 = np.sum(rw0, axis = 0)
				results_dict['novel_reward_list'].append(rw0)


				print('Training Iter {:>4} | Reward: {:6.3f} | Pol. Loss: {:6.3f} | Mean h: {:6.3f} | y>0: {:6.3f}'.format(\
					i, np.mean(np.sum(reward, axis=0)), pol_loss, np.mean(h), np.mean(y>0)))



				print('Time ', time.time() -t0, ' Testing Iter {:>4} | Reward: {:6.3f} '.format(i, np.mean(np.sum(reward0, axis=0))))
				t0 = time.time()
				weights = sess.run(model.var_dict)
				pickle.dump(results_dict, open('./results_hist_60tasks_fast_v1.pkl','wb'))
				#pickle.dump(weights, open('./saved_weights_l2rl.pkl','wb'))
				pickle.dump(weights, open('./saved_weights_60tasks_fast_v1.pkl','wb'))






	print('Model complete.\n')

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
