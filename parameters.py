### Authors: Nicolas Y. Masse, Gregory D. Grant

import numpy as np
import pickle

print('\n--> Loading parameters...')

##############################
### Independent parameters ###
##############################

par = {
	# Setup parameters
	'savedir'				: './savedir/',
	'save_fn'				: '80_tasks_v0',
	'train'					: True,
	'save_weights'			: True,
	'load_weights'			: False,
	'learning_method'		: 'RL', # 'RL' or 'SL'
	'LSTM_init'				: 0.02,
	'w_init'				: 0.02,

	# Network shape
	'num_motion_tuned'		: 32,
	'num_fix_tuned'			: 1,
	'num_rule_tuned'		: 0,
	'n_hidden'				: 256,
	'n_lstm_out'			: 64,
	'n_striatum'			: 256,
	'striatum_top_k'		: 1,
	'n_val'					: 1,
	'top_down'				: True,
	'use_striatum'			: True,


	# Timings and rates
	'learning_rate'			: 1e-3,
	'drop_rate'				: 0.2,
	'grad_clip_val'			: 5,

	# Variance values
	'input_mean'			: 0.0,
	'noise_in'				: 0.,
	'noise_rnn'				: 0.0,
	'n_filters'				: 1,

	# Task specs
	'task'					: 'multistim',
	'trial_length'			: 1300,
	'mask_duration'			: 0,
	'dead_time'				: 100,
	'dt'					: 100,
	'trials_per_seq'		: 5,
	'train_task_list'		: [a for a in range(1,61)],
	'test_task_list'		: [0],
	'dead_trials'			: 2,

	# RL parameters
	'fix_break_penalty'     : -1.,
	'wrong_choice_penalty'  : -0.01,
	'correct_choice_reward' : 1.,
	'discount_rate'         : 0.,
	'n_unique_vals'			: 4,

	# Tuning function data
	'num_motion_dirs'		: 8,
	'tuning_height'			: 1.0,

	# Cost values
	'sparsity_cost'         : 5e-2, # was 5e-5
	'rec_cost'				: 1.,  # was 1e-2
	'weight_cost'           : 1e-9,  # was 1e-9
	'entropy_cost'          : 0.01,
	'val_cost'              : 0.01,

	# Training specs
	'batch_size'			: 180,
	'n_iters'				: 25000000,		# 1500 to train straight cortex

	'load_encoder'			: True,
	'train_encoder'			: False,

}


############################
### Dependent parameters ###
############################

def update_parameters(updates, verbose=True, update_deps=True):
	""" Updates parameters based on a provided
		dictionary, then updates dependencies """

	par.update(updates)
	if verbose:
		print('Updating parameters:')
		for (key, val) in updates.items():
			print('{:<24} --> {}'.format(key, val))

	if update_deps:
		update_dependencies()



def load_encoder_weights():

	fn = './savedir/80_tasks_v0_model_weights.pkl'
	results = pickle.load(open(fn, 'rb'))
	print('Weight keys ', results['weights'].keys())
	par['W0_init'] = results['weights']['W0']
	par['W1_init'] = results['weights']['W1']
	par['b0_init'] = results['weights']['b0']

def load_all_weights():

	fn = './savedir/80_tasks_v1_model_weights.pkl'
	results = pickle.load(open(fn, 'rb'))
	print('Weight keys ', results['weights'].keys())
	for k, v in results['weights'].items():
		par[k + '_init'] = v



def update_weights(var_dict):

	print('Setting weight values manually; disabling training and weight saving.')
	par['train'] = False
	par['save_weights'] = False
	for key, val in var_dict['weights'].items():
		print(key, val.shape)
		if not 'A_' in key:
			par[key+'_init'] = val


def update_dependencies():
	""" Updates all parameter dependencies """

	# Reward map, for hippocampus reward one-hot conversion
	par['reward_map'] = {
		par['fix_break_penalty']		: 0,
		par['wrong_choice_penalty']		: 1,
		0.								: 2,
		par['correct_choice_reward']	: 3
	}

	par['num_reward_types'] = len(par['reward_map'].keys())
	par['reward_map_matrix'] = np.zeros([par['num_reward_types'],1]).astype(np.float32)
	for key, val in par['reward_map'].items():
		par['reward_map_matrix'][val,:] = key

	# Set input and output sizes
	par['n_input']  = par['num_motion_tuned'] + par['num_fix_tuned'] + par['num_rule_tuned']
	par['n_pol'] = par['num_motion_dirs'] + 1
	par['n_output'] = par['n_pol']

	# Set trial step length
	par['num_time_steps'] = par['trial_length']//par['dt']


update_dependencies()
print('--> Parameters successfully loaded.\n')
