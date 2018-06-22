import numpy as np
import tensorflow as tf
import os

print("--> Loading parameters...")

"""
Independent parameters
"""

par = {
    # Setup parameters
    'save_dir'              : './',
    #'conv_weight_fn'        : 'C:\\Users\\Krithika\\Documents\\RNNs\\learning_to_learn\\conv_weights.pkl',
    #'conv_weight_fn'        : '/home/bpeysakhovich/Documents/rnn_modeling/learning_to_learn/conv_weights.pkl',
    'conv_weight_fn'        : '/home/masse/learning_to_learn/conv_weights.pkl',
    #'conv_weight_fn'        : 'C:\\Users\\Freedmanlab\\barbara\\learning_to_learn\\conv_weights.pkl',
    'analyze_model'         : True,

    # Network configuration
    'synapse_config'        : None, # Full is 'std_stf'
    'exc_inh_prop'          : 0.8,       # Literature 0.8, for EI off 1
    'var_delay'             : False,
    'LSTM'                  : True,
    'synapse_config'        : 'std_stf', # 'std_stf' or None

    # Synthetic data gpu_options
    'synthetic_data'        : False,
    'synthetic_size'         : 20,

    # Network shape
    'n_input'               : [2048, 1000],
    'n_hidden'              : 100,
    'n_pol'                 : 3,
    'n_val'                 : 1,
    'include_ff_layer'      : False,

    # Timings and rates
    'dt'                    : 20,
    'learning_rate'         : 5e-3,
    'membrane_time_constant': 100,
    'discount_rate'         : 0.,

    # Variance values
    'clip_max_grad_val'     : 20,
    'noise_in_sd'           : 0.0,
    'noise_rnn_sd'          : 0.05,
    'drop_keep_pct'         : 0.8,

    # Cost parameters
    'spike_cost'            : 1e-6,
    'entropy_cost'          : 0.005,

    # Synaptic plasticity specs
    'tau_fast'              : 200,
    'tau_slow'              : 1500,
    'U_stf'                 : 0.15,
    'U_std'                 : 0.45,

    # Training specs
    'batch_size'            : 1024,
    'num_iterations'        : 40000,
    'iters_between_outputs' : 100,
    'trials_per_sequence'   : 4,
    'trials_per_grad_update': 1,

    # Task specs
    'trial_type'            : 'task1',
    'switch_every_ep'       : False,
    'iters_before_im_switch': 1,

    # Reward parameters
    'fix_break_penalty'     : -1.,
    'wrong_choice_penalty'  : -0.01,
    'correct_choice_reward' : 1.,

    # Parameters for convolutional layer
    'conv_filters'          : [16,16,32,32],
    'kernel_size'           : [3, 3],
    'pool_size'             : [2,2],
    'stride'               : 1,

    # Save paths
    'save_fn'               : 'model_results.pkl',

}


"""
Dependent parameters
"""

def update_parameters(updates):
    """
    Takes a list of strings and values for updating parameters in the parameter dictionary
    Example: updates = [(key, val), (key, val)]
    """
    print('Updating parameters...')
    for key, val in updates.items():
        par[key] = val
        #print('Updating ', key)

    update_trial_params()
    update_dependencies()

def update_trial_params():

    """
    Update all the trial parameters given trial_type
    """

    if par['trial_type'] == 'task1':
        par['ITI'] = 200
        par['fix'] = 200
        par['stim'] = 200
        par['delay'] = 0
        par['resp'] = 400

        par['ITI'] = 60
        par['fix'] = 40
        par['stim'] = 40
        par['delay'] = 0
        par['resp'] = 100


        par['trial_length'] = par['ITI'] + par['fix'] + par['stim'] + par['delay'] + par['resp']
        par['n_time_steps'] = par['trial_length']//par['dt']
        par['sequence_time_steps'] = par['n_time_steps']*par['trials_per_sequence']

    else:
        print(par['trial_type'], ' not a recognized trial type')
        quit()


def update_dependencies():
    """
    Updates all parameter dependencies
    """


    if par['synthetic_data']:
        par['n_input'] = [par['synthetic_size']]
        par['include_ff_layer'] = False
        print('Using synthetic data')

    # If num_inh_units is set > 0, then neurons can be either excitatory or
    # inihibitory; is num_inh_units = 0, then the weights projecting from
    # a single neuron can be a mixture of excitatory or inhibitory
    if par['exc_inh_prop'] < 1 and not par['LSTM']:
        par['EI'] = True
    elif par['LSTM']:
        print('Using LSTM networks; setting to EI to False')
        par['EI'] = False
        par['exc_inh_prop'] = 1.
        par['synapse_config'] = False


    par['num_exc_units'] = int(np.round(par['n_hidden']*par['exc_inh_prop']))
    par['num_inh_units'] = par['n_hidden'] - par['num_exc_units']

    par['EI_list'] = np.ones(par['n_hidden'], dtype=np.float32)
    par['EI_list'][-par['num_inh_units']:] = -1.

    par['EI_matrix'] = np.diag(par['EI_list'])

    # Membrane time constant of RNN neurons
    par['alpha'] = np.float32(par['dt'])/par['membrane_time_constant']
    # The standard deviation of the Gaussian noise added to each RNN neuron
    # at each time step
    par['noise_rnn'] = np.sqrt(2*par['alpha'])*par['noise_rnn_sd']
    par['noise_in'] = np.sqrt(2/par['alpha'])*par['noise_in_sd'] # since term will be multiplied by par['alpha']

    # The time step in seconds
    par['dt_sec'] = par['dt']/1000


    ####################################################################
    ### Setting up assorted intial weights, biases, and other values ###
    ####################################################################

    par['h_init'] = 0.1*np.ones((par['batch_size'], par['n_hidden']), dtype=np.float32)
    par['c_init'] = 0.1*np.ones((par['batch_size'], par['n_hidden']), dtype=np.float32) # cell state (only used for LSTM units)

    # Initialize input weights
    c = 0.05
    if par['include_ff_layer']:
        par['W_in0_init'] =  c*np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_input'][0], par['n_input'][1]]))
        par['b_in0_init'] = np.zeros((1, par['n_input'][1]), dtype = np.float32)
        par['W_in1_init'] =  c*np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_input'][1], par['n_hidden']]))

    else:
        #par['W_in1_init'] =  c*np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_hidden'], par['n_input'][0]]))
        par['W_in1_init'] =  c*np.float32(np.random.uniform(-c, c, size = [par['n_input'][0], par['n_hidden']]))

        # Testing whether sparser connectivity helps.... []
        connectivity_prob = 1.
        q = np.float32(np.random.rand(par['n_input'][0], par['n_hidden']) < connectivity_prob)
        print(type(q))
        print(q.shape)
        par['W_in1_init'] *= q

    if par['EI']:
        par['W_rnn_init'] =  c*np.float32(np.random.gamma(shape=0.25, scale=1.0, size = [par['n_hidden'], par['n_hidden']]))
        par['w_rnn_mask'] = np.ones((par['n_hidden'], par['n_hidden']), dtype=np.float32) - np.eye(par['n_hidden'])
        par['W_rnn_init'] *= par['w_rnn_mask']
    else:
        par['W_rnn_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_hidden']]))
        par['w_rnn_mask'] = np.ones((par['n_hidden'], par['n_hidden']), dtype=np.float32)

    par['W_pol_out_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_pol']]))
    par['W_val_out_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_val']]))

    par['b_pol_out_init'] = np.zeros((1, par['n_pol']), dtype = np.float32)
    par['b_val_out_init'] = np.zeros((1, par['n_val']), dtype = np.float32)


    if par['LSTM']:
        c = 0.05
        par['Wf_init'] =  c*np.float32(np.random.uniform(-c, c, size = [par['n_input'][0], par['n_hidden']]))
        par['Wi_init'] =  c*np.float32(np.random.uniform(-c, c, size = [par['n_input'][0], par['n_hidden']]))
        par['Wo_init'] =  c*np.float32(np.random.uniform(-c, c, size = [par['n_input'][0], par['n_hidden']]))
        par['Wc_init'] =  c*np.float32(np.random.uniform(-c, c, size = [par['n_input'][0], par['n_hidden']]))

        par['Uf_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_hidden']]))
        par['Ui_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_hidden']]))
        par['Uo_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_hidden']]))
        par['Uc_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_hidden'], par['n_hidden']]))

        par['Wf_reward_pos_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_val'], par['n_hidden']]))
        par['Wi_reward_pos_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_val'], par['n_hidden']]))
        par['Wo_reward_pos_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_val'], par['n_hidden']]))
        par['Wc_reward_pos_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_val'], par['n_hidden']]))

        par['Wf_reward_neg_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_val'], par['n_hidden']]))
        par['Wi_reward_neg_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_val'], par['n_hidden']]))
        par['Wo_reward_neg_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_val'], par['n_hidden']]))
        par['Wc_reward_neg_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_val'], par['n_hidden']]))

        par['Wf_action_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_pol'], par['n_hidden']]))
        par['Wi_action_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_pol'], par['n_hidden']]))
        par['Wo_action_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_pol'], par['n_hidden']]))
        par['Wc_action_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_pol'], par['n_hidden']]))

        par['bf_init'] = np.zeros((1, par['n_hidden']), dtype = np.float32)
        par['bi_init'] = np.zeros((1, par['n_hidden']), dtype = np.float32)
        par['bo_init'] = np.zeros((1, par['n_hidden']), dtype = np.float32)
        par['bc_init'] = np.zeros((1, par['n_hidden']), dtype = np.float32)


    else:
        par['W_reward_pos_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_val'], par['n_hidden']]))
        par['W_reward_neg_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_val'], par['n_hidden']]))
        par['W_action_init'] =  np.float32(np.random.uniform(-c, c, size = [par['n_pol'], par['n_hidden']]))

        par['b_rnn_init'] = np.zeros((1, par['n_hidden']), dtype = np.float32)


    """
    Setting up synaptic parameters
    0 = static
    1 = facilitating
    2 = depressing
    """
    par['synapse_type'] = np.zeros(par['n_hidden'], dtype=np.int8)

    # only facilitating synapses
    if par['synapse_config'] == 'stf':
        par['synapse_type'] = np.ones(par['n_hidden'], dtype=np.int8)

    # only depressing synapses
    elif par['synapse_config'] == 'std':
        par['synapse_type'] = 2*np.ones(par['n_hidden'], dtype=np.int8)

    # even numbers facilitating, odd numbers depressing
    elif par['synapse_config'] == 'std_stf':
        par['synapse_tparaype'] = np.ones(par['n_hidden'], dtype=np.int8)
        par['ind'] = range(1,par['n_hidden'],2)
        par['synapse_type'][par['ind']] = 2

    par['alpha_stf'] = np.ones((1, par['n_hidden']), dtype=np.float32)
    par['alpha_std'] = np.ones((1, par['n_hidden']), dtype=np.float32)
    par['U'] = np.ones((1, par['n_hidden']), dtype=np.float32)

    # initial synaptic values
    par['syn_x_init'] = np.zeros((par['batch_size'], par['n_hidden']), dtype=np.float32)
    par['syn_u_init'] = np.zeros((par['batch_size'], par['n_hidden']), dtype=np.float32)

    for i in range(par['n_hidden']):
        if par['synapse_type'][i] == 1:
            par['alpha_stf'][0, i] = par['dt']/par['tau_slow']
            par['alpha_std'][0, i] = par['dt']/par['tau_fast']
            par['U'][0, i] = 0.15
            par['syn_x_init'][:, i] = 1
            par['syn_u_init'][:, i] = par['U'][0, i]

        elif par['synapse_type'][i] == 2:
            par['alpha_stf'][0, i] = par['dt']/par['tau_fast']
            par['alpha_std'][0, i] = par['dt']/par['tau_slow']
            par['U'][0, i] = 0.45
            par['syn_x_init'][:, i] = 1
            par['syn_u_init'][:, i] = par['U'][0, i]




def spectral_radius(A):

    return np.max(abs(np.linalg.eigvals(A)))

update_trial_params()
update_dependencies()

print("--> Parameters successfully loaded.\n")
