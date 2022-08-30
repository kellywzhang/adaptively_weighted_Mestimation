import math
import os
import pickle
import argparse
import json

import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize
from scipy.special import expit
import matplotlib.pyplot as plt
from scipy.special import expit, logit

parser = argparse.ArgumentParser()
parser.add_argument('--strategy', type=str, default='TS', \
        choices=['TS', 'independent'])
parser.add_argument('--N', type=int, default=5000,
        help='Number of monte carlo simulations')
parser.add_argument('--T', type=int, default=100,
        help='Number of timesteps')
parser.add_argument('--d', type=int, default=3,
        help='Number of dimension')
parser.add_argument('--params', type=str, default='1,1,1,0,0,0',
        help='Expected rewards for each arm')
parser.add_argument('--var', type=float, default=1.0,
        help='Reward variance (sigma^2)')
parser.add_argument('--clipping', type=float, default=0,
        help='Clipping value in [0, 1)')
parser.add_argument('--reward', type=str, default='normal', \
        choices=['bernoulli', 'normal', 'poisson', 'uniform', 't-dist'])
parser.add_argument('--no_zeros', type=int, default=0) 
parser.add_argument('--Z_dist', type=str, default='uniform', \
        help='Context distribution', \
         choices=['normal', 'uniform', 'bernoulli', 'ones'] )

parser.add_argument('--path', type=str, default='./context_simulations')

args = parser.parse_args()
print(vars(args))
assert args.clipping < 1
assert args.clipping > 0

nonsave_args = ['path']
if not args.no_zeros:
    nonsave_args.append( 'no_zeros' )
save_args = ['{}={}'.format(key, val) for key, val in vars(args).items() if key not in nonsave_args]
save_str = '_'.join(save_args)

path = args.path
if not os.path.isdir(path):
    os.mkdir(path)
save_f = os.path.join(path, save_str)
if not os.path.isdir(save_f):
    os.mkdir(save_f)

# Save arguments
with open(save_f+'/args.json', 'w') as f:
    json.dump(vars(args), f, indent=4)

# Process expected rewards for each arm in stationary and non-stationary cases
params_str = args.params.replace('n', '-', args.d*2)
params = np.array( [float(x) for x in params_str.split(',')] )
assert len(params) % 2 == 0
assert len(params)/2 == args.d

print('Parameters:', params)

def get_pis_TS(states, post_invcov1, post_invcov0, total_reward1, total_reward0, clipping=0):
    post_cov1 = np.linalg.inv(post_invcov1)
    post_cov0 = np.linalg.inv(post_invcov0)

    post_mean1_reward = np.einsum('ijk,ik->ij', post_cov1, total_reward1 )
    post_mean0_reward = np.einsum('ijk,ik->ij', post_cov0, total_reward0 ) 
  
    # Calculate posterior to get probability of sampling arm
    mean1 = np.einsum('ij,ij->i', states, post_mean1_reward)
    mean0 = np.einsum('ij,ij->i', states, post_mean0_reward)
    var1 = np.einsum('ij,ij->i', np.einsum('ijk,ik->ij', post_cov1, states), states)
    var0 = np.einsum('ij,ij->i', np.einsum('ijk,ik->ij', post_cov0, states), states)

    post_mean = mean1 - mean0
    post_var = var1 + var0
   
    # Calculate sampling probability
    # For Z_1, Z_2 i.i.d standard normal
    # P(p1 > p0) = P(mu1-mu0 + Z1 s1 - Z0 s0 > 0) =  P(mu1-mu0 + sqrt(s1^2 + s0^2)Z > 0)
        # =  P([mu1-mu0]/sqrt(s1^2 + s0^2) + Z > 0)
    ratio = np.divide(post_mean, np.sqrt(post_var))
    pis = stats.norm.cdf(ratio)

    if clipping > 0:
        pis = np.minimum(np.maximum(pis, clipping), 1-clipping)
    
    return pis



###################
# sample contexts #
###################
if args.d == 1:
    all_Z = np.ones((args.N, args.T, 1))

else:
    if args.Z_dist == 'ones':
        all_Z = np.ones((args.N, args.T, args.d))
    elif args.Z_dist == 'bernoulli':
        all_Z = np.random.binomial(1,0.5,size=(args.N, args.T, args.d-1))+1
    elif args.Z_dist == 'uniform':
        all_Z = np.random.uniform(0,5,size=(args.N, args.T, args.d-1))
    elif args.Z_dist == 'normal':
        Z_mean = np.ones(args.d-1)*3
        Z_cov = np.eye(args.d-1) 
        all_Z = np.random.multivariate_normal(Z_mean, Z_cov, size=(args.N, args.T))
    
    all_Z = np.concatenate( [np.ones((args.N,args.T,1)), all_Z], axis=2 )

base_reward = np.dot(all_Z, params[:args.d])
advantage = np.dot(all_Z, params[args.d:])

all_pis = []

all_actions = None
all_rewards = None

if args.strategy == 'TS':
    # Prior of variance is identity matrix
    post_invcov1 = np.zeros((args.N, args.d, args.d)) + np.eye(args.d)
    post_invcov0 = np.zeros((args.N, args.d, args.d)) + np.eye(args.d)
    total_reward1 = np.zeros((args.N, args.d))
    total_reward0 = np.zeros((args.N, args.d))
else:
    raise ValueError("Invalid action selection strategy")

for t in range(1, args.T+1):
    if t % 50 == 0:
        print('T={}'.format(t))
        print(np.mean(pis), np.var(pis))

        plt.hist( pis, bins=100 )
        plt.savefig( os.path.join( save_f, 'pis_t={}'.format(t) ) )
        plt.xlim(0,1)
        plt.close()

    ###########
    # get pis #
    ###########
    if t == 2 and args.no_zeros:
        pis = 1-pis
    elif args.strategy == 'TS':
        pis = get_pis_TS(all_Z[:,t-1], post_invcov1, post_invcov0, \
                total_reward1, total_reward0, clipping=args.clipping)
    elif args.strategy == 'independent':
        pis = 0.5*np.ones(args.N)
    else:
        raise ValueError('Invalid Strategy')
    all_pis.append(pis)

    # Sample arms
    try:
        actions = np.random.binomial(1, pis, args.N)
    except:
        import ipdb; ipdb.set_trace()
    
    # Sample reward noise
    natural_param = base_reward[:,t-1] + advantage[:,t-1]*actions
    
    if args.reward == 'normal':
        # Linear model of expected rewards
        noise = np.random.normal(0, math.sqrt(args.var), args.N)
        rewards = natural_param + noise
        TS_rewards = rewards
    elif args.reward == 'bernoulli':
        # Binary rewards: inverse link is logistic function
        rewards = np.random.binomial( 1, expit(natural_param) )
        TS_rewards = rewards*2 - 1
        # expit(x) = 1/(1+np.exp(-x)) )
    elif args.reward == 'poisson':
        # Poisson rewards: inverse link is exponential function
        rewards = np.random.poisson( np.exp( natural_param ) )
        TS_rewards = 0.6*rewards
    elif args.reward == 'uniform':
        # Linear model of expected rewards
        maxval = np.sqrt(args.var*12) / 2
        noise = np.random.uniform(-maxval, maxval, args.N)
        rewards = natural_param + noise
        TS_rewards = rewards
    elif args.reward == 't-dist':
        noise = stats.t.rvs(df=5, size=args.N)
        rewards = natural_param + noise
        TS_rewards = rewards
    else:
        raise ValueError("invalid reward type")


    # Track posterior
    if args.strategy == 'TS':
        current_Z = all_Z[:,t-1]
        current_ZZ = np.einsum('ij,ik->ijk', current_Z, current_Z)
        current_reward1 = np.expand_dims(actions*TS_rewards, axis=1)*current_Z
        current_reward0 = np.expand_dims( (1-actions) *TS_rewards, axis=1)*current_Z
        expand_actions = np.expand_dims(actions, (1,2))
        
        post_invcov1 = expand_actions * current_ZZ + post_invcov1
        post_invcov0 = (1-expand_actions) * current_ZZ + post_invcov0
        total_reward1 += current_reward1
        total_reward0 += current_reward0


    # Track history
    if t == 1:
        all_actions = np.expand_dims(actions,1)
        all_rewards = np.expand_dims(rewards,1)
    else:
        all_actions = np.concatenate( [all_actions, np.expand_dims(actions,1)], axis=1 )
        all_rewards = np.concatenate( [all_rewards, np.expand_dims(rewards,1)], axis=1 )


ave_reward = np.mean( np.sum(all_rewards, axis=1) )
print( 'ave total reward', ave_reward )

simulation_data = {
        'all_actions': all_actions,
        'all_rewards': all_rewards,
        'all_Z': all_Z,
        'all_pis': np.array(all_pis).T,
        }

print('Saving...')
with open(save_f+'/simulation_data.p', 'wb') as fp:
    pickle.dump(simulation_data, fp)

print(save_f)
print( np.mean( np.array(all_pis) == args.clipping ) )

