import math
import os
import pickle
import argparse
import json

import numpy as np
import scipy.stats as stats

parser = argparse.ArgumentParser()
parser.add_argument('--strategy', type=str, default='TS', \
        help='Action selection strategy', \
        choices=['TS', 'epsilon', 'independent', 'TS_hodges'])
parser.add_argument('--N', type=int, default=100000, 
        help='Number of monte carlo simulations')
parser.add_argument('--T', type=int, default=5,
        help='Number of timesteps')
parser.add_argument('--means', type=str, default='0,0',
        help='Expected rewards for each arm')
parser.add_argument('--var', type=float, default=1,
        help='Reward variance (sigma^2)')
parser.add_argument('--clipping', type=float, default=0,
        help='Clipping value in [0, 1)')
parser.add_argument('--reward', type=str, default='normal', \
        choices=['bernoulli', 'normal', 'uniform', 'tdist'])
parser.add_argument('--pi1', type=float, default=0.5,
        help='Sampling probability for first batch') 
parser.add_argument('--prior_means', type=str, default='0,0',
        help='Prior means on rewards for each arm used for Thompson Sampling')
parser.add_argument('--prior_vars', type=str, default='1,1',
        help='Prior variances on rewards for each arm used for Thompson Sampling')
parser.add_argument('--alg_var', type=float, default=1,
        help='Variance of rewards assumed by Thompson Sampling')
parser.add_argument('--no_zeros', type=int, default=0) 

parser.add_argument('--path', type=str, default='./mab_simulations')

args = parser.parse_args()
print(vars(args))
assert args.clipping < 1

nonsave_args = ['path', 'pi1', 'prior_means', 'prior_vars', 'alg_var']
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
true_means = [float(x) for x in args.means.split(',')]
assert len(true_means) == 2

print('Arm Means:', true_means)

prior_means = [float(x) for x in args.prior_means.split(',')]
prior_vars = [float(x) for x in args.prior_vars.split(',')]
assert len(prior_means) == 2
assert len(prior_vars) == 2

def get_pis_TS(Rsum1, Rsum0, N1, N0, var, clipping=0):
    sum_rewards = [ Rsum0, Rsum1 ]
    actions = [ N0, N1 ]

    pm = []
    pv = []
    for k in range(2):
        # Posterior mean
        pm_temp = np.divide(prior_means[k]*var + prior_vars[k]*sum_rewards[k], var + prior_vars[k]*actions[k])
        pm.append(pm_temp)
        
        # Posterior variance
        pv_temp = np.divide(prior_vars[k]*var, var + prior_vars[k]*actions[k])
        pv.append(pv_temp)
   
    post_mean = pm[1] - pm[0]
    post_var = pv[1] + pv[0]
    
    # Calculate sampling probability
    ratio = np.divide(post_mean, np.sqrt(post_var))
    pis = stats.norm.cdf(ratio)

    if clipping > 0:
        pis = np.minimum(np.maximum(pis, clipping), 1-clipping)
    
    return pis


def get_pis_TS_hodges(Rsum1, Rsum0, N1, N0, var, t, clipping=0):
    # Calculate posterior to get probability of sampling arm
    sum_rewards = [ Rsum0, Rsum1 ]
    actions = [ N0, N1 ]

    pm = []
    pv = []
    for k in range(2):
        # Posterior mean
        pm_temp = np.divide(prior_means[k]*var + prior_vars[k]*sum_rewards[k], var + prior_vars[k]*actions[k])
        pm.append(pm_temp)
        
        # Posterior variance
        pv_temp = np.divide(prior_vars[k]*var, var + prior_vars[k]*actions[k])
        pv.append(pv_temp)
   
    post_mean = pm[1] - pm[0]
    post_var = pv[1] + pv[0]
   
    c = 1.5
    shrink_range = ( post_mean <= c/ np.sqrt( np.sqrt(t) ) ) * ( post_mean >= -c/ np.sqrt( np.sqrt(t) ) )
    post_mean = (1-shrink_range)*post_mean

    # Calculate sampling probability
    ratio = np.divide(post_mean, np.sqrt(post_var))
    pis = stats.norm.cdf(ratio)

    if clipping > 0:
        pis = np.minimum(np.maximum(pis, clipping), 1-clipping)
    
    return pis


def get_pis_epsilon(Rsum1, Rsum0, N1, N0, clipping=0):
    sum_rewards = [ Rsum0, Rsum1 ]
    actions = [ N0, N1 ]

    all_means = []
    for k in range(2):
        means = np.divide( sum_rewards[k], actions[k] )
        all_means.append( np.expand_dims(means,axis=1) )

    all_means = np.nan_to_num(all_means)
    all_means = np.concatenate(all_means, axis=1)

    pis = np.ones((args.N,2))*clipping
    max_vals = np.broadcast_to(np.expand_dims(np.max(all_means, axis=1), 1), (args.N, 2))
    pis += (1-2*clipping)*(np.equal(max_vals, all_means))
   
    return pis[:, 1]



# pis ( N x K )
if args.no_zeros:
    pis = np.random.binomial(1, 0.5, args.N)
else:
    pis = args.pi1 * np.ones(args.N)
all_pis = []
all_pis.append(pis)

all_actions = []; all_rewards = []
Rsum1 = np.zeros(args.N); Rsum0 = np.zeros(args.N)
N1 = np.zeros(args.N); N0 = np.zeros(args.N)

for t in range(1, args.T+1):
    if t % 500 == 0:
        print('T={}'.format(t))

    # Sample arms
    actions = np.random.binomial(1, pis, args.N)
    
    # Sample reward noise
    if args.reward == 'normal':
        noise = np.random.normal(0, math.sqrt(args.var), args.N)
    elif args.reward == 'bernoulli':
        noise = math.sqrt(args.var)*(np.random.binomial(1, 0.5, size=args.N) - 0.5)/2
    elif args.reward == 'uniform':
        noise = math.sqrt(args.var*12)*(np.random.uniform(0, 1, size=args.N) - 0.5) 
    elif args.reward == 'tdist':
        noise = np.random.standard_t(0, math.sqrt(args.var), args.N)
    else:
        raise ValueError("invalid reward type")

    rewards = noise + actions*true_means[1] + (1-actions)*true_means[0]

    all_actions.append( actions )
    all_rewards.append( rewards )

    Rsum1 += rewards*actions
    Rsum0 += rewards*(1-actions)
    N1 += actions
    N0 += (1-actions)

    if t != args.T:
        if t == 1 and args.no_zeros:
            pis = 1-pis
        elif args.strategy == 'TS':
            pis = get_pis_TS(Rsum1, Rsum0, N1, N0, args.alg_var, clipping=args.clipping)
        elif args.strategy == 'epsilon':
            pis = get_pis_epsilon(Rsum1, Rsum0, N1, N0, clipping=args.clipping)
        elif args.strategy == 'independent':
            pis = 0.5*np.ones(args.N)
        elif args.strategy == 'TS_hodges':
            pis = get_pis_TS_hodges(Rsum1, Rsum0, N1, N0, args.alg_var, t, clipping=args.clipping)
        else:
            raise ValueError('Invalid Strategy')
        all_pis.append(pis)


all_actions = np.vstack( all_actions ).T
all_rewards = np.vstack( all_rewards ).T

simulation_data = {
        'all_actions': all_actions, # N x T
        'all_rewards': all_rewards,
        'all_pis': np.array(all_pis),
        }

print('Saving...')
with open(save_f+'/simulation_data.p', 'wb') as fp:
    pickle.dump(simulation_data, fp)


