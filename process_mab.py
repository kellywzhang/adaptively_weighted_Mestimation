import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
import os
import argparse
import json
import time
import pickle
from scipy.stats import chi2

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
parser.add_argument('--clipping', type=float, default=0.0,
        help='Clipping value in [0, 1)')
parser.add_argument('--reward', type=str, default='normal', \
        choices=['bernoulli', 'normal', 'uniform', 't-dist'])
parser.add_argument('--no_zeros', type=int, default=0)

# non-save arguments
parser.add_argument('--path', type=str, default='./mab_simulations',
        help='Where to save results' )
parser.add_argument('--load_results', type=int, default=0,
        help='Only load results from a previous run of process_mab.py')
parser.add_argument('--verbose', type=int, default=0,
        help='Prints more details')
parser.add_argument('--estvar', type=int, default=0,
        help='Estimate the variance')
parser.add_argument('--adjust', type=int, default=1,
        help='Use adjusted power to only allow feasible solutions (proper Type-1 error control)')
parser.add_argument('--ols', type=int, default=1,
        help='Use OLS estimator')
parser.add_argument('--awaipw', type=int, default=0,
        help='Use AW-AIPW estimator')
parser.add_argument('--awls', type=int, default=1,
        help='Use AWLS estimator')
parser.add_argument('--null_means', type=str, default='0,0',
        help='Null expected rewards for each arm')
parser.add_argument('--sparseT', type=str, default=None,
        help='Evaluate estimators not at every batch')

args = parser.parse_args()
print( vars(args) )

plt.rcParams.update({'font.size': 18})
path = args.path
nonsave_args = ['path', 'load_results', 'estvar', 'adjust', 'awaipw', \
        'verbose', 'sparseT', 'bols', 'ols', 'awls', 'null_means', \
        'prior_means', 'prior_vars', 'alg_var']
if not args.no_zeros:
    nonsave_args.append( 'no_zeros' )
save_args = [ '{}={}'.format(key, val) for key, val in vars(args).items() if key not in nonsave_args ]
save_str = '_'.join( save_args )
save_f_load = os.path.join( path, save_str)
if args.estvar: 
    save_f = os.path.join( path, save_str, 'estimate_variance')
else:
    save_f = os.path.join( path, save_str, 'known_variance')

if not os.path.isdir( save_f ):
    os.mkdir( save_f )

alphas = [ 0.05, 0.1 ]
if args.sparseT is None:
    Tvals = [t for t in range(1, args.T+1)]
else:
    Tvals = [ int(x) for x in args.sparseT.split(",") ]

true_means = [ float(x) for x in args.means.split(',') ]
margin = true_means[1]-true_means[0]        # margin := Treatment effect
null = margin == 0
save_raw_null = os.path.join( path, save_str.replace("means={}".format(args.means), \
        "means={}".format(args.null_means)) )
if args.estvar:
    save_f_null = os.path.join( save_raw_null, 'estimate_variance' )
else:
    save_f_null = os.path.join( save_raw_null, 'known_variance' )
print("Difference in Arm Means", margin)

if null and args.adjust:
    if not os.path.isdir( os.path.join( save_f, 'cutoff_adjustments' ) ):
        os.mkdir( os.path.join( save_f, 'cutoff_adjustments' ) )


def make_hist(name, vals, plot_normal=(0,1), density=True, power=False, title_size=18, hist_type='errors'):
    alphas = [0.05, 0.1]
    if hist_type=='pis':
        alphas = [0.05]
    for alpha in alphas:
        fig = plt.figure( figsize=(8,5) )
        mu = np.mean(vals)
        var = np.var( vals, ddof=1 )
        title_string = "Distribution under {}".format( strategy2name[args.strategy] )
        if 'ols' in name:
            title_string = "Distribution of {}".format( 'OLS Estimator' )
        elif 'awls' in name:
            title_string = "Distribution of {}".format( 'AW-Least Squares Estimator')
        else:
            title_string = "Distribution of {}".format( 'Empirical Average Allocation\nUnder {}'.format(strategy2name[args.strategy]))
           
        plt.hist( vals, bins=99, density=density, label='Empirical Distribution', color='C2')
   
        if power:
            cutoff = math.fabs( stats.norm.ppf( alpha / 2 ) )
            plt.axvline( x= cutoff, color='k', label='{}% Confidence Interval\nfor standard normal'.format( int(100-alpha*100)) )
            plt.axvline( x= -cutoff, color='k' )

            power = np.greater( np.abs( vals ), cutoff ).mean()
            power = np.round(power*100,1)

        if density:
            mu = 0
            variance = 1
            sigma = math.sqrt(variance)
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            plt.plot(x, stats.norm.pdf(x, mu, sigma), label="Standard Normal")
            plt.xlim(left=-5,right=5)

        if 'awls' in name:
            plt.legend(loc='center right', fontsize=15,
                        bbox_to_anchor=(1.25, 0.5))
        if hist_type=='errors':
            if args.strategy == 'epsilon':
                plt.text(-4.3, 0.2, "Type-1 error:\n{} %".format( power ), fontsize=18)
            else:
                plt.text(-4.5, 0.2, "Type-1 error:\n{} %".format( power ), fontsize=18)
                #plt.text(-4.5, 0.2, "Coverage\nProbability:\n{} %".format( 100-power ), fontsize=15)
        plt.title( title_string , fontsize=title_size )
        if not hist_type=='pis':
            alph_str = '_alpha={}'.format(int(alpha*100))
        else:
            alph_str = ''
            plt.xlim(0,1)
        plt.savefig( os.path.join( save_f, name+alph_str ) + '.png', bbox_inches='tight' )
        plt.close()


def print_results(t, alphas, print_dict):
    print( "-----------------------\nt={}".format(t) )
    for alpha in alphas:
        print( 'alpha={}'.format(alpha))
        if null:
            print('Type-1 Error={}'.format(print_dict[t][alpha]) )
        else:
            print('Power={}'.format(print_dict[t][alpha]) )


def calculate_power(alphas, cutoffs, estimates, save_dict):
    for A, cutoff in zip(alphas, cutoffs):
        vals = np.abs( estimates ) 
        ht = np.greater( vals, cutoff )
        power = ht.mean()
        se = np.std( ht, ddof=1) / math.sqrt(args.N)
        save_dict[A] = (power, se)
    return save_dict


def calculate_cutoff_adjustment(alphas, vals, orig_cutoffs=None):
    cutoffs = {}
    for k, A in enumerate(alphas):
        adjusted_cutoff = np.quantile( np.abs(vals), 1-A )
        if orig_cutoffs is not None:
            # We do not adjust the cutoff if using the original cutoff does not inflate the Type-1 error
            cutoffs[A] = max( adjusted_cutoff, orig_cutoffs[k] )
        else:
            cutoffs[A] = adjusted_cutoff
    return cutoffs


def ols_inference(simulation_dict, alphas, power_dict):
    power_dict['ols'] = { t:{ alpha: {} for alpha in alphas } for t in Tvals }
    power_dict['ols1'] = { t:{ alpha: {} for alpha in alphas } for t in Tvals }
    power_dict['ols_error'] = { t:{ alpha: {} for alpha in alphas } for t in Tvals }
    power_dict['ols1_error'] = { t:{ alpha: {} for alpha in alphas } for t in Tvals }
    ols_dict = { 'est0': [], 'est1': [], 'stat': [], 'mse': [], 'mse1': [], 'var1': [] }
    
    all_actions = simulation_data[ 'all_actions' ]
    all_rewards = simulation_data[ 'all_rewards' ]
    all_pis = simulation_data[ 'all_pis' ]
   
    if args.adjust:
        if null:
            adjusted_cutoffs = {}
        else:
            with open( os.path.join( save_f_null, 'cutoff_adjustments', 'ols.json' ), 'r' ) as f:
                adjusted_cutoffs = json.load( f )
    
    all_bias = []
    print( '\nOLS' )
    Rsum1 = np.zeros(args.N); Rsum0 = np.zeros(args.N)
    N1 = np.zeros(args.N); N0 = np.zeros(args.N)
    for t, t0 in zip(Tvals, [0]+Tvals[:-1]):

        Rsum1 += np.sum( all_rewards[:,t0:t]*all_actions[:,t0:t], axis=1 )
        Rsum0 += np.sum( all_rewards[:,t0:t]*(1-all_actions[:,t0:t]), axis=1 )
        N1 += np.sum( all_actions[:,t0:t], axis=1 )
        N0 += np.sum( (1-all_actions[:,t0:t]), axis=1 )
        sum_rewards = [ Rsum0, Rsum1 ]
        actions = [ N0, N1 ]

        all_means = []
        for k in range(2):
            means = np.divide( sum_rewards[k], actions[k], where=actions[k]!=0 )
            all_means.append( np.expand_dims(means,axis=1) )

        all_means = np.nan_to_num(all_means)
        all_means = np.concatenate(all_means, axis=1)
        
        if args.estvar and t!=1:
            means = actions[1]*all_means[:,1] + actions[0]*all_means[:,0]
            means = np.expand_dims(means,axis=1)
            residuals = all_rewards[:,:t] - means
            noise_std = np.std(residuals, ddof=2, axis=1)
        else:
            noise_std = np.ones(args.N)*math.sqrt(args.var)

        ols_std = np.sqrt( t / ( actions[0] * actions[1] ) ) * noise_std
        ols_margin_stat = ( all_means[:,1] - all_means[:,0] ) / ols_std
        ols_margin_error = ( all_means[:,1] - all_means[:,0] - margin ) / ols_std
        cutoffs = [ math.fabs( scipy.stats.norm.ppf( alpha / 2 ) ) for alpha in alphas ]
    
        ols1_std = np.sqrt( 1 / actions[1] ) * noise_std
        ols1_stat = ( all_means[:,1] ) / ols1_std
        ols1_error = ( all_means[:,1] - true_means[1] ) / ols1_std
        ols_mse = np.mean( np.square( all_means[:,1] - true_means[1] ) + np.square( all_means[:,0] - true_means[0] ) )
        ols_mse_std = np.std( np.square( all_means[:,1] - true_means[1] ) + np.square( all_means[:,0] - true_means[0] ) ) / np.sqrt(args.N)
        
        ols_dict['est0'].append( all_means[:,0] )
        ols_dict['est1'].append( all_means[:,1] )
        ols_dict['stat'].append( ols_margin_stat )
        ols_dict['mse'].append( (ols_mse, ols_mse_std) )
        ols_dict['mse1'].append( ( np.mean( np.square(all_means[:,1]-true_means[1]) ), np.std( all_means[:,1] ) / np.sqrt(args.N) ) )
        ols_dict['var1'].append( ( np.var(all_means[:,1]), np.var(all_means[:,1]) / args.N ) )
        
        print( 'ols mse', ols_mse )
        print( 'ols bias', np.mean( all_means[:,1] ), np.std( all_means[:,1] ) / np.sqrt(args.N) )
        print('ols var1', np.var(all_means[:,1]) )
        all_bias.append( ( np.mean( all_means[:,1] ), np.std( all_means[:,1] ) / np.sqrt(args.N) ) )

        if args.adjust:
            if null:
                adjusted_cutoffs[t] = calculate_cutoff_adjustment(alphas, ols_margin_stat, \
                        orig_cutoffs=cutoffs)
            else:
                # get adjusted cutoffs
                cutoffs = [ v for k, v in adjusted_cutoffs[str(t)].items() if float(k) in alphas ]
       
        calculate_power(alphas, cutoffs, ols_margin_stat, power_dict['ols'][t])
        calculate_power(alphas, cutoffs, ols_margin_error, power_dict['ols_error'][t])
        calculate_power(alphas, cutoffs, ols1_error, power_dict['ols1_error'][t])

        print_results(t, alphas, print_dict=power_dict['ols_error'])
        make_hist( 'ols_distribution_t={}'.format(t), ols_margin_error, power=True )
        make_hist( 'ols1_distribution_t={}'.format(t), ols1_error, power=True )
    
    ols_dict['bias'] = all_bias

    if args.adjust:
        if null:
            with open( os.path.join( save_f, 'cutoff_adjustments', 'ols.json' ), 'w' ) as f:
                json.dump( adjusted_cutoffs, f, indent=4 )

    return ols_dict, noise_std



def awls_inference(simulation_dict, alphas, power_dict, weight='sqrtpi'):
    power_dict['awls_'+weight] = { t:{ alpha: {} for alpha in alphas } for t in Tvals }
    power_dict['awls_error_'+weight] = { t:{ alpha: {} for alpha in alphas } for t in Tvals }
    awls_dict = { 'est0': [], 'est1': [], 'stat': [], 'mse': [], 'mse_raw': [], 'mse1': [], 'var1': [] }
    
    all_actions = simulation_data[ 'all_actions' ]
    all_rewards = simulation_data[ 'all_rewards' ]
    all_pis = np.transpose(simulation_data[ 'all_pis' ])
    ave_pis = np.mean(all_pis, axis=0) 
    ave_pis = np.expand_dims( ave_pis, axis=0 )

    if args.adjust:
        if null:
            adjusted_cutoffs = {}
        else:
            with open( os.path.join( save_f_null, 'cutoff_adjustments', 'awls_{}.json'.format(weight) ), 'r' ) as f:
                adjusted_cutoffs = json.load( f )

    print( '\nAWLS' )
    if weight=='sqrtpi':
        weights1 = np.divide( all_actions, np.sqrt(all_pis), where=all_pis!=0)
        weights0 = np.divide( 1-all_actions, np.sqrt(1-all_pis), where=all_pis!=1)
        weighted_rewards1 = np.divide( all_actions*all_rewards, np.sqrt(all_pis), where=all_pis!=0)
        weighted_rewards0 = np.divide( (1-all_actions)*all_rewards, np.sqrt(1-all_pis), where=all_pis!=1)
    elif weight == 'pieval':
        weights1 = np.multiply( all_actions, np.sqrt(ave_pis/all_pis), where=all_pis!=0)
        weights0 = np.multiply( 1-all_actions, np.sqrt((1-ave_pis)/(1-all_pis)), where=all_pis!=1)
        weighted_rewards1 = np.multiply( all_actions*all_rewards, np.sqrt(ave_pis/all_pis), where=all_pis!=0)
        weighted_rewards0 = np.multiply( (1-all_actions)*all_rewards, np.sqrt((1-ave_pis)/(1-all_pis)), where=all_pis!=1)

    all_bias = []
    print('max weights', np.max(weights0), np.max(weights1))
    for t in Tvals:
        V1 = np.sum( weights1[:,:t], axis=1 )
        V0 = np.sum( weights0[:,:t], axis=1 )
        
        awls_est1 = np.divide( np.sum( weighted_rewards1[:,:t], axis=1 ), V1, where=V1!=0 )
        awls_est0 = np.divide( np.sum( weighted_rewards0[:,:t], axis=1 ), V0, where=V0!=0 )
        awls_margin = awls_est1 - awls_est0

        if args.estvar and t!=1:
            sum_rewards1 = np.sum( (all_rewards*all_actions)[:,:t], axis=1)
            sum_rewards0 = np.sum( (all_rewards*(1-all_actions))[:,:t], axis=1)
            sum_rewards = [ sum_rewards0, sum_rewards1 ]
            actions = [ np.sum(1-all_actions[:,:t], axis=1), np.sum(all_actions[:,:t], axis=1) ]
            
            all_means = []
            for k in range(2):
                means = np.divide( sum_rewards[k], actions[k], where=actions[k]!=0 )
                all_means.append( np.expand_dims(means,axis=1) )

            all_means = np.nan_to_num(all_means)
            all_means = np.concatenate(all_means, axis=1)

            means = actions[1]*all_means[:,1] + actions[0]*all_means[:,0]
            means = np.expand_dims(means,axis=1)
            residuals = all_rewards[:,:t] - means
            noise_std = np.std(residuals, axis=1)
        else:
            noise_std = np.ones(args.N)*math.sqrt(args.var)
        
        if weight == 'pieval':
            awls_std = np.sqrt( np.mean(ave_pis)/np.square(V1) + np.mean(1-ave_pis)/np.square(V0) ) * noise_std
        else:
            awls_std = np.sqrt( 1/np.square(V1) + 1/np.square(V0) ) * noise_std
        awls_stat = 1/np.sqrt(t) * awls_margin / awls_std
        awls_error = 1/np.sqrt(t) * ( awls_margin - margin ) / awls_std

        awls_mse_raw = np.square( awls_est1 - true_means[1] ) + np.square( awls_est0 - true_means[0] )
        awls_mse = np.mean( awls_mse_raw )
        awls_mse_std = np.std( np.square( awls_est1 - true_means[1] ) + np.square( awls_est0 - true_means[0] ) ) / np.sqrt(args.N)

        awls_dict['est0'].append( awls_est0 )
        awls_dict['est1'].append( awls_est1 )
        awls_dict['stat'].append( awls_stat )
        awls_dict['mse'].append( (awls_mse, awls_mse_std) )
        awls_dict['mse_raw'].append( awls_mse_raw )

        awls_dict['mse1'].append( ( np.mean( np.square(awls_est1-true_means[1]) ), np.std( awls_est1 ) / np.sqrt(args.N) ) )
        awls_dict['var1'].append( ( np.var(awls_est1), np.var(awls_est1) / args.N ) )

        print( 'awls_'+weight+" error", np.mean( np.square(awls_margin-margin) ) )
        print( 'awls_'+weight+' mse', awls_mse )
        print( 'awls bias', np.mean( awls_est1 ), np.std( awls_est1 ) / np.sqrt(args.N) )
        all_bias.append( ( np.mean( awls_est1 ), np.std( awls_est1 ) / np.sqrt(args.N) ) )
        print( 'awls var1', np.var(awls_est1) )

        cutoffs = [ math.fabs( scipy.stats.norm.ppf( alpha / 2 ) ) for alpha in alphas ]
       
        awls1_stat = (awls_est1-true_means[1]) * V1 / np.sqrt(t)
        awls0_stat = (awls_est0-true_means[0]) * V0 / np.sqrt(t)
        if weight == 'pieval':
            awls1_stat = awls1_stat / np.sqrt( np.mean(ave_pis) )
            awls0_stat = awls0_stat / np.sqrt( np.mean(1-ave_pis) )
        make_hist( 'awls1_{}_distribution_t={}'.format(weight, t), awls1_stat, power=True, hist_type='errors' )
        make_hist( 'awls0_{}_distribution_t={}'.format(weight, t), awls0_stat, power=True, hist_type='errors' )

        if args.adjust:
            if null:
                adjusted_cutoffs[t] = calculate_cutoff_adjustment(alphas, awls_stat, \
                        orig_cutoffs=cutoffs)
            else:
                # get adjusted cutoffs
                cutoffs = [ v for k, v in adjusted_cutoffs[str(t)].items() if float(k) in alphas ]
        
        calculate_power(alphas, cutoffs, awls_stat, power_dict['awls_'+weight][t])
        calculate_power(alphas, cutoffs, awls_error, power_dict['awls_error_'+weight][t])
       
        make_hist( 'awls_{}_distribution_t={}'.format(weight, t), awls_error, power=True )
        print_results(t, alphas, print_dict=power_dict['awls_'+weight])
        
    
    awls_dict['bias'] = all_bias

    if args.adjust:
        if null:
            with open( os.path.join( save_f, 'cutoff_adjustments', 'awls_{}.json'.format(weight) ), 'w' ) as f:
                json.dump( adjusted_cutoffs, f, indent=4 )

    return awls_dict



def awaipw_inference(simulation_dict, alphas, power_dict):
    power_dict['awaipw'] = { t:{ alpha: {} for alpha in alphas } for t in Tvals }
    power_dict['awaipw_error'] = { t:{ alpha: {} for alpha in alphas } for t in Tvals }
    awaipw_dict = { 'estdelta':[], 'est0': [], 'est1': [], 'stat': [] }
    
    all_actions = simulation_data[ 'all_actions' ]
    all_rewards = simulation_data[ 'all_rewards' ]
    all_pis = np.transpose(simulation_data[ 'all_pis' ])
   
    if args.adjust:
        if null:
            adjusted_cutoffs = {}
        else:
            with open( os.path.join( save_f_null, 'cutoff_adjustments', 'awaipw.json' ), 'r' ) as f:
                adjusted_cutoffs = json.load( f )

    print( '\nAW-AIPW' )
    # Update model mu
    cumsum_sum1 = np.cumsum(all_actions*all_rewards, axis=1)
    cumsum_sum0 = np.cumsum( (1-all_actions)*all_rewards, axis=1)
    cumsum_count1 = np.cumsum(all_actions, axis=1)
    cumsum_count0 = np.cumsum(1-all_actions, axis=1)
    mu1_raw = np.divide( cumsum_sum1, cumsum_count1, where=cumsum_count1!=0 )
    mu0_raw = np.divide( cumsum_sum0, cumsum_count0, where=cumsum_count0!=0 )
    mu1 = np.hstack( [np.zeros((args.N,1)), mu1_raw] )[:,:-1]
    mu0 = np.hstack( [np.zeros((args.N,1)), mu0_raw] )[:,:-1]

    mu1 = np.zeros((args.N,args.T))
    mu0 = np.zeros((args.N,args.T))

    weights = np.sqrt(all_pis*(1-all_pis))
    G1 = np.divide( all_actions*(all_rewards - mu1), all_pis, where=all_pis!=0) + mu1
    G0 = np.divide( (1-all_actions)*(all_rewards - mu0), 1-all_pis, where=all_pis!=1) + mu0
    Gdiff = G1 - G0
  
    # AW-Diff, with uniform weighting
    for t in Tvals:
        weights_tmp = weights[:,:t]
        Gdiff_tmp = Gdiff[:,:t]
        numerator = np.sum(weights_tmp*Gdiff_tmp, axis=1)
        deltahat = np.divide( numerator, np.sum(weights_tmp, axis=1) )
        awaipw_dict['estdelta'].append(deltahat)

        denominator = np.sum( np.square(weights_tmp) * \
                np.square( Gdiff_tmp - np.expand_dims(deltahat,axis=1) ),  axis=1 )
        
        awaipw_stat = np.divide( numerator, np.sqrt(denominator) )
        awaipw_error = np.divide( np.sum(weights_tmp*(Gdiff_tmp-margin), axis=1), np.sqrt(denominator) )

        # Calculate test statistic
        cutoffs = [ math.fabs( scipy.stats.norm.ppf( alpha / 2 ) ) for alpha in alphas ]
        
        if args.adjust:
            if null:
                adjusted_cutoffs[t] = calculate_cutoff_adjustment(alphas, awaipw_stat, \
                        orig_cutoffs=cutoffs)
            else:
                # get adjusted cutoffs
                cutoffs = [ v for k, v in adjusted_cutoffs[str(t)].items() if float(k) in alphas ]
        
        calculate_power(alphas, cutoffs, awaipw_stat, power_dict['awaipw'][t])
        calculate_power(alphas, cutoffs, awaipw_error, power_dict['awaipw_error'][t])
        
        make_hist( 'awaipw_distribution_t={}'.format(t), awaipw_error, power=True )
        print_results(t, alphas, print_dict=power_dict['awaipw'])
        
    
    if args.adjust:
        if null:
            with open( os.path.join( save_f, 'cutoff_adjustments', 'awaipw.json' ), 'w' ) as f:
                json.dump( adjusted_cutoffs, f, indent=4 )

    return awaipw_dict     





strategy2name = {
            'TS': 'Thompson Sampling',
            'epsilon': r'$\epsilon$'+'-Greedy',
            'independent': 'Independently Sampled',
            'TS_hodges': 'Thompson Sampling Hodges',
        }

key2color = { 
    'ols': 'C0',
    'ols1': 'C0',
    'awaipw': 'C7',
    'awls_sqrtpi': 'C8',
    'awls_pieval': 'r',
}

key2name = {
    'ols': 'OLS',
    'ols1': 'OLS1',
    'awaipw': 'AW-AIPW', 
    'awls_sqrtpi': 'AW-LS (uniform stabilizing \u03C0)', 
    'awls_pieval': 'AW-LS (expected stabilizing \u03C0)', 
}

order_index = {
    'ols': 1,
    'ols1': 2,
    'awaipw': 5,
    'awls_sqrtpi': 15,
    'awls_pieval': 16,
}

def power_plots(plot_keys, power_dict, alphas):
    # Power plots
    title_size = 18
    label_size = 18
    
    keys = [k for k in power_dict.keys() if 'error' not in k and 'reward' not in k and 'ave_allocation' not in k]
    keys.sort(key=lambda x: order_index[x] )

    for alpha in alphas:
        fig = plt.figure( figsize=(10,5) )
        all_se = []
        for key in keys:
            if key in plot_keys:
                vals, se = zip(* [ power_dict[key][t][alpha] for t in Tvals] )
                plt.plot(Tvals, vals, label=key2name[key], color=key2color[key])
                all_se.append( np.max( se ) )
        plt.xlabel('Timesteps (T)', fontsize=label_size)
        if null:
            plt.ylabel('Type-1 Error', fontsize=label_size)
            plt.hlines( alpha, 1, args.T, label='Nominal '+r'$\alpha$', color='k' )
            if alpha == 0.1:
                plt.ylim(top=0.22)
            else:
                plt.ylim(top=0.12)
        else:
            plt.ylabel('Power', fontsize=label_size)
            plt.ylim(bottom=0, top=1)
            plt.hlines( 0, 1, args.T, label='Nominal '+r'$\alpha$', color='k' )
        if args.estvar:
            estvarstr = "estimated variance"
        else:
            estvarstr = "known variance"
        plt.xticks( [y for x, y in enumerate(Tvals) if x % 2 == 1 ] )
        plt.title("{}, Margin {}".format( strategy2name[args.strategy], true_means[1]-true_means[0] ), fontsize=title_size)
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        plt.legend(fontsize='large')
        if null:
            plt.savefig( os.path.join( save_f, 'type1error_alpha={}.png'.format(alpha) ),
                    bbox_inches='tight')
        else:
            plt.savefig( os.path.join( save_f, 'power_alpha={}_adjusted={}.png'.format(alpha, args.adjust) ),
                    bbox_inches='tight')
        plt.close()

    return np.max(all_se)



########################
# Load simulation data #
########################
if not args.load_results:
    print('Loading...')
    with open(save_f_load+'/simulation_data.p', 'rb') as fp:
        simulation_data = pickle.load(fp)
    print('Done loading!')

    # Plot action selection probabilities
    all_pis = simulation_data[ 'all_pis' ]
    make_hist('sampling_probabilities_t={}'.format(args.T), all_pis[args.T-1], plot_normal=False, 
              density=False, hist_type='pis')

    all_rewards = simulation_data[ 'all_rewards' ]
    all_actions = simulation_data[ 'all_actions' ]

    # Plot average allocation
    for t in Tvals:
        if t > 1:
            ave_allocation = np.mean(all_actions[:,:t],axis=1)
            make_hist('average_allocation_t={}'.format(t), ave_allocation, plot_normal=False, 
                      density=False, hist_type='pis')
            
    all_ave_allocation = []
    for t in Tvals:
        ave_allocation = np.mean(all_actions[:,:t], axis=1)
        all_ave_allocation.append( (np.mean(ave_allocation), np.var(ave_allocation)) )

    # Perform inference
    power_dict = {}
    if args.ols:
        ols_dict, ols_noise_std = ols_inference(simulation_data, alphas, power_dict)
    if args.awls:
        awls_dict = awls_inference(simulation_data, alphas, power_dict)
        awls_pieval_dict = awls_inference(simulation_data, alphas, power_dict, weight='pieval')
    if args.awaipw:
        awaipw_dict = awaipw_inference(simulation_data, alphas, power_dict)

    pickle.dump( power_dict, open( os.path.join( save_f, \
            'power_dict_adjust={}.p'.format(args.adjust)), 'wb' ) )
else:
    power_dict = pickle.load( open( os.path.join( save_f, \
            'power_dict_adjust={}.p'.format(args.adjust)), 'rb' ) )


# Prepare final results
plot_keys = []
if args.ols:
    plot_keys.append('ols')
    plot_keys.append('ols_error')
    plot_keys.append('ols1_error')
if args.awls:
    plot_keys.append('awls_sqrtpi')
    plot_keys.append('awls_pieval')
if args.awaipw:
    plot_keys.append('awaipw')

print('plot_keys', plot_keys)

# Print final type-1 error / power
for alpha in alphas:
    print('\n\n\n=======================================================')
    if null:
        print("Timestep {}, Type-1 Error:".format(args.T))
    else:
        print("Timestep {}, Power:".format(args.T))
    print('alpha={}'.format(alpha))
    print('=======================================================')
    for key in plot_keys:
        print(key, [ power_dict[key][t][alpha][0] for t in Tvals])
    print('=======================================================')

    
if args.awls:
    title_size = 18
    label_size = 18
   
    fig = plt.figure( figsize=(10,5) )

    plt.errorbar(Tvals, [x[0] for x in awls_dict['mse']], yerr=[x[1] for x in awls_dict['mse']], 
            label=key2name['awls_sqrtpi'], color=key2color['awls_sqrtpi'])
    plt.errorbar(Tvals, [x[0] for x in awls_pieval_dict['mse']], yerr=[x[1] for x in awls_pieval_dict['mse']], 
            label=key2name['awls_pieval'], color=key2color['awls_pieval'])
    plt.xlabel('T', fontsize=label_size)
    plt.ylabel('MSE', fontsize=label_size)
    plt.title("Mean Squared Error for All Parameters", fontsize=title_size)
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.legend(fontsize='medium')
    plt.savefig( os.path.join( save_f, 'MSE.png' ),
                bbox_inches='tight')
    plt.close()


