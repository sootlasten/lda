import numpy as np
import matplotlib.pyplot as plt
from colgibbs import col_gibbs 
from stdgibbs import std_gibbs
from statsmodels.graphics.tsaplots import plot_acf


def _plot_joint_and_preds(filename, title, logjoints, logpreds):
    """Plots the log-joint probabilities and log-pred probabilites in a single window
       and saves the plot as 'filename'."""
    fig = plt.figure()
    plt.suptitle(title)

    ax1 = fig.add_subplot(211)
    ax1.set_ylabel('Log-joint prob')
    ax1.plot(logjoints)

    ax2 = fig.add_subplot(212)
    ax2.set_xlabel('Gibbs iteration')
    ax2.set_ylabel('Log-pred prob')
    ax2.plot(logpreds)

    plt.savefig(filename)


def plot_autocorr(filename, title, logjoints, logpreds, lags=np.arange(30)):
    fig = plt.figure()
    plt.suptitle(title)

    ax1 = fig.add_subplot(211)
    ax1.set_ylabel('Autocorr of log-joint prob')
    plot_acf(logjoints, ax=ax1, lags=lags, title='')

    ax2 = fig.add_subplot(212)
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Autocorr of log-pred prob')
    plot_acf(logpreds, ax=ax2, lags=lags, title='')
    
    plt.savefig(filename)


if __name__ == '__main__':
    burn_in = 20
    K = 3

    data = np.loadtxt('data/toyexample.data', dtype=np.int)
    
    # standard Gibbs 
    # logjoints, logpreds, _, _, _, _, _, _ = std_gibbs(data, K, nb_iters=1000)
    # plot_autocorr('plots/std_autocorr.png', 'standard LDA Gibbs autocorrelation', logjoints[burn_in:], logpreds[burn_in:])
    # _plot_joint_and_preds('plots/std_lda_gibbs.png', 'standard LDA Gibbs', logjoints, logpreds)
    
    # collapsed Gibbs
    logjoints, logpreds, _, _, _, _ = col_gibbs(data, K, nb_iters=1000)
    plot_autocorr('plots/col_autocorr.png', 'collapsed LDA Gibbs autocorrelation', logjoints[burn_in:], logpreds[burn_in:])
    _plot_joint_and_preds('plots/collapsed_lda_gibbs.png', 'collapsed LDA Gibbs', logjoints, logpreds)

