# ========================================================================
# Created by:
# Felipe Uribe
# ========================================================================
# Version 2023
# ========================================================================
import numpy as np
import arviz as az
import scipy.stats as sps
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
matplotlib.rcParams.update({'font.size': 16})
#matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#matplotlib.rcParams['text.usetex'] = True



#=======================================================================================
def plotmatrix(data, datap=None, plotprint=False):
    if type(datap) is np.ndarray:
        prplot = True
    else:
        prplot = False
    d = data.shape[0]
    n = d**2 
    M = np.eye(d)
    diag = np.where(M.flatten()==1)
    diag = diag[0] + 1
    j, idx = 0, 0
    #
    fig = plt.figure(figsize=(15, 8))
    fig.subplots_adjust(wspace=0.6, hspace=0.7)
    for i in range(1, n+1):
        # plot histogram
        if (i in diag):
            # yy = sps.gaussian_kde(data[j, :])(binn)  
            # cc, dd = yy.min(), yy.max()
            # yyp = sps.gaussian_kde(datap[j, :])(binnp)
            plt.subplot(d, d, i)
            ax = plt.gca()

            # posterior
            aa, bb = data[j, :].min(), data[j, :].max()
            hist, _, widths, center = histogram(data[j, :])
            bars = plt.bar(center, hist, width=widths, align='center', color='royalblue')
            for b in bars:
                b.set_alpha(0.8)
                b.set_edgecolor("black")
                b.set_linewidth(0.5)
            plt.yticks([])

            # prior
            if prplot:
                aap, bbp = datap[j, :].min(), datap[j, :].max()
                low_x, upp_x = min(aa, aap), max(bb, bbp)
                hist, _, widths, center = histogram(datap[j, :])
                bars = plt.bar(center, hist, width=widths, align='center', color='crimson')
                for b in bars:
                    b.set_alpha(0.6)
                    b.set_edgecolor("black")
                    b.set_linewidth(0.5)
                plt.yticks([])
                plt.xlim([low_x, upp_x])
                plt.xticks(np.linspace(low_x, upp_x, 4))
            else:
                plt.xlim([aa, bb])
                plt.xticks(np.linspace(aa, bb, 4))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            idx += 1
            k = j+1  # jump one space for the next subplot

        # plot scatter
        else: 
            if (j > k):   # plot only a lower matrix
                aa, bb = data[k, :].min(), data[k, :].max()
                cc, dd = data[j, :].min(), data[j, :].max()
                plt.subplot(d, d, i)
                ax = plt.gca()
                plt.plot(data[k, :], data[j, :], '.', color='navy', markersize=1, rasterized=True)

                # prior
                if prplot:
                    aap, bbp = datap[k, :].min(), datap[k, :].max()
                    ccp, ddp = datap[j, :].min(), datap[j, :].max()
                    low_x, upp_x = min(aa, aap), max(bb, bbp)
                    low_y, upp_y = min(cc, ccp), max(dd, ddp)
                    plt.plot(datap[k, :], datap[j, :], '.', color='crimson', markersize=1, rasterized=True)
                    plt.xlim([low_x, upp_x])
                    plt.ylim([low_y, upp_y])
                else:
                    plt.xlim([aa, bb])
                    plt.ylim([cc, dd])
                ax.set_aspect('equal', adjustable='datalim', anchor='SW')
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            k += 1

        if (k == d):
            j += 1   # rows
            k  = 0   # cols

    plt.tight_layout()
    if plotprint:
        plt.savefig('marginals.pdf', format='pdf', dpi=150, bbox_inches='tight')
        plt.pause(2)



#=======================================================================================
def histogram(data, Nh=None, type=0):
    N = len(data)
    if (Nh == None):
        Nh = int(np.ceil(np.sqrt(N)))
    # histogram
    a, b = data.min(), data.max(),
    binn = np.linspace(a, b, num=Nh)
    h, bins = np.histogram(data, bins=binn, density=True)
    widths = np.diff(bins)
    center = (bins[:-1] + bins[1:]) / 2  
    if (type == 0):
        return h, bins, widths, center
    elif (type == 1):
        plt.figure()
        plt.step(bins[:-1], h, 'k', where="mid", linewidth=1)
        plt.ylim(0, h.max()) 
        plt.xlim(a, b)
        plt.xlabel(r'Data')
        plt.ylabel(r'Frequency')
        plt.pause(2)
    elif (type == 2):
        plt.figure()
        plt.bar(bins[:-1], h, zorder=1, align='edge', width=widths, edgecolor='k', fill=False, linewidth=0.7)  
        plt.ylim(0, h.max()) 
        plt.xlim(a, b)
        plt.xlabel(r'Data')
        plt.ylabel(r'Frequency')
        plt.pause(2)
        


#=======================================================================================
def posterior_stats_1D(data, name='Data', HDIorCI=1, lag=50, plotprint=False):
    N = len(data)
    #
    data_t = az.convert_to_inference_data(data.reshape(1, N))
    #post = data_t.posterior
    stats_t = az.summary(data_t)
    print(stats_t)

    # main stats
    mu = np.mean(data)
    med = np.median(data)
    std = np.std(data, ddof=1)
    HDI = az.hdi(data, hdi_prob=.95)
    CI_95 = np.percentile(data, [2.5, 97.5])
    bfmi = az.bfmi(data)
    mcse = az.mcse(data)

    # ergodic mean and autocorrelation
    mu_erg = np.array([np.mean(data[:i+1]) for i in range(N)])
    R = az.autocorr(data)

    # histogram
    h, bins, _, _ = histogram(data)
    dh = h[1]-h[0]
    aa, bb = bins.min(), bins.max()
    cc, dd = 0, h.max()+dh    

    # histogram
    plt.figure(figsize=(13,5))
    ax1 = plt.subplot(141)
    ax1.step(bins[:-1], h, 'k', where="mid", linewidth=1)
    ax1.vlines(mu, cc, dd, colors='blue', linestyles=':')
    if HDIorCI == 0:
        ax1.axvspan(CI_95[0], CI_95[1], alpha=0.25, color='blue')
    elif HDIorCI == 1:
        ax1.axvspan(HDI[0], HDI[1], alpha=0.25, color='blue')
    elif HDIorCI == 2:
        ax1.axvspan(CI_95[0], CI_95[1], alpha=0.25, color='red')
        ax1.axvspan(HDI[0], HDI[1], alpha=0.25, color='blue')
    ax1.set_xlim(aa, bb)
    ax1.set_ylim(cc, dd)
    ax1.set_xlabel(r"${}$".format(name))
    ax1.set_ylabel(r'Frequency')

    # chain
    ax1 = plt.subplot(142)
    ax1.plot(data, 'k-', linewidth=1)
    ax1.set_xlim([0, N])
    # ax1.set_ylim([0.006, 0.035])
    # ax1.set_ylabel(r'$\tau$')
    ax1.set_title('Chain')
    ax1.set_xlabel('Sample index')

    # erg mean
    ax1 = plt.subplot(143)
    ax1.plot(mu_erg, 'k-', linewidth=1.5)
    # ax1.axhspan(CI_95_s[0], CI_95_s[1], alpha=0.25, color='blue')
    ax1.set_xlim([0, N])
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax1.set_title('Cumulative mean')
    ax1.set_xlabel('Sample index')
    
    # autocorr
    ax1 = plt.subplot(144)
    ax1.plot(R, 'k-', linewidth=1.5)
    ax1.set_xlim(0, lag) 
    ax1.set_ylim(0, 1)
    ax1.set_title('Autocorrelation')
    ax1.set_xlabel('Lag')
    #
    plt.tight_layout()
    if plotprint:
        plt.savefig('posterior_stats_{}.pdf'.format(name), format='pdf', dpi=150, bbox_inches='tight')
    plt.pause(2)

    return mu, med, std, bfmi, mcse