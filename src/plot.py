#!/usr/bin/env python -O
# -*- coding: ascii -*-

import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import seaborn as sns

from matplotlib.ticker import MultipleLocator
from typing import Dict

TRUE_VAL = np.array([
    -72.125000000017080, -70.696428571445720, -68.655612244915050,
    -66.352405247830460, -63.936745106222890, -61.472890759819610,
    -58.988381754218224, -56.495020751817634, -53.997866036503090,
    -51.499085444225365, -48.999608047534934, -46.499832020381845,
    -43.999928008744890, -41.499969146614816, -38.999986777130460,
    -36.499994333065740, -33.999997571323620, -31.499998959148467,
    -28.999999553930515, -26.499999808837163, -23.999999918082842,
    -21.499999964902470, -18.999999984968035, -16.499999993567513,
    -13.999999997253013, -11.499999998832486, -8.9999999995094540,
    -6.4999999997995520, -3.9999999999238590, -1.4999999999771570
])
TRUE_VAR = np.array([
    359.29687500541900, 358.68463010744995, 354.81124401833270,
    347.49005981787820, 337.67209732445195, 326.32150062137530,
    314.11106921816054, 301.44427024323870, 288.54406797250280,
    275.52759897718585, 262.45433738761653, 249.35375055335020,
    236.24017319157810, 223.12047998101315, 209.99793062346714,
    196.87405645796747, 183.74957134101663, 170.62480588342095,
    157.49991234941513, 144.37496052676053, 131.24998226665320,
    118.12499205194244, 104.99999644629384, 91.874998415584140,
    78.749999296421340, 65.624999689718660, 52.499999865041730,
    39.374999943083310, 26.249999977772850, 13.124999993171010
])
STEADY_STATE = np.array([
    0.03418803, 0.03418803, 0.03418803, 0.03418803, 0.03418803,
    0.03418803, 0.03418803, 0.03418803, 0.03418803, 0.03418803,
    0.03418803, 0.03418803, 0.03418803, 0.03418802, 0.03418799,
    0.03418793, 0.03418779, 0.03418747, 0.03418672, 0.03418497,
    0.03418089, 0.03417136, 0.03414912, 0.03409724, 0.03397619,
    0.03369373, 0.03303467, 0.03149685, 0.02790860, 0.01953602
])


def plot_performance(directory: str, filename: str) -> None:
    raw_data = dict()
    for file in ['first_predictions', 'second_predictions', 'direct_predictions']:
        raw_data[file] = np.load('{0}/1/{1}.npy'.format(directory, file))

    # compress data to relevant metrics
    data = dict()
    for key1, key2 in [('first_predictions', 'value'),
                       ('second_predictions', 'indirect'),
                       ('direct_predictions', 'direct')]:
        data[key2] = dict()
        rmsve = np.copy(raw_data[key1])
        for i in range(rmsve.shape[0]):
            for j in range(rmsve.shape[1]):
                if key1 == 'first_predictions':
                    rmsve[i, j, :] -= TRUE_VAL
                elif key1 == 'second_predictions':
                    rmsve[i, j, :] -= np.square(raw_data['first_predictions'][i, j, :])
                    rmsve[i, j, :] -= TRUE_VAR
                else:
                    assert key1 == 'direct_predictions'
                    rmsve[i, j, :] -= TRUE_VAR
        rmsve = np.sqrt(np.average(np.square(rmsve), axis=2, weights=STEADY_STATE))
        data[key2]['mean'] = np.mean(rmsve, axis=0)
        data[key2]['sem'] = st.sem(rmsve, axis=0)

    # setup figure
    sns.set_style('ticks')
    sns.set_context('talk')
    plt.rc('text', usetex=True)
    c_map = sns.color_palette('colorblind')
    fig, ax1 = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
    ax1.set_xscale('log')
    ax2 = ax1.twinx()
    timesteps = np.arange(0, 3000000, 30)

    # plot value
    ax1.plot(timesteps,
             data['value']['mean'],
             color=c_map[0],
             label='Value')
    ax1.fill_between(timesteps,
                     data['value']['mean'] - data['value']['sem'],
                     data['value']['mean'] + data['value']['sem'],
                     color=c_map[0],
                     alpha=0.5)

    # plot indirect variance
    ax2.plot(timesteps,
             data['indirect']['mean'],
             color=c_map[1],
             label='VTD')
    ax2.fill_between(timesteps,
                     data['indirect']['mean'] - data['indirect']['sem'],
                     data['indirect']['mean'] + data['indirect']['sem'],
                     color=c_map[1],
                     alpha=0.5)

    # plot direct variance
    ax2.plot(timesteps,
             data['direct']['mean'],
             color=c_map[2],
             label='Direct')
    ax2.fill_between(timesteps,
                     data['direct']['mean'] - data['direct']['sem'],
                     data['direct']['mean'] + data['direct']['sem'],
                     color=c_map[2],
                     alpha=0.5)

    # tidy up plot
    ax1.set_xlabel('Timestep', labelpad=10)
    ax1.set_ylabel('Value RMSVE', labelpad=10)
    ax1.set_xlim(10 ** 2, 3000000)
    ax1.set_ylim(0, 50)
    ax1.tick_params(axis='both', which='major', pad=15)
    ax2.set_ylabel('Variance RMSVE', fontsize=14, labelpad=10)
    ax2.set_ylim(0, 50 ** 2)
    ax2.tick_params(axis='both', which='major', pad=5)
    fig.legend(loc=(0.575, 0.725), frameon=False)

    # save figure for performance
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def plot_quality(directory: str, filename: str) -> None:
    raw_data = dict()
    for file in ['first_predictions', 'second_predictions', 'direct_predictions']:
        raw_data[file] = np.load('{0}/1/{1}.npy'.format(directory, file))

    # compress data
    data = dict()
    data['indirect'] = dict()
    data['indirect']['mean'] = np.mean(raw_data['second_predictions'][:, - 1, :] -
                                       np.square(raw_data['first_predictions'][:, - 1, :]), axis=0)
    data['indirect']['sem'] = st.sem(raw_data['second_predictions'][:, - 1, :] -
                                     np.square(raw_data['first_predictions'][:, - 1, :]), axis=0)
    data['direct'] = dict()
    data['direct']['mean'] = np.mean(raw_data['direct_predictions'][:, - 1, :], axis=0)
    data['direct']['sem'] = st.sem(raw_data['direct_predictions'][:, - 1, :], axis=0)

    # setup figure
    sns.set_style('ticks')
    sns.set_context('talk')
    plt.rc('text', usetex=True)
    c_map = sns.color_palette('colorblind')
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
    states = np.arange(30) + 1

    # plot true variance
    ax.plot(states,
            TRUE_VAR,
            color=c_map[0],
            label='True Variance')

    # plot indirect variance
    ax.errorbar(states,
                data['indirect']['mean'],
                yerr=data['indirect']['sem'],
                color=c_map[1],
                label='VTD')

    # plot direct variance
    ax.errorbar(states,
                data['direct']['mean'],
                yerr=data['direct']['sem'],
                color=c_map[2],
                label='Direct')

    # tidy up plot
    ax.set_xlabel('State', labelpad=10)
    ax.set_ylabel('Variance', labelpad=10)
    ax.tick_params(axis='both', which='major', pad=15)
    fig.legend(loc=(0.26, 0.23), frameon=False)

    # save figure for performance
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def plot_sensitivity(directory: str, filename: str) -> None:
    powers_of_two = ['0.0000305176', '0.0000610352', '0.0001220703', '0.0002441406',
                     '0.0004882812', '0.0009765625', '0.0019531250', '0.0039062500',
                     '0.0078125000', '0.0156250000', '0.0312500000', '0.0625000000',
                     '0.1250000000', '0.2500000000', '0.5000000000', '1.0000000000']
    raw_data = dict()
    for file in ['first_predictions', 'second_predictions', 'direct_predictions']:
        raw_data[file] = dict()
        for val_alpha in powers_of_two:
            raw_data[file][val_alpha] = np.load(
                '{0}/0/0.0004882812/{1}/{2}.npy'.format(directory, val_alpha, file))

    # compress data
    data = dict()
    for key1, key2 in [('second_predictions', 'indirect'), ('direct_predictions', 'direct')]:
        data[key2] = dict()
        data[key2]['mean'] = np.zeros(len(powers_of_two))
        data[key2]['sem'] = np.zeros(len(powers_of_two))
        for i, val_alpha in enumerate(powers_of_two):
            rmsve = np.copy(raw_data[key1][val_alpha])
            for j in range(rmsve.shape[0]):
                for k in range(rmsve.shape[1]):
                    if key1 == 'second_predictions':
                        rmsve[j, k, :] -= np.square(
                            raw_data['first_predictions'][val_alpha][j, k, :])
                        rmsve[j, k, :] -= TRUE_VAR
                    else:
                        assert key1 == 'direct_predictions'
                        rmsve[j, k, :] -= TRUE_VAR
            rmsve = np.sqrt(np.average(np.square(rmsve), axis=2, weights=STEADY_STATE))
            data[key2]['mean'][i] = np.mean(np.sum(rmsve, axis=1) * 3e3, axis=0)
            data[key2]['sem'][i] = st.sem(np.sum(rmsve, axis=1) * 3e3, axis=0)

    # setup figure
    sns.set_style('ticks')
    sns.set_context('talk')
    plt.rc('text', usetex=True)
    c_map = sns.color_palette('colorblind')
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)

    # plot indirect variance
    ax.errorbar([float(v) for v in powers_of_two],
                data['indirect']['mean'],
                yerr=data['indirect']['sem'],
                color=c_map[1],
                label='VTD')

    # plot direct variance
    ax.errorbar([float(v) for v in powers_of_two],
                data['direct']['mean'],
                yerr=data['direct']['sem'],
                color=c_map[2],
                label='Direct')

    # tidy up plot
    ax.set_xlabel('Step Size', labelpad=10)
    ax.set_ylabel('Total RMSVE', labelpad=10)
    ax.tick_params(axis='both', which='major', pad=15)
    fig.legend(loc=(0.325, 0.785), frameon=False)

    # save figure for performance
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def main(args: Dict[str, str]) -> None:
    directory = args['directory']
    plot_performance(directory, '{}/performance.png'.format(directory))
    plot_quality(directory, '{}/quality.pdf'.format(directory))
    plot_sensitivity(directory, '{}/sensitivity.pdf'.format(directory))


def parse_args() -> Dict[str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str)
    return vars(parser.parse_args())


if __name__ == '__main__':
    np.seterr(divide='raise', over='ignore', under='raise', invalid='raise')
    main(parse_args())
