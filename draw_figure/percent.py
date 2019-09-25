from pandas import DataFrame

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def main():
    datasets = ['COLLAB', 'DD', 'ENZYMES', 'IMDB-BINARY', 'IMDB-MULTI',
                'MUTAG', 'PROTEINS', 'PTC_MR', 'REDDIT-MULTI-5K',
                'REDDIT-MULTI-12K']
    evals = ['train', 'test']
    models = ['GCN', 'GFN']
    percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for dataset in datasets:
        # plt.cla()
        plt.figure(dpi=100)
        data = []
        for percent in percents:
            data_dict = {'GCN_train': np.zeros(100),
                         'GCN_test': np.zeros(100),
                         'GFN_train': np.zeros(100),
                         'GFN_test': np.zeros(100)}
            for model in models:
                for eval in evals:
                    with open("../save_data/{}_Res{}_{}_{}.txt".format(
                            dataset, model, percent, eval
                    ), "r") as f:
                        count = 0
                        for line in f.readlines():
                            data_dict['{}_{}'.format(model, eval)][count % 100] = \
                                data_dict['{}_{}'.format(model, eval)][count % 100] + \
                                float(line.rstrip())
                            count = count + 1
            GCN_idx = np.argmax(data_dict['GCN_test'])
            GFN_idx = np.argmax(data_dict['GFN_test'])
            print(GCN_idx, GFN_idx)
            for model in models:
                for eval in evals:
                        with open("../save_data/{}_Res{}_{}_{}.txt".format(
                                dataset, model, percent, eval
                        )) as f:
                            count = 0
                            for line in f.readlines():
                                if count % 100 == GCN_idx and model == 'GCN':
                                    data.append([float(line.rstrip()),
                                                 '{}({})'.format(model, eval),
                                                 int(count / 100),
                                                 percent])
                                elif count % 100 == GFN_idx and model == 'GFN':
                                    data.append([float(line.rstrip()),
                                                 '{}({})'.format(model, eval),
                                                 int(count / 100),
                                                 percent])
                                count = count + 1
        print(len(data))
        dataframe = DataFrame(data,
                              columns=['Accuracy',
                                       'Method',
                                       'Fold',
                                       'Training data size'])
        sns.set(style="ticks",
                rc={"lines.linewidth": 2.5})
        ax = sns.lineplot(x='Training data size',
                          y='Accuracy',
                          hue='Method',
                          style='Method',
                          data=dataframe,
                          style_order=["GCN(train)",
                                       "GCN(test)",
                                       "GFN(train)",
                                       "GFN(test)"],
                          dashes=[(4, 2),
                                  (4, 2),
                                  '',
                                  ''])
        if dataset == 'DD':
            plt.ylim(0.5, 1.1)
        elif dataset == 'ENZYMES':
            plt.ylim(0.1, 1.1)
        elif dataset == 'IMDB-MULTI':
            plt.ylim(0.2, 0.8)
        elif dataset == 'MUTAG':
            plt.ylim(0.5, 1.1)
        elif dataset == 'PTC_MR':
            plt.ylim(0.3, 1.1)
        elif dataset == 'REDDIT-MULTI-12K':
            plt.ylim(0.0, 1.0)
        plt.legend(loc='lower right',
                   prop={'size': 16},
                   labels=["GCN(train)",
                           "GCN(test)",
                           "GFN(train)",
                           "GFN(test)"])
        plt.tick_params(labelsize=16)
        plt.xlabel('Training data size', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.savefig('../perf vs percent/{}.pdf'.format(dataset),
                    bbox_inches="tight")
        plt.savefig('../perf vs percent/{}.png'.format(dataset),
                    bbox_inches="tight")


if __name__ == '__main__':
    main()
