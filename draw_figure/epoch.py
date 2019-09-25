from pandas import DataFrame

import matplotlib.pyplot as plt
import seaborn as sns
import os


def epoch_for_reddit():
    datasets = ['REDDIT-MULTI-5K',
                'REDDIT-MULTI-12K',
                'REDDIT-BINARY']
    evals = ['train', 'test']
    models = ['GCN', 'GFN']
    for dataset in datasets:
        # plt.cla()
        plt.figure(dpi=100)
        data = []
        for model in models:
            for eval in evals:
                with open("../save_data_REDDIT/{}_Res{}_1.0_{}.txt".format(
                        dataset, model, eval), "r") as f:
                    count = 0
                    for line in f.readlines():
                        data.append([count % 100 + 1,
                                     float(line.rstrip()),
                                     "{}({})".format(model, eval),
                                     int(count / 100)])
                        count = count + 1
        dataframe = DataFrame(data, columns=['Epoch',
                                             'Accuracy',
                                             'Method',
                                             'Fold'])
        sns.set(style="ticks",
                rc={"lines.linewidth": 2.5})
        ax = sns.lineplot(x='Epoch',
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
                                  ''],
                          ci='sd')
        if dataset == 'REDDIT-BINARY':
            plt.ylim(0, 1.0)
        elif dataset == 'REDDIT-MULTI-5K':
            plt.ylim(0, 0.9)
        elif dataset == 'REDDIT-MULTI-12K':
            plt.ylim(0.4, 0.8)
        plt.legend(loc='lower right',
                   prop={'size': 16},
                   labels=["GCN(train)",
                           "GCN(test)",
                           "GFN(train)",
                           "GFN(test)"])
        plt.tick_params(labelsize=16)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.savefig('../perf vs iter/reddit_epoch_{}.pdf'.format(dataset),
                    bbox_inches="tight")
        plt.savefig('../perf vs iter/reddit_epoch_{}.png'.format(dataset),
                    bbox_inches="tight")


def main():
    datasets = ['COLLAB', 'DD', 'ENZYMES',
                'IMDB-BINARY', 'IMDB-MULTI',
                'MUTAG', 'NCI1', 'PROTEINS',
                'PTC_MR', 'REDDIT-MULTI-5K',
                'REDDIT-MULTI-12K', 'REDDIT-BINARY']
    evals = ['train', 'test']
    models = ['GCN', 'GFN']
    for dataset in datasets:
        # plt.cla()
        plt.figure(dpi=100)
        data = []
        for model in models:
            for eval in evals:
                with open("../save_data/{}_Res{}_{}.txt".format(dataset, model, eval),
                          "r") as f:
                    count = 0
                    for line in f.readlines():
                        data.append([count % 100 + 1,
                                     float(line.rstrip()),
                                     "{}({})".format(model, eval),
                                     int(count / 100)])
                        count = count + 1
        dataframe = DataFrame(data, columns=['Epoch',
                                             'Accuracy',
                                             'Method',
                                             'Fold'])
        sns.set(style="ticks",
                rc={"lines.linewidth": 2.5})
        ax = sns.lineplot(x='Epoch',
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
                                  ''],
                          ci='sd')
        if dataset == 'PTC_MR':
            plt.ylim(0, 1)
        elif dataset == 'PROTEINS':
            plt.ylim(0.2, 1.0)
        elif dataset == 'REDDIT-MULTI-12K':
            plt.ylim(0.0, 0.8)
        elif dataset == 'DD':
            plt.ylim(0.4, 1.0)
        elif dataset == 'COLLAB':
            plt.ylim(0.5, 1.0)
        elif dataset == 'IMDB-MULTI':
            plt.ylim(0.2, 0.7)
        elif dataset == 'NCI1':
            plt.ylim(0.5, 1.0)
        elif dataset == 'REDDIT-MULTI-5K':
            plt.ylim(0, 0.9)
        plt.legend(loc='lower right',
                   prop={'size': 16},
                   labels=["GCN(train)",
                           "GCN(test)",
                           "GFN(train)",
                           "GFN(test)"])
        plt.tick_params(labelsize=16)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.savefig('../perf vs iter/epoch_{}.pdf'.format(dataset),
                    bbox_inches="tight")
        plt.savefig('../perf vs iter/epoch_{}.png'.format(dataset),
                    bbox_inches="tight")


if __name__ == '__main__':
    if not os.path.exists('../perf vs iter'):
        os.makedirs('../perf vs iter')
    main()
    epoch_for_reddit()
