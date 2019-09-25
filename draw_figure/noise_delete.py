from pandas import DataFrame

import matplotlib.pyplot as plt
import seaborn as sns
import os


def main():
    dir = '../perf-vs-noises/noises_delete'
    clss = ['BIO', 'SOCIAL']  # bio or social
    models = ['GCN', 'GFN']   # gcn or gfn
    randds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    data_dict = {
        'BIO_GCN_train': [],
        'BIO_GCN_test': [],
        'BIO_GFN_train': [],
        'BIO_GFN_test': [],
        'SOCIAL_GCN_train': [],
        'SOCIAL_GCN_test': [],
        'SOCIAL_GFN_train': [],
        'SOCIAL_GFN_test': []
    }

    for cls in clss:
        for model in models:
            with open(os.path.join(dir, 'noises_{}_{}.log'.format(cls.lower(),
                                                                  model.lower())), 'r') as f:
                for line in f.readlines():
                    if line.startswith('0'):
                        line = line.rstrip()
                        data_dict['{}_{}_train'.format(cls, model)].append(
                            float(line.split(',')[0][-5:])
                        )
                        data_dict['{}_{}_test'.format(cls, model)].append(
                            float(line.split(',')[1][-5:])
                        )

    for cls in clss:
        if cls == 'BIO':
            datasets = ['MUTAG', 'NCI1', 'PROTEINS', 'DD', 'ENZYMES', 'PTC_MR']
        elif cls == 'SOCIAL':
            datasets = ['COLLAB', 'IMDB-BINARY', 'IMDB-MULTI',
                        'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K', 'REDDIT-BINARY']
        else:
            raise ValueError('cls error')
        for dataset in datasets:
            # plt.cla()
            plt.figure(dpi=100)
            data = []
            for model in models:
                train_acc = data_dict['{}_{}_train'.format(cls, model)]
                test_acc = data_dict['{}_{}_test'.format(cls, model)]
                assert len(train_acc) == len(test_acc)
                count = 0
                while count < len(train_acc):
                    if dataset == datasets[int(count / 1100)]:
                        data.append([randds[int(count / 110) % 10],
                                     train_acc[count],
                                     '{}(train)'.format(model)])
                    count = count + 1
                count = 0
                while count < len(test_acc):
                    if dataset == datasets[int(count / 1100)]:
                        data.append([randds[int(count / 110) % 10],
                                     test_acc[count],
                                     '{}(test)'.format(model)])
                    count = count + 1
            dataframe = DataFrame(data,
                                  columns=['Noise Level',
                                           'Accuracy',
                                           'Method'])
            sns.set(style="ticks",
                    rc={"lines.linewidth": 2.5})
            sns_line = sns.lineplot(x='Noise Level',
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
            if dataset == 'COLLAB':
                plt.ylim(0.3, 1.0)
            elif dataset == 'DD':
                plt.ylim(0.5, 1.0)
            elif dataset == 'MUTAG':
                plt.ylim(0.4, 1.0)
            elif dataset == 'NCI1':
                plt.ylim(0.4, 1.0)
            elif dataset == 'PROTEINS':
                plt.ylim(0.4, 1.0)
            elif dataset == 'ENZYMES':
                plt.ylim(0.0, 1.0)
            elif dataset == 'IMDB-BINARY':
                plt.ylim(0.3, 0.9)
            elif dataset == 'IMDB-MULTI':
                plt.ylim(0.0, 0.9)
            elif dataset == 'PTC_MR':
                plt.ylim(0.0, 1.0)
            elif dataset == 'REDDIT-MULTI-5K':
                plt.ylim(0.0, 1.0)
            elif dataset == 'REDDIT-MULTI-12K':
                plt.ylim(0.2, 0.7)

            plt.legend(loc='lower right',
                       prop={'size': 16},
                       labels=["GCN(train)",
                               "GCN(test)",
                               "GFN(train)",
                               "GFN(test)"])
            plt.tick_params(labelsize=16)
            plt.xlabel('Rand', fontsize=16)
            plt.ylabel('Accuracy', fontsize=16)
            plt.savefig('../noise_delete/{}.pdf'.format(dataset),
                        bbox_inches="tight")
            plt.savefig('../noise_delete/{}.png'.format(dataset),
                        bbox_inches="tight")


if __name__ == '__main__':
    main()
