from pandas import DataFrame

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


def mean_generate():
    dir = '../perf-vs-noises/noises_delete'
    clss = ['BIO', 'SOCIAL']  # bio or social
    models = ['GCN', 'GFN']  # gcn or gfn
    randds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    data_dict = {
        'BIO_GCN_train': [],
        'BIO_GCN_test': [],
        'BIO_GCN_error': [],
        'BIO_GFN_train': [],
        'BIO_GFN_test': [],
        'BIO_GFN_error': [],
        'SOCIAL_GCN_train': [],
        'SOCIAL_GCN_test': [],
        'SOCIAL_GCN_error': [],
        'SOCIAL_GFN_train': [],
        'SOCIAL_GFN_test': [],
        'SOCIAL_GFN_error': []
    }

    for cls in clss:
        for model in models:
            with open(os.path.join(dir,
                                   'noises_{}_{}.log'.format(cls.lower(),
                                                             model.lower())), 'r') as f:
                for line in f.readlines():
                    if line.startswith('Train Acc'):
                        line = line.rstrip()
                        data_dict['{}_{}_train'.format(cls, model)].append(
                            float(line.split(',')[0][-6:])
                        )
                        data_dict['{}_{}_test'.format(cls, model)].append(
                            float(line.split(',')[1][-13:-8])
                        )
                        data_dict['{}_{}_error'.format(cls, model)].append(
                            float(line.split(',')[1][-5:])
                        )

    for cls in clss:
        if cls == 'BIO':
            datasets = ['MUTAG', 'NCI1', 'PROTEINS',
                        'DD', 'ENZYMES', 'PTC_MR']
        elif cls == 'SOCIAL':
            datasets = ['COLLAB', 'IMDB-BINARY', 'IMDB-MULTI',
                        'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K',
                        'REDDIT-BINARY']
        else:
            raise ValueError('cls error')
        for dataset in datasets:
            plt.figure(dpi=100)
            data = []
            idx = datasets.index(dataset)
            for model in models:
                train_acc = data_dict['{}_{}_train'.format(
                    cls, model)][idx * len(randds): (idx + 1) * len(randds)]
                test_acc = data_dict['{}_{}_test'.format(
                    cls, model)][idx * len(randds): (idx + 1) * len(randds)]
                error = data_dict['{}_{}_error'.format(
                    cls, model)][idx * len(randds): (idx + 1) * len(randds)]
                data_len = len(train_acc)
                assert len(train_acc) == len(test_acc) == len(error)
                for i in range(data_len):
                    data_rand = error[i] * np.random.randn(10000) + test_acc[i]
                    assert abs(np.std(data_rand) - error[i]) < 0.01
                    assert abs(np.mean(data_rand) - test_acc[i]) < 0.01
                    data.append([randds[i],
                                 train_acc[i],
                                 '{}(train)'.format(model)])
                    for j in range(len(data_rand)):
                        data.append([randds[i],
                                     data_rand[j],
                                     '{}(test)'.format(model)])
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
                                            ''],
                                    ci='sd')
            plt.ylim(0.0, 1.0)
            plt.legend(loc='lower right',
                       prop={'size': 16},
                       labels=["GCN(train)",
                               "GCN(test)",
                               "GFN(train)",
                               "GFN(test)"])
            plt.tick_params(labelsize=16)
            plt.xlabel('Noise Level', fontsize=16)
            plt.ylabel('Accuracy', fontsize=16)
            plt.savefig('../noise_delete_mean/{}.pdf'.format(dataset),
                        bbox_inches="tight")
            plt.savefig('../noise_delete_mean/{}.png'.format(dataset),
                        bbox_inches="tight")
            plt.close()


def mean():
    dir = '../perf-vs-noises/noises_delete'
    clss = ['BIO', 'SOCIAL']  # bio or social
    models = ['GCN', 'GFN']   # gcn or gfn
    randds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    data_dict = {
        'BIO_GCN_train': [],
        'BIO_GCN_test': [],
        'BIO_GCN_error': [],
        'BIO_GFN_train': [],
        'BIO_GFN_test': [],
        'BIO_GFN_error': [],
        'SOCIAL_GCN_train': [],
        'SOCIAL_GCN_test': [],
        'SOCIAL_GCN_error': [],
        'SOCIAL_GFN_train': [],
        'SOCIAL_GFN_test': [],
        'SOCIAL_GFN_error': []
    }

    for cls in clss:
        for model in models:
            with open(os.path.join(dir,
                                   'noises_{}_{}.log'.format(cls.lower(),
                                                             model.lower())), 'r') as f:
                for line in f.readlines():
                    if line.startswith('Train Acc'):
                        line = line.rstrip()
                        data_dict['{}_{}_train'.format(cls, model)].append(
                            float(line.split(',')[0][-6:])
                        )
                        data_dict['{}_{}_test'.format(cls, model)].append(
                            float(line.split(',')[1][-13:-8])
                        )
                        data_dict['{}_{}_error'.format(cls, model)].append(
                            float(line.split(',')[1][-5:])
                        )
                        print(line)
                        print(line.split(',')[0][-6:],
                              line.split(',')[1][-13:-8],
                              line.split(',')[1][-5:])

    for cls in clss:
        if cls == 'BIO':
            datasets = ['MUTAG', 'NCI1', 'PROTEINS',
                        'DD', 'ENZYMES', 'PTC_MR']
        elif cls == 'SOCIAL':
            datasets = ['COLLAB', 'IMDB-BINARY', 'IMDB-MULTI',
                        'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K',
                        'REDDIT-BINARY']
        else:
            raise ValueError('cls error')
        for dataset in datasets:
            idx = datasets.index(dataset)
            plt.figure(dpi=100)
            plt.ylabel('Noise Level')
            plt.xlabel('Accuracy')
            for model in models:
                train_acc = data_dict['{}_{}_train'.format(
                    cls, model)][idx * len(randds): (idx + 1) * len(randds)]
                test_acc = data_dict['{}_{}_test'.format(
                    cls, model)][idx * len(randds): (idx + 1) * len(randds)]
                error = data_dict['{}_{}_error'.format(
                    cls, model)][idx * len(randds): (idx + 1) * len(randds)]
                print(train_acc)
                print(error)
                if model == 'GCN':
                    plt.plot(np.array(randds[:len(train_acc)]),
                             np.array(train_acc),
                             alpha=0.8,
                             linewidth=2.5,
                             dashes=(4, 2),
                             label='GCN(train)')
                    plt.plot(np.array(randds[:len(test_acc)]),
                             np.array(test_acc),
                             alpha=0.8,
                             linewidth=2.5,
                             dashes=(4, 2),
                             label='GCN(test)')
                    plt.fill_between(np.array(randds[:len(test_acc)]),
                                     np.array(test_acc) + np.array(error),
                                     np.array(test_acc) - np.array(error),
                                     alpha=0.5)
                else:
                    plt.plot(np.array(randds[:len(train_acc)]),
                             np.array(train_acc),
                             alpha=0.8,
                             linewidth=2.5,
                             label='GFN(train)')
                    plt.plot(np.array(randds[:len(test_acc)]),
                             np.array(test_acc),
                             alpha=0.8,
                             linewidth=2.5,
                             label='GFN(test)')
                    plt.fill_between(np.array(randds[:len(test_acc)]),
                                     np.array(test_acc) + np.array(error),
                                     np.array(test_acc) - np.array(error),
                                     alpha=0.5)
            plt.ylim(0.0, 1.0)
            plt.legend(loc='lower right',
                       prop={'size': 16},
                       labels=["GCN(train)",
                               "GCN(test)",
                               "GFN(train)",
                               "GFN(test)"])
            plt.tick_params(labelsize=16)
            plt.xlabel('Noise Level', fontsize=16)
            plt.ylabel('Accuracy', fontsize=16)
            plt.savefig('../noise_delete/{}.pdf'.format(dataset),
                        bbox_inches="tight")
            plt.savefig('../noise_delete/{}.png'.format(dataset),
                        bbox_inches="tight")
            plt.close()


def last():
    dir = '../perf-vs-noises/noises_delete'
    clss = ['BIO', 'SOCIAL']  # bio or social
    models = ['GCN', 'GFN']  # gcn or gfn
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
                    if len(line.split('/')) > 1 and line.split('/')[1][:3] == '100':
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
                    if dataset == datasets[int(count / 100)]:
                        data.append([randds[int(count / 10) % 10],
                                     train_acc[count],
                                     '{}(train)'.format(model)])
                    count = count + 1
                count = 0
                while count < len(test_acc):
                    if dataset == datasets[int(count / 100)]:
                        data.append([randds[int(count / 10) % 10],
                                     test_acc[count],
                                     '{}(test)'.format(model)])
                    count = count + 1
            if dataset == 'REDDIT-MULTI-12K':
                print(data)
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
                                            ''],
                                    ci='sd')
            plt.ylim(0.0, 1.0)
            plt.legend(loc='lower right',
                       prop={'size': 16},
                       labels=["GCN(train)",
                               "GCN(test)",
                               "GFN(train)",
                               "GFN(test)"])
            plt.tick_params(labelsize=16)
            plt.xlabel('Noise Level', fontsize=16)
            plt.ylabel('Accuracy', fontsize=16)
            plt.savefig('../noise_delete_last/{}.pdf'.format(dataset),
                        bbox_inches="tight")
            plt.savefig('../noise_delete_last/{}.png'.format(dataset),
                        bbox_inches="tight")
            plt.close()


if __name__ == '__main__':
    mean_generate()
    mean()
    last()