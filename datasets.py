import os.path as osp
import re

import torch
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.utils import degree
import torch_geometric.transforms as T
from feature_expansion import FeatureExpander
from image_dataset import ImageDataset
from tu_dataset import TUDatasetExt
from qm9_dataset import QM9Ext
from tosca_dataset import TOSCAEXT
from modelnet_dataset import ModelNetExT


def get_dataset(name, sparse=True, feat_str="deg+ak3+reall", root=None):
    if root is None or root == '':
        path = osp.join(osp.expanduser('~'), 'pyG_data', name)
    else:
        path = osp.join(root, name)
    degree = feat_str.find("deg") >= 0
    onehot_maxdeg = re.findall("odeg(\d+)", feat_str)
    onehot_maxdeg = int(onehot_maxdeg[0]) if onehot_maxdeg else None
    k = re.findall("an{0,1}k(\d+)", feat_str)
    k = int(k[0]) if k else 0
    groupd = re.findall("groupd(\d+)", feat_str)
    groupd = int(groupd[0]) if groupd else 0
    remove_edges = re.findall("re(\w+)", feat_str)
    remove_edges = remove_edges[0] if remove_edges else 'none'
    centrality = feat_str.find("cent") >= 0
    coord = feat_str.find("coord") >= 0

    pre_transform = FeatureExpander(
        degree=degree, onehot_maxdeg=onehot_maxdeg, AK=k,
        centrality=centrality, remove_edges=remove_edges,
        group_degree=groupd).transform

    if 'MNIST' in name or 'CIFAR' in name:
        if name == 'MNIST_SUPERPIXEL':
            train_dataset = MNISTSuperpixels(path, True,
                                             pre_transform=pre_transform,
                                             transform=T.Cartesian())
            test_dataset = MNISTSuperpixels(path, False,
                                            pre_transform=pre_transform,
                                            transform=T.Cartesian())
        else:
            train_dataset = ImageDataset(path, name, True,
                                         pre_transform=pre_transform, coord=coord,
                                         processed_file_prefix="data_%s" % feat_str)
            test_dataset = ImageDataset(path, name, False,
                                        pre_transform=pre_transform, coord=coord,
                                        processed_file_prefix="data_%s" % feat_str)
        dataset = (train_dataset, test_dataset)
    elif 'QM9' in name:
        dataset = QM9Ext(path, pre_transform=pre_transform,
                         processed_filename="data_%s.pt" % feat_str)
    elif 'ModelNet' in name:
        pre_transform = FeatureExpander(
            degree=degree, onehot_maxdeg=onehot_maxdeg, AK=k,
            centrality=centrality, remove_edges=remove_edges,
            group_degree=groupd).cloud_point_transform
        train_dataset = ModelNetExT(path, train=True, pre_transform=pre_transform,
                                    processed_file_prefix="data_%s" % feat_str)
        test_dataset = ModelNetExT(path, train=True, pre_transform=pre_transform,
                                   processed_file_prefix="data_%s" % feat_str)
        dataset = (train_dataset, test_dataset)
    elif 'TOSCA' in name:
        # pre_transform = FeatureExpander(
        #     degree=degree, onehot_maxdeg=onehot_maxdeg, AK=k,
        #     centrality=centrality, remove_edges=remove_edges,
        #     group_degree=groupd).cloud_point_transform
        dataset = TOSCAEXT(path, pre_transform=pre_transform,
                           processed_file_prefix="data_%s" % feat_str)
    else:
        dataset = TUDatasetExt(
            path, name, pre_transform=pre_transform,
            use_node_attr=True, processed_filename="data_%s.pt" % feat_str)

        dataset.data.edge_attr = None

    return dataset
