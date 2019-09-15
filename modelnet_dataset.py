import os
import os.path as osp
import glob
import random

import torch
from torch_geometric.datasets import ModelNet
from torch_geometric.read import read_off
from torch_geometric.utils import to_undirected


class ModelNetExT(ModelNet):
    r"""The ModelNet10/40 datasets from the `"3D ShapeNets: A Deep
    Representation for Volumetric Shapes"
    <https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf>`_ paper,
    containing CAD models of 10 and 40 categories, respectively.

    .. note::

        Data objects hold mesh faces instead of edge indices.
        To convert the mesh to a graph, use the
        :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
        To convert the mesh to a point cloud, use the
        :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
        sample a fixed number of points on the mesh faces according to their
        face area.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string, optional): The name of the dataset (:obj:`"10"` for
            ModelNet10, :obj:`"40"` for ModelNet40). (default: :obj:`"10"`)
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    urls = {
        '10':
        'http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip',
        '40': 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
    }

    def __init__(self,
                 root,
                 name='10',
                 num=100,
                 train=True,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 processed_file_prefix='data'):
        assert name in ['10', '40']
        self.num = num
        self.processed_file_prefix = processed_file_prefix
        super(ModelNetExT, self).__init__(root, name, train,
                                          transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        return ['{}_training.pt'.format(self.processed_file_prefix),
                '{}_test.pt'.format(self.processed_file_prefix)]

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset):
        categories = glob.glob(osp.join(self.raw_dir, '*', ''))
        categories = sorted([x.split(os.sep)[-2] for x in categories])

        data_list = []
        for target, category in enumerate(categories):
            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob('{}/{}_*.off'.format(folder, category))
            for path in paths:
                data = read_off(path)
                data.y = torch.tensor([target])

                pos = data.pos
                face = data.face.contiguous() - 1

                random_list = sorted(random.sample(range(int(face.size(1)) - 1),
                                                   int(face.size(1) / 10)))
                face = face[:, random_list]
                face_set_a = set(face[0, :])
                face_set_b = set(face[1, :])
                face_set_c = set(face[2, :])
                pos_list = sorted(list(face_set_a | face_set_b | face_set_c))
                for i, _ in enumerate(pos_list):
                    pos_list[i] = int(pos_list[i])
                dict_key = {}
                for i, j in enumerate(pos_list):
                    dict_key[int(j)] = i
                for i in range(face.size(1)):
                    face[0][i] = dict_key[int(face[0][i])]
                    face[1][i] = dict_key[int(face[1][i])]
                    face[2][i] = dict_key[int(face[2][i])]
                pos = pos[pos_list]

                assert pos.size(1) == 3 and face.size(0) == 3

                edge_index = torch.cat([face[:2], face[1:], face[::2]], dim=1)
                edge_index = to_undirected(edge_index, num_nodes=self.num)

                data.pos = pos
                data.face = face
                data.edge_index = edge_index

                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)


