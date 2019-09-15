import os
import os.path as osp
import random
import glob

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TOSCA
from torch_geometric.read import read_txt_array
from torch_geometric.utils import to_undirected


class TOSCAEXT(TOSCA):
    r"""The TOSCA dataset from the `"Numerical Geometry of Non-Ridig Shapes"
        <https://www.amazon.com/Numerical-Geometry-Non-Rigid-Monographs-Computer/
        dp/0387733000>`_ book, containing 80 meshes.
        Meshes within the same category have the same triangulation and an equal
        number of vertices numbered in a compatible way.

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
            categories (list, optional): List of categories to include in the
                dataset. Can include the categories :obj:`"Cat"`, :obj:`"Centaur"`,
                :obj:`"David"`, :obj:`"Dog"`, :obj:`"Gorilla"`, :obj:`"Horse"`,
                :obj:`"Michael"`, :obj:`"Victoria"`, :obj:`"Wolf"`. If set to
                :obj:`None`, the dataset will contain all categories. (default:
                :obj:`None`)
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

    url = 'http://tosca.cs.technion.ac.il/data/toscahires-asci.zip'

    categories = [
        'cat', 'centaur', 'david', 'dog', 'gorilla', 'horse', 'michael',
        'victoria', 'wolf'
    ]

    def __init__(self,
                 root,
                 num=1000,
                 categories=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 processed_file_prefix='data'):
        categories = self.categories if categories is None else categories
        categories = [cat.lower() for cat in categories]
        for cat in categories:
            assert cat in self.categories
        self.num = num
        self.categories = categories
        self.processed_file_prefix = processed_file_prefix
        super(TOSCAEXT, self).__init__(root, categories,
                                       transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        return '{}_{}.pt'.format(self.processed_file_prefix,
                                 '_'.join([cat[:2] for cat in self.categories]))

    def process(self):
        data_list = []
        for cat in self.categories:
            paths = glob.glob(osp.join(self.raw_dir, '{}*.tri'.format(cat)))
            paths = [path[:-4] for path in paths]
            paths = sorted(paths, key=lambda e: (len(e), e))

            for path in paths:
                pos = read_txt_array('{}.vert'.format(path))
                face = read_txt_array('{}.tri'.format(path), dtype=torch.long)

                face = face.t().contiguous() - 1

                random_list = sorted(random.sample(range(int(face.size(1) - 1)),
                                                   self.num))
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

                data = Data(pos=pos, face=face,
                            edge_index=edge_index,
                            y=self.categories.index(cat))

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


