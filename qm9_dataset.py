from torch_geometric.datasets import QM9


class QM9Ext(QM9):
    r"""The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
        Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
        about 130,000 molecules with 13 regression targets.
        Each molecule includes complete spatial information for the single low
        energy conformation of the atoms in the molecule.
        In addition, we provide the atom features from the `"Neural Message
        Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.

        Args:
            root (string): Root directory where the dataset should be saved.
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

    url = 'http://www.roemisch-drei.de/qm9.tar.gz'

    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 processed_filename='data.pt'):
        self.processed_filename = processed_filename
        super(QM9Ext, self).__init__(root, transform,
                                     pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        return self.processed_filename
