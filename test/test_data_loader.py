import pickle
import pytest

from vocabulary import Vocabulary
from data_loader import get_loader

import config


@pytest.mark.skip(reason="Unpickling fails in the pytest environment")
class TestDataLoader(object):
    def setUp(self) -> None:
        with open(config.vocab_file_path, 'rb') as f:
            self.vocab = pickle.load(f)

    def tearDown(self):
        pass

    def test_data_loader_runs(self):
        self.setUp()
        data_loader = get_loader(root=config.data_dir,
                                 vocab=self.vocab,
                                 img_report_path=config.img_report_file,
                                 transform=None,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0)




if __name__ == '__main__':
    tdl = TestDataLoader()

    tdl.test_data_loader_runs()
