import os

from utils.file_path import FilePath
from utils.util import read_file


class ReadConfig:
    def __init__(self):
        pass

    def read_experiment_1_path(self):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR, os.pardir))
        path = "{0}{1}".format(ROOT_DIR, '/config/experiments/experiment_1.yml')
        return path

    def read_experiment_2_path(self):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR, os.pardir))
        path = "{0}{1}".format(ROOT_DIR, '/config/experiments/experiment_2.yml')
        return path

    def read_data_path(self):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR, os.pardir))
        path = "{0}{1}".format(ROOT_DIR, '/config/data_config/data_path.yaml')
        content_dict = read_file(file_path=path)['contents']

        return content_dict

    def parse_file_path(self):
        content_dict = self.read_data_path()
        file_path = FilePath()
        file_path.common_path = content_dict['common_path']
        file_path.raw_data_dir = file_path.common_path + content_dict[
            'raw_data_dir']
        file_path.series_dir = file_path.common_path + content_dict[
            'series_dir']
        file_path.spectrogram_dir = file_path.common_path + content_dict[
            'spectrogram_dir']
        file_path.training_dir = file_path.common_path + content_dict[
            'training_dir']
        return file_path


if __name__ == "__main__":
    rc = ReadConfig()
    data = rc.read_data_path()
    print(data)
