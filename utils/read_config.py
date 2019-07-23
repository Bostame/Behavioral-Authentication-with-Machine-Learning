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
        file_path.raw_data = content_dict['raw_data']
        file_path.output_path = content_dict['output_path']
        file_path.test_name = content_dict['test_name']
        file_path.output_path = file_path.output_path + "/" + file_path.test_name
        file_path.series_dir = file_path.output_path + "/series"
        file_path.spectrogram_dir = file_path.output_path + "/spectrogram"
        file_path.threeChannelSpectrogram = file_path.output_path + \
                                            "/3CTrackings"
        file_path.training_dir = file_path.output_path + "/training"
        file_path.model_dir = file_path.output_path + "/model"
        return file_path


if __name__ == "__main__":
    rc = ReadConfig()
    data = rc.read_data_path()
    print(data)
