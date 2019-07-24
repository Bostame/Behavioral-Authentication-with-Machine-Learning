from prep.data_prep.preprocessing import SpectrogramGeneration
from prep.data_prep.process_data import ProcessData
from prep.data_prep.records_generator import RecordsGenerator
from utils.read_config import ReadConfig

if __name__ == '__main__':
    rc = ReadConfig()
    file_path = rc.parse_file_path()

    # Create Series Data
    proc_data = ProcessData()
    proc_data.create_series(file_path.raw_data, file_path.series_dir)

    # Create Spectrogram Data
    sp_gen = SpectrogramGeneration()
    sp_gen.trackings_generator(series_path=file_path.series_dir,
                               spectrogram_path=file_path.spectrogram_dir)
    # Image Size: 50* 100
    sp_gen.generate_3channel_longer(spectrogram_dir=file_path.spectrogram_dir,
                                    threeChannelSpectrogram=file_path.threeChannelSpectrogram)

    # Generate TFRecords
    rg = RecordsGenerator()
    if file_path.train_test_split:
        rg._20190712_generate_records_10classes_train_test_split(
            file_path.threeChannelSpectrogram, file_path.training_dir)
    else:
        rg._20190714_generate_records_10classes_acgy_longer(
            file_path.threeChannelSpectrogram, file_path.training_dir)
