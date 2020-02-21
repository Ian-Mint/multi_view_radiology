from utils import create_input_files

if __name__ == '__main__':
        # Create input files (along with word map)
        create_input_files(dataset='iu-x-ray',
                           karpathy_json_path='data/report.json',
                           image_folder='data/png',
                           captions_per_image=1,
                           min_word_freq=2,
                           output_folder='data/medical_data_new/',
                           max_len=100)
