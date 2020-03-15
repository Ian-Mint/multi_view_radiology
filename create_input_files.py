from utils import create_input_files, create_input_files_pair

if __name__ == '__main__':
        # Create input files (along with word map)
        create_input_files_pair(dataset='iu-x-ray',
                               karpathy_json_path='preprocessing/Img_Report2.json',
                               image_folder='data/png',
                               captions_per_image=1,
                               min_word_freq=2,
                               output_folder='data/medical_data_new_pair/',
                               max_len=100)
