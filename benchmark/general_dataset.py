import os

class GeneralDataset():
    def __init__(self, folder_name, folder_path=None):
        self.folder_name = folder_name
        self.input_has_image, self.input_has_text = False, False
        self.output_has_image, self.output_has_text = False, False

        dir_path = folder_path+'{}/'.format(str(self.folder_name))

        #used to load the images
        def image_loader(path):
            image_paths = []
            file_names = os.listdir(path)
            file_len = len(file_names)
            for file in range(file_len):
                path = f"{path}{str(file)}.jpg"
                image_paths.append(path)

            return image_paths

        #used to load the txt files
        def text_loader(path):
            text = []
            with open(path) as f:
                lines = f.readlines()
                for line in lines:
                    text.append(line)
            f.close()
            return text

        #used to select image_loader or text loader
        def file_loader(path: str, source: str = 'input'):
            files = {
                "text": [],
                "image": []
            }

            file_names = os.listdir(path)
            for file in file_names:
                if file[-3:] == 'txt':
                    files['text'].append(text_loader(path+file))
                else:
                    files['image'].append(image_loader(path+file+'/'))

            return files

        input_path = dir_path + 'inputs/'
        output_path = dir_path + 'outputs/'

        test_path = os.listdir(output_path)

        self.input_files = file_loader(input_path, "input")
        self.output_files = file_loader(output_path, "output")




    def __len__(self):
        return len(self.input_files[0])

    def __getitem__(self, idx):
        #transform the images to embeddings
        input_files = {
            "text": self.input_files['text'][idx] if len(self.input_files['text']) > 0 else None,
            "image": self.input_files['image'][idx] if len(self.input_files['image']) > 0 else None
        }
        output_files = {
            "text": self.output_files['text'][idx] if len(self.output_files['text']) > 0 else None,
            "image": self.output_files['image'][idx] if len(self.output_files['image']) > 0 else None
        }

        return {'input':input_files, 'output':output_files}
