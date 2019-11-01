import os
from glob import glob
from shutil import rmtree
import torch


class Checkpoint:
    def __init__(self, args):
        self.args = args
        self.checkpoints_folder = self.__create_folder()

        self.__file_sort_key = lambda path: int(os.path.basename(path).split('_')[0])
        self.__checkpoint_file_pattern = os.path.join(self.checkpoints_folder, '{}_chkpt.tar')
        self.__best_file_pattern = os.path.join(self.checkpoints_folder, '{}_best.tar')

        self.last_checkpoint = self.__get_last_saved_file(self.__checkpoint_file_pattern)
        self.best_checkpoint = self.__get_last_saved_file(self.__best_file_pattern)

    def __create_folder(self):
        folder = '{model_name}_{class_num}way_{sample_num_per_class}shot'.format(**vars(self.args))
        folder = os.path.join(self.args.checkpoints_folder, folder)
        if not (self.args.resume or self.args.test):
            try:
                rmtree(folder)
            except FileNotFoundError:
                pass
        os.makedirs(folder, exist_ok=True)
        return folder

    def __get_last_saved_file(self, pattern):
        checkpoints = glob(pattern.format('*'))
        try:
            return max(checkpoints, key=self.__file_sort_key)
        except ValueError:
            return None

    def save(self, **kwargs):
        file_path = self.__save_file(self.__checkpoint_file_pattern, **kwargs)
        self.last_checkpoint = file_path
        self.__remove_old_files(self.__checkpoint_file_pattern)

    def save_best(self, **kwargs):
        file_path = self.__save_file(self.__best_file_pattern, **kwargs)
        self.best_checkpoint = file_path
        self.__remove_old_files(self.__best_file_pattern)

    def __save_file(self, pattern, **kwargs):
        file_path = pattern.format(kwargs['episode'])
        for k, v in kwargs.items():
            try:
                kwargs[k] = v.state_dict()
            except AttributeError:
                pass
        torch.save(kwargs, file_path)
        return file_path

    def __remove_old_files(self, pattern, keep=3):
        files = glob(pattern.format('*'))
        files = sorted(files, key=self.__file_sort_key)
        for f in files[:-keep]:
            os.remove(f)

    def load(self, **kwargs):
        checkpoint = self.__load(self.last_checkpoint, **kwargs)
        self.args.start_episode = checkpoint['episode'] + 1

    def load_best(self, **kwargs):
        self.__load(self.best_checkpoint, **kwargs)

    def __load(self, checkpoint_file, **kwargs):
        try:
            checkpoint = torch.load(checkpoint_file)
        except AttributeError:
            raise FileNotFoundError('Checkpoint not found.')

        for k, v in kwargs.items():
            v.load_state_dict(checkpoint[k])
        print(f'Checkpoint at episode {checkpoint["episode"]} loaded')
        return checkpoint

    def get_saved_accuracy(self):
        try:
            accuracy = torch.load(self.best_checkpoint)['accuracy']
        except (OSError, AttributeError):
            accuracy = 0.0
        return accuracy
