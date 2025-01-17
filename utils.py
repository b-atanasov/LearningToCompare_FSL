import os
from glob import glob
from shutil import rmtree
import sys
import logging
import torch


class LogDir:
    def __init__(self, args):
        self.args = args
        self.__create()

    def __create(self):
        path = self.get_path()
        path_exists = os.path.exists(path)
        if path_exists and not (self.args.resume or self.args.test):
            self.__delete(path)    
        os.makedirs(path, exist_ok=True)
        return path

    def get_path(self):
        path = '{class_num}way_{sample_num_per_class}shot_{backbone}_{img_size}_{loss_type}'
        path = path.format(**vars(self.args))
        if self.args.enable_ctm:
            path += '_ctm'
        path = os.path.join('logs', path)
        return path

    def __delete(self, path):
        if self.__confirm_delete(path):
            rmtree(path)
        else:
            sys.exit()

    def __confirm_delete(self, path):
        input_text = f'Are you sure you want to delete the checkpoints in {path}? (y/n)'
        while True:
            confirmation = input(input_text).lower()
            if confirmation in ['y', 'n']:
                break
        return confirmation == 'y'


def configure_logging(args):
    log_dir = LogDir(args).get_path()
    log_file = 'test.log' if args.test else 'train.log'
    log_file = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    handlers = [file_handler, console_handler]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(message)s', handlers=handlers,
                        datefmt='%Y-%m-%dT%H:%M:%S')
    return log_dir


class Checkpoint:
    def __init__(self, args, log_dir):
        self.args = args
        self.checkpoints_folder = self.__create_folder(log_dir)

        self.__file_sort_key = lambda path: int(os.path.basename(path).split('_')[0])
        self.__checkpoint_file_pattern = os.path.join(self.checkpoints_folder, '{}_chkpt.tar')
        self.__best_file_pattern = os.path.join(self.checkpoints_folder, '{}_best.tar')

        self.last_checkpoint = self.__get_last_saved_file(self.__checkpoint_file_pattern)
        self.best_checkpoint = self.__get_last_saved_file(self.__best_file_pattern)

    def __create_folder(self, log_dir):
        folder = os.path.join(log_dir, 'checkpoints')
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
        logging.info('Checkpoint at episode %d loaded', checkpoint["episode"])
        return checkpoint

    def get_saved_accuracy(self):
        try:
            accuracy = torch.load(self.best_checkpoint)['accuracy']
        except (OSError, AttributeError):
            accuracy = 0.0
        return accuracy


class Accuracy:
    def __init__(self):
        self.number_correct_predicitons = 0
        self.total_number_of_predictions = 0

    def add_batch(self, input, target):
        self.number_correct_predicitons += (input.argmax(1) == target).sum().data.item()
        self.total_number_of_predictions += input.size()[0]

    def calculate(self):
        return self.number_correct_predicitons / self.total_number_of_predictions
