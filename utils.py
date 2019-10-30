import os
from glob import glob
from shutil import rmtree
import torch


class Checkpoint:
    def __init__(self, args):
        self.args = args
        self.model_checkpoints_folder = os.path.join(args.checkpoints_folder, args.model_name)
        if not args.resume:
            rmtree(self.model_checkpoints_folder)
        os.makedirs(self.model_checkpoints_folder, exist_ok=True)
        self.last_checkpoint = self.__get_last_saved_checkpoint(best=False)
        self.best_checkpoint = self.__get_last_saved_checkpoint(best=True)

    def __get_last_saved_checkpoint(self, best):
        file_path_pattern = self.__get_file_path('*', best)
        all_checkpoints = glob(file_path_pattern)
        return max(all_checkpoints) if all_checkpoints else None

    def __get_file_path(self, episode, best):
        file_name = ('{class_num}way_{sample_num_per_class}shot_'
                     'ep{episode}.tar').format(**vars(self.args), episode=episode)
        if best:
            file_name = 'best_' + file_name
        return os.path.join(self.model_checkpoints_folder, file_name)

    def save(self, best, **kwargs):      
        file_path = self.__get_file_path(kwargs['episode'], best)
        for k, v in kwargs.items():
            try:
                kwargs[k] = v.state_dict()
            except AttributeError:
                pass
        torch.save(kwargs, file_path)
        if best:
            self.best_checkpoint = file_path
        else:
            self.last_checkpoint = file_path
        self.__remove_old_checkpoints(best)

    def __remove_old_checkpoints(self, best, keep=3):
        file_path_pattern = self.__get_file_path('*', best)
        all_checkpoints = glob(file_path_pattern)
        for c in sorted(all_checkpoints)[:-keep]:
            os.remove(c)

    def load(self, best, **kwargs):
        checkpoint_file = self.best_checkpoint if best else self.last_checkpoint
        try:
            checkpoint = torch.load(checkpoint_file)
        except AttributeError:
            raise FileNotFoundError('Best model checkpoint not found.')

        for k, v in kwargs.items():
            v.load_state_dict(checkpoint[k])

        if not best:
            self.args.start_episode = checkpoint['episode'] + 1
        print(f'Checkpoint at episode {checkpoint["episode"]} loaded')

    def get_saved_accuracy(self):
        try:
            accuracy = torch.load(self.best_checkpoint)['accuracy']
        except (OSError, AttributeError):
            accuracy = 0.0
        return accuracy