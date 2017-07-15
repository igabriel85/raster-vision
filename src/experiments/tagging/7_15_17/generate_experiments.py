from os.path import join
from copy import deepcopy

from rastervision.experiment_generator import (
    ExperimentGenerator, get_parent_dir)


class TestExperimentGenerator(ExperimentGenerator):
    def generate_experiments(self):
        base_exp = {
            'batch_size': 8,
            'git_commit': '7f79ca',
            'problem_type': 'tagging',
            'dataset_name': 'planet_kaggle',
            'generator_name': 'jpg',
            'active_input_inds': [0, 1, 2],
            'use_pretraining': True,
            'optimizer': 'sgd',
            'lr_schedule': [[0, 1e-2], [10, 1e-3], [20, 1e-4]],
            'nesterov': True,
            'momentum': 0.9,
            'train_ratio': 0.8,
            'nb_eval_plot_samples': 100,
            'epochs': 30,
            'validation_steps': 480,
            'augment_methods': ['hflip', 'vflip', 'rotate90'],
            'run_name': 'tagging/7_15_17/sgd',
            'steps_per_epoch': 2400
        }

        model_types = ['baseline_resnet', 'densenet121']

        exps = []
        exp_count = 0
        for model_type in model_types:
            exp = deepcopy(base_exp)
            exp['run_name'] = join(exp['run_name'], str(exp_count))
            exp['model_type'] = model_type
            if model_type == 'baseline_resnet':
                exp['batch_size'] = int(exp['batch_size'] * 4)
                exp['validation_steps'] = int(exp['validation_steps'] / 4)
                exp['steps_per_epoch'] = int(exp['steps_per_epoch'] / 4)
            exps.append(exp)
            exp_count += 1

        return exps


if __name__ == '__main__':
    path = get_parent_dir(__file__)
    gen = TestExperimentGenerator().run(path)
