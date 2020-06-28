import os

from notebooks.helpers.classes.data_augmentation import DataAugmentation
from smoother import Smoother


class DataProcessor:

    @staticmethod
    def main(strategy_name):
        project_dir = "/Users/allarviinamae/EduWorkspace/openpose-jupyter-data-exploration"
        wrapper_input_dir = "centered-keypoints"

        input_full_path = f"{project_dir}/{wrapper_input_dir}"

        frames_dirs = [name for name in os.listdir(input_full_path) if os.path.isdir(f"{input_full_path}/{name}")]

        if 'smoother' == strategy_name:
            strategy = Smoother()
        elif 'augmentation' == strategy_name:
            strategy = DataAugmentation()
        else:
            strategy = DataAugmentation()

        for frames_dir in frames_dirs:
            strategy.run(frames_dir, project_dir, wrapper_input_dir)


if __name__ == '__main__':
    DataProcessor.main(strategy_name="augmentation")
