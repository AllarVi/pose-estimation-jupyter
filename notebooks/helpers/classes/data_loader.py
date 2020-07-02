from glob import glob

import numpy as np
import pandas as pd
from keras.utils import to_categorical
from path import Path


class DataLoader:

    @staticmethod
    def filter_active_keypoints(frame_df):
        # active_keypoints = [0, 1, 8, 2, 3, 4, 5, 6, 7] # trunk

        active_keypoints = [0, 1, 8, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]

        filtered_df = frame_df[frame_df.index.isin(active_keypoints)]
        # filtered_df = frame_df

        return filtered_df

    @staticmethod
    def get_frame_data(filepath):
        frame_data = pd.read_csv(filepath)

        x_data = DataLoader.filter_active_keypoints(frame_data.iloc[:, 0])
        y_data = DataLoader.filter_active_keypoints(frame_data.iloc[:, 1])

        # print(f"x_data shape: {x_data.shape}")
        # print(f"y_data shape: {y_data.shape}")

        h_stacked = np.hstack((x_data, y_data))
        # h_stacked = np.hstack((y_data))

        return h_stacked

    @staticmethod
    def get_frame_file_path(frame_file_path_template, frame_idx):
        frame_file_path = frame_file_path_template.replace("[frame_idx]", str(frame_idx))
        return frame_file_path

    @staticmethod
    def get_label(sample_dir_name):
        return 0 if sample_dir_name[0] == 'b' else 1

    @staticmethod
    def get_y_labels(sample_dir_names):
        return [DataLoader.get_label(sample_dir_name) for sample_dir_name in sample_dir_names]

    @staticmethod
    def get_categorical_y_labels(y_labels):
        y_labels_stacked = np.dstack(y_labels)
        print(f"y_labels_stacked shape: {y_labels_stacked.shape}")

        y_labels_categorical = to_categorical(y_labels_stacked)
        print(f"y_labels_categorical shape: {y_labels_categorical.shape}")

        y_labels_squeezed = np.squeeze(y_labels_categorical)
        print(f"y_labels_squeezed shape {y_labels_squeezed.shape}")

        (y_rows, y_cols) = y_labels_squeezed.shape
        y_labels_list = [[y_labels_squeezed[i, 0], y_labels_squeezed[i, 1]] for i in range(y_rows)]

        for idx, y_label in enumerate(y_labels_list):
            if idx == 5:
                break
            print(f"y_label categorical: {y_label}")

        return y_labels_list

    @staticmethod
    def get_frames_count(root_path, sample_dir_name):
        return len([Path(f).abspath() for f in glob(f"{root_path}/{sample_dir_name}" + '/*')])

    @staticmethod
    def get_frames(root_path, sample_dir_name):
        frames = []
        for frame_idx in range(0, DataLoader.get_frames_count(root_path, sample_dir_name)):
            frame_file_path_template = f"{root_path}/{sample_dir_name}/{sample_dir_name}.mov-[frame_idx]-0.csv"
            frame_file_path = DataLoader.get_frame_file_path(frame_file_path_template, frame_idx)
            frame_data = DataLoader.get_frame_data(frame_file_path)

            frames.append(frame_data)

        frames = np.dstack(frames)
        squeezed = np.squeeze(frames)
        axes_swapped = np.swapaxes(squeezed, 0, 1)

        return axes_swapped

    @staticmethod
    def get_sample_idx_by_frames_count(frames_count, samples):
        sample_idx = 0

        for sample in samples:
            if len(sample) == frames_count:
                return sample_idx
            sample_idx = sample_idx + 1

    @staticmethod
    def get_sample_name_by_frames_count(frames_count, samples, sample_dir_names):
        sample_idx = DataLoader.get_sample_idx_by_frames_count(frames_count, samples)
        return sample_dir_names[sample_idx]

    @staticmethod
    def get_samples_list(sample_dir_names, root_path):
        samples = []
        sample_dir_names_count = len(sample_dir_names)
        for sample_dir_name_idx, sample_dir_name in enumerate(sample_dir_names):
            print(f"Loading frames for {sample_dir_name_idx}/{sample_dir_names_count}")
            frames = DataLoader.get_frames(root_path, sample_dir_name)
            samples.append(frames)

        return samples
