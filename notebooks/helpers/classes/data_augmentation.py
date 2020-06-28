import os
from glob import glob
from os import path

import pandas as pd
from path import Path


class DataAugmentation:

    @staticmethod
    def run(frames_dir, project_dir, wrapper_input_dir):
        wrapper_output_dir = "augmented-keypoints"

        frames_dir_full_path = f"{project_dir}/{wrapper_input_dir}/{frames_dir}"
        frame_file_full_path = f"{frames_dir_full_path}/{frames_dir}.mov-[frame_idx]-[person_idx].csv"

        print(f"Frame file full path={frame_file_full_path}")

        input_files = [Path(f).abspath() for f in glob(frames_dir_full_path + '/*')]

        print(f"Input path includes {len(input_files)} files")

        all_persons_files = DataAugmentation.get_person_files(input_files, frame_file_full_path)

        # Collecting frame data for the person with index 0
        person_idx_to_collect = 0
        person_frame_files = all_persons_files[person_idx_to_collect]

        frames_list = [DataAugmentation.get_frame_data(frame_file) for frame_file in person_frame_files]

        print(f"Imported data for {len(frames_list)} frames")

        # for body_part_idx in range(0, 25):
        #   frames_list = Smoother.smooth_average(frames_list, body_part_idx)

        # for body_part_idx in range(0, 25):
        #    frames_list = Smoother.fill_body_part_data_with_averages(frames_list, body_part_idx)

        DataAugmentation.save_new_sample(frames_dir,
                                         frames_list,
                                         project_dir,
                                         wrapper_output_dir)

    @staticmethod
    def save_new_sample(frames_dir, frames_list, project_dir, wrapper_output_dir):
        output_frames_root_path = DataAugmentation.create_output_dirs(frames_dir, project_dir, wrapper_output_dir)

        for idx, fixed_frame in enumerate(frames_list):
            frame_file_full_path = f"{output_frames_root_path}/{frames_dir}.mov-[frame_idx]-0.csv"
            frame_file_full_path = frame_file_full_path.replace('[frame_idx]', str(idx))

            fixed_frame.to_csv(frame_file_full_path, index=False)

    @staticmethod
    def create_output_dirs(frames_dir, project_dir, wrapper_output_dir):
        output_wrapper_path = f"{project_dir}/{wrapper_output_dir}"

        if not path.exists(output_wrapper_path):
            print(f"Creating output wrapper dir={output_wrapper_path}")
            os.mkdir(output_wrapper_path)

        output_frames_root_path = f"{output_wrapper_path}/{frames_dir}"

        if not path.exists(output_frames_root_path):
            print(f"Creating output dir={output_frames_root_path}")
            os.mkdir(output_frames_root_path)

        return output_frames_root_path

    @staticmethod
    def get_frame_data(frame_file):
        try:
            frame_data = pd.read_csv(frame_file)

            return frame_data
        except FileNotFoundError:
            print(f"Frame data not found for frame {frame_file}")
            return pd.DataFrame()

    @staticmethod
    def get_person_files(input_files, frame_data_path):
        all_persons_files = {}
        max_persons = 10

        for input_file_idx in range(len(input_files)):
            for person_idx in range(max_persons):
                expected_file = frame_data_path.replace("[frame_idx]", str(input_file_idx))
                expected_file = expected_file.replace("[person_idx]", str(person_idx))

                if expected_file in input_files:
                    person_files = []

                    if person_idx in all_persons_files:
                        person_files = all_persons_files[person_idx]

                    person_files.append(expected_file)

                    all_persons_files[person_idx] = person_files

        persons_indeces = all_persons_files.keys()
        print(f"Found {len(persons_indeces)} persons")
        for person_idx in persons_indeces:
            print(f"Person {person_idx} has {len(all_persons_files[person_idx])} frame files")

        return all_persons_files


if __name__ == '__main__':
    DataAugmentation.run(frames_dir="backflip-1-allar",
                         project_dir="/Users/allarviinamae/EduWorkspace/openpose-jupyter-data-exploration",
                         wrapper_input_dir="centered-keypoints")
