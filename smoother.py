import os
from glob import glob
from os import path
from statistics import mean

import pandas as pd
from path import Path

from body_part_not_available import BodyPartPointNotAvailable
from normalizer import Normalizer


class Smoother:

    @staticmethod
    def run(frames_dir, project_dir, wrapper_input_dir):
        wrapper_output_dir = "centered-keypoints"

        frames_dir_full_path = f"{project_dir}/{wrapper_input_dir}/{frames_dir}"
        frame_file_full_path = f"{frames_dir_full_path}/{frames_dir}.mov-[frame_idx]-[person_idx].csv"

        print(f"Frame file full path={frame_file_full_path}")

        input_files = [Path(f).abspath() for f in glob(frames_dir_full_path + '/*')]

        print(f"Input path includes {len(input_files)} files")

        all_persons_files = Smoother.get_person_files(input_files, frame_file_full_path)

        # Collecting frame data for the person with index 0
        person_idx_to_collect = 0
        person_frame_files = all_persons_files[person_idx_to_collect]

        frames_list = [Smoother.get_frame_data(frame_file) for frame_file in person_frame_files]

        print(f"Imported data for {len(frames_list)} frames")

        # for body_part_idx in range(0, 25):
        #   frames_list = Smoother.smooth_average(frames_list, body_part_idx)

        # for body_part_idx in range(0, 25):
        #    frames_list = Smoother.fill_body_part_data_with_averages(frames_list, body_part_idx)

        first_frame_df = frames_list[0]
        old_hip_x = float(first_frame_df.iloc[8, 0:1])
        old_hip_y = float(first_frame_df.iloc[8, 1:2])

        for body_part_idx in range(0, 25):
            frames_list = Normalizer.normalize_to_center(frames_list,
                                                         old_hip_x=old_hip_x,
                                                         old_hip_y=old_hip_y,
                                                         body_part_nr=body_part_idx)

        output_frames_root_path = f"{project_dir}/{wrapper_output_dir}/{frames_dir}"

        if not path.exists(output_frames_root_path):
            print(f"Creating output dir={output_frames_root_path}")
            os.mkdir(output_frames_root_path)

        for idx, fixed_frame in enumerate(frames_list):
            frame_file_full_path = f"{output_frames_root_path}/{frames_dir}.mov-[frame_idx]-[person_idx].csv"
            frame_file_full_path = frame_file_full_path.replace('[frame_idx]', str(idx))
            frame_file_full_path = frame_file_full_path.replace('[person_idx]', str(person_idx_to_collect))

            fixed_frame.to_csv(frame_file_full_path, index=False)

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

    @staticmethod
    def get_body_part_y_data(frame, body_part_nr=0):
        return float(frame.iloc[body_part_nr, 1:2])

    @staticmethod
    def get_body_part_x_data(frame, body_part_nr=0):
        return float(frame.iloc[body_part_nr, 0:1])

    @staticmethod
    def get_new_body_part_data(body_part_data):
        new_body_part_data = []

        for idx, body_part in enumerate(body_part_data):
            if idx == 0:  # Handle first data point
                try:
                    new_body_part_data.append(Smoother.get_next_available_body_part_point(idx, body_part_data))
                except BodyPartPointNotAvailable:
                    print("WARN! Body part not available at all...")
                    new_body_part_data = body_part_data
                    break
                continue

            previous_body_part = new_body_part_data[idx - 1]

            # print(
            #    f"Body part frame={idx} {body_part} - {previous_body_part} = {body_part - previous_body_part} is {body_part == 0}")

            if body_part == 0:
                try:
                    next_available = Smoother.get_next_available_body_part_point(idx, body_part_data)
                except BodyPartPointNotAvailable:
                    next_available = previous_body_part

                new_average = mean([previous_body_part, next_available])

                new_body_part_data.append(new_average)
            else:
                new_body_part_data.append(body_part)

        return new_body_part_data

    @staticmethod
    def get_next_available_body_part_point(start, body_part_data):
        for idx in range(start, len(body_part_data)):
            if (idx + 1) == len(body_part_data):
                raise BodyPartPointNotAvailable

            if body_part_data[idx]:
                return body_part_data[idx]

    @staticmethod
    def smooth_average(frame_data, body_part_nr=0):
        current_body_part_x_data = [Smoother.get_body_part_x_data(frame, body_part_nr) for idx, frame in
                                    enumerate(frame_data)]
        current_body_part_y_data = [Smoother.get_body_part_y_data(frame, body_part_nr) for idx, frame in
                                    enumerate(frame_data)]

        body_part_x_data_series = pd.Series(current_body_part_x_data)
        body_part_y_data_series = pd.Series(current_body_part_y_data)

        rolling_x = body_part_x_data_series.rolling(window=3)
        rolling_x_mean = rolling_x.mean()

        rolling_y = body_part_y_data_series.rolling(window=3)
        rolling_y_mean = rolling_y.mean()

        # Manual fixing
        rolling_x_mean[0] = current_body_part_x_data[0]
        rolling_x_mean[1] = current_body_part_x_data[1]
        rolling_y_mean[0] = current_body_part_y_data[0]
        rolling_y_mean[1] = current_body_part_y_data[1]

        # Substitute old body part data with new
        return Smoother.substitute_body_part_data(body_part_nr, frame_data, rolling_x_mean,
                                                  rolling_y_mean)

    @staticmethod
    def fill_body_part_data_with_averages(frame_data, body_part_nr=0):
        current_body_part_y_data = [Smoother.get_body_part_y_data(frame, body_part_nr) for idx, frame in
                                    enumerate(frame_data)]
        current_body_part_x_data = [Smoother.get_body_part_x_data(frame, body_part_nr) for idx, frame in
                                    enumerate(frame_data)]

        new_body_part_y_data = Smoother.get_new_body_part_data(current_body_part_y_data)
        new_body_part_x_data = Smoother.get_new_body_part_data(current_body_part_x_data)

        # Substitute old body part data with new
        return Smoother.substitute_body_part_data(body_part_nr, frame_data, new_body_part_x_data,
                                                  new_body_part_y_data)

    @staticmethod
    def substitute_body_part_data(body_part_nr, frame_data, new_body_part_x_data, new_body_part_y_data):
        return [Smoother.get_new_frame(old_frame, body_part_nr, new_body_part_x_data[frame_idx],
                                       new_body_part_y_data[frame_idx]) for frame_idx, old_frame in
                enumerate(frame_data)]

    @staticmethod
    def get_new_frame(old_frame, body_part_nr, new_body_part_x_frame_data, new_body_part_y_frame_data):
        new_frame = old_frame.copy()

        new_frame.iloc[body_part_nr, 0:1] = new_body_part_x_frame_data
        new_frame.iloc[body_part_nr, 1:2] = new_body_part_y_frame_data

        return new_frame


if __name__ == '__main__':
    Smoother.run(frames_dir="backflip-1-allar",
                 project_dir="/Users/allarviinamae/EduWorkspace/openpose-jupyter-data-exploration",
                 wrapper_input_dir="raw-keypoints")
