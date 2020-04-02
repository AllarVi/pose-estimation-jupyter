import os
from glob import glob
from os import path

import pandas as pd
from path import Path


class Smoother:

    @staticmethod
    def run(frames_dir):
        frames_root_path = f"/Users/allarviinamae/EduWorkspace/openpose-jupyter-data-exploration/raw-keypoints/{frames_dir}"
        output_frame_data_path = f"{frames_root_path}/{frames_dir}.mov-[frame_idx]-[person_idx].csv"

        print(f"Frame data path={output_frame_data_path}")

        input_files = [Path(f).abspath() for f in glob(frames_root_path + '/*')]

        print(f"Root path includes {len(input_files)} files")

        all_persons_files = Smoother.get_person_files(input_files, output_frame_data_path)

        # Collecting frame data for the person with index 0
        person_idx_to_collect = 0
        person_frame_files = all_persons_files[person_idx_to_collect]

        frame_data = [Smoother.get_frame_data(frame_file) for frame_file in person_frame_files]

        print(f"Imported data for {len(frame_data)} frames")

        fixed_frame_data = [frame.copy() for frame in frame_data]

        for idx in range(0, 25):
            fixed_frame_data = Smoother.fix_body_part_data(fixed_frame_data, idx)

        output_frames_root_path = f"/Users/allarviinamae/EduWorkspace/openpose-jupyter-data-exploration/output-keypoints/{frames_dir}"

        if not path.exists(output_frames_root_path):
            print(f"Creating output dir={output_frames_root_path}")
            os.mkdir(output_frames_root_path)

        for idx, fixed_frame in enumerate(fixed_frame_data):
            output_frame_data_path = f"{output_frames_root_path}/{frames_dir}.mov-[frame_idx]-[person_idx].csv"
            output_frame_data_path = output_frame_data_path.replace('[frame_idx]', str(idx))
            output_frame_data_path = output_frame_data_path.replace('[person_idx]', str(person_idx_to_collect))

            fixed_frame.to_csv(output_frame_data_path, index=False)

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
            if idx == 0:
                new_body_part_data.append(body_part)
                continue

            previous_body_part = new_body_part_data[idx - 1]

            print(
                f"Body part frame={idx} {body_part} - {previous_body_part} = {body_part - previous_body_part} is {body_part == 0}")

            if body_part == 0:
                new_body_part_data.append(previous_body_part)
            else:
                new_body_part_data.append(body_part)

        return new_body_part_data

    @staticmethod
    def fix_body_part_data(frame_data, body_part_nr=0):
        body_part_y_data = [Smoother.get_body_part_y_data(frame, body_part_nr) for idx, frame in enumerate(frame_data)]
        body_part_x_data = [Smoother.get_body_part_x_data(frame, body_part_nr) for idx, frame in enumerate(frame_data)]

        new_body_part_y_data = Smoother.get_new_body_part_data(body_part_y_data)
        new_body_part_x_data = Smoother.get_new_body_part_data(body_part_x_data)

        for idx, frame in enumerate(frame_data):
            frame.iloc[body_part_nr, 0:1] = new_body_part_x_data[idx]
            frame.iloc[body_part_nr, 1:2] = new_body_part_y_data[idx]

            frame_data[idx] = frame

        return frame_data


if __name__ == '__main__':
    Smoother.run(frames_dir="backflip-1-allar")
