import pandas as pd


class Normalizer:

    @staticmethod
    def get_frame_data(frame_file):
        try:
            frame_data = pd.read_csv(frame_file)

            return frame_data
        except FileNotFoundError:
            print(f"Frame data not found for frame {frame_file}")
            return pd.DataFrame()

    @staticmethod
    def get_body_part_y_data(frame, body_part_nr=0):
        return float(frame.iloc[body_part_nr, 1:2])

    @staticmethod
    def get_body_part_x_data(frame, body_part_nr=0):
        return float(frame.iloc[body_part_nr, 0:1])

    @staticmethod
    def normalize_to_center(frame_data, old_hip_x, old_hip_y, body_part_nr=0):
        current_body_part_x_data = [Normalizer.get_body_part_x_data(frame, body_part_nr) for frame in
                                    frame_data]
        current_body_part_y_data = [Normalizer.get_body_part_y_data(frame, body_part_nr) for frame in
                                    frame_data]

        new_body_part_x_data = list(map(lambda x: x - old_hip_x, current_body_part_x_data))

        new_body_part_y_data = list(map(lambda x: x - old_hip_y, current_body_part_y_data))

        # Substitute old body part data with new
        return Normalizer.substitute_body_part_data(body_part_nr, frame_data, new_body_part_x_data,
                                                    new_body_part_y_data)

    @staticmethod
    def substitute_body_part_data(body_part_nr, frame_data, new_body_part_x_data, new_body_part_y_data):
        return [Normalizer.get_new_frame(old_frame, body_part_nr, new_body_part_x_data[frame_idx],
                                         new_body_part_y_data[frame_idx]) for frame_idx, old_frame in
                enumerate(frame_data)]

    @staticmethod
    def get_new_frame(old_frame, body_part_nr, new_body_part_x_frame_data, new_body_part_y_frame_data):
        new_frame = old_frame.copy()

        new_frame.iloc[body_part_nr, 0:1] = new_body_part_x_frame_data
        new_frame.iloc[body_part_nr, 1:2] = new_body_part_y_frame_data

        return new_frame


if __name__ == '__main__':
    Normalizer.run(frames_dir="backflip-1-allar",
                   project_dir="/Users/allarviinamae/EduWorkspace/openpose-jupyter-data-exploration",
                   wrapper_input_dir="raw-keypoints")
