import os

from smoother import Smoother


def main():
    project_dir = "/Users/allarviinamae/EduWorkspace/openpose-jupyter-data-exploration"
    wrapper_input_dir = "raw-keypoints"

    input_full_path = f"{project_dir}/{wrapper_input_dir}"

    frames_dirs = [name for name in os.listdir(input_full_path) if os.path.isdir(f"{input_full_path}/{name}")]

    smoother = Smoother()

    for frames_dir in frames_dirs:
        smoother.run(frames_dir, project_dir, wrapper_input_dir)


if __name__ == '__main__':
    main()
