import numpy as np
import pandas as pd


class Padder:

    @staticmethod
    def pad_with_zeros(orig_ndarray, desired_rows_count):
        (frames, features) = orig_ndarray.shape

        sample_df = pd.DataFrame(data=orig_ndarray,
                                 index=np.arange(frames),
                                 columns=np.arange(features))

        zeros_df = pd.DataFrame(0,
                                index=np.arange(desired_rows_count),
                                columns=np.arange(features),
                                dtype='float')

        for i in range(features):
            zeros_df[i] = sample_df[i].astype(float)

        padded_df = zeros_df.fillna(0)

        return padded_df.to_numpy()

    @staticmethod
    def get_padded_samples(samples):
        padded_samples_list = []

        for idx, sample_ndarray in enumerate(samples):
            desired_rows_count = 110

            padded_ndarray = Padder.pad_with_zeros(sample_ndarray, desired_rows_count)

            padded_samples_list = padded_samples_list + [padded_ndarray]

        return padded_samples_list
