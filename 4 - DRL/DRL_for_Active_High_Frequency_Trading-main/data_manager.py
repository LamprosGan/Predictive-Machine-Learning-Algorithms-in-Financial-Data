import copy
import os
import random

import numpy as np
import pandas as pd


def scale(df):
    mins = pd.read_csv('/Volumes/L96/DRL/days/mins.csv', header=None)
    maxes = pd.read_csv('/Volumes/L96/DRL/days/maxes.csv', header=None)
    min = np.array(mins.min(axis=0))
    max = np.array(maxes.max(axis=0))

    # Normalize the data
    df = (np.matrix(df) - min) / (
            max - min)

    return pd.DataFrame(df)


class Data:
    def __init__(self, args, rescale):
        self.data_dir = os.path.join(os.getcwd(), args.data_dir)
        self.start_end_clip = args.start_end_clip
        self.last_n_ticks = args.last_n_ticks
        self.snapshots = self.samples = self.test_files = []
        self.snapshots_per_day = args.snapshots_per_day
        self.snapshot_size = args.snapshot_size
        self.total_snapshots = args.tot_snapshots

        if rescale:
            self.load_and_scale()

        # If we want to resample snapshots or we don't have any saved
        if args.resample or not os.listdir(os.path.join(self.data_dir, 'clean_train_data')):
            self.load_unscaled_train_files()
            indexes = random.sample(range(len(self.samples)), self.total_snapshots)
            print("2 ", indexes)
            self.save_snapshots(indexes)

        self.load_snapshots()

    def load_and_scale(self):

        if os.path.exists(self.data_dir + "/maxes.csv"):
            os.remove(self.data_dir + "/maxes.csv")
        if os.path.exists(self.data_dir + "/mins.csv"):
            os.remove(self.data_dir + "/mins.csv")

        for f in os.listdir(os.path.join(self.data_dir, 'train_data')):
            self.min_max_scaling_compute(f)

    def min_max_scaling_compute(self, filename):

        df = copy.deepcopy(pd.read_csv(os.path.join(os.path.join(self.data_dir, 'train_data'), filename), header=None)).reset_index(drop=True)

        mins = pd.DataFrame().append(pd.Series(df.to_numpy().min(axis=0)), ignore_index=True)
        maxes = pd.DataFrame().append(pd.Series(df.to_numpy().max(axis=0)), ignore_index=True)

        mins.to_csv(self.data_dir + "/mins.csv", index=False, header=False, mode='a')
        maxes.to_csv(self.data_dir + "/maxes.csv", index=False, header=False, mode='a')

    def select_windows(self, df, k, n):
        # Compute drawdown and keep only the windows with the biggest ones

        min = df.mid_price[self.last_n_ticks - 1:].rolling(k).min()
        max = df.mid_price[self.last_n_ticks - 1:].rolling(k).max()
        drawdown = max - min

        indx = drawdown.drop_duplicates().nlargest(n)

        for i in indx.index:
            window_start = i - k + 1 - self.last_n_ticks - 1
            window_end = i + 1

            # Check if window_start is negative
            if window_start < 0:
                print(f"Skipping window at i={i} because window_start={window_start} is <0")
                continue

            window_df = df.iloc[window_start:window_end]

            # Print window details
            print(f"select_windows: i={i}, window_start={window_start}, window_end={window_end}, window_length={len(window_df)}")

            # Ensure window has enough rows
            if len(window_df) < self.last_n_ticks:
                print(f"Skipping window at i={i} because it has only {len(window_df)} rows, need at least {self.last_n_ticks}")
                continue

            # Proceed with scaling and appending
            scaled_df = scale(window_df.drop(columns=['mid_price']))
            unscaled_df = window_df.drop(columns=['mid_price'])

            # Print shapes of scaled and unscaled DataFrames
            print(f"Appending snapshot with shape scaled={scaled_df.shape}, unscaled={unscaled_df.shape}")

            self.samples.append((scaled_df, unscaled_df))


    def load_unscaled_train_files(self):
        for filename in os.listdir(os.path.join(self.data_dir, 'train_data')):
            df = copy.deepcopy(
                pd.read_csv(os.path.join(os.path.join(self.data_dir, 'train_data'), filename), header=None, encoding='utf-8').reset_index(drop=True))
            df['mid_price'] = abs(df.iloc[:, 0] + df.iloc[:, 2]) / 2
            self.select_windows(df, self.snapshot_size, self.snapshots_per_day)
        print("1 ", len(self.samples))

    def load_test_file(self, i):
        filename = sorted(os.listdir(os.path.join(self.data_dir, 'test_data')))[i]
        df = pd.read_csv(os.path.join(os.path.join(self.data_dir, 'test_data'), filename), header=None).reset_index(drop=True)
        return filename, [scale(df), df]

    def save_snapshots(self, indexes):
        for j, i in enumerate(indexes):
            self.samples[i][0].to_csv(os.path.join(self.data_dir, f'clean_train_data/scaled_{j}.csv'), index=False)
            self.samples[i][1].to_csv(os.path.join(self.data_dir, f'clean_train_data/unscaled_{j}.csv'), index=False)
        print("3 ", len(self.samples))
        self.samples.clear()
        print("4 ", len(self.snapshots))

    def load_snapshots(self):
        for i in range(self.total_snapshots):
            scaled = pd.read_csv(os.path.join(self.data_dir, f'clean_train_data/scaled_{i}.csv'))
            unscaled = pd.read_csv(os.path.join(self.data_dir, f'clean_train_data/unscaled_{i}.csv'))
            self.snapshots.append([scaled, unscaled])
        print("5 ", len(self.samples))
