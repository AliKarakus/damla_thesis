import numpy as np


class SnapshotData:

    def __init__(self, us, vs):
        self.us = us
        self.vs = vs

    def U(self):
        return np.vstack((self.us, self.vs))


class Snapshots:

    def __init__(self, num_snaps):
        self.snaps = [None] * num_snaps
        self.num_snaps = num_snaps

    def add_snapshot_data(self, time_idx, uvals, vvals):
        self.snaps[time_idx] = SnapshotData(uvals, vvals)

    # Override [] operator.
    def __getitem__(self, idx):
        return self.snaps[idx]

    # Override iterator for 'in' keyword
    def __iter__(self):
        return iter(self.snaps)