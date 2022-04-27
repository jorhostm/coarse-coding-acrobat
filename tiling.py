import numpy as np
from numpy import sin, cos, pi, floor


class Tiling:
    def __init__(self, bounds, bins):
        self.bounds = []
        self.bins = bins
        self.dims = 0
        for bound in bounds:
            if bound is None:
                self.bounds.append(None)

            else:
                a, b = bound
                self.bounds.append(np.linspace(a, b, bins))
                self.dims += 1

        self.til = 0

    def get_num_tiles(self):
        return (self.bins + 1) ** self.dims

    def get_index(self, state):
        e = 0
        index = 0
        for i, arr in enumerate(self.bounds):
            if arr is not None:
                k = np.digitize(state[i], arr)
                index += k * 6 ** e
                e += 1

        return index


def features(tils, s):
    offset = 0
    indices = []

    for tiling in tils:
        indices.append(offset + tiling.get_index(s))
        offset += tiling.get_num_tiles()

    return indices


def init_tilings(num_tilings, bins_angle, bins_velocity):
    bin_distance_angle = 2*pi / (bins_angle + 1)
    bin_increment_angle = bin_distance_angle / num_tilings

    bin_distance_v1 = 8 * pi / (bins_velocity + 1)
    bin_increment_v1 = bin_distance_v1 / num_tilings

    bin_distance_v2 = 18 * pi / (bins_velocity + 1)
    bin_increment_v2 = bin_distance_v2 / num_tilings

    tilings = []

    for i in range(num_tilings):
        bound_angle = (-pi + (i+1) * bin_increment_angle, pi - (num_tilings - i) * bin_increment_angle)
        bound_v1 = (-4 * pi + (i+1) * bin_increment_v1, 4*pi - (num_tilings - i) * bin_increment_v1)
        bound_v2 = (-9 * pi + (i+1) * bin_increment_v2, 9*pi - (num_tilings - i) * bin_increment_v2)
        tiling = Tiling([bound_angle, bound_v1, bound_angle, bound_v2], bins_angle)

        tilings.append(tiling)

    return tilings

