from collections import defaultdict
import numpy as np
import h5py
import torch

from sklearn.neighbors import BallTree

class LeisureNetworkLoader:
    def __init__(self, june_world_path, k=1):
        self.june_world_path = june_world_path
        self._super_area_coordinates = self._get_super_area_coordinates()
        self._super_area_ids = self._get_super_area_ids()
        self._ball_tree = self._generate_ball_tree()
        self.k = k

    def _get_super_area_coordinates(self):
        with h5py.File(self.june_world_path, "r") as f:
            super_area_coords = f["geography"]["super_area_coordinates"][:]
        super_area_coords = np.array(
            [np.deg2rad(coordinates) for coordinates in super_area_coords]
        )
        return super_area_coords

    def _get_super_area_ids(self):
        with h5py.File(self.june_world_path, "r") as f:
            super_area_ids = f["geography"]["super_area_id"][:]
        return super_area_ids

    def _get_people_per_super_area(self):
        ret = {}
        with h5py.File(self.june_world_path, "r") as f:
            people_super_area = f["population"]["super_area"][:]
            for super_area_id in self._super_area_ids:
                people_in_super_area = np.where(people_super_area == super_area_id)[0]
                ret[super_area_id] = list(people_in_super_area)
        return ret

    def _generate_ball_tree(self):
        ball_tree = BallTree(self._super_area_coordinates, metric="haversine")
        return ball_tree

    def _get_closest_super_areas(self, super_area, k=3):
        coordinates = self._super_area_coordinates[super_area]
        dist, ind = self._ball_tree.query(coordinates.reshape(1, -1), k=k)
        return ind[0]

    def _get_close_people_per_super_area(self, k):
        ret = {}
        people_per_super_area = self._get_people_per_super_area()
        for super_area in self._super_area_ids:
            closest = self._get_closest_super_areas(super_area, k=k)
            people = []
            for sa in closest:
                people += people_per_super_area[sa]
            ret[super_area] = people
        return ret

    def load_network(self, data):
        close_people_per_super_area = self._get_close_people_per_super_area(k=self.k)
        ret = torch.empty((2, 0), dtype=torch.long)
        for super_area in close_people_per_super_area:
            people = torch.tensor(
                close_people_per_super_area[super_area], dtype=torch.long
            )
            edges = torch.vstack(
                (people, super_area * torch.ones(len(people), dtype=torch.long))
            )
            ret = torch.hstack((ret, edges))
        data["agent", "attends_leisure", "leisure"].edge_index = ret
        data["leisure"].id = torch.tensor(list(close_people_per_super_area.keys()))
        data["leisure"].people = torch.tensor(
            [len(close_people_per_super_area[sa]) for sa in close_people_per_super_area]
        )

#    def load_network(self, data):
#        close_people_per_super_area = self._get_close_people_per_super_area(k=self.k)
#        ret = torch.empty((2, 0), dtype=torch.long)
#        for super_area in close_people_per_super_area:
#            to_choose = int(len(close_people_per_super_area[super_area]) / self.k)
#            indices = torch.randperm(len(close_people_per_super_area[super_area]))[
#                :to_choose
#            ]
#            people = torch.tensor(
#                close_people_per_super_area[super_area], dtype=torch.long
#            )[indices]
#            edges = torch.vstack(
#                (people, super_area * torch.ones(len(people), dtype=torch.long))
#            )
#            ret = torch.hstack((ret, edges))
#        data["agent", "attends_leisure", "leisure"].edge_index = ret
#        data["leisure"].id = torch.tensor(list(close_people_per_super_area.keys()))
#        data["leisure"].people = torch.tensor(
#            [len(close_people_per_super_area[sa]) for sa in close_people_per_super_area]
#        )
#
#
