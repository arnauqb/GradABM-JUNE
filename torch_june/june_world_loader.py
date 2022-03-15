import torch_geometric.transforms as T
from torch_june.utils import generate_erdos_renyi
from collections import defaultdict
import numpy as np
import h5py
import torch
from tqdm import tqdm

from sklearn.neighbors import BallTree


class AgentDataLoader:
    def __init__(self, june_world_path):
        self.june_world_path = june_world_path

    def load_agent_data(self, data):
        with h5py.File(self.june_world_path, "r") as f:
            population = f["population"]
            data["agent"].id = population["id"][:]
            data["agent"].age = population["age"][:]
            data["agent"].sex = population["sex"][:].astype(str)


class NetworkLoader:
    spec = None
    plural = None

    def __init__(self, june_world_path):
        self.june_world_path = june_world_path

    def _get_people_per_group(self):
        ret = defaultdict(lambda: [])
        with h5py.File(self.june_world_path, "r") as f:
            group_ids = f["population"]["group_ids"][:, 1]
            group_specs = f["population"]["group_specs"][:, 1]
            for (i, (group_id, group_spec)) in enumerate(zip(group_ids, group_specs)):
                if group_spec.decode() != self.spec:
                    continue
                ret[group_id].append(i)
        return ret

    def _get_group_ids(self):
        with h5py.File(self.june_world_path, "r") as f:
            group_ids = f[self.plural]["id"][:]
        return group_ids

    def load_network(self, data):
        people_per_group = self._get_people_per_group()
        adjlist_i = []
        adjlist_j = []
        for (group_id, people) in people_per_group.items():
            for person in people:
                adjlist_i.append(person)
                adjlist_j.append(group_id)
        data[self.spec].id = self._get_group_ids()
        data["agent", f"attends_{self.spec}", self.spec].edge_index = torch.vstack(
            (torch.tensor(adjlist_i), torch.tensor(adjlist_j))
        )


class HouseholdNetworkLoader(NetworkLoader):
    spec = "household"
    plural = "households"

    def __init__(self, june_world_path):
        self.june_world_path = june_world_path

    def _get_people_per_group(self):
        ret = defaultdict(lambda: [])
        with h5py.File(self.june_world_path, "r") as f:
            group_ids = f["population"]["group_ids"][:, 0]
            for i, group_id in enumerate(group_ids):
                ret[group_id].append(i)
        return ret


class CompanyNetworkLoader(NetworkLoader):
    spec = "company"
    plural = "companies"


class SchoolNetworkLoader(NetworkLoader):
    spec = "school"
    plural = "schools"


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
        for super_area in tqdm(close_people_per_super_area):
            people = torch.tensor(close_people_per_super_area[super_area])
            p = 5 / len(people)
            edges = generate_erdos_renyi(nodes=people, edge_prob=p)
            ret = torch.hstack((ret, edges))
        data["agent", f"attends_leisure", "leisure"].edge_index = ret


class GraphLoader:
    def __init__(self, june_world_path):
        self.june_world_path = june_world_path

    def load_graph(self, data):
        for network_loader_class in [
            HouseholdNetworkLoader,
            CompanyNetworkLoader,
            SchoolNetworkLoader,
            LeisureNetworkLoader,
        ]:
            print(f"Loading {network_loader_class}...")
            network_loader = network_loader_class(self.june_world_path)
            network_loader.load_network(data)
        data = T.ToUndirected()(data)
        return data
