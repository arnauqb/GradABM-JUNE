import torch_geometric.transforms as T
from collections import defaultdict
import numpy as np
import h5py
import torch


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
    plurals = {
        "pub": "pubs",
        "gym": "gyms",
        "grocery": "groceries",
        "cinema": "cinemas",
    }

    def __init__(self, june_world_path, specs):
        self.june_world_path = june_world_path
        self.specs = specs

    def _get_people_per_area(self):
        ret = {}
        with h5py.File(self.june_world_path, "r") as f:
            area_ids = f["geography"]["area_id"]
            people_area = f["population"]["area"]
            for area_id in area_ids:
                people_in_area = np.where(people_area == area_id)[0]
                ret[area_id] = list(people_in_area)
        return ret

    def _get_area_of_venues_per_spec(self):
        ret = defaultdict(lambda: defaultdict(list))
        with h5py.File(self.june_world_path, "r") as f:
            for spec in self.specs:
                plural = self.plurals[spec]
                areas = f["social_venues"][plural]["area"][:]
                ids = f["social_venues"][plural]["id"][:]
                for (id, area) in zip(ids, areas):
                    ret[spec][id] = area
        return ret

    def _get_nearby_venues_per_area_per_spec(self):
        ret = defaultdict(lambda: defaultdict(list))
        with h5py.File(self.june_world_path, "r") as f:
            areas = f["geography"]["area_id"][:]
            sv_specs = f["geography"]["social_venues_specs"][:].astype(str)
            sv_ids = f["geography"]["social_venues_ids"][:]
            for i, area in enumerate(areas):
                for (spec, id) in zip(sv_specs[i], sv_ids[i]):
                    if spec not in self.specs:
                        continue
                    ret[spec][area].append(id)
        return ret

    def _get_people_per_area_per_social_spec(self):
        ret = defaultdict(lambda: defaultdict(list))
        people_per_area = self._get_people_per_area()
        nearby_venues_per_area_per_spec = self._get_nearby_venues_per_area_per_spec()
        area_of_venues_per_spec = self._get_area_of_venues_per_spec()
        for spec in self.specs:
            for area in people_per_area:
                venues = nearby_venues_per_area_per_spec[spec][area]
                for venue in venues:
                    area_of_venue = area_of_venues_per_spec[spec][venue]
                    ret[area_of_venue][spec] += people_per_area[area]
        return ret

    def _get_group_ids(self, spec):
        plural = self.plurals[spec]
        with h5py.File(self.june_world_path, "r") as f:
            group_ids = f["social_venues"][plural]["id"][:]
        return group_ids

    def load_network(self, data):
        people_per_area_per_social_spec = self._get_people_per_area_per_social_spec()
        for spec in self.specs:
            adjlist_i = []
            adjlist_j = []
            area_ids = []
            for area in people_per_area_per_social_spec:
                people = people_per_area_per_social_spec[area][spec]
                if len(people) != 0:
                    area_ids.append(area)
                for person in people:
                    adjlist_i.append(person)
                    adjlist_j.append(area)
            data[spec].id = np.sort(area_ids)
            data["agent", f"attends_{spec}", spec].edge_index = torch.vstack(
                (torch.tensor(adjlist_i), torch.tensor(adjlist_j))
            )


class GraphLoader:
    def __init__(self, june_world_path, leisure_specs=("pub", "gym")):
        self.june_world_path = june_world_path
        self.leisure_specs = leisure_specs

    def load_graph(self, data):
        for network_loader_class in [
            HouseholdNetworkLoader,
            CompanyNetworkLoader,
            SchoolNetworkLoader,
        ]:
            network_loader = network_loader_class(self.june_world_path)
            network_loader.load_network(data)
        leisure_network_loader = LeisureNetworkLoader(
            self.june_world_path, self.leisure_specs
        )
        leisure_network_loader.load_network(data)
        data = T.ToUndirected()(data)
        return data
