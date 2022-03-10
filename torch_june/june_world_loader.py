from collections import defaultdict
import h5py
import torch

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
