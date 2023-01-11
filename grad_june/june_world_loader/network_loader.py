import h5py
import torch
from collections import defaultdict

class NetworkLoader:
    spec = None
    plural = None
    columns = None

    def __init__(self, june_world_path):
        self.june_world_path = june_world_path

    def _get_people_per_group(self):
        ret = defaultdict(lambda: [])
        with h5py.File(self.june_world_path, "r") as f:
            for column in self.columns:
                group_ids = f["population"]["group_ids"][:, column]
                group_specs = f["population"]["group_specs"][:, column]
                for i, (group_id, group_spec) in enumerate(zip(group_ids, group_specs)):
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
        for group_id, people in people_per_group.items():
            for person in people:
                adjlist_i.append(person)
                adjlist_j.append(group_id)
        data[self.spec].id = self._get_group_ids()
        data[self.spec].people = torch.tensor(
            [len(people_per_group[i]) for i in data[self.spec].id]
        )
        edge_type = ("agent", f"attends_{self.spec}", self.spec)
        new_edges = torch.vstack((torch.tensor(adjlist_i), torch.tensor(adjlist_j)))
        data[edge_type].edge_index = new_edges
