import h5py
import torch
import numpy as np


class AgentDataLoader:
    def __init__(self, june_world_path):
        self.june_world_path = june_world_path

    def _get_socioeconomic_indices(self):
        bins = [0, 0.20, 0.4, 0.6, 0.8, 1.0]
        with h5py.File(self.june_world_path, "r") as f:
            agent_area_ids = f["population"]["area"][:]
            socio_indices = f["geography"]["area_socioeconomic_indices"][:][
                agent_area_ids
            ]
            socio_indices = np.digitize(socio_indices, bins)
        return torch.tensor(socio_indices, dtype=torch.int8)

    def load_agent_data(self, data):
        with h5py.File(self.june_world_path, "r") as f:
            population = f["population"]
            data["agent"].id = torch.tensor(population["id"][:])
            data["agent"].age = torch.tensor(population["age"][:])
            data["agent"].ethnicity = population["ethnicity"][:].astype("U")
            data["agent"].socioeconomic_index = self._get_socioeconomic_indices()
            area_ids = population["area"][:]
            area_names = f["geography"]["area_name"][:][area_ids].astype("U")
            data["agent"].area = area_names
            sexes = population["sex"][:].astype(str).astype(object)
            sexes[sexes == "m"] = 0
            sexes[sexes == "f"] = 1
            data["agent"].sex = torch.tensor(sexes.astype(int))
