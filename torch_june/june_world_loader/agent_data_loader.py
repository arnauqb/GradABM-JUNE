import h5py
import torch


class AgentDataLoader:
    def __init__(self, june_world_path):
        self.june_world_path = june_world_path

    def load_agent_data(self, data):
        with h5py.File(self.june_world_path, "r") as f:
            population = f["population"]
            data["agent"].id = torch.tensor(population["id"][:])
            data["agent"].age = torch.tensor(population["age"][:])
            area_ids = population["area"][:]
            area_names = f["geography"]["area_name"][:][area_ids].astype("U")
            data["agent"].area = area_names
            sexes = population["sex"][:].astype(str).astype(object)
            sexes[sexes == "m"] = 0
            sexes[sexes == "f"] = 1
            data["agent"].sex = torch.tensor(sexes.astype(int))
