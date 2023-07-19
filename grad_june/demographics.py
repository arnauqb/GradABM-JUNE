import torch
import torch_geometric
from torch_geometric.data import HeteroData

def store_differentiable_deaths(data: HeteroData, dead_idx: int):
    """
    Returns differentiable deaths. The results are stored
    in data["results"]
    """
    symptoms = data["agent"].symptoms
    #dead_idx = self.model.symptoms_updater.stages_ids[-1]
    deaths = (
        (symptoms["current_stage"] == dead_idx)
        * symptoms["current_stage"]
        / dead_idx
    )
    if data["results"]["deaths_per_timestep"] is not None:
        data["results"]["deaths_per_timestep"] = torch.hstack(
            (data["results"]["deaths_per_timestep"], deaths.sum())
        )
    else:
        data["results"]["deaths_per_timestep"] = deaths.sum()

def get_cases_by_age(data: HeteroData, age_bins: torch.Tensor):
    device = age_bins.device
    ret = torch.zeros(age_bins.shape[0] - 1, device=device)
    for i in range(1, age_bins.shape[0]):
        mask1 = data["agent"].age < age_bins[i]
        mask2 = data["agent"].age > age_bins[i - 1]
        mask = mask1 * mask2
        ret[i - 1] = (data["agent"].is_infected * mask).sum()
    return ret

def get_people_by_age(ages: torch.Tensor, age_bins: torch.Tensor):
    ret = {}
    for i in range(1, age_bins.shape[0]):
        mask1 = ages < age_bins[i]
        mask2 = ages > age_bins[i - 1]
        mask = mask1 * mask2
        ret[int(age_bins[i].item())] = mask.sum()
    return ret

def get_cases_by_ethnicity(data: HeteroData, ethnicities):
    device = ethnicities.device
    ret = torch.zeros(len(ethnicities), device=device)
    for i, ethnicity in enumerate(ethnicities):
        mask = torch.tensor(
            data["agent"].ethnicity == ethnicity, device=device
        )
        ret[i] = (mask * data["agent"].is_infected).sum()
    return ret

def get_people_per_area(agent_ids: torch.Tensor, area_ids: torch.Tensor):
    """Gets people ids in each area.
    
    **Arguments:**

    - `agent_ids`: Ids of all agents.
    - `area_ids`: Area ids of all agents.
    """
    people_per_area = {}
    for area_id in torch.unique(area_ids):
        people_per_area[area_id.item()] = agent_ids[area_ids == area_id]
    return people_per_area