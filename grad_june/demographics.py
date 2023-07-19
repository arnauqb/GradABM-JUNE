import torch

def get_people_per_area(agent_ids: torch.Tensor, area_ids: torch.Tensor):
    """Gets people ids in each area.
    
    **Arguments:**

    - `agent_ids`: Ids of all agents.
    - `area_ids`: Area ids of all agents.
    """
    people_per_area = {}
    for area_id in torch.unique(area_ids):
        people_per_area[area_id] = agent_ids[area_ids == area_id]
    return people_per_area