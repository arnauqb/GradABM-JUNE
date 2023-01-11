import torch_geometric.transforms as T

from grad_june.june_world_loader.household_loader import HouseholdNetworkLoader
from grad_june.june_world_loader.care_home_loader import CareHomeNetworkLoader
from grad_june.june_world_loader.company_loader import CompanyNetworkLoader
from grad_june.june_world_loader.school_loader import SchoolNetworkLoader 
from grad_june.june_world_loader.university_loader import UniversityNetworkLoader
from grad_june.june_world_loader.leisure_loader import LeisureNetworkLoader 


class GraphLoader:
    def __init__(self, june_world_path, k_leisure=3):
        self.june_world_path = june_world_path
        self.k_leisure = k_leisure

    def load_graph(
        self,
        data,
        load_leisure=True,
        loaders=(
            HouseholdNetworkLoader,
            CareHomeNetworkLoader,
            CompanyNetworkLoader,
            SchoolNetworkLoader,
            UniversityNetworkLoader,
        ),
    ):
        for network_loader_class in loaders:
            print(f"Loading {network_loader_class}...")
            network_loader = network_loader_class(self.june_world_path)
            network_loader.load_network(data)
        if load_leisure:
            print("Loading leisure ...")
            leisure_loader = LeisureNetworkLoader(
                self.june_world_path, k=self.k_leisure
            )
            leisure_loader.load_network(data)
        data = T.ToUndirected()(data)
        return data
