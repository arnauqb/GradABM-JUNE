import torch
import pytest
from pytest import fixture
from torch_geometric.data import HeteroData

from grad_june.june_world_loader.agent_data_loader import AgentDataLoader
from grad_june.june_world_loader.graph_loader import GraphLoader
from grad_june.june_world_loader.household_loader import HouseholdNetworkLoader
from grad_june.june_world_loader.care_home_loader import CareHomeNetworkLoader
from grad_june.june_world_loader.company_loader import CompanyNetworkLoader
from grad_june.june_world_loader.school_loader import SchoolNetworkLoader
from grad_june.june_world_loader.university_loader import UniversityNetworkLoader
from grad_june.june_world_loader.leisure_loader import LeisureNetworkLoader


class TestLoadAgentData:
    @fixture(name="agent_data_loader")
    def make_agent_loader(self, june_world_path):
        return AgentDataLoader(june_world_path)

    def test__agent_properties(self, agent_data_loader):
        data = HeteroData()
        agent_data_loader.load_agent_data(data)
        assert len(data["agent"]["id"]) == 769
        assert len(data["agent"]["area"]) == 769
        assert len(data["agent"]["age"]) == 769
        assert len(data["agent"]["sex"]) == 769
        assert data["agent"]["age"][14] == 6
        assert data["agent"]["sex"][14] == 1
        assert data["agent"]["age"][22] == 8
        assert data["agent"]["sex"][22] == 0
        assert data["agent"]["area"][14] == "E00023664"
        assert data["agent"]["area"][300] == "E00079478"


class TestNetworks:
    @pytest.mark.parametrize(
        "loader_class, ids, expected",
        [
            (HouseholdNetworkLoader, (0, 50), ([272], [220, 248])),
            (
                CompanyNetworkLoader,
                (0, 11),
                ([177, 551], [69, 75, 136, 570, 695]),
            ),
            (
                SchoolNetworkLoader,
                (0,),
                ([4, 5],),
            ),
            (
                UniversityNetworkLoader,
                (38,),
                ([57, 58, 65, 59, 60, 61, 64, 62, 63],),
            ),
        ],
    )
    def test__get_people_per_group(self, june_world_path, loader_class, ids, expected):
        loader = loader_class(june_world_path)
        print(loader)
        ret = loader._get_people_per_group()
        for id, exp in zip(ids, expected):
            assert set(exp).issubset(set(ret[id]))

    @pytest.mark.parametrize(
        "spec, loader_class, n_groups, total_people, group_ids, n_people_per_group",
        [
            ("household", HouseholdNetworkLoader, 355, 745, (2, 20, 209), (1, 6, 1)),
            ("care_home", CareHomeNetworkLoader, 1, 27, (0,), (27,)),
            ("company", CompanyNetworkLoader, 1980, 333, (0, 10, 1455), (2, 0, 7)),
            ("school", SchoolNetworkLoader, 1, 78, (0,), (78,)),
            ("university", UniversityNetworkLoader, 39, 43, (38, 23), (9, 17)),
        ],
    )
    def test__load_network(
        self,
        june_world_path,
        spec,
        loader_class,
        n_groups,
        total_people,
        group_ids,
        n_people_per_group,
    ):
        data = HeteroData()
        loader = loader_class(june_world_path)
        loader.load_network(data)
        assert len(data[spec].id) == n_groups
        assert len(data[f"attends_{spec}"].edge_index[0]) == total_people
        for group_id, n_people in zip(group_ids, n_people_per_group):
            assert data[spec].people[group_id] == n_people


class TestLeisureNetwork:
    @fixture(name="leisure_loader")
    def make_leisure_loader(self, june_world_path):
        return LeisureNetworkLoader(june_world_path, k=3)

    def test__get_people_per_super_area(self, leisure_loader):
        ret = leisure_loader._get_people_per_super_area()
        assert len(ret[0]) == 294
        assert 50 in ret[0]
        assert len(ret[2]) == 325
        assert 464 in ret[2]

    def test__generate_super_area_nearest_neighbor(self, leisure_loader):
        closest_sas = leisure_loader._get_closest_super_areas(0, k=3)
        assert (closest_sas == [0, 2, 1]).all()
        closest_sas = leisure_loader._get_closest_super_areas(1, k=3)
        assert (closest_sas == [1, 0, 2]).all()
        closest_sas = leisure_loader._get_closest_super_areas(2, k=3)
        assert (closest_sas == [2, 0, 1]).all()

    def test__get_close_people_per_super_area(self, leisure_loader):
        close_people_per_sa = leisure_loader._get_close_people_per_super_area(k=3)
        assert len(close_people_per_sa[0]) == 769
        assert len(close_people_per_sa[1]) == 769
        assert len(close_people_per_sa[2]) == 769
        close_people_per_sa = leisure_loader._get_close_people_per_super_area(k=2)
        assert len(close_people_per_sa[1]) == 444

    def test__load_leisure_network(self, leisure_loader):
        data = HeteroData()
        leisure_loader.load_network(data)
        assert len(data["attends_leisure"]["edge_index"][0]) > 1500
        assert len(data["leisure"]["id"]) == 3
        assert data["leisure"]["people"][0] == 769
        assert data["leisure"]["people"][2] == 769
        assert (data["leisure"]["super_area"] == ['E02000978', 'E02003270', 'E02003353']).all()


class TestLoadGraph:
    @fixture(name="graph_loader")
    def make_graph_loader(self, june_world_path):
        return GraphLoader(june_world_path, k_leisure=1)

    def test__graph_loader(self, graph_loader):
        data = HeteroData()
        data = graph_loader.load_graph(data)
        assert len(data["household"]["id"]) == 355
        assert len(data["school"]["id"]) == 1
        assert len(data["company"]["id"]) == 1980
        assert len(data["attends_company"]["edge_index"][0]) == 333
        assert len(data["rev_attends_company"]["edge_index"][0]) == 333
        assert len(data["attends_school"]["edge_index"][0]) == 78
        assert len(data["rev_attends_school"]["edge_index"][0]) == 78
        assert len(data["attends_household"]["edge_index"][0]) == 745
        assert len(data["rev_attends_household"]["edge_index"][0]) == 745
        assert (
            len(data["attends_care_home"]["edge_index"][0]) == 27
        )  # residents + workers
        assert len(data["rev_attends_care_home"]["edge_index"][0]) == 27
        assert len(data["attends_leisure"]["edge_index"][0]) == 769
        assert len(data["rev_attends_leisure"]["edge_index"][0]) == 769

        goes_to_school = set(data["attends_school"].edge_index[0, :].numpy())
        goes_to_company = set(data["attends_company"].edge_index[0, :].numpy())
        assert len(goes_to_school.intersection(goes_to_company)) == 0

        goes_to_household = set(data["attends_household"].edge_index[0, :].numpy())
        goes_to_care_home = set(data["attends_care_home"].edge_index[0, :].numpy())
        assert len(goes_to_care_home.intersection(goes_to_household)) == 3
