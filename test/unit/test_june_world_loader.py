import torch
from pytest import fixture
from torch_geometric.data import HeteroData

from torch_june.june_world_loader import (
    AgentDataLoader,
    HouseholdNetworkLoader,
    CompanyNetworkLoader,
    SchoolNetworkLoader,
    LeisureNetworkLoader,
    GraphLoader,
)


class TestLoadAgentData:
    @fixture(name="agent_data_loader")
    def make_agent_loader(self, june_world_path):
        return AgentDataLoader(june_world_path)

    def test__agent_properties(self, agent_data_loader):
        data = HeteroData()
        agent_data_loader.load_agent_data(data)
        assert len(data["agent"]["id"]) == 6640
        assert len(data["agent"]["age"]) == 6640
        assert len(data["agent"]["sex"]) == 6640
        assert data["agent"]["age"][1234] == 24
        assert data["agent"]["sex"][1234] == "m"
        assert data["agent"]["age"][2234] == 47
        assert data["agent"]["sex"][2234] == "f"


class TestHouseholdNetwork:
    @fixture(name="household_loader")
    def make_household_loader(self, june_world_path):
        return HouseholdNetworkLoader(june_world_path)

    def test__get_people_per_household(self, household_loader):
        ret = household_loader._get_people_per_group()
        assert set(ret[10]) == set([154, 314, 421])
        assert set(ret[50]) == set([113, 275])
        assert set(ret[125]) == set([227, 370, 402])

    def test__load_household_network(self, household_loader):
        data = HeteroData()
        household_loader.load_network(data)
        assert len(data["household"]["id"]) == 2367  # number of households
        assert (
            len(data["attends_household"]["edge_index"][0]) == 6640
        )  # everyone has a household
        assert data["household"]["people"][10] == 3
        assert data["household"]["people"][50] == 2
        assert data["household"]["people"][125] == 3


class TestCompanyNetwork:
    @fixture(name="company_loader")
    def make_company_loader(self, june_world_path):
        return CompanyNetworkLoader(june_world_path)

    def test__get_people_per_company(self, company_loader):
        ret = company_loader._get_people_per_group()
        assert set(ret[69]) == set([4116, 4554, 5648])
        assert set(ret[33]) == set([1277, 2957, 3265, 3660, 4540, 5715])

    def test__load_company_network(self, company_loader):
        data = HeteroData()
        company_loader.load_network(data)
        assert len(data["company"]["id"]) == 130
        assert len(data["attends_company"]["edge_index"][0]) == 2871
        assert data["company"]["people"][69] == 3
        assert data["company"]["people"][33] == 6


class TestSchoolNetwork:
    @fixture(name="school_loader")
    def make_school_loader(self, june_world_path):
        return SchoolNetworkLoader(june_world_path)

    def test__get_people_per_school(self, school_loader):
        ret = school_loader._get_people_per_group()
        people_in_school1 = set(ret[0])
        people_in_school2 = set(ret[1])
        assert len(people_in_school1) == 1055
        assert len(people_in_school2) == 565
        assert set([1213, 1808, 2134, 2460, 3154]).issubset(people_in_school1)
        assert set([4409, 5022, 6340, 6350]).issubset(people_in_school2)

    def test__load_school_network(self, school_loader):
        data = HeteroData()
        school_loader.load_network(data)
        assert len(data["school"]["id"]) == 2
        assert len(data["attends_school"]["edge_index"][0]) == 1620
        assert data["school"]["people"][0] == 1055
        assert data["school"]["people"][1] == 565


class TestLeisureNetwork:
    @fixture(name="leisure_loader")
    def make_leisure_loader(self, june_world_path_only_people):
        return LeisureNetworkLoader(june_world_path_only_people, k=3)

    def test__get_people_per_super_area(self, leisure_loader):
        ret = leisure_loader._get_people_per_super_area()
        assert len(ret[0]) == 8483
        assert 6103 in ret[0]
        assert len(ret[2]) == 6640
        assert 15780 in ret[2]

    def test__generate_super_area_nearest_neighbor(self, leisure_loader):
        closest_sas = leisure_loader._get_closest_super_areas(0, k=3)
        assert (closest_sas == [0, 3, 1]).all()
        closest_sas = leisure_loader._get_closest_super_areas(1, k=3)
        assert (closest_sas == [1, 3, 0]).all()
        closest_sas = leisure_loader._get_closest_super_areas(2, k=3)
        assert (closest_sas == [2, 0, 1]).all()
        closest_sas = leisure_loader._get_closest_super_areas(3, k=3)
        assert (closest_sas == [3, 1, 0]).all()

    def test__get_close_people_per_super_area(self, leisure_loader):
        close_people_per_sa = leisure_loader._get_close_people_per_super_area(k=3)
        assert len(close_people_per_sa[0]) == 22164
        assert 7456 in close_people_per_sa[0]
        assert len(close_people_per_sa[2]) == 22417
        assert 16776 in close_people_per_sa[2]

    def test__load_leisure_network(self, leisure_loader):
        data = HeteroData()
        leisure_loader.load_network(data)
        assert len(data["attends_leisure"]["edge_index"][0]) > (6640 + 8483)
        assert len(data["leisure"]["id"]) == 4
        assert data["leisure"]["people"][0] == 22164
        assert data["leisure"]["people"][2] == 22417


class TestLoadGraph:
    @fixture(name="graph_loader")
    def make_graph_loader(self, june_world_path):
        return GraphLoader(june_world_path, k_leisure=1)

    def test__graph_loader(self, graph_loader):
        data = HeteroData()
        data = graph_loader.load_graph(data)
        assert len(data["household"]["id"]) == 2367
        assert len(data["school"]["id"]) == 2
        assert len(data["company"]["id"]) == 130
        assert len(data["attends_company"]["edge_index"][0]) == 2871
        assert len(data["rev_attends_company"]["edge_index"][0]) == 2871
        assert len(data["attends_school"]["edge_index"][0]) == 1620
        assert len(data["rev_attends_school"]["edge_index"][0]) == 1620
        assert len(data["attends_household"]["edge_index"][0]) == 6640
        assert len(data["rev_attends_household"]["edge_index"][0]) == 6640
        assert len(data["attends_leisure"]["edge_index"][0]) == 6640
        assert len(data["rev_attends_leisure"]["edge_index"][0]) == 6640
