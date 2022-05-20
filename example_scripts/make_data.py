from torch_geometric.data import HeteroData
from torch_june import GraphLoader, AgentDataLoader
import pickle
import sys

june_world_path = sys.argv[1]

data = HeteroData()
data = GraphLoader(june_world_path, k_leisure=1).load_graph(data)
AgentDataLoader(june_world_path).load_agent_data(data)

with open("./data.pkl", "wb") as f:
    pickle.dump(data, f)
