import pickle
import sys
from torch_june.utils import create_simple_connected_graph

data = create_simple_connected_graph(int(sys.argv[1]))
pickle.dump(data, open("./worlds/data_small.pkl", "wb"))
