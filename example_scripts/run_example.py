import matplotlib.pyplot as plt

from torch_june.runner import Runner

runner = Runner.from_file()
runner.run()

results = runner.results
dates = results["dates"]
cases_per_timestep = results["cases_per_timestep"].detach().cpu().numpy()
runner.save_results()

#fig, ax = plt.subplots()
#ax.plot(dates, cases_per_timestep)
#fig.autofmt_xdate()
#plt.show()
