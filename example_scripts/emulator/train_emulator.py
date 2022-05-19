import pickle
import numpy as np
from tqdm import tqdm
import torch
import gpytorch

n_samples = 240


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_tasks):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=n_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=n_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def get_training_data(device):
    samples_x, samples_y = pickle.load(open(f"./samples_ne_{n_samples}.pkl", "rb"))
    samples_x = samples_x.float().to(device)
    samples_y = samples_y.float().to(device)
    n_train = int(len(samples_x) * 1.0)
    train_x = samples_x[:n_train]
    train_y = samples_y[:n_train]
    val_x = samples_x[n_train:]
    val_y = samples_y[n_train:]
    return train_x, train_y, val_x, val_y


def train_emulator(train_x, train_y, val_x, val_y):
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks).to(
        device
    )
    model = MultitaskGPModel(train_x, train_y, likelihood, n_tasks=n_tasks).to(device)
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    max_training_iter = 10_000
    # for i in range(training_iter):
    previous_loss = np.inf
    pbar = tqdm(range(max_training_iter))
    for i in pbar:
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        if i >= max_training_iter:
            break
        loss.backward()
        pbar.set_description(f"{i}: {loss:.3e}\t {previous_loss:.3e}")
        # print('Iter %d/%d - Loss: %.3e' % (i + 1, max_training_iter, loss.item()))

        optimizer.step()
        if i % 100 == 0:
            if loss < previous_loss:
                previous_loss = loss
            else:
                break

        #    model.eval()
        #    likelihood.eval()
        #    output_val = model(val_x)
        #    loss_val = -mll(output_val, val_y)
        #    #print(f"{i}:\tVal loss: {loss_val:.2e}")
        #    if previous_loss is not None:
        #        if previous_loss <= loss_val:
        #            print("should stop")
        #            break
        #    previous_loss = loss_val
        #    model.train()
        #    likelihood.train()

    torch.save(model.state_dict(), f"./emulator_{n_samples}.pth")


if __name__ == "__main__":
    time_stamps = [10, 30, 60, 90]
    device = "cuda:0"
    n_tasks = len(time_stamps)
    train_x, train_y, val_x, val_y = get_training_data(device)
    train_emulator(train_x, train_y, val_x, val_y)
