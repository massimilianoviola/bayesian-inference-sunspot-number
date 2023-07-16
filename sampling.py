import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["savefig.dpi"] = 400
plt.style.use("bmh")

# read data and convert to numpy array
data = pd.read_csv("Sunspots_observations.csv", skiprows=6)
data = data.loc[data["Year CE.1"].notnull(), ["Year CE.1", "SNm"]].to_numpy()

# t: time in years, y: monthly mean total sunspot number
t = data[:, 0]
y = data[:, 1]
y = y - y.mean()  # normalize

# load data from fourier decomposition with n cosines
z = np.load("fourier.npz")
# get components by their name. sol has shape=(3n+1,)
sol = np.concatenate((z["estimates_A"], z["estimates_F"], z["estimates_P"], z["sigma"]))
n = len(sol) // 3

# initialize the walkers in a tiny gaussian ball around the fourier decomposition
nwalkers = 128
pos = sol + 1e-3 * np.random.randn(nwalkers, len(sol)) * sol  # use variance proportional to magnitude
nwalkers, ndim = pos.shape


# Jeffreys prior on sigma, uninformative priors on the rest
def log_prior(theta):
    sigma = theta[-1]
    return -np.log(sigma) if sigma > 0 else -np.inf


def log_likelihood(theta, y, t, n):
    n_samples = len(y)
    A, w, phi, sigma = theta[:n], theta[n:2*n], theta[2*n:3*n], theta[-1]
    return -n_samples/2 * np.log(2 * np.pi * sigma**2) - 1/(2 * sigma**2) * ((y - (A * np.cos(np.outer(t, 2 * np.pi * w) + phi)).sum(axis=1))**2).sum()


def log_probability(theta, y, t, n):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, y, t, n)


# initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(y, t, n), moves=emcee.moves.StretchMove(a=2.0))

# run a few “burn-in” steps to let the walkers explore the parameter space and get settled into the maximum of the density
#print("Running burn-in steps...")
#state = sampler.run_mcmc(pos, 100, progress=True)  # save the final position
#sampler.reset()
# run true production run from the new initial position
#sampler.run_mcmc(state, 500, progress=True)

state = sampler.run_mcmc(pos, 25000, progress=True)

tau = sampler.get_autocorr_time(quiet=True)

fig, axes = plt.subplots(4, figsize=(10, 8), sharex=True)
samples = sampler.get_chain()
print(samples.shape)
labels = ["w1", "w2", "w3", "w4"]
for i in range(4):
    ax = axes[i]
    ax.plot(samples[:, :, 15+i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("step number")

samples = sampler.get_chain(flat=True)
print(samples.shape)

for i in range(n, 2*n):
    plt.figure(figsize=(12, 4))
    plt.hist(samples[:, i], 100, color="k", histtype="step")
    plt.xlabel(fr"$\omega_{{{i-n+1}}}$")
    plt.ylabel(fr"$p(\omega_{{{i-n+1}}})$")
    plt.gca().set_yticks([])
    plt.tight_layout()

    perc = np.percentile(samples[:, i], [16, 50, 84])
    print(f"omega_{i-n+1} interval: {perc[0], perc[2]}")

print(
    "Mean acceptance fraction: {0:.3f}".format(
        np.mean(sampler.acceptance_fraction)
    )
)

plt.show()