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
sol = np.concatenate((z["estimates_A"], z["estimates_T"], z["estimates_P"], z["sigma"]))
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
    A, T, phi, sigma = theta[:n], theta[n:2*n], theta[2*n:3*n], theta[-1]
    return -n_samples/2 * np.log(2 * np.pi * sigma**2) - 1/(2 * sigma**2) * ((y - (A * np.cos(np.outer(t, 2 * np.pi / T) + phi)).sum(axis=1))**2).sum()


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

samples = sampler.get_chain()
print(samples.shape)

##################################################

M = 5  # maximum number of subplots per figure
num_figures = (n - 1) // M + 1

# plot positions of each walker as a function of the number of steps for A
for fig_num in range(num_figures):
    start_index = fig_num * M
    end_index = min((fig_num + 1) * M, n)
    fig, axes = plt.subplots(end_index - start_index, figsize=(12, 1.5 * (end_index - start_index)), sharex=True)
    labels = [f"$A_{{{i+1}}}$" for i in range(start_index, end_index)]
    for i in range(len(labels)):
        ax = axes[i]
        ax.plot(samples[:, :, i + start_index], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
    axes[-1].set_xlabel("Step number")
    axes[0].set_title("$A$ chains")
    plt.tight_layout()

# plot positions of each walker as a function of the number of steps for T
for fig_num in range(num_figures):
    start_index = fig_num * M
    end_index = min((fig_num + 1) * M, n)
    fig, axes = plt.subplots(end_index - start_index, figsize=(12, 1.5 * (end_index - start_index)), sharex=True)
    labels = [f"$T_{{{i+1}}}$" for i in range(start_index, end_index)]
    for i in range(len(labels)):
        ax = axes[i]
        ax.plot(samples[:, :, i + n + start_index], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
    axes[-1].set_xlabel("Step number")
    axes[0].set_title("$T$ chains")
    plt.tight_layout()

# plot positions of each walker as a function of the number of steps for phi
for fig_num in range(num_figures):
    start_index = fig_num * M
    end_index = min((fig_num + 1) * M, n)
    fig, axes = plt.subplots(end_index - start_index, figsize=(12, 1.5 * (end_index - start_index)), sharex=True)
    labels = [f"$\phi_{{{i+1}}}$" for i in range(start_index, end_index)]
    for i in range(len(labels)):
        ax = axes[i]
        ax.plot(samples[:, :, i + 2*n + start_index], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
    axes[-1].set_xlabel("Step number")
    axes[0].set_title("$\phi$ chains")
    plt.tight_layout()

# plot positions of each walker as a function of the number of steps for sigma
plt.figure(figsize=(12, 2))
plt.plot(samples[:, :, -1], "k", alpha=0.3)
plt.xlim(0, len(samples))
plt.ylabel("$\sigma$")
plt.xlabel("Step number")
plt.title("$\sigma$ chain")
plt.tight_layout()

##################################################

del samples
logprob = sampler.get_log_prob()  # log posterior chains
# plot posterior values of each walker as a function of the number of steps
plt.figure(figsize=(12, 4))
plt.plot(logprob, "k", alpha=0.3)
plt.xlim(0, len(logprob))
plt.xlabel("Step number")
plt.ylabel("Log posterior")
plt.title("Log posterior chains")
plt.tight_layout()

##################################################

# estimate of the integrated autocorrelation time
tau = sampler.get_autocorr_time(quiet=True)
print(f"Mean autocorrelation time: {np.mean(tau):.3f} steps")
tau_max = int(tau.max())

flat_samples = sampler.get_chain(flat=True)
print(flat_samples.shape)

flat_logprob = sampler.get_log_prob(flat=True)
theta_max = flat_samples[np.argmax(flat_logprob)]  # MAP estimate

##################################################

# project the sampling results into the observed data space
plt.figure(figsize=(12, 4))
inds = np.random.randint(len(flat_samples), size=500)
for ind in inds:
    sample = flat_samples[ind]
    A, T, phi, sigma = sample[:n], sample[n:2*n], sample[2*n:3*n], sample[-1]
    signal = np.zeros(len(t))
    for magnitude, period, phase in zip(A, T, phi):
        signal += magnitude * np.cos(2 * np.pi / period * t + phase)
    plt.plot(t, signal, "C1", alpha=0.1)

# add the initial estimate obtained from fourier
A, T, phi, sigma = sol[:n], sol[n:2*n], sol[2*n:3*n], sol[-1]
signal = np.zeros(len(t))
for magnitude, period, phase in zip(A, T, phi):
    signal += magnitude * np.cos(2 * np.pi / period * t + phase)
plt.plot(t, signal, "C9", label="Initial estimate", lw=2)

# add the estimate with maximum value of posterior (MAP estimate)
A, T, phi, sigma = theta_max[:n], theta_max[n:2*n], theta_max[2*n:3*n], theta_max[-1]
map_signal = np.zeros(len(t))
for magnitude, period, phase in zip(A, T, phi):
    map_signal += magnitude * np.cos(2 * np.pi / period * t + phase)
plt.plot(t, map_signal, "C8", label="MAP estimate", lw=2)

# add observed data in the background
plt.plot(t, y, "k", label="Data points", alpha=0.3)
plt.legend(fontsize=14)
plt.xlabel("Time")
plt.ylabel("Normalized count")
plt.title("Projection of the sampling results into the observed data space")
plt.tight_layout()

##################################################

# plot 2 sigma posterior spread into the observed data space
models = []
thetas = flat_samples[inds]
for theta in thetas:
    A, T, phi, sigma = theta[:n], theta[n:2*n], theta[2*n:3*n], theta[-1]
    signal = np.zeros(len(t))
    for magnitude, period, phase in zip(A, T, phi):
        signal += magnitude * np.cos(2 * np.pi / period * t + phase)
    models.append(signal)
spread = np.std(models, axis=0)
med_model = np.median(models, axis=0)

plt.figure(figsize=(12, 4))
plt.plot(t, y, "k", label="Data points", alpha=0.3)
plt.plot(t, map_signal, label="MAP estimate", c="C1")
plt.fill_between(t, med_model - 2*spread, med_model + 2*spread, color="C7", alpha=0.5, label=r"$2\sigma$ posterior spread")
plt.legend(fontsize=14)
plt.xlabel("Time")
plt.ylabel("Normalized count")
plt.title(r"$2\sigma$ posterior spread into the observed data space")
plt.tight_layout()

##################################################

# print confidence intervals for the periods
# use 16th, 50th, and 84th percentiles of the samples in the marginalized distributions
jupiter = 11.862  # target period
for i in range(n, 2*n):
    perc = np.percentile(flat_samples[:, i], [16, 50, 84])
    print(f"T_{i-n+1} 68% confidence interval: ({perc[0]:.3f}, {perc[2]:.3f})")
    
    if flat_samples[:, i].min() < jupiter < flat_samples[:, i].max():  # compatible T
        plt.figure(figsize=(12, 4))
        plt.hist(flat_samples[:, i], 100, color="k", histtype="step")
        plt.axvline(jupiter, c="C1", label="Jupiter orbital period")
        plt.xlabel(fr"$T_{{{i-n+1}}}$")
        plt.ylabel(fr"$p(T_{{{i-n+1}}})$")
        plt.title(fr"Distribution of $T_{{{i-n+1}}}$ samples")
        plt.gca().set_yticks([])
        plt.legend(fontsize=14)
        plt.tight_layout()

print(
    "Mean acceptance fraction: {0:.3f}".format(
        np.mean(sampler.acceptance_fraction)
    )
)

plt.show()