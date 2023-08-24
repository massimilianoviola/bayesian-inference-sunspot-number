import emcee
import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
import numpy as np
import pandas as pd
import scipy


plt.rcParams["savefig.dpi"] = 400
plt.style.use("bmh")

# read data and convert to numpy array
data = pd.read_csv("data/Sunspots_observations.csv", skiprows=6)
data = data.loc[data["Year CE.1"].notnull(), ["Year CE.1", "SNm"]].to_numpy()

# t: time in years, y: monthly mean total sunspot number
t = data[:, 0]
y = data[:, 1]
y = y - y.mean()  # normalize

# load data from fourier decomposition with n cosines
z = np.load("data/fourier.npz")
# get components by their name. sol has shape=(3n+1,)
sol = np.concatenate((z["estimates_A"], z["estimates_T"], z["estimates_P"], z["sigma"]))
n = len(sol) // 3
n_samples = len(y)

# initialize the walkers in a tiny gaussian ball around the fourier decomposition
nwalkers = 128
pos = sol + 1e-5 * np.random.randn(nwalkers, len(sol)) * sol  # use variance proportional to magnitude
nwalkers, ndim = pos.shape


# Jeffreys prior on sigma, uninformative priors on A and T, zero-mean gaussian prior on phi
# the gaussian prior corresponds to an L2 penalty, favoring small values
phi_std = np.pi  # the standard deviation of the gaussian, higher value means less regularization

def log_prior(theta):
    A, T, phi, sigma = theta[:n], theta[n:2*n], theta[2*n:3*n], theta[-1]
    if sigma > 0 and all(A > 0) and all(T > 0):
        return -np.log(sigma) - 1/(2 * phi_std**2) * np.sum(phi**2)
    else:
        return -np.inf


def log_likelihood(theta):
    A, T, phi, sigma = theta[:n], theta[n:2*n], theta[2*n:3*n], theta[-1]
    return -n_samples/2 * np.log(2 * np.pi * sigma**2) - 1/(2 * sigma**2) * ((y - (A * np.cos(np.outer(t, 2 * np.pi / T) + phi)).sum(axis=1))**2).sum()


def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


if __name__ == "__main__":

    ncpu = cpu_count()  # number of cores
    print(f"Using {ncpu} CPUs for sampling...")

    # set up the backend, clear it in case the file already exists
    filename = "data/chains.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    with Pool() as pool:  # run sampling using multiprocessing
        # initialize the sampler
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, moves=emcee.moves.DEMove(), pool=pool, backend=backend
        )
        state = sampler.run_mcmc(pos, 50000, progress=True)

    samples = sampler.get_chain()
    print(f"Samples shape: {samples.shape}")

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
    burnin = int(5 * np.max(tau))  # throw away a few times max autocorrelation as burn-in
    thin = int(0.5 * np.min(tau))  # thin by half the min autocorrelation time
    print(f"Burn-in: {burnin} steps")
    print(f"Thin by: {thin} steps")
    thinned_samples = sampler.get_chain(discard=burnin, thin=thin)
    tau_thin = emcee.autocorr.integrated_time(thinned_samples, quiet=True)
    print(f"Mean autocorrelation time (after burn-in and thin): {np.mean(tau_thin):.3f} steps")
    del thinned_samples

    flat_samples = sampler.get_chain(flat=True, discard=burnin, thin=thin)
    print(f"Flat samples shape (after burn-in and thin): {flat_samples.shape}\n")

    flat_logprob = sampler.get_log_prob(flat=True, discard=burnin, thin=thin)
    theta_max = flat_samples[np.argmax(flat_logprob)]  # MAP estimate

    ##################################################

    # compare fourier initialization vs MAP estimate
    print("Fourier initialization vs MAP estimate:")
    print(f"Log prior went from {log_prior(sol):.3f} to {log_prior(theta_max):.3f}")
    print(f"Log likelihood went from {log_likelihood(sol):.3f} to {log_likelihood(theta_max):.3f}")
    print(f"Log posterior went from {log_probability(sol):.3f} to {log_probability(theta_max):.3f}\n")
    # define vectorized np functions
    prior_v = np.vectorize(log_prior,signature="(n) -> ()")
    likelihood_v = np.vectorize(log_likelihood, signature="(n) -> ()")
    posterior_v = np.vectorize(log_probability, signature="(n) -> ()")
    # compare average walker behavior
    print("Average walker behavior:")
    print(f"Average log prior went from {np.mean(prior_v(pos)):.3f} to {np.mean(prior_v(flat_samples)):.3f}")
    print(f"Average log likelihood went from {np.mean(likelihood_v(pos)):.3f} to {np.mean(likelihood_v(flat_samples)):.3f}")
    print(f"Average log posterior went from {np.mean(posterior_v(pos)):.3f} to {np.mean(flat_logprob):.3f}\n")
    print("-"*50)

    ##################################################

    plt.figure(figsize=(8, 5))
    s = np.arange(-4*phi_std, 4*phi_std, 0.01)
    f = scipy.stats.norm.pdf(s, 0, phi_std)
    plt.plot(s,f, lw=3, c="k", label="Prior")
    plt.ylabel("Density")
    plt.title(r"Sample distribution vs prior for $\phi$")
    for i in range(n):
        col = np.random.random(3)  # random color
        plt.hist(flat_samples[:, 2*n+i], 100, color=col, alpha=0.5, label=f"$\phi_{{{i+1}}}$", density=True)
    plt.legend(fontsize=8)
    plt.tight_layout()

    ##################################################

    # project the sampling results into the observed data space
    plt.figure(figsize=(12, 5))
    inds = np.random.randint(len(flat_samples), size=200)
    for ind in inds:
        sample = flat_samples[ind]
        A, T, phi, sigma = sample[:n], sample[n:2*n], sample[2*n:3*n], sample[-1]
        signal = np.zeros(len(t))
        for magnitude, period, phase in zip(A, T, phi):
            signal += magnitude * np.cos(2 * np.pi / period * t + phase)
        plt.plot(t, signal, "C7", alpha=0.1, label="Samples") if ind == inds[0] else plt.plot(t, signal, "C7", alpha=0.1)

    # add the initial estimate obtained from fourier
    A, T, phi, sigma = sol[:n], sol[n:2*n], sol[2*n:3*n], sol[-1]
    signal = np.zeros(len(t))
    for magnitude, period, phase in zip(A, T, phi):
        signal += magnitude * np.cos(2 * np.pi / period * t + phase)
    plt.plot(t, signal, "C8", label="Initial estimate")

    # add the estimate with maximum value of posterior (MAP estimate)
    A, T, phi, sigma = theta_max[:n], theta_max[n:2*n], theta_max[2*n:3*n], theta_max[-1]
    map_signal = np.zeros(len(t))
    for magnitude, period, phase in zip(A, T, phi):
        map_signal += magnitude * np.cos(2 * np.pi / period * t + phase)
    plt.plot(t, map_signal, "C1", label="MAP estimate")

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

    plt.figure(figsize=(12, 5))
    plt.plot(t, map_signal, label="MAP estimate", c="C1")
    plt.fill_between(t, med_model - 2*spread, med_model + 2*spread, color="C7", alpha=0.5, label=r"$2\sigma$ posterior spread")
    plt.plot(t, y, "k", label="Data points", alpha=0.3)
    plt.legend(fontsize=14)
    plt.xlabel("Time")
    plt.ylabel("Normalized count")
    plt.title(r"$2\sigma$ posterior spread into the observed data space")
    plt.tight_layout()

    ##################################################

    # print confidence intervals for the periods
    # use percentiles of the samples in the marginalized distributions
    jupiter = 11.862  # target period
    for i in range(n, 2*n):
        perc = np.percentile(flat_samples[:, i], [2.5, 16, 84, 97.5])
        print(f"T_{i-n+1} 95% confidence interval: ({perc[0]:.3f}, {perc[-1]:.3f})")
        
        if flat_samples[:, i].min() < jupiter < flat_samples[:, i].max():  # compatible T
            plt.figure(figsize=(8, 4))
            plt.hist(flat_samples[:, i], 100, color="k", histtype="step")
            plt.axvline(jupiter, c="C1", label="Jupiter")
            plt.axvline(perc[1], c="C7", label="68% CI")
            plt.axvline(perc[-2], c="C7")
            plt.axvline(perc[0], c="C9", label="95% CI")
            plt.axvline(perc[-1], c="C9")
            plt.xlabel(fr"$T_{{{i-n+1}}}$")
            plt.ylabel(fr"$p(T_{{{i-n+1}}})$")
            plt.title(fr"Distribution of $T_{{{i-n+1}}}$ samples")
            plt.gca().set_yticks([])
            plt.legend(fontsize=14)
            plt.tight_layout()

    print("-"*50, "\n")
    print(
        "Mean acceptance fraction: {0:.3f}".format(
            np.mean(sampler.acceptance_fraction)
        )
    )

    plt.show()