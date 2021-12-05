import numpy as np
import matplotlib.pyplot as plt

def load_loss(exp_number):
	result_dir = "results"
	return np.loadtxt(f"{result_dir}/experiment_{exp_number}/avg_cum_loss.txt")

def plot_loss(start, T, args, title):

	x = np.arange(start, T)

	for (label, loss) in args.items():
		plt.plot(x, loss[start:], '-', alpha=1, label=label)

	plt.xscale("log")
	plt.yscale("log")

	plt.title(title)
	plt.legend()
	plt.show()

if __name__ == "__main__":

	# parameters
	start = 20
	plot_1 = False
	plot_2 = False
	plot_3 = True 
	plot_4 = True
	plot_5 = True

	# -------- Simple Tuning, Mixed Entropy; Different Delays ---------
	zero = load_loss(1)
	T = len(zero)

	# 1. Gaussian delays with different mean
	if plot_1:
		Gaussian_100_25 = load_loss(2)
		Gaussian_1000_250 = load_loss(3)
		Gaussian_10000_2500 = load_loss(4)
		plot_loss(start, T, title="simple tuning, mixed entropy, Gaussian w. different means", 
			args={
				"no delay": zero,
				"mean: 100, std: 25": Gaussian_100_25,
				"mean: 1000, std: 250": Gaussian_1000_250,
				"mean: 10000, std: 2500": Gaussian_10000_2500
			})

	# 2. Gaussian delays with mean=10000 and different variance
	if plot_2:
		Gaussian_10000_500 = load_loss(5)
		Gaussian_10000_2500 = load_loss(4)
		Gaussian_10000_5000 = load_loss(6)
		Gaussian_10000_10000 = load_loss(7)
		plot_loss(start, T, title="simple tuning, mixed entropy, mean=10000, different variance", 
			args={
				"no delay": zero,
				"std: 500": Gaussian_10000_500,
				"std: 2500": Gaussian_10000_2500,
				"std: 5000": Gaussian_10000_5000,
				"std: 10000": Gaussian_10000_10000,
			})

	# -------- Simple Tuning; Different Regularizer ---------
	# ---no delay
	# without delay, eta_inv = 0 << sqrt(t), so we should expect the mixed ~= Tsallis
	if plot_3:
		mixed = zero
		negative = load_loss(8)
		Tsallis = load_loss(11)
		Orabona = load_loss(14)
		plot_loss(start, T, title="simple tuning, no delay, different regularizer", 
			args={
				"hybrid": mixed,
				"negative": negative,
				"Tsallis": Tsallis,
				"Orabona": Orabona,
			})
	
	# --- Gaussian_1000_250
	if plot_4:
		mixed = load_loss(3)
		negative = load_loss(9)
		Tsallis = load_loss(12)
		Orabona = load_loss(15)
		plot_loss(start, T, title="simple tuning, Gaussian_1000_250, different regularizer", 
			args={
				"hybrid": mixed,
				"negative": negative,
				"Tsallis": Tsallis,
				"Orabona": Orabona,
			})

	# --- Gaussian_10000_2500
	# with large delay, eta_inv ~ O(t) >> sqrt(t), so we should expect the performance of 
	# the mixed entropy to converge to the negative entropy
	if plot_5:	
		mixed = load_loss(4)
		negative = load_loss(10)
		Tsallis = load_loss(13)
		Orabona = load_loss(16)
		plot_loss(start, T, title="simple tuning, Gaussian_10000_2500, different regularizer", 
			args={
				"hybrid": mixed,
				"negative": negative,
				"Tsallis": Tsallis,
				"Orabona": Orabona,
			})