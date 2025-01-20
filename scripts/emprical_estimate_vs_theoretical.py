import numpy as np
from scipy.stats import multivariate_normal, ncx2
from sklearn.neighbors import BallTree
from functools import partial
import matplotlib.pyplot as plt
from multiprocessing import Pool

random_seed = 42
np.random.seed(random_seed)

def generate_datasets(d, n1=50, n2=5000):
    D1 = np.random.multivariate_normal(np.zeros(d), np.eye(d), n1)
    D2 = np.random.multivariate_normal(np.zeros(d), np.eye(d), n2)
    return D1, D2

def create_centered_linear_classifier(center, d):
    normal_vector = np.random.randn(d)
    normal_vector /= np.linalg.norm(normal_vector)
    bias = -np.dot(normal_vector, center)
    return normal_vector, bias

def linear_classifier(x, normal_vector, bias):
    return np.dot(x, normal_vector) + bias > 0

def integral_cap_mc(n_samples, dimension, radius, center, mean, l_classifier):
    points = np.random.uniform(-radius, radius, (n_samples, dimension)) + center
    distances = np.linalg.norm(points - center, axis=1)
    in_ball = distances <= radius
    in_cap = in_ball & l_classifier(points)
    prob_density = multivariate_normal.pdf(points, mean=mean, cov=1)
    return np.mean(prob_density[in_cap]) * (2 * radius) ** dimension

def empirical_accuracy(center, points, tree, radius, l_classifier):
    indices = tree.query_radius([center], r=radius)[0]
    ball_points = points[indices]
    if ball_points.size == 0:
        return 0, 0
    predictions = l_classifier(ball_points)
    return np.mean(predictions), len(ball_points)

def process_point(args):
    x, D2, tree, dimension, radius, n_samples_mc, mean = args
    normal_vector, bias = create_centered_linear_classifier(x, dimension)
    l_classifier = partial(linear_classifier, normal_vector=normal_vector, bias=bias)
    
    # Monte Carlo estimation
    mass_cap = integral_cap_mc(n_samples_mc, dimension, radius, x, mean, l_classifier)
    mass_ball = ncx2.cdf(radius**2, dimension, np.linalg.norm(x)**2)
    theoretical_estimate = mass_cap / mass_ball

    # Empirical estimation
    emp_acc, n_points_in_ball = empirical_accuracy(x, D2, tree, radius, l_classifier)
    return theoretical_estimate, emp_acc, n_points_in_ball

def compute_estimates(D1, D2, dimension, radius, n_samples_mc, mean):
    tree = BallTree(D2)
    args = [(x, D2, tree, dimension, radius, n_samples_mc, mean) for x in D1]

    # Use multiprocessing to parallelize computations
    with Pool() as pool:
        results = pool.map(process_point, args)

    theoretical_estimates, empirical_estimates, n_points_in_ball = zip(*results)
    
    # Compute averages and errors
    avg_theoretical = np.mean(theoretical_estimates)
    avg_empirical = np.mean(empirical_estimates)
    se_empirical = (np.array(theoretical_estimates) - np.array(empirical_estimates)) ** 2
    mse_empirical = np.mean(se_empirical)
    
    return avg_theoretical, avg_empirical, mse_empirical, theoretical_estimates, empirical_estimates, se_empirical

def plot_res(theoretical_estimates, empirical_estimates, se_empirical, title=' ', save=False):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    x = np.linspace(0, 1, 100)
    plt.plot(x, x, 'r--')
    plt.scatter(theoretical_estimates, empirical_estimates)
    plt.xlabel('Theoretical Estimates')
    plt.ylabel('Empirical Estimates')
    plt.title(title)

    plt.subplot(1, 2, 2)
    plt.hist(se_empirical, bins=20)
    plt.title('Histogram of Squared Errors')
    if save:
        plt.savefig(f'{title}.png')
    plt.show()

# Main simulation parameters
n_samples_mc = 1_000_000
n_samples_d1 = 50
radius = 5
dimension = 10
mean = np.zeros(dimension)
ls_n_samples_d2 = np.linspace(100, 100_000, 30, dtype=int)

# Store results
ls_theoretical_estimates, ls_empirical_estimates, ls_se = [], [], []
ls_avg_empirical, ls_avg_theoretical, ls_mse = [], [], []

for n_samples_d2 in ls_n_samples_d2:
    D1, D2 = generate_datasets(dimension, n_samples_d1, n_samples_d2)
    avg_theoretical, avg_empirical, mse_empirical, theoretical_estimates, empirical_estimates, se_empirical = compute_estimates(
        D1, D2, dimension, radius, n_samples_mc, mean
    )
    
    ls_avg_theoretical.append(avg_theoretical)
    ls_avg_empirical.append(avg_empirical)
    ls_se.append(se_empirical)
    ls_empirical_estimates.append(empirical_estimates)
    ls_theoretical_estimates.append(theoretical_estimates)
    ls_mse.append(mse_empirical)

# Convert results to arrays
ls_theoretical_estimates = np.array(ls_theoretical_estimates)
ls_empirical_estimates = np.array(ls_empirical_estimates)
ls_se = np.array(ls_se)
ls_avg_theoretical = np.array(ls_avg_theoretical).reshape(-1, 1)
ls_avg_empirical = np.array(ls_avg_empirical).reshape(-1, 1)
ls_mse = np.array(ls_mse).reshape(-1, 1)

print("Shapes:")
print(ls_theoretical_estimates.shape, ls_avg_empirical.shape, ls_se.shape, ls_avg_theoretical.shape, ls_avg_empirical.shape, ls_mse.shape)

# Save results
np.save(f'ls_theoretical_estimates_radius{radius}_dim_{dimension}_nd2{ls_n_samples_d2[0]}-{ls_n_samples_d2[-1]}.npy', ls_theoretical_estimates)
np.save(f'ls_empirical_estimates_radius{radius}_dim_{dimension}_nd2{ls_n_samples_d2[0]}-{ls_n_samples_d2[-1]}.npy', ls_empirical_estimates)
np.save(f'ls_se_radius{radius}_dim_{dimension}_nd2{ls_n_samples_d2[0]}-{ls_n_samples_d2[-1]}.npy', ls_se)
np.save(f'ls_mse_radius{radius}_dim_{dimension}_nd2{ls_n_samples_d2[0]}-{ls_n_samples_d2[-1]}.npy', ls_mse)
np.save(f'ls_avg_empirical_radius{radius}_dim_{dimension}_nd2{ls_n_samples_d2[0]}-{ls_n_samples_d2[-1]}.npy', ls_avg_empirical)
np.save(f'ls_avg_theoretical_radius{radius}_dim_{dimension}_nd2{ls_n_samples_d2[0]}-{ls_n_samples_d2[-1]}.npy', ls_avg_theoretical)


# %%
# random_seed = 42
# np.random.seed(random_seed)
# n_samples_mc=10_000_000
# n_samples_d1 = 50
# n_samples_d2 = 5000
# dimensions = np.linspace(2, 40, 10, dtype=int)
# ls_theoretical_estimates, ls_empirical_estimates, ls_se = [], [], []
# ls_avg_empirical, ls_avg_theoretical, ls_mse = [], [], []
# radius = 5

# for dimension in dimensions:
#     mean = np.zeros(dimension)
#     D1, D2 = generate_datasets(dimension, n_samples_d1, n_samples_d2)
#     avg_theoretical, avg_empirical, mse_empirical, theoretical_estimates, empirical_estimates, se_empirical = compute_estimates(D1, D2, dimension, radius, n_samples_mc, mean)
#     ls_theoretical_estimates.append(avg_theoretical)
#     ls_empirical_estimates.append(avg_empirical)
#     ls_se.append(mse_empirical)
#     ls_mse.append(mse_empirical)
#     ls_avg_empirical.append(empirical_estimates)
#     ls_avg_theoretical.append(theoretical_estimates)

# ls_theoretical_estimates = np.array(ls_theoretical_estimates)  # Shape (10, 50)
# ls_empirical_estimates = np.array(ls_empirical_estimates)      # Shape (10, 50)
# ls_se = np.array(ls_se)                                        # Shape (10, 50)
# ls_avg_theoretical = np.array(ls_avg_theoretical).reshape(-1, 1)  # Shape (10, 1)
# ls_avg_empirical = np.array(ls_avg_empirical).reshape(-1, 1)      # Shape (10, 1)
# ls_mse = np.array(ls_mse).reshape(-1, 1) 

# np.save(f'ls_theoretical_estimates_radius{radius}_dim_{dimensions[0]}-{dimensions[-1]}_nd2{n_samples_d2}.npy', ls_theoretical_estimates)
# np.save(f'ls_empirical_estimates_radius{radius}_dim_{dimensions[0]}-{dimensions[-1]}_nd2{n_samples_d2}.npy', ls_empirical_estimates)
# np.save(f'ls_se_radius{radius}_dim_{dimensions[0]}-{dimensions[-1]}_nd2{n_samples_d2}.npy', ls_se)
# np.save(f'ls_mse_radius{radius}_dim_{dimensions[0]}-{dimensions[-1]}_nd2{n_samples_d2}.npy', ls_mse)
# np.save(f'ls_avg_empirical_radius{radius}_dim_{dimensions[0]}-{dimensions[-1]}_nd2{n_samples_d2}.npy', ls_avg_empirical)
# np.save(f'ls_avg_theoretical_radius{radius}_dim_{dimensions[0]}-{dimensions[-1]}_nd2{n_samples_d2}.npy', ls_avg_theoretical)

# # Plot the squared errors for each dimension
# plt.figure(figsize=(12, 6))
# plt.plot(dimensions, mse_empirical, label='Squared Errors')
# plt.xlabel('Dimension')
# plt.ylabel('MSE')
# plt.title('Mean Squared Errors for Different Dimensions')
# plt.legend()
# plt.show()


# # Plot the squared errors for each dimension
# plt.figure(figsize=(12, 6))
# plt.plot(ls_n_samples_d2, ls_mse, label='Squared Errors')
# plt.xlabel('Number of samples of D2')
# plt.ylabel('MSE')
# plt.title('Mean Squared Errors for different samples sizes of D2')
# plt.legend()

