import mixture_tests as tests
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from scipy.stats import norm
import os
import numpy as np
from helper_functions import *

def get_initial_means(array, k):
    """
    Picks k random points from the 2D array 
    (without replacement) to use as initial 
    cluster means

    params:
    array = numpy.ndarray[numpy.ndarray[float]] - m x n | datapoints x features

    k = int

    returns:
    initial_means = numpy.ndarray[numpy.ndarray[float]]
    """
    targets = [int(np.random.rand() * 10) for i in range(k)]
    return(array[targets])
    
def k_means_step(X, k, means):
    """
    A single update/step of the K-means algorithm
    Based on a input X and current mean estimate,
    predict clusters for each of the pixels and 
    calculate new means. 
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n | pixels x features (already flattened)
    k = int
    means = numpy.ndarray[numpy.ndarray[float]] - k x n

    returns:
    (new_means, clusters)
    new_means = numpy.ndarray[numpy.ndarray[float]] - k x n
    clusters = numpy.ndarray[int] - m sized vector
    """
    clusters = [sorted([(c,np.linalg.norm((j-means[c]), 2)) 
                for c in range(k)], key = lambda x: x[1])[0][0] 
        for j in X]
    new_means = np.array([np.mean(X[np.array(clusters) == i, :], 0) for i in set(clusters)])
    return(new_means, np.array(clusters))

def K_means_2D_dataset(dataset_index, K):
    # Load the dataset from data folder͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    X = np.loadtxt("data/%d_dataset_X.csv" % dataset_index, delimiter=",")
    print("The dataset is of a size:", X.shape)

    # Load the labels͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    # Clustering is unsupervised method, where no labels are provided͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    # However, since we generated the data outselves we know the clusters,͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    # and load them for illustration purposes.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    y = np.int16(np.loadtxt("data/%d_dataset_y.csv" % dataset_index, delimiter=","))

    # Feel free to edit the termination condition for the K-means algorithm͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    # Currently is just runs for n_iterations, before terminating͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    n_iterations = 10
    m,n = X.shape
    means = get_initial_means(X,K)
    clusters = np.zeros([n])
    # keeping track of how clusters and means changed, for visualization purposes͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    means_history = [means]
    clusters_history = [clusters] 
    for iteration_i in range(n_iterations):
        means, clusters = k_means_step(X, K, means)
        clusters_history.append(clusters)

    return X, y, means_history, clusters_history


def k_means_segment(image_values, k=3, initial_means=None):
    """
    Separate the provided RGB values into
    k separate clusters using the k-means algorithm,
    then return an updated version of the image
    with the original values replaced with
    the corresponding cluster values.

    params:
    image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - r x c x ch
    k = int
    initial_means = numpy.ndarray[numpy.ndarray[float]] or None

    returns:
    updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - r x c x ch
    """
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    raise NotImplementedError()

def compute_sigma(X, MU):
    """
    Calculate covariance matrix, based in given X and MU values
    
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    
    returns:
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    """
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    raise NotImplementedError()

def initialize_parameters(X, k):
    """
    Return initial values for training of the GMM
    Set component mean to a random
    pixel's value (without replacement),
    based on the mean calculate covariance matrices,
    and set each component mixing coefficient (PIs)
    to a uniform values
    (e.g. 4 components -> [0.25,0.25,0.25,0.25]).
    
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    
    returns:
    (MU, SIGMA, PI)
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k 
    """
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    # Hint: for initializing SIGMA you could choose to use compute_sigma͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    raise NotImplementedError()

def prob(x, mu, sigma):
    """Calculate the probability of x (a single
    data point or an array of data points) under the
    component with the given mean and covariance.
    The function is intended to compute multivariate
    normal distribution, which is given by N(x;MU,SIGMA).

    params:
    x = numpy.ndarray[float] (for single datapoint) 
        or numpy.ndarray[numpy.ndarray[float]] (for array of datapoints)
    mu = numpy.ndarray[float]
    sigma = numpy.ndarray[numpy.ndarray[float]]

    returns:
    probability = float (for single datapoint) 
                or numpy.ndarray[float] (for array of datapoints)
    """
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    raise NotImplementedError()

def E_step(X,MU,SIGMA,PI,k):
    """
    E-step - Expectation 
    Calculate responsibility for each
    of the data points, for the given 
    MU, SIGMA and PI.
    
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k
    k = int
    
    returns:
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    raise NotImplementedError()

def M_step(X, r, k):
    """
    M-step - Maximization
    Calculate new MU, SIGMA and PI matrices
    based on the given responsibilities.
    
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    r = numpy.ndarray[numpy.ndarray[float]] - k x m
    k = int
    
    returns:
    (new_MU, new_SIGMA, new_PI)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k
    """
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    raise NotImplementedError()

def loglikelihood(X, PI, MU, SIGMA, k):
    """Calculate a log likelihood of the 
    trained model based on the following
    formula for posterior probability:
    
    log(Pr(X | mixing, mean, stdev)) = sum((i=1 to m), log(sum((j=1 to k),
                                      mixing_j * N(x_i | mean_j,stdev_j))))

    Make sure you are using natural log, instead of log base 2 or base 10.
    
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k
    k = int

    returns:
    log_likelihood = float
    """
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    raise NotImplementedError()


def train_model(X, k, convergence_function, initial_values = None):
    """
    Train the mixture model using the 
    expectation-maximization algorithm. 
    E.g., iterate E and M steps from 
    above until convergence.
    If the initial_values are None, initialize them.
    Else it's a tuple of the format (MU, SIGMA, PI).
    Convergence is reached when convergence_function
    returns terminate as True,
    see default convergence_function example 
    in `helper_functions.py`

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    convergence_function = func
    initial_values = None or (MU, SIGMA, PI)

    returns:
    (new_MU, new_SIGMA, new_PI, responsibility)
    new_MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    new_SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    new_PI = numpy.ndarray[float] - k
    responsibility = numpy.ndarray[numpy.ndarray[float]] - k x m
    """
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    raise NotImplementedError()

def cluster(r):
    """
    Based on a given responsibilities matrix
    return an array of cluster indices.
    Assign each datapoint to a cluster based,
    on component with a max-likelihood 
    (maximum responsibility value).
    
    params:
    r = numpy.ndarray[numpy.ndarray[float]] - k x m - responsibility matrix
    
    return:
    clusters = numpy.ndarray[int] - m x 1 
    """
    # TODO: finish this͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    raise NotImplementedError()

def segment(X, MU, k, r):
    """
    Segment the X matrix into k components. 
    Returns a matrix where each data point is 
    replaced with its max-likelihood component mean.
    E.g., return the original matrix where each pixel's
    intensity replaced with its max-likelihood 
    component mean. (the shape is still mxn, not 
    original image size)

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    k = int
    r = numpy.ndarray[numpy.ndarray[float]] - k x m - responsibility matrix

    returns:
    new_X = numpy.ndarray[numpy.ndarray[float]] - m x n
    """
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    raise NotImplementedError()

def best_segment(X,k,iters):
    """Determine the best segmentation
    of the image by repeatedly
    training the model and
    calculating its likelihood.
    Return the segment with the
    highest likelihood.

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    k = int
    iters = int

    returns:
    (likelihood, segment)
    likelihood = float
    segment = numpy.ndarray[numpy.ndarray[float]]
    """
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    raise NotImplementedError()

def GMM_2D_dataset(dataset_index, K):
    # Load the dataset from data folder͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    X = np.loadtxt("data/%d_dataset_X.csv" % dataset_index, delimiter=",")
    print("There are %d datapoints in the current dataset, each of a size %d" % X.shape)
    print("""\nNote that that the Gaussian Ellipses and Normal Curves may not share the
same color as the points they represent (within the same chart).
In fact, the Gaussian Ellipses and Normal Curves represent the clusters
in the top left chart (and thus share colors with those points).""")
    # Load the labels͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    # Clustering is unsupervised method, where no labels are provided͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    # However, since we generated the data outselves we know the labels,͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    # and load them for illustration purposes.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    y = np.int16(np.loadtxt("data/%d_dataset_y.csv" % dataset_index, delimiter=","))
    # Feel free to edit the termination condition for the EM algorithm͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    # Currently is just runs for n_iterations, before terminating͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    
    MU, SIGMA, PI = initialize_parameters(X, K)
    
    clusters_history = []
    statistics_history = []
    for _ in range(200):
        r = E_step(X,MU,SIGMA,PI,K)
        new_MU, new_SIGMA, new_PI = M_step(X, r, K)
        PI, MU, SIGMA = new_PI, new_MU, new_SIGMA
        clusters = cluster(r)
        clusters_history.append(clusters)
        statistics_history.append((PI, MU, SIGMA))

    return X, y, clusters_history, statistics_history

def setup_subplot(plt, i, title, plot_number):
    ax = plt.subplot(2, 2, plot_number)
    ax.set_title(title)
    ax.patch.set_facecolor('gray')
    ax.patch.set_alpha(0.1)
    return ax

def plot_gaussian_ellipse(k, mean, covar, ax2, colors):
    v,w = np.linalg.eig(covar)
        
    angle = np.arctan(w[1,0] / w[0,0])
    angle = 180 * angle / np.pi
    
    color = colors[k % len(colors)]
    for i in range(3,8):
        plot_v = i * np.sqrt(v)
        ellipse = pat.Ellipse(mean, plot_v[0], plot_v[1], angle, fill = True, alpha = 0.10, lw = 1.0, ls = 'dashdot', ec = 'black', fc = color, zorder = 0)
        ax2.add_artist(ellipse)

def plot_gaussian(X, mean, var, X_min, X_max, ax):
    samples = np.linspace(X_min, X_max, 100)
    ax.plot(samples, norm.pdf(samples, mean, var))
    

def bayes_info_criterion(X, PI, MU, SIGMA, k):
    """
    See description above
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
     numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k
    k = int
    
    return:
    bayes_info_criterion = int
    """
    # TODO: finish this function͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    raise NotImplementedError()

def BIC_likelihood_model_test(image_matrix, comp_means):
    """Returns the number of components
    corresponding to the minimum BIC 
    and maximum likelihood with respect
    to image_matrix and comp_means.
    
    params:
    image_matrix = numpy.ndarray[numpy.ndarray[float]] - m x n
    comp_means = list(numpy.ndarray[numpy.ndarray[float]]) - list(k x n) (means for each value of k)

    returns:
    (n_comp_min_bic, n_comp_max_likelihood)
    n_comp_min_bic = int
    n_comp_max_likelihood = int
    """
    # TODO: finish this method͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    raise NotImplementedError()

def return_your_name():
    return("Kiavosh Peynabard")
