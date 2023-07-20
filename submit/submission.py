
    #################################################
    ### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
    #################################################
    # file to edit: solution.ipynbimport numpy as np
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
    
    N = array.shape[0]
    indx = np.random.choice(N,size = k, replace = False)

    return(array[indx,:])

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
################ END OF LOCAL TEST CODE SECTION ######################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃

def compute_distance(X, centroids, k):
    #Used by ML course HW distance calculation.
    distance = np.zeros((X.shape[0], k))
    for k in range(k):
        row_norm = np.linalg.norm(X - centroids[k, :], axis=1)
        distance[:, k] = np.square(row_norm)
    return distance

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

    distance = compute_distance(X, means, k)
    y = np.argmin(distance, axis=1)

    new_mean = np.zeros((k, X.shape[1]))
    for k_ in range(k):
        new_mean[k_, :] = np.mean(X[y == k_, :], axis=0)

    return(new_mean, y)

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
################ END OF LOCAL TEST CODE SECTION ######################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃

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
    not_converged = True
    steps = 0
    max_ittr = 250
    X = flatten_image_matrix(image_matrix=image_values)

    if (initial_means).any() is None:
        old_mean = get_initial_means(X, k=k)
    else:
        old_mean = initial_means

    while not_converged and steps < max_ittr:
        new_mean, y = k_means_step(X, k, old_mean)
        if np.all(new_mean == old_mean):
            not_converged = False
        old_mean = new_mean
        steps += 1

    im_val = X

    for i in range(k):
        im_val[y == i] = old_mean[i]

    return im_val.reshape(image_values.shape)

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
################ END OF LOCAL TEST CODE SECTION ######################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃

def compute_sigma(X, MU):
    """
    Calculate covariance matrix, based in given X and MU values

    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n

    returns:
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    """
    body = np.zeros((MU.shape[0], X.shape[0], X.shape[1]))
    for k in range(MU.shape[0]):
        body[k] = np.array([X[m] - MU[k] for m in range(X.shape[0])])
    sigma = np.array([np.matmul(body[k].T, body[k]) for k in range(body.shape[0])])/X.shape[0]
    return sigma
########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
################ END OF LOCAL TEST CODE SECTION ######################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃

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
    MU = get_initial_means(X, k)
    SIGMA = compute_sigma(X, MU)
    PI = np.array([1/k for t in range(1, k+1)])
    return(MU,SIGMA,PI)

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
################ END OF LOCAL TEST CODE SECTION ######################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃

def prob(x, mu, sigma):
    """Calculate the probability of x (a single
    data point or an array of data points) under the
    component with the given mean and covariance.
    The function is intended to compute multivariate
    normal distribution, which is given by N(x;MU,SIGMA).

    params:
    x = numpy.ndarray[float] (for single datapoint) a
        or numpy.ndarray[numpy.ndarray[float]] (for array of datapoints)
    mu = numpy.ndarray[float]
    sigma = numpy.ndarray[numpy.ndarray[float]]

    returns:
    probability = float (for single datapoint)
                or numpy.ndarray[float] (for array of datapoints)
    """
    probib = None
    bridge = x
    if len(x.shape) == 1 :
        bridge = np.empty((1,len(x)))
        bridge = np.array([x])
    body = bridge - mu
    exponent = -1/2 * np.einsum('ij,ji->i', np.dot(body, np.linalg.pinv(sigma)), body.T)
    dinom = 1/(np.sqrt((2*np.pi)**bridge.shape[1] *(np.linalg.det(sigma))))
    probib = (dinom)*np.exp(exponent)
    if len(x.shape) == 1:
        return probib[0]

    return probib

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
################ END OF LOCAL TEST CODE SECTION ######################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃

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
    m, n = X.shape
    responsibility = np.zeros((k, m))
    responsibility = np.array([PI[i] * prob(X, mu=MU[i, :], sigma=SIGMA[i, :, :]) for i in range(k)])
    responsibility /= np.sum(responsibility, axis=0)
    return responsibility

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
################ END OF LOCAL TEST CODE SECTION ######################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃

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
    m, n = X.shape
    new_PI = np.mean(r, axis=1)
    new_MU = np.zeros((k, n))
    new_MU = np.array([np.sum(r[k].reshape(-1, 1) * X, axis=0) / np.sum(r[k]) for k in range(k)])
    new_SIGMA = np.zeros((k, n, n))
    for i in range(k):
        X_centered = X - new_MU[i]
        new_SIGMA[i] = (X_centered.T * r[i]).dot(X_centered) / np.sum(r[i])

    return (new_MU, new_SIGMA, new_PI)


########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
################ END OF LOCAL TEST CODE SECTION ######################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃

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

    weighted = 0
    for j in range(k):
        weighted += PI[j] *prob(X, mu=MU[j], sigma=SIGMA[j])
    return np.sum(np.log(weighted))


########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
################ END OF LOCAL TEST CODE SECTION ######################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃

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

    MU, SIGMA, PI = initialize_parameters(X, k)
    if initial_values is not None:
        MU, SIGMA, PI = initial_values

    old_likelihood = 0
    new_likelihood = loglikelihood(X, PI, MU, SIGMA, k)
    count = 0
    convrged = False
    while not convrged:
        responsibility = E_step(X, MU, SIGMA, PI, k)
        MU, SIGMA, PI = M_step(X, responsibility, k)
        old_likelihood = new_likelihood
        new_likelihood = loglikelihood(X, PI, MU, SIGMA, k)
        count, convrged = convergence_function(old_likelihood, new_likelihood, count)

    return MU, SIGMA, PI, responsibility

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
################ END OF LOCAL TEST CODE SECTION ######################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃

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
    return(np.argmax(r, axis=0))

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
################ END OF LOCAL TEST CODE SECTION ######################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃

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
    clusters = cluster(r)
    return(MU[clusters])

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
################ END OF LOCAL TEST CODE SECTION ######################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃

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

    best_likelihood = 0
    best_segment = None

    for i in range(iters):
        MU, SIGMA, PI, responsibility = train_model(X, k, convergence_function=default_convergence)
        print(MU.shape)

        likelihood = loglikelihood(X, PI, MU, SIGMA, k)
        if likelihood > best_likelihood:
            best_likelihood = likelihood
            best_segment = segment(X,MU,k, responsibility)

    return (best_likelihood, best_segment)

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
################ END OF LOCAL TEST CODE SECTION ######################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃

def bayes_info_criterion(X, PI, MU, SIGMA, k):
    """
    See description above
    params:
    X = numpy.ndarray[numpy.ndarray[float]] - m x n
    MU = numpy.ndarray[numpy.ndarray[float]] - k x n
    SIGMA = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]] - k x n x n
    PI = numpy.ndarray[float] - k
    k = int

    return:
    bayes_info_criterion = int
    """
    m, n = X.shape
    likelihood = loglikelihood(X, PI, MU, SIGMA, MU.shape[0])
    num_params = k * (n + n*(n+1)//2 + 1) - 1
    return np.log(m)* num_params -2*likelihood

########## DON'T WRITE ANY CODE OUTSIDE THE FUNCTION! ################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
##### CODE BELOW IS USED FOR RUNNING LOCAL TEST DON'T MODIFY IT ######͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
################ END OF LOCAL TEST CODE SECTION ######################͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃

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
    n_comp_min_bic = 0
    n_comp_max_likelihood = 0
    min_bic = np.inf
    max_likelihood = 0
    for k, means in enumerate(comp_means, start=1):
        sigma = compute_sigma(image_matrix, means)
        pi = np.array([1/k for t in range(1, k+1)])
        mu, sigma, pi, r = train_model(X = image_matrix, k=k, convergence_function=default_convergence , initial_values=(means, sigma, pi))
        bic = bayes_info_criterion(image_matrix, pi, mu, sigma, k)
        #minizizing cost
        if bic < min_bic:
            min_bic = bic
            n_comp_min_bic = k
        #maximizing likelihood
        likelihood = loglikelihood(image_matrix, pi, mu, sigma, mu.shape[0])
        if likelihood > max_likelihood:
            max_likelihood = likelihood
            n_comp_max_likelihood = k

    return n_comp_min_bic, n_comp_max_likelihood

def return_your_name():
    return("Kiavosh Peynabard")