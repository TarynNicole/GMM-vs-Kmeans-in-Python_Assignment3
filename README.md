# GMM-vs-KNN-in-Python_Assignment3
**Comparing Gaussian Mixture Model vs KNN**

This notebook is based on the article by (Patel & Kushwaha, 2020) which compares K-means and Gaussian Mixture Models to evaluate cluster representations of the two methods for heterogeneity in resource usage of Cloud workloads.

# Firstly, what is clustering?
Clustering can be defined as the process of grouping data together based on some similarity measure. It can be hierarchical or partitional, overlapping or fuzzy, exclusive, complete or partial. Clusters may also be well-seperated, prototype-based, graph-based, or density based.

## **K-means**

K-means is defined under partitional clustering techniques, whereby data objects are divided into non-overlapping groups. It is also a prototype-based clustering technique.

The K-means algorithm uses Expectation Maximization algorithm to determine the cluster membership which reduces the Sum of Squared Errors (SSE), which is the SSE of each datapoint and its nearest centroid.

Given k which is the number of clusters, the K-means clustering algorithm can be defined as follows:

Randomly initialize k centroids from the dataset.
Repeat until convergence:

1. For each data point, recompute the distance to each centroid and assign each point to the cluster with the nearest centroid.
2. Recompute the mean for each cluster and update the cluster centroids.

K-means is a simple and relatively easy to understand algorithm. However, its simplicity does indeed lead to practical challenges in its application. Its use of simple distance-from-cluster-center to assign cluster membership also leads to poor performance for many real world situations.

**Further challenges related to K-means Algorithm:**

1. K-Means has no mechanism to handle the uncertainty when a data point is close to more than one cluster centroid.
2. K-Means fails to produce optimal clusters for complex, non-linear decision boundaries.
3. It is sensitive to initial guess of centroids. Different initializations may lead to different clusters.

**Gaussian Mixture Models (GMM)**

A GMM is an unsupervised clustering technique that forms ellipsoidal shaped clusters based on probability density estimations using the Expectation-Maximization. Each cluster is modelled as a Gaussian distribution. The mean and the covariance rather than only the mean as in K-Means, give GMMs the ability to provide a better quantitative measure of fitness per number of clusters. A GMM is represented as a linear combination of the basic Gaussian probability distribution, also known as the Normal Distribution.

The Expectation Maximization algorithm gives maximum likelihood estimates for GM in terms of the mean vector µ, the covariance matrix Σ and the mixing coefficients π. Multiple random initializations is one way to prevent EM from converging to local maxima for log likelihood function with multiple local maximum

The most important thing to know about GMs is that the convergence of this model is based on the EM (expectation-maximization) algorithm. It is somewhat similar to K-Means and it can be summarized as follows:

1. Initialize μ, ∑, and mixing coefficient π and evaluate the initial value of the log likelihood L
2. Evaluate the responsibility function using current parameters
3. Obtain new μ, ∑, and π using newly obtained responsibilities
4. Compute the log likelihood L again. Repeat steps 2–3 until the convergence. The Gaussian Mixtures will also converge to a local minimum.


## Comparing K-means and GMM in terms of key K-means challenges:
**Key challenges with K-Means clustering**

**Optimal Number of Clusters:** The Elbow method is a technique for finding number of clusters, k. In the plot of within-cluster SSE for different number of clusters, k is that value beyond which the distortions begin to approach a constant value. In scikit-learn, SSE is obtained by the inertia attribute of the KMeans model.
Initial Centroids Selection: Initial cluster centroids are determined using the K-Means++ algorithm, which is a technique for obtaining initial guesses for centroids. An inappropriate choice of initial centroids may lead to bad cluster quality and slow convergence rates.

**Convergence Rate:** This can be controlled by setting a maximum number of iterations for EM. In scikit-learn, the K-Means algorithm stops if it converges earlier than the maximum number of iterations. In situations where K-Means does not converge, the algorithm stops when changes in within-cluster SSE is less than a tolerance value. In the experiments, this value is set to 1e-05.
Clustering with Gaussian Mixtures

**Optimal number of Components:** GMM is a generative model that gives a probability distribution for the data set. An optimal number of components avoids overfitting or underfitting and can be determined by evaluating the model likelihood using cross-validation or analytic criterion.
The Akaike Information Criterion (AIC) and the Bayesian Information criterion (BIC)

These are analytic methods that estimate the goodness-of-fit of statistical models relative to each other for a given data set. They provide a quantitative measure of how general the model is, in terms of accuracy of representing future data using the process that generated the current data. AIC and BIC use a penalty for overfitting and under-fitting and this value is larger for BIC than that by AIC

# Objective 
**`Since we do not have access to the dataset used in the paper, we will first implement the two methods on randomly generated data to explain the two methods, with their pros and cons, and then we will futher explore it by implementing it on the Popular Iris dataset to compare the two methods on that dataset. Thus our objective is to evaluate which clustering method between the two has more benefits and performs best when compared to one another.`**
