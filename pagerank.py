import numpy as np

def _createWeights(arr):
    '''
        Given a list of edges in a graph, returns the labels of distinct nodes
        and a dictionary of their weights.
    '''
    labels, counts = np.unique(arr[:, 0], return_counts=True)   # Finds the unique entries in first column, returns their values
                                                                # and their counts to calculate weights
    weight_dict = dict()
    for index, label in enumerate(labels):                      # Creates a dictionary that holds the normalization factor,
        weight_dict[label] = 1 / counts[index]                  # 1/(number of outgoing links) for each node

    return labels, weight_dict


def _createTransitionMatrix(arr, labels, weight_dict):
    '''
        Create a transition matrix based on the transitions in arr of shape (n, 2).
        Labels is the list of distinct nodes in the original graph.
        Weight_dict is a dictionary where the weight of a link for each node in labels are kept.  
    '''
    transitionMatrix = np.zeros((labels.size, labels.size))

    for i, label in enumerate(labels):
        links_from_label = arr[np.nonzero(label == arr[:, 0]), 1][0]
    #    if links_from_label.shape == (0,):                              # For dangling nodes, nodes that have no outgoing links
    #        weight_dict[label] = 1 / labels.size                        # Not necessary while using with a damping factor
    #        links_to_label = labels                                  
        transitionMatrix[np.searchsorted(labels, links_from_label), i] = weight_dict[label]

    return transitionMatrix


def _checkStochastic(matrix, eps=1e-10, verbose=False):
    '''
        Checks if a given matrix is stochastic, within an error of eps.
        Returns the number of columns that violates the condition.
        If verbose is set to True, prints the results.
    '''
    if verbose:
        print(f"Sums of columns that do not add to 1 in a reasonable error margin {eps} will be shown.")
    nonstochastic_columns = 0
    for i in matrix.sum(0):
        if not (i - 1 < eps):
            nonstochastic_columns += 1
            if verbose: print(i)
    if verbose:
        print(f"Calculation finished, average value of sums of all columns is {np.mean(matrix.sum(0))}.")
    return nonstochastic_columns

def _checkSparsity(matrix, verbose=False):
    '''
        Checks the sparsity of a given matrix. Returns the sparcity rate.
        Set verbose to true to print the results.
    '''
    zeros = np.count_nonzero(matrix==0)
    elements = matrix.size
    sparse_rate = np.divide(zeros,elements)
    if verbose:
        print(f"There are {elements} elements in total, {zeros} of them are zero.")
        print(f"Sparsity rate of this matrix is {sparse_rate*100}%")
    return sparse_rate


def addDamping(transitionMatrix, labels, p = 0.15):
    '''
        Adds damping vector to the given matrix. Damping vector simulates
        the random surfing between two nodes, even if they are not linked.
    '''
    if(p < 0 or p > 1):
        throw ValueError("Please try again with a damping factor in interval [0,1].")
    rankMatrix = (1 - p) * transitionMatrix + p * ((1/labels.size) * np.ones((labels.size,labels.size)))
    return rankMatrix


def solveRank(rankMatrix, tol=1e-7, verbose=False):
    '''
    Given the stochastic transition matrix of a graph, returns the PageRank vector of the graph,
    using the power method until the average distance between two vectors are less than 'tol'.

    Set verbose to True in order the see the iteration number and the difference of last two vectors.
    '''
    v0 = np.ones(rankMatrix.shape[0]) / rankMatrix.shape[0]

    iteration_counter = 1
    while True:
        v = np.dot(rankMatrix, v0)
        v = v / np.linalg.norm(v)
        if (np.mean(np.abs(np.subtract(v,v0))) < 2 * tol):
            break        
        iteration_counter += 1
        v0 = v
        if verbose:
            print(f"Iteration: {iteration_counter}, Error: {np.mean(np.abs(np.subtract(v,v0)))}")
    
    if verbose:
        print(f"Appropriate vector found in {iteration_counter} iterations, final difference between two iteration result vectors was less than {tol}.")
    return v


def calculateRankVector(data, damping=0):
    '''
        Calculates the PageRank ranking vector on a (n, 2) array, which
        represents the edges of a directed graphs. Each row represents an edge
        as a tuple (node_from, node_to).
        Set damping to a positive value < 1 to apply damping on the final matrix.
    '''
    labels, weights = createWeights(data)
    rankMatrix = createTransitionMatrix(data, labels, weights)
    if damping = 0:
        rankMatrix = addDamping(rankMatrix, labels, damping)
    resultVector = solveRank(rankMatrix)
    return resultVector


def rankItemsByRankings(labels, final):
    '''
        Ranks the items in 'labels' using the values in the 'final' vector, in descending order. 
    '''
    return labels[final.argsort()][::-1]