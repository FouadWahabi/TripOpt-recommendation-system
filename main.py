from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import tensorflow as tf
from random import choice, shuffle
from pymongo import MongoClient
from bson.objectid import ObjectId
from tensorflow.tensorboard.tensorboard import FLAGS


def _connect_mongo(host, db, port=27017, username=None, password=None):
    """ Creating a connection with a Mongo database """
    if username and password:
        mongo_uri = 'mongodb://%s:%s@%s:%s/%s' % (username, password, host, port, db)
        conn = MongoClient(mongo_uri)
    else:
        conn = MongoClient(host, port)

    return conn[db]


def load_activities_categories(db):
    """ Load the list of activities from the db """
    activities_categories = db['CategoryActivity'].find()
    activities = {}
    categories = set([])
    for activity_category in activities_categories:
        categories.add(activity_category['categoryId'])
        if not activities.get(activity_category['activityId']):
            activities[activity_category['activityId']] = [activity_category['categoryId']]
        else:
            activities[activity_category['activityId']].append(activity_category['categoryId'])
    return categories, activities


def create_activities_categories_array(categories, activities):
    return np.array(
        [[1 if category in activities[activity] else 0 for category in categories] for activity in activities])


def load_train_set(db):
    # Like value
    like = 0.002
    """ Result train_set """
    train_set = {}

    (categories, activities) = load_activities_categories(db)

    """ Load the list of votes from the db"""
    votes = db['Vote'].find()

    for vote in votes:
        if not train_set.get(vote['userId']):
            train_set[vote['userId']] = {}

        activity_categories = activities[vote['activityId']]
        value = like if bool(vote['value']) else -like
        for category in categories:
            if not train_set[vote['userId']].get(category):
                train_set[vote['userId']][category] = 0

        for activity_category in activity_categories:
            if not train_set[vote['userId']].get(activity_category):
                train_set[vote['userId']][activity_category] += value

    return np.array([user for user in sorted(train_set)]), np.array(
        [[train_set[user][category] for category in sorted(train_set[user])] for user in sorted(train_set)])


def k_means_cluster(vectors, noofclusters):
    """ Clustering users based on there category weights using the K-Means method """

    noofclusters = int(noofclusters)
    assert noofclusters < len(vectors)

    dim = len(vectors[0])

    vector_indices = list(range(len(vectors)))
    shuffle(vector_indices)

    # Create a tf Graph

    graph = tf.Graph()

    with graph.as_default():

        # Create tf session

        sess = tf.Session()

        # Initialize centroid vectors
        centroids = [tf.Variable((vectors[vector_indices[i]]))
                     for i in range(noofclusters)]
        # Node to Create centroid variables
        centroid_value = tf.placeholder("float64", [dim])
        cent_assigns = []
        for centroid in centroids:
            cent_assigns.append(tf.assign(centroid, centroid_value))

        # Initialize assignments vectors to zeros
        assignments = [tf.Variable(0) for i in range(len(vectors))]

        # Assign values to to assignments
        assignment_value = tf.placeholder("int32")
        cluster_assigns = []
        for assignment in assignments:
            cluster_assigns.append(tf.assign(assignment,
                                             assignment_value))

        # Node to Compute the mean
        # The placeholder for the input
        mean_input = tf.placeholder("float", [None, dim])
        # The Node/op takes the input and computes a mean along the 0th
        # dimension
        mean_op = tf.reduce_mean(mean_input, 0)

        # Node for computing Euclidean distances
        # Placeholders for input
        v1 = tf.placeholder("float", [dim])
        v2 = tf.placeholder("float", [dim])
        euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(
            v1, v2), 2)))

        # This node will figure out which cluster to assign a vector to based on
        # Euclidean distances of the vector from the centroids
        # Placeholder for input
        centroid_distances = tf.placeholder("float", [noofclusters])
        cluster_assignment = tf.argmin(centroid_distances, 0)

        # Initialize state variables
        init_op = tf.global_variables_initializer()

        # Initialize all variables
        sess.run(init_op)

        # Start the clustering iteration

        # Now perform the Expectation-Maximization steps of K-Means clustering
        # iterations. To keep things simple, we will only do a set number of
        # iterations, instead of using a Stopping Criterion.
        noofiterations = 100
        for iteration_n in range(noofiterations):

            # Expectation step
            ##Based on the centroid locations till last iteration, compute
            ##the _expected_ centroid assignments.
            # Iterate over each vector
            for vector_n in range(len(vectors)):
                vect = vectors[vector_n]
                # Compute Euclidean distance between this vector and each centroid
                distances = [sess.run(euclid_dist, feed_dict={
                    v1: vect, v2: sess.run(centroid)})
                             for centroid in centroids]
                # Now use the cluster assignment node, with the distances
                # as the input
                assignment = sess.run(cluster_assignment, feed_dict={
                    centroid_distances: distances})
                # Now assign the value to the appropriate state variable
                sess.run(cluster_assigns[vector_n], feed_dict={
                    assignment_value: assignment})

            # Maximisation step
            # Based on the expected state computed from the Expectation Step,
            # compute the locations of the centroids so as to maximize the
            # overall objective of minimizing within-cluster Sum-of-Squares
            for cluster_n in range(noofclusters):
                # Collect all the vectors assigned to this cluster
                assigned_vects = [vectors[i] for i in range(len(vectors))
                                  if sess.run(assignments[i]) == cluster_n]

                if assigned_vects:
                    # Compute new centroid location
                    new_location = sess.run(mean_op, feed_dict={
                        mean_input: array(assigned_vects)})
                    # Assign value to appropriate variable
                    sess.run(cent_assigns[cluster_n], feed_dict={
                        centroid_value: new_location})

        # Return centroids and assignments
        centroids = sess.run(centroids)
        assignments = sess.run(assignments)
        return centroids, assignments


def redim(x, layer_sizes):
    # Build the encoding layers
    next_layer_input = x

    encoding_matrices = []
    for dim in layer_sizes:
        input_dim = int(next_layer_input.get_shape()[1])

        # Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)]
        W = tf.Variable(
            tf.random_uniform([input_dim, dim], -1.0 / math.sqrt(input_dim), 1.0 / math.sqrt(input_dim)))

        # Initialize b to zero
        b = tf.Variable(tf.zeros([dim]))

        # We are going to use tied-weights so store the W matrix for later reference.
        encoding_matrices.append(W)

        output = tf.nn.tanh(tf.matmul(next_layer_input, W) + b)

        # the input into the next layer is the output of this layer
        next_layer_input = output

    # The fully encoded x value is now stored in the next_layer_input
    encoded_x = next_layer_input

    # build the reconstruction layers by reversing the reductions
    layer_sizes.reverse()
    encoding_matrices.reverse()

    for i, dim in enumerate(layer_sizes[1:] + [int(x.get_shape()[1])]):
        # we are using tied weights, so just lookup the encoding matrix for this step and transpose it
        W = tf.transpose(encoding_matrices[i])
        b = tf.Variable(tf.zeros([dim]))
        output = tf.nn.tanh(tf.matmul(next_layer_input, W) + b)
        next_layer_input = output

    # the fully encoded and reconstructed value of x is here:
    reconstructed_x = next_layer_input

    return {
        'encoded': encoded_x,
        'decoded': reconstructed_x,
        'cost': tf.sqrt(tf.reduce_mean(tf.square(x - reconstructed_x)))
    }


def create_cluster_assignments(train_set, users_idx, assignments, noofclusters):
    """ Group datapoints by clustes """
    cluster_assignments = [[] for h in xrange(noofclusters)]
    cluster_users = [[] for h in xrange(noofclusters)]
    for assignment_idx in xrange(len(assignments)):
        cluster_assignments[assignments[assignment_idx]].append(train_set[assignment_idx])
        cluster_users[assignments[assignment_idx]].append(users_idx[assignment_idx])
    return cluster_assignments, cluster_users


def plot_clusters(data_points, users_idx, centroids, assignments):
    import matplotlib.pyplot as plt
    """ Plot out the different clusters """

    # Choose a different colour for each cluster
    colour = plt.cm.rainbow(np.linspace(0, 1, len(centroids)))

    cluster_assignments, cluster_users = create_cluster_assignments(data_points, users_idx, assignments, len(centroids))

    for i, assignement in enumerate(cluster_assignments):
        centroid = centroids[i]
        print(np.array(assignement)[:, 0])
        plt.scatter(np.array(assignement)[:, 0], np.array(assignement)[:, 1], c=colour[i])
        # Also plot centroid
        plt.plot(centroid[0], centroid[1], markersize=35, marker="x", color='k', mew=10)
        plt.plot(centroid[0], centroid[1], markersize=30, marker="x", color='m', mew=5)
    plt.show()


def elbow_algorithm(train_set, users_idx, max_k=15):
    # Initialize the error array
    sse = [0 for h in xrange(3, max_k + 1)]
    centroids_k = []
    assignments_k = []
    # Iterate over the possible number of clusters
    for i in range(3, max_k + 1):
        # Run the clustering algorithm
        (centroids, assignments) = k_means_cluster(train_set, i)
        centroids_k.append(centroids)
        assignments_k.append(assignments)
        # Construct an array of clusters and there data points
        cluster_assignments, cluster_users = create_cluster_assignments(train_set, users_idx, assignments, i)
        # For each cluster compute the mean squared error
        for idx, cluster in enumerate(cluster_assignments):
            mean = centroids[idx]
            for data_point in cluster:
                sse[i - 3] += ((data_point - mean) ** 2).mean()
    best_sse, best_k = min((val, idx) for (idx, val) in enumerate(sse))
    return best_k + 3, centroids_k[best_k], assignments_k[best_k]


def save_rec_data(db, train_set, users_idx, assignments, centroids, best_k):
    # Cleanup rec cb
    db.centroids.delete_many({})
    db.assignments.delete_many({})
    # Group users and assignments by clusters
    cluster_assignments, cluster_users = create_cluster_assignments(train_set, users_idx, assignments, best_k)
    # Insert new rec data
    for idx, centroid in enumerate(centroids):
        centroid_doc = db.centroids.insert_one({
            "data": centroid.tolist()
        })
        assignment_docs = db.assignments.insert_many([{
                                                          "userId": user,
                                                          "centroidId": centroid_doc.inserted_id
                                                      } for user in cluster_users[idx]])


def load_rec_data(db):
    centroids = []
    assignments = []
    centroids_data = db.centroids.find({})
    assignments_data = db.assignments.find({})
    for centroid_data in centroids_data:
        centroids.append(np.array(centroid_data['data']))

    for assignment_data in assignments_data:
        assignments.append({'userId': assignment_data['userId'], 'centroidId': assignment_data['centroidId']})


train_db = _connect_mongo("localhost", "trip_opt")
users_idx, train_set = load_train_set(train_db)
best_k, centroids, assignments = elbow_algorithm(train_set, users_idx, 4)
rec_db = _connect_mongo("localhost", "trip_opt_rec")
save_rec_data(rec_db, train_set, users_idx, assignments, centroids, best_k)


"""

## Load the rec data
rec_db = _connect_mongo("localhost", "trip_opt_rec")
(centroids, assignmetns) = load_rec_data(rec_db)

"""