'''
K-Means clustering using tensorflow
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import json
import random


# import data
with open('k_means.json') as f:
  raw_x = json.loads(f.read())

# Visualize Data
plt.plot([i[0] for i in raw_x], [i[1] for i in raw_x], 'bo')
plt.show()

# Convert input data to Tensor
x = tf.Variable(raw_x)

# Set Constants
K = 3                   # Number of clusters
N = int(x.shape[1])     # Number of features (in our case = 2)
M = int(x.shape[0])     # Number of training examples
MAX_ITER = 500          # Number of iterations to run K-Means

#  Default Cluster Assignment. Initialize every point to be in the same cluster.
cluster_assignment = tf.Variable(tf.zeros([M], dtype=tf.int64))

# initialize cluster centroids to random K points in our data set. 
centroids = tf.Variable(random.sample(raw_x,K))

# Reshape points and centroid tensors to dimension [M,N,K] 
rep_centroids = tf.reshape(tf.tile(centroids, [M, 1]), [M, N, K])
rep_points = tf.reshape(tf.tile(x, [1, K]), [M, N, K])

# euclidean distances between K centroid and those M points each. 
# Get a Tensor (dimension = [M,K]) 
distances = tf.reduce_sum(tf.square(rep_points - rep_centroids), axis=1)

# New cluster assignment based on the minimum distance. 
new_clusters = tf.argmin(distances, axis=1)

# Has anything changed since last iteration?
cluster_changed = tf.reduce_any(tf.not_equal(new_clusters, cluster_assignment))

# Get the mean of data given buckets
def bucket_mean(data, bucket_ids, num_buckets):
    total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
    count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
    return total / count

# New Centroids
new_centroids = bucket_mean(x, new_clusters, K)

# If changed, simultaneous update
with tf.control_dependencies([cluster_changed]):
    do_updates = tf.group(
        centroids.assign(new_centroids),
        cluster_assignment.assign(new_clusters)
        )

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Optimize Clustering
changed=True
i = 0
while changed and i < MAX_ITER:
    i += 1
    [changed, _] = sess.run([cluster_changed, do_updates])
    print([i])

[x, cluster_assignment, centroids] = sess.run([x, cluster_assignment, centroids])

# Visualize
series = {}
for i in range(len(cluster_assignment)):
    if cluster_assignment[i] not in series.keys():
        series[cluster_assignment[i]] = []
    series[cluster_assignment[i]].append(x[i])

for k in series:
    plt.plot([i[0] for i in series[k]], [i[1] for i in series[k]], 'o')
plt.plot([i[0] for i in centroids], [i[1] for i in centroids], 'bx')

plt.show()


'''
Reference taken from : https://gist.github.com/dave-andersen/265e68a5e879b5540ebc
'''




