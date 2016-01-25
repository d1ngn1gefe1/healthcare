import numpy as np
# from threading import Thread as worker
from multiprocessing import Process as worker 

num_threads = 20
out_dir = './features/'

def map_features(X, theta_u, theta_v, images, thread_index):
    normalize = 100
    m = X.shape[0]
    num_features = theta_u.shape[0]
    features = np.zeros((m,num_features))
    for i in range(0,m):
        index = X[i][3]
        width = images[index].shape[0]
        height = images[index].shape[1]
        if (i % 100 == 0):
          print('width: ' + str(width) + ' height: ' + str(height))
        for j in range(0,num_features):
            left = X[i][:1] + theta_u[j] / (X[i][2]/normalize)
            right =  X[i][:1] + theta_v[j] / (X[i][2]/normalize)

            left_new = np.minimum(left, [width-1, height-1])
            right_new = np.minimum(right, [width-1, height-1])

            features[i][j] = float(images[index][left_new[0], left_new[1]]) - float(images[index][right_new[0], right_new[1]])
            if (i % 100 == 0):
              print(' left: ' + str(left_new) + ' right: ' + str(right_new))
              print('image[i][left]: ' + str(images[index][left_new[0], left_new[1]]) + ' image[i][right]: ' + str(images[index][right_new[0], right_new[1]]))
    np.save(out_dir + str(thread_index) + '.npy', features)
    return features

def map_features_thread(X, theta_u, theta_v, images):
  X_split = np.array_split(X, num_threads)
  processes = []

  for i in range(num_threads):
    processes.append(
      worker(
        target = map_features,
        name="Thread #%d" % i,
        args=(X_split[i], theta_u, theta_v, images, i)
      )  
    )
  [t.start() for t in processes]
  [t.join() for t in processes]
