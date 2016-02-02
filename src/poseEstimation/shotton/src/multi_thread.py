import numpy as np
import os
import random
import time
import threading
import collections
from Queue import Queue
from itertools import chain

"""
To use this file, call the randomBatch() function. Example:
    A1, A2 = randomBatch(split='train', batch_size=100, num_threads=8)

Returns:
    A1: python array of numpy arrays
        shape: [batch_size, numpy(??, 4096)]

    A2: python array X of python arrays Y where each element of Y is a numpy array
        shape: [batch_size, [4, numpy(??, 4096)]]

    where ?? is variable length due to differing video lengths
"""

RGB_FEATURES_DIR = '/mnt2/data/youtube/rgb_features'

# Loads RGB features for a particular subset/sequence from a video
def getPartitionRgbFeatures(path, partition, num_partitions):
    npz = np.load(path)
    parts = np.array_split(npz['rgb_features'], num_partitions)
    return parts[partition]

def randomBatch(split, batch_size, num_threads):
    ls = os.listdir(os.path.join(RGB_FEATURES_DIR, split))
    q1 = Queue()
    q2 = Queue()
    wargs = {
        'rgb_features_dir': RGB_FEATURES_DIR,
        'split': split,
        # Directory listing of all videos
        'ls': ls,
        # Number of parts to split each video
        'num_partitions': 3,
        # Number of Agent 2 video sequences (could inculde 2 sequences from same video)
        'num_a2_videos': 4,
        # Batch size
        'batch_size': batch_size,
        # Resulting queue
        'a1': q1,
        'a2': q2,
    }

    parts = np.array_split(np.arange(min(batch_size, len(ls))), num_threads)

    # Start the worker threads
    threads = []
    print "[INFO] Starting threads..."
    for tid in xrange(num_threads):
        num_to_generate = len(parts[tid])
        thread = threading.Thread(target=randomBatchWorker, args=[tid, num_to_generate, wargs])
        thread.start()
        threads.append(thread)

    print "[INFO] Waiting for threads to finish..."
    for thread in threads:
        thread.join()

    def queue_to_list(q):
        l = []
        while q.qsize() > 0:
            l.append(q.get())
        return l


    A1 = list(chain.from_iterable(queue_to_list(q1)))
    A2 = list(chain.from_iterable(queue_to_list(q2)))

    return A1, A2


# Creates a batch of (A1 sequence, A2 sequences) pairs
def randomBatchWorker(tid, num_to_generate, wargs):
    batch = []
    num_examples = len(wargs['ls'])

    A1 = []
    A2 = []
    start_time = time.time()
    # Generate the examples
    for i in xrange(num_to_generate):
        # Select a random A1 video and partition
        # We use minus 2 because A2 needs to have the next consecutive partition
        A1_idx = random.randint(0, num_examples-1)
        A1_path = os.path.join(wargs['rgb_features_dir'], wargs['split'], wargs['ls'][A1_idx])
        A1_partition = random.randint(0, wargs['num_partitions']-2)
        A1.append(getPartitionRgbFeatures(A1_path, A1_partition, wargs['num_partitions']))

        # Add the correct (next) sequence to A2
        A2_data = []
        A2_data.append(getPartitionRgbFeatures(A1_path, A1_partition + 1, wargs['num_partitions']))

        # Generate the remaining random A2 sequneces
        for j in xrange(wargs['num_a2_videos']-1):
            while True:
                A2_idx = random.randint(0, num_examples-1)
                A2_partition = random.randint(0, wargs['num_partitions']-1)
                if A2_idx != A1_idx and A2_partition != A1_partition:
                    break

            a2_path = os.path.join(wargs['rgb_features_dir'], wargs['split'], wargs['ls'][A2_idx])
            A2_data.append(getPartitionRgbFeatures(a2_path, A2_partition, wargs['num_partitions']))

        A2.append(A2_data)

        elapsed_time = time.time() - start_time
        print '[INFO] Thread %i   Finished %i of %i   Elapsed: %f seconds' % \
            (tid, i+1, num_to_generate, elapsed_time)

    wargs['a1'].put(A1)
    wargs['a2'].put(A2)

if __name__ == '__main__':
    random.seed(None)  # Seed 'None" will use system time
    A1, A2 = randomBatch(split='train', batch_size=1400, num_threads=1)