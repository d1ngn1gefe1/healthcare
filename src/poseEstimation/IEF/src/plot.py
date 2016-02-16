import numpy as np
import matplotlib.pyplot as plt

def parse_log(log_path):
  log = np.genfromtxt(log_path, delimiter=',', dtype='str')
  loss = log[:,-1].astype(np.float)
  itr = np.array(range(45)) * 10
  return (loss, itr)

def main():
  log_path = 'loss.txt'
  loss, itr = parse_log(log_path)
  plt.scatter(itr, loss)
  plt.plot(itr, loss)
  plt.title('base lr: 1e-7')
  plt.xlabel('Num. iterations (150 itr per IEF)')
  plt.ylabel('Training loss')
  plt.savefig('train_loss.jpg')

if __name__ == "__main__":
  main()


