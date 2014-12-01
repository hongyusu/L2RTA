


# Wrapper function to profile developed Random Spanning Tree Approximation algorithm parallelly on interactive cluster,for the purpose of multiple parameter and datasets.
# The script uses Python thread and queue package.
# Implement worker class and queuing system.
# The framework looks at each parameter combination as a job and pools all jobs in a queue.
# It generates a group of workers (computing nodes). 
# Each worker will always take and process the first job from the queue.
# In case that job is not completed by the worker, it will be push back to the queue, and will be processed later on.


import math
import re
import Queue
from threading import ThreadError
from threading import Thread
import os
import sys
import commands
sys.path.append('/cs/taatto/group/urenzyme/workspace/netscripts/')
from get_free_nodes import get_free_nodes
import multiprocessing
import time
import logging
import random
logging.basicConfig(format='%(asctime)s %(filename)s %(funcName)s %(levelname)s:%(message)s', level=logging.INFO)


job_queue = Queue.Queue()


# Worker class
# job is a tuple of parameters
class Worker(Thread):
  def __init__(self, job_queue, node):
    Thread.__init__(self)
    self.job_queue  = job_queue
    self.node = node
    pass # def
  def run(self):
    all_done = 0
    while not all_done:
      try:
        job = self.job_queue.get(0)
        time.sleep(random.randint(5000,6000) / 1000.0)  # sleep random time
        singleRSTA(self.node, job)
      except Queue.Empty:
        all_done = 1
      pass # while
    pass # def
  pass # class


def singleRSTA(node, job):
  (n,filename,graph_type,t,kth_fold,l_norm,kappa) = job
  try:
    if os.path.isfile("../outputs/%s_%s_%s_f%s_l%s_k%s_pfRSTAs.log" % (filename,graph_type,t,kth_fold,l_norm,kappa)):
      logging.info('\t--< (node)%s,(f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s' %( node,filename,graph_type,t,kth_fold,l_norm,kappa))
    else:
      logging.info('\t--> (node)%s,(f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s' %( node,filename,graph_type,t,kth_fold,l_norm,kappa))
      os.system(""" ssh -o StrictHostKeyChecking=no %s 'cd /cs/taatto/group/urenzyme/workspace/colt2014/experiments/random_spanning_tree_approximation/inference_codes/; rm -rf /var/tmp/.matlab; export OMP_NUM_THREADS=32; nohup matlab -nodisplay -r "profile_RSTA '%s' '%s' '%s' '0' '%s' '%s' '%s' " > /var/tmp/tmp_%s_%s_%s_f%s_l%s_k%s_pfRSTAs' """ % (node,filename,graph_type,t,kth_fold,l_norm,kappa,filename,graph_type,t,kth_fold,l_norm,kappa) )
      logging.info('\t--| (node)%s,(f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s' %( node,filename,graph_type,t,kth_fold,l_norm,kappa))
  except:
    job_queue.put((job))
    logging.info('\t--X (node)%s,(f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s' %( node,filename,graph_type,t,kth_fold,l_norm,kappa))
  if not os.path.isfile("../outputs/%s_%s_%s_f%s_l%s_k%s_pfRSTAs.log" % (filename,graph_type,t,kth_fold,l_norm,kappa)):
    job_queue.put((job))
    logging.info('\t--x (node)%s,(f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s' %( node,filename,graph_type,t,kth_fold,l_norm,kappa))
  time.sleep(10)
  pass # def


def run():
  cluster = get_free_nodes()[0]
  #cluster = ['dave']
  jobs=[]
  n=0
  is_main_run_factor=5

  filenames=['cancer','ArD20','ArD30','toy10','toy50','emotions','yeast','medical','scene','enron','cal500','fp']
  n=0
  # generate jobs
  for kth_fold in ['1']:#,'2','3','4','5']:
    for filename in filenames:
      graph_type = 'tree'
      for kappa in [180,280]:#['2','4','8','16','20','32','40','50','60']:
        for l_norm in ['2']:
          #for t in [5]:#range(0,41,10):
          for t in [1,5] + range(10,41,10):
            if t==0:
              t=1
            para_t="%d" % (t)
            try:
              with open("../outputs/%s_%s_%s_f%s_l%s_k%s_pfRSTAs.log" % (filename,graph_type,para_t,kth_fold,l_norm,kappa)): pass
              continue
            except:
              n=n+1
              job_queue.put((n,filename,graph_type,para_t,kth_fold,l_norm,kappa))
            pass # for |T|
          pass # for l
        pass # for kappa
      pass # for datasets
    pass # for k fole
  # running jobs
  job_size = job_queue.qsize()
  logging.info( "\t\tprocessing %d jobs" % (job_size))
  threads = []
  for i in range(len(cluster)):
    if job_queue.empty():
      break
    t = Worker(job_queue, cluster[i])
    time.sleep(is_main_run_factor)
    try:
      t.start()
      threads.append(t)
    except ThreadError:
      logging.warning("\t\tError: thread error caught!")
    pass
  for t in threads:
    t.join()
    pass
  pass # def



if __name__ == "__main__":
  run()
  pass


