




# Wrapper function to run developed Random Spanning Tree Approximation algorithm parallelly on interactive cluster, for the purpose of multiple parameters and datasets.
# The script uses Python thread and queue package.
# Implement worker class and queuing system.
# The framework looks at each parameter combination as a job and pools all job_queue in a queue.
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


job_queue = Queue.PriorityQueue()

# Worker class
# job is a tuple of parameters
class Worker(Thread):
  def __init__(self, job_queue, node):
    Thread.__init__(self)
    self.job_queue  = job_queue
    self.node = node
    self.penalty = 0 # penalty parameter which prevents computing node with low computational resources getting job_queue from job queue
    pass # def
  def run(self):
    all_done = 0
    while not all_done:
      try:
        time.sleep(random.randint(5000,6000) / 1000.0)  # get some rest :-)
        time.sleep(self.penalty*120) # bad worker will rest more
        job = self.job_queue.get(0)
        add_penalty = singleRSTA(self.node, job)
        self.penalty += add_penalty
        if self.penalty < 0:
          self.penalty = 0
      except Queue.Empty:
        all_done = 1
      pass # while
    pass # def
  pass # class


global_rundir = ''

# function to check if the result file already exist in the destination folder
def checkfile(filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method):
  file_exist = 0
  file_exist += os.path.isfile("%s/%s_%s_%s_f%s_l%s_k%s_c%s_s%s_n%s_RSTAs.log" % (global_rundir,filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))
  file_exist += os.path.isfile("%s/%s/c%s/%s_%s_%s_f%s_l%s_k%s_c%s_s%s_n%s_RSTAs.log" % (global_rundir,filename,slack_c,filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))
  if file_exist > 0:
    return 1
  else:
    return 0
  pass # checkfile


def singleRSTA(node, job):
  (priority, job_detail) = job
  (filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method) = job_detail
  try:
    if checkfile(filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method):
      logging.info('\t--< (priority) %d (node)%s,(f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s,(c)%s,(s)%s,(n)%s' % ( priority, node,filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))
      fail_penalty = 0
    else:
      logging.info('\t--> (priority) %d (node)%s,(f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s,(c)%s,(s)%s,(n)%s' %( priority, node,filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))
      os.system(""" ssh -o StrictHostKeyChecking=no %s 'cd /cs/taatto/group/urenzyme/workspace/colt2014/experiments/L2RTA/inference_codes/; rm -rf /var/tmp/.matlab; export OMP_NUM_THREADS=32; nohup matlab -nodisplay -nosplash -r "run_RSTA '%s' '%s' '%s' '0' '%s' '%s' '%s' '%s' '%s' '%s' '/var/tmp' '%s'" > /var/tmp/tmp_%s_%s_%s_f%s_l%s_k%s_c%s_s%s_n%s_RSTAs' """ % (node,filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method,global_rundir,filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method) )
      logging.info('\t--| (priority) %d (node)%s,(f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s,(c)%s,(s)%s,(n)%s' %( priority, node,filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))
      fail_penalty = -1
  except Exception as excpt_msg:
    print excpt_msg
    job_queue.put((priority, job_detail))
    logging.info('\t--= (priority) %d (node)%s,(f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s,(c)%s,(s)%s,(n)%s' %( priority, node,filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))
    fail_penalty = 1
  if not os.path.isfile("%s/%s_%s_%s_f%s_l%s_k%s_c%s_s%s_n%s_RSTAs.log" % (global_rundir,filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method)):
    job_queue.put((job))
    logging.info('\t--x (priority) %d (node)%s,(f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s,(c)%s,(s)%s,(n)%s' %( priority, node,filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))
    fail_penalty = 1
  time.sleep(10)
  return fail_penalty
  pass # def


def run():
  is_main_run_factor=5
  #filenames=['toy10','toy50','emotions','medical','enron','yeast','scene','cal500','fp','cancer']
  #filenames=['cancer']
  filenames=['toy10','toy50','emotions','yeast','scene','fp']
  n=0
  # generate job_queue
  logging.info('\t\tGenerating priority queue.')
  for newton_method in ['1','0']:
    for filename in filenames:
      for slack_c in ['1', '10', '0.1']:
        for t in [1, 5, 10, 20, 30]:
          para_t="%d" % (t)
          graph_type = 'tree'
          for kappa in ['1','4','6','8','10','12','14','16']:
            for l_norm in ['2']:
              #for kth_fold in ['1','2','3','4','5']:
              for kth_fold in ['1']:
                for loss_scaling_factor in ['0.1','1']:
                  if checkfile(filename,graph_type,para_t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method):
                    continue
                  else:
                    n=n+1
                    job_queue.put( (n, (filename,graph_type,para_t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method)) )
                  pass # for newton_method
                pass # for loss_scaling_factor
              pass # for slack_c
            pass # for |T|
          pass # for l
        pass # for kappa
      pass # for datasets
    pass # for k fole
  # get computing nodes
  cluster = get_free_nodes()[0] # if you have access to some interactive computer cluster, get the list of hostnames of the cluster
  #cluster = ['melkinkari'] # if you don't have access to any computer cluster, just use your machine as the only computing node
  # running job_queue
  job_size = job_queue.qsize()
  logging.info( "\t\tProcessing %d job_queue" % (job_size))
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


# It's actually not necessary to have '__name__' space, but whatever ...
if __name__ == "__main__":
  global_rundir = sys.argv[1]
  run()
  pass


