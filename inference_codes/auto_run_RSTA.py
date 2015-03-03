


# Wrapper function to run developed Random Spanning Tree Approximation algorithm parallelly on interactive cluster, for the purpose of multiple parameters and datasets.
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
    self.penalty = 0 # penalty parameter which prevents computing node with low computational resources getting jobs from job queue
    pass # def
  def run(self):
    all_done = 0
    while not all_done:
      try:
        time.sleep(random.randint(5000,6000) / 1000.0)  # sleep random time
        time.sleep(self.penalty*120)
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


def checkfile(filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method):
  file_exist = 0
  file_exist += os.path.isfile("../outputs/%s_%s_%s_f%s_l%s_k%s_c%s_s%s_n%s_RSTAs.log" % (filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))        # temporatory folder
  file_exist += os.path.isfile("../outputs/phase1/%s_%s_%s_f%s_l%s_k%s_c%s_s%s_n%s_RSTAs.log" % (filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method)) # 100
  file_exist += os.path.isfile("../outputs/phase2/%s_%s_%s_f%s_l%s_k%s_c%s_s%s_n%s_RSTAs.log" % (filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method)) # 1
  file_exist += os.path.isfile("../outputs/phase3/%s_%s_%s_f%s_l%s_k%s_c%s_s%s_n%s_RSTAs.log" % (filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method)) # 0.1
  file_exist += os.path.isfile("../outputs/phase4/%s_%s_%s_f%s_l%s_k%s_c%s_s%s_n%s_RSTAs.log" % (filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method)) # 10
  file_exist += os.path.isfile("../outputs/phase5/%s_%s_%s_f%s_l%s_k%s_c%s_s%s_n%s_RSTAs.log" % (filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method)) # 0.01
  file_exist += os.path.isfile("../outputs/phase6/%s_%s_%s_f%s_l%s_k%s_c%s_s%s_n%s_RSTAs.log" % (filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method)) # 50
  file_exist += os.path.isfile("../outputs/phase7/%s_%s_%s_f%s_l%s_k%s_c%s_s%s_n%s_RSTAs.log" % (filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method)) # 0.5
  file_exist += os.path.isfile("../outputs/phase8/%s_%s_%s_f%s_l%s_k%s_c%s_s%s_n%s_RSTAs.log" % (filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method)) # 20
  file_exist += os.path.isfile("../outputs/phase9/%s_%s_%s_f%s_l%s_k%s_c%s_s%s_n%s_RSTAs.log" % (filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method)) # 0.05
  file_exist += os.path.isfile("../outputs/phase10/%s_%s_%s_f%s_l%s_k%s_c%s_s%s_n%s_RSTAs.log" % (filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method)) # 5
  if file_exist > 0:
    return 1
  else:
    return 0
  pass # def


def singleRSTA(node, job):
  (n,filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method) = job
  try:
    if checkfile(filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method):
      logging.info('\t--< (node)%s,(f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s,(c)%s,(s)%s,(n)%s' %( node,filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))
      fail_penalty = 0
    else:
      logging.info('\t--> (node)%s,(f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s,(c)%s,(s)%s,(n)%s' %( node,filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))
      os.system(""" ssh -o StrictHostKeyChecking=no %s 'cd /cs/taatto/group/urenzyme/workspace/colt2014/experiments/L2RTA/inference_codes/; rm -rf /var/tmp/.matlab; export OMP_NUM_THREADS=32; nohup matlab -nodisplay -r "run_RSTA '%s' '%s' '%s' '0' '%s' '%s' '%s' '%s' '%s' '%s'" > /var/tmp/tmp_%s_%s_%s_f%s_l%s_k%s_c%s_s%s_n%s_RSTAs' """ % (node,filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method,filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method) )
      logging.info('\t--| (node)%s,(f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s,(c)%s,(s)%s,(n)%s' %( node,filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))
      fail_penalty = -1
  except Exception as excpt_msg:
    print excpt_msg
    job_queue.put((job))
    logging.info('\t--= (node)%s,(f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s,(c)%s,(s)%s,(n)%s' %( node,filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))
    fail_penalty = 1
  if not os.path.isfile("../outputs/%s_%s_%s_f%s_l%s_k%s_c%s_s%s_n%s_RSTAs.log" % (filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method)):
    job_queue.put((job))
    logging.info('\t--x (node)%s,(f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s,(c)%s,(s)%s,(n)%s' %( node,filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))
    fail_penalty = 1
  time.sleep(10)
  return fail_penalty
  pass # def


def run():
  jobs=[]
  n=0
  is_main_run_factor=5
  filenames=['cancer','ArD20','ArD30','toy10','toy50','emotions','yeast','medical','scene','enron','cal500','fp']
  filenames=['scene']
  n=0
  # generate jobs
  logging.info('\t\tGenerating job queue.')
  #for slack_c in ['100','1','0.1','10','0.01','50','0.5','20','0.05','5']:
  for slack_c in ['1']:
    for kth_fold in ['1','2','3','4','5']:
      for filename in filenames:
        graph_type = 'tree'
        for kappa in ['2','8','16','20']:
          for l_norm in ['2']:
            for t in range(0,41,10):
              if t==0:
                t=1
              para_t="%d" % (t)
              for loss_scaling_factor in range(0,11,2):
                if loss_scaling_factor ==0:
                  loss_scaling_factor = 1
                for newton_method in ['1']:
                  if checkfile(filename,graph_type,para_t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method):
                    continue
                  else:
                    n=n+1
                    job_queue.put((n,filename,graph_type,para_t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))
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
  # running jobs
  job_size = job_queue.qsize()
  logging.info( "\t\tProcessing %d jobs" % (job_size))
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
  run()
  pass


