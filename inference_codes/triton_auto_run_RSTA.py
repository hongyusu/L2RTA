




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
import multiprocessing
import time
import logging
import random
logging.basicConfig(format='%(asctime)s %(filename)s %(funcName)s %(levelname)s:%(message)s', level=logging.INFO)




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


def singleRSTA(job, tmpdir):
  (n,filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method) = job
  try:
    if checkfile(filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method):
      logging.info('\t--< (f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s,(c)%s,(s)%s,(n)%s' %( filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))
    else:
      logging.info('\t--> (f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s,(c)%s,(s)%s,(n)%s' %( filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))
      os.system(""" export OMP_NUM_THREADS=32; nohup matlab -nodisplay -nosplash -r "run_RSTA '%s' '%s' '%s' '0' '%s' '%s' '%s' '%s' '%s' '%s' '%s'" > %s/tmp_%s_%s_%s_f%s_l%s_k%s_c%s_s%s_n%s_RSTAs """ % (filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method,tmpdir,tmpdir,filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method) )
      logging.info('\t--| (f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s,(c)%s,(s)%s,(n)%s' %( filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))
  except Exception as excpt_msg:
    print excpt_msg
    logging.info('\t--= (f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s,(c)%s,(s)%s,(n)%s' %( filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))
    fail_penalty = 1
  if not os.path.isfile("../outputs/%s_%s_%s_f%s_l%s_k%s_c%s_s%s_n%s_RSTAs.log" % (filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method)):
    logging.info('\t--x (f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s,(c)%s,(s)%s,(n)%s' %( filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))
    #singleRSTA(job)
  pass # def


def run(job_id, tmpdir):
  jobs=[]
  is_main_run_factor=5
  filenames=['toy10','ArD20','ArD30','toy50','emotions','medical','enron','cal500','fp','cancer','yeast','scene']
  #filenames=['scene','yeast','cancer']
  n=0
  # generate jobs
  #logging.info('\t\tGenerating job queue.')
  for filename in filenames:
    for slack_c in ['100','1','0.1','10','0.01','50','0.5','20','0.05','5']:
    #for slack_c in ['1']:
      for t in range(0,41,10):
        if t==0:
          t=1
        t=10
        para_t="%d" % (t)
        graph_type = 'tree'
        for kappa in ['2','8','16','20']:
          for l_norm in ['2']:
            for kth_fold in ['1','2','3','4','5']:
              for loss_scaling_factor in range(0,11,2):
                if loss_scaling_factor ==0:
                  loss_scaling_factor = 1
                for newton_method in ['0']:
                  if checkfile(filename,graph_type,para_t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method):
                    continue
                  else:
                    n=n+1
                    jobs.append((n,filename,graph_type,para_t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))
                  pass # for newton_method
                pass # for loss_scaling_factor
              pass # for slack_c
            pass # for |T|
          pass # for l
        pass # for kappa
      pass # for datasets
    pass # for k fole
  # running jobs
  if job_id > len(jobs):
    return
  #logging.info( "\t\tProcessing %d jobs" % (len(jobs)))
  singleRSTA(jobs[job_id-1], tmpdir)
  pass # def


# It's actually not necessary to have '__name__' space, but whatever ...
if __name__ == "__main__":
  run(eval(sys.argv[1]), sys.argv[2])
  pass


