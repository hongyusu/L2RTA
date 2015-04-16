




# The wrapper script will enable running with different parameters parallelly .


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


# function to run RSTA algorithm on one set of parameters
def singleRSTA(job, tmpdir, rundir):
  (n,filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method) = job
  try:
    if checkfile(filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method):
      logging.info('\t--< (f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s,(c)%s,(s)%s,(n)%s' %( filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))
    else:
      logging.info('\t--> (f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s,(c)%s,(s)%s,(n)%s' %( filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))
      os.system(""" export OMP_NUM_THREADS=32; matlab -nodisplay -nosplash -nojvm -r "run_RSTA '%s' '%s' '%s' '0' '%s' '%s' '%s' '%s' '%s' '%s' '%s' '%s'" > %s/tmp_%s_%s_%s_f%s_l%s_k%s_c%s_s%s_n%s_RSTAs """ % (filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method,tmpdir,rundir,tmpdir,filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method) )
      logging.info('\t--| (f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s,(c)%s,(s)%s,(n)%s' %( filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))
  except Exception as excpt_msg:
    print excpt_msg
    logging.info('\t--= (f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s,(c)%s,(s)%s,(n)%s' %( filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))
    fail_penalty = 1
  if not os.path.isfile("../outputs/compare_run/%s_%s_%s_f%s_l%s_k%s_c%s_s%s_n%s_RSTAs.log" % (filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method)):
    logging.info('\t--x (f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(k)%s,(c)%s,(s)%s,(n)%s' %( filename,graph_type,t,kth_fold,l_norm,kappa,slack_c,loss_scaling_factor,newton_method))
  pass # singleRSTA


def run(job_id, tmpdir, rundir):
  jobs=[]
  is_main_run_factor=5
  filenames=['toy10','toy50','emotions','medical','enron','yeast','scene','cal500','fp','cancer']
  filenames=['toy10','toy50','emotions','yeast','scene']
  n=0
  # generate jobs
  for filename in filenames:
    #for slack_c in ['1','100','0.1','10','0.01','50','0.5','20','0.05','5']:
    for slack_c in ['1','100','0.1','10','0.01']:
      for t in range(0,21,5):
        if t==0:
          t=1
        para_t="%d" % (t)
        graph_type = 'tree'
        #for kappa in ['2','8','16','20']:
        for kappa in ['1','2','4','8','16']:
          for l_norm in ['2']:
            for kth_fold in ['1','2','3','4','5']:
              for loss_scaling_factor in ['0.5','0.1','1','5','10']:
                for newton_method in ['1','0']:
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
  # for job in jobs:
  # start jobs
  if not job_id > len(jobs):
    singleRSTA(jobs[job_id-1], tmpdir, rundir)
    time.sleep(1)
  pass # def


# It's actually not necessary to have '__name__' space, but whatever ...
if __name__ == "__main__":
  # the main function will take in parameters: job_id, tmpdir
  global_rundir = sys.argv[3]
  run(eval(sys.argv[1]), sys.argv[2], sys.argv[3])
  pass


