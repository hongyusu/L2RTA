
import os
import sys
import commands
sys.path.append('/home/group/urenzyme/workspace/netscripts/')
from get_free_nodes import get_free_nodes
import multiprocessing
import time
import logging
logging.basicConfig(format='%(asctime)s %(filename)s %(funcName)s %(levelname)s:%(message)s', level=logging.INFO)



def singleRSTA(filename,graph_type,t,node,kth_fold,l_norm,in_i,in_c):
  try:
    with open("../parameters/%s_%s_%s_f%s_l%s_i%s_RSTAp.log" % (filename,graph_type,t,kth_fold,l_norm,in_i)): pass
    logging.info('\t--< (node)%s,(f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(i)%s' %( node,filename,graph_type,t,kth_fold,l_norm,in_i))
  except:
    logging.info('\t--> (node)%s,(f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(i)%s' %( node,filename,graph_type,t,kth_fold,l_norm,in_i))
    os.system(""" ssh -o StrictHostKeyChecking=no %s 'cd /home/group/urenzyme/workspace/colt2014/experiments/inference_codes/; rm -rf /var/tmp/.matlab; export OMP_NUM_THREADS=32; nohup matlab -nodisplay -r "parameter_selection_RSTA '%s' '%s' '%s' '0' '%s' '%s' '%s' '%s' " > /var/tmp/tmp_%s_%s_%s_f%s_l%s_i%s' """ % (node,filename,graph_type,t,kth_fold,l_norm,in_i,in_c,filename,graph_type,t,kth_fold,l_norm,in_i) )
    logging.info('\t--| (node)%s,(f)%s,(type)%s,(t)%s,(f)%s,(l)%s,(i)%s' %( node,filename,graph_type,t,kth_fold,l_norm,in_i))
    time.sleep(5)
  pass

def parameter_selection():
  cluster = get_free_nodes()[0]
  #cluster = ['dave']
  jobs=[]
  n=0
  is_main_run=0

  #filenames=['emotions','yeast','scene','enron','medical','toy10','toy50']#,'cal500','fp','cancer'] 
  filenames=['toy10','toy50','emotions','yeast','scene','enron','medical']
  n=0
  for filename in filenames:
    for graph_type in ['tree']:
      for l_norm in ['2']:
        in_i=0
        for in_c in ['100','75','50','20','10','5','1','0.5','0.25','0.1','0.01']:
          in_i+=1
          for kth_fold in ['1','2','3','4','5']:
            node=cluster[n%len(cluster)]
            n+=1
            p=multiprocessing.Process(target=singleRSTA, args=(filename,graph_type,'1',node,kth_fold,l_norm,"%d" % in_i,in_c,))
            jobs.append(p)
            p.start()
            time.sleep(2) # fold
            pass
        time.sleep(2) # c
        pass
      time.sleep(2*is_main_run) # lnorm
      pass
    time.sleep(60*is_main_run) # tree
    for job in jobs:
      job.join()
      pass
    pass




parameter_selection()


