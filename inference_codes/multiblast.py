
"""
	multiple threads BLAST
                1. single BLAST thread can use up to x cores, as specified in single_blast function
                2. threads are started with time interval 5 or 6 second randomly to avoid heavy load (reading through same pipe) on cluster
                3. all hits for a given sequenes which satisfy given evalue threshold are extracted
	by Hongyu SU

        Corrections:
                28.03.2012      correct evalue default value in parser command line section
                30.08.2012      correct file path due to server update
                30.08.2012      correct E-value checkup logic to allow E-value above 1 (useful to do db/db search)
                31.08.2012      add word size parameter
                31.08.2012      correct global parameter evalue's missing 'e'
                08.01.2013      correct blast search returen 500 aligned sequence by default, too bad for search with long contigs 
                08.01.2013      add log functon with logging package
"""


#!/usr/bin/env python
#-*- coding: iso-8859-15 -*-


# use packages
import sys
import os
import math
import re
import Queue
import time
import optparse
import random
import commands
import logging
from threading import ThreadError
from threading import Thread
sys.path.append('/home/group/urenzyme/workspace/netscripts/')
from get_free_nodes import get_free_nodes


# parse command line arguments
parser = optparse.OptionParser(version = "%prog 1.1 on 30.11.2011 by Hongyu SU", description = "python script for multiple thread blast. Example usage: python myblast.py -i /fs/group/urenzyme/workspace/metageno/samples/data/MAAOC.fasta -l 5 -o /fs/group/urenzyme/workspace/metageno/blast/results/MAAOC.silva-all -b /fs/b/su/softwares/blast/bin/blastn -e 0.05 -a 0 -d /fs/group/urenzyme/workspace/metageno/databases/silva/all/silva-all")
parser.add_option("-i", "--ifname", dest="ifname", default="", help="sample file name")
parser.add_option("-o", "--ofname", dest="ofname", default="", help="result file name")
parser.add_option("-l", "--load", dest="load", default="5", help="average load(threads per node in cluster)")
parser.add_option("-b", "--blast", dest="blast", default="", help="blastx or blastn script path")
parser.add_option("-e", "--evalue", dest="evalue", default="0.05", help="evalue of extracting blast search results")
parser.add_option("-a", "--alignment", dest="alignment", default="0", help="whether store blast alignment or not")
parser.add_option("-d", "--database", dest="db", default="", help="path of database folder")
parser.add_option("-w", "--wordsize", dest="wordsize", default="2", help="word size of BLAST search")
(options, arguments) = parser.parse_args()
if options.ifname == "":
        parser.error("Error: no input sample file!")
        exit(0)
if options.ofname == "":
        parser.error("Error: no output result file!")
        exit(0)
if eval(options.load) < 1:
        parser.error("Error: negative load!")
        exit(0)
if options.blast == "":
        parser.error("Error: no blastn or blastx script path!")
        exit(0)
if eval(options.evalue)<0:
        parser.error("Error: evalue makes no sense!")
        exit(0)
if eval(options.alignment) not in [0,1]:
        parser.error("Error: wrong alignment parameter!")
        exit(0)
if options.db == "":
        parser.error("Error: no database path")
        exit(0)
if eval(options.wordsize) <2:
        parser.error("Error: word size is too small")
        exit(0)

# global parameters
logging.basicConfig(format='%(asctime)s %(filename)s %(funcName)s %(levelname)s:%(message)s', level=logging.INFO)
scriptfolder = "/home/group/urenzyme/workspace/metageno/blast/"
job_queue = Queue.Queue()
max_thread = 120
block_size = 500 


# worker class
# job is a tuple of parameters
class Worker(Thread):
	def __init__(self, job_queue, node):
		Thread.__init__(self)
		self.job_queue  = job_queue
		self.node = node
	def run(self):
		all_done = 0
		while not all_done:
			try:
				job = self.job_queue.get(0)
				time.sleep(random.randint(5000,6000) / 1000.0)	# sleep random time
				single_thread(self.node, job)
			except Queue.Empty:
				all_done = 1
	pass


# single thread blast
def single_thread(node, job):
        #print evalue
	(job_id, jobfname) = job
	jobresfname = jobfname+'.res'
	# run blast on one thread
	logging.info(" %s->%s" % (node, job_id))
	if alignment == 1:
                try:
		        singleres = commands.getoutput("ssh -o StrictHostKeyChecking=no %s '%s -num_descriptions 10 -word_size %d -num_alignments 1 -evalue %.2f -num_threads 4 -outfmt %s -db %s -query %s -out %s'" % (node, blast, wordsize, evalue, '\"6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qseq sseq \&\"\&', db, jobfname, jobresfname))
		        #singleres = commands.getoutput("ssh -o StrictHostKeyChecking=no %s '%s -num_descriptions 1000000000 -word_size %d -num_alignments 1 -evalue %.2f -num_threads 4 -outfmt %s -db %s -query %s -out %s'" % (node, blast, wordsize, evalue, '\"6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qseq sseq \&\"\&', db, jobfname, jobresfname))
                        os.system("echo '--->single blast job %d end on node %s' >> %s" % (job_id, node, jobresfname))
                except:
                        job_queue.put((job))
                        logging.warning(">%s (node) put back %d (job)" % (node, job_id))
	else:
                try:
		        singleres = commands.getoutput("ssh -o StrictHostKeyChecking=no %s '%s -num_descriptions 10 -word_size %d -evalue %.2f -num_threads 4 -outfmt %s -db %s -query %s -out %s'" % (node, blast, wordsize, evalue, '\"6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore \&\"\&', db, jobfname, jobresfname))
		        #singleres = commands.getoutput("ssh -o StrictHostKeyChecking=no %s '%s -num_descriptions 1000000000 -word_size %d -evalue %.2f -num_threads 4 -outfmt %s -db %s -query %s -out %s'" % (node, blast, wordsize, evalue, '\"6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore \&\"\&', db, jobfname, jobresfname))
                        os.system("echo '--->single blast job %d end on node %s' >> %s" % (job_id, node, jobresfname))
                except:
                        job_queue.put((job))
                        logging.warning(">%s (node) put back %d (job)" % (node, job_id))
        time.sleep(random.randint(1000,5000) / 1000.00)
	logging.info(" %s-|%s" % (node, job_id))
        pass


# function for multiple blast processing
def multiple_thread_blast(ifname, load, ofname):
        alljob = 0
        old_job_size = 0
        job_size = 9999999
        while not old_job_size == job_size and not job_size == 0:
                cluster = get_free_nodes()[0]
                old_job_size = job_size
        	# get job queue
	        logging.info( "generate job queue ...")
                if not os.path.exists(tmpresfolder):    # always queue jobs
                        os.system("mkdir %s" % tmpresfolder)
        	        fin = open(ifname)
	                job_id = 0
	                cur_size = 0
	                jobfname = tmpresfolder+re.sub(r'.*/','',ifname)+'.'+"%d"%job_id
	                job_queue.put((job_id,jobfname))
	                fout = open(jobfname, 'w') 
                        for line in fin:
		                if len(line.strip()) == 0:
			                continue
                                if line.strip().startswith('>') and cur_size >= block_size:
			                fout.close()
			                job_id += 1
			                cur_size = 0
			                jobfname = tmpresfolder+re.sub(r'.*/','',ifname)+'.'+"%d"%job_id
			                job_queue.put((job_id,jobfname))
			                fout = open(jobfname, 'w')
		                fout.write("%s" % line)
                                if line.strip().startswith('>'):
		                        cur_size += 1
	                fout.close()
	                fin.close()
                        alljob = job_id
                else:   # conditionally queue jobs
        	        fin = open(ifname)
	                job_id = 0
	                cur_size = 0
	                jobfname = tmpresfolder+re.sub(r'.*/','',ifname)+'.'+"%d"%job_id
	                fout = open(jobfname, 'w') 
                        for line in fin:
		                if len(line.strip()) == 0:
			                continue
                                if line.strip().startswith('>') and cur_size >= block_size:
			                fout.close()
                                        queue_job_opt = 1
                                        if os.path.exists("%s.res" % jobfname):
                                                if not len(re.findall('end', commands.getoutput("tail -1 %s.res" % jobfname))) == 0:
                                                        queue_job_opt = 0
                                        if queue_job_opt == 1:
	                                        job_queue.put((job_id,jobfname))
        			        job_id += 1
	        		        cur_size = 0
		        	        jobfname = tmpresfolder+re.sub(r'.*/','',ifname)+'.'+"%d"%job_id
			                fout = open(jobfname, 'w')
		                fout.write("%s" % line)
                                if line.strip().startswith('>'):
		                        cur_size += 1
        	        fout.close()
                        queue_job_opt = 1
                        if os.path.exists("%s.res" % jobfname):
                                if not len(re.findall('end', commands.getoutput("tail -1 %s.res" % jobfname))) == 0:
                                        queue_job_opt = 0
                        if queue_job_opt == 1:
                                job_queue.put((job_id,jobfname))
	                fin.close()
                        alljob = job_id
        	# processing	
                job_size = job_queue.qsize()
	        logging.info( "processing %d jobs" % (job_size))
	        threads = []
		for j in range(load):
                        if job_queue.empty():
                                break
	                for i in range(max_thread):
                                if job_queue.empty():
                                        break
			        t = Worker(job_queue, cluster[i%len(cluster)])
        			time.sleep(1)
                                try:
		        	        t.start()
			                threads.append(t)
                                except ThreadError:
                                        logging.warning( "\t\tError: thread error caught!")
	        for t in threads:
		        t.join()
                logging.info("%d : %d"  %(old_job_size, job_size))
	# combine results
	logging.info ("combining results ...")
	for i in range(alljob+1):
		jobfname = tmpresfolder+re.sub(r'.*/', '', ifname)+".%d"%i+'.res'
		if i== 0:
                        try:
			        os.system("cat %s |sed '/end on/d' > %s" % (jobfname, ofname))
                        except:
                                logging.warning( "\t%s not there" % jobfname)
		else:
                        try:
			        os.system("cat %s |sed '/end on/d' >> %s" % (jobfname, ofname))
                        except:
                                logging.warning("\t%s not there" % jobfname)
	# cleaning
	os.system('rm -rf %s' % tmpresfolder)
	pass

#
if __name__ == "__main__":
        global blast
        global evalue
        global alignment
        global db
        global wordsize
        global tmpresfolder
        blast = options.blast
        evalue = eval(options.evalue)
        alignment = eval(options.alignment)
        db = options.db
        wordsize = eval(options.wordsize)
	tmpresfolder = scriptfolder + "tmpres_%s/"%(re.sub(r'.*/','',options.ifname))
	multiple_thread_blast(options.ifname, eval(options.load), options.ofname)
	pass


	
	
