
/* 
 *
 * Ver 0.0
 *
 * MATLAB gateway function: compute_topk_omp.c
 *
 * March 2014
 *
 * Implemented with C OpenMP library for multiple processes.
 *
 * Given edge list of a single tree as well as score that corresponds to the edge labels --,-+,+-,++, compute the K best multilabel that maximizes the sums of edge score.
 *
 * Input:   
 *      1. edge score: 
 *          (1) edge score matrix of 4*|E| dimension, where rows are score of edge labels --,-+,+-,++, columns are edges,
 *          (2) reshape the edge score matrix into 4|E|*1 dimension
 *      2. K: 
 *          the number of best multilabel from a single tree
 *      3. edge list: 
 *          marix of |E|*2 dimension, processed by 'roottree function' in run_RSTA.m , example:
 *           E =
 *              1    13
 *             13     9
 *              9     8
 *              8     5
 *              8    12
 *              5     2
 *             12     7
 *             12    10
 *              7     3
 *              7     4
 *             10     6
 *              4    11
 *      4. node degree: 
 *          matrix of 1*|V| dimension, example:
 *           node_degree =
 *               1     1     1     2     2     1     3     3     2     2     1     3     2
 *
 * Output:
 *      1. multilabels: 
 *          k best multilabels, matrix of |Y|*K
 *      2. score of multilabels:
 *          k best score, matrix of 1*K
 *
 * Example usage in matlab:
 *      [Y_tmp,Y_tmp_val] = compute_topk_omp(in_gradient,kappa,E,node_degree);
 *
 * Compile into MATLAB function with the following command :
 *      mex compute_topk_omp.c forward_alg_omp.c backward_alg_omp.c  CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" CC="/usr/local/bin/gcc -std=c99"
 *
 * NOTE:
 *      1. There is no memeory lead, last check on 26/03/2014
 *      2. Add detailed comment on 25/04/2014
 *      3. check memory lead on 31.05.2014
 *
 *
 *
 */

#include "matrix.h"
#include "mex.h"
#include "backward_alg_omp.h"
#include "forward_alg_omp.h"
#include "stdio.h"
#include "omp.h"
#include "math.h"

// matlab gateway function
void mexFunction ( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    // FOR THE CONVINIENT OF DEALING WITH INPUT AND OUTPUT
    #define IN_gradient         prhs[0]
    #define IN_K                prhs[1]
    #define IN_E                prhs[2]
    #define IN_node_degree      prhs[3]
    #define OUT_Ymax            plhs[0]
    #define OUT_YmaxVal         plhs[1]
    
    // INPUT
    // gradient
    double * gradient_i;
    gradient_i = mxGetPr(IN_gradient);
    // K
    int K;
    K = mxGetScalar(IN_K);
    // E
    double * E;
    E = mxGetPr(IN_E);
    // node_degree
    double * node_degree;
    node_degree = mxGetPr(IN_node_degree);
    // mm
    int E_nrow = mxGetM(IN_E);
    int gradient_len = mxGetM(IN_gradient);
    int mm;
    mm = gradient_len/4/E_nrow;
    // nlabel
    int nlabel;
    nlabel = mxGetN(IN_node_degree);
    // MIN_GRADIENT_VAL & local copy of gradient
    
    double min_gradient_val;
    double * gradient;
    gradient = (double *) malloc (sizeof(double) * gradient_len);
    min_gradient_val = 1000000000000;
    
    for(int ii=0;ii<gradient_len;ii++)
    {
        gradient[ii] = gradient_i[ii];
        if(gradient_i[ii]<min_gradient_val)
        {min_gradient_val = gradient_i[ii];}
    }
    min_gradient_val -= 0.00001;
    for(int ii=0;ii<gradient_len;ii++)
    {
        gradient[ii] = gradient[ii]- min_gradient_val;
    }
    
    
    // Ymax
    double * Ymax;
    OUT_Ymax = mxCreateDoubleMatrix(mm,K*nlabel,mxREAL);
    Ymax = mxGetPr(OUT_Ymax);
    // YmaxVal
    double * YmaxVal;
    OUT_YmaxVal = mxCreateDoubleMatrix(mm,K,mxREAL);
    YmaxVal = mxGetPr(OUT_YmaxVal);
    // MAX_NODE_DEGREE
    int max_node_degree;
    max_node_degree = 0;
    for(int ii=0;ii<nlabel;ii++)
    {
        if(max_node_degree<node_degree[ii])
        {max_node_degree = node_degree[ii];}
    }
    
    // OMP LOOP THROUGH EXAMPLES
    int nn = 50;
    int nworker = (mm-2)/nn;
    //printf("data: %d worker: %d\n", mm,nworker);
    if(nworker <1){nworker=1;};
    int * start_pos = (int *) malloc (sizeof(int) * (nworker));
    int * stop_pos = (int *) malloc (sizeof(int) * (nworker));
    start_pos[0]=0;
    stop_pos[0]=nn;
    for(int ii=1;ii<nworker;ii++)
    {
        start_pos[ii]=ii*nn;
        stop_pos[ii]=(ii+1)*nn;
    }
    stop_pos[nworker-1]=mm;
    int share_i;
    

    int num_cores = omp_get_num_procs();
    omp_set_dynamic(0);
    omp_set_num_threads(num_cores);
    #pragma omp parallel for private(share_i)
    for(share_i=0;share_i<nworker;share_i++)
    {
        //printf("id %d max %d num %d cpu %d\n", omp_get_thread_num(),omp_get_max_threads(),omp_get_num_threads(),omp_get_num_procs());
        for(int training_i=start_pos[share_i];training_i<stop_pos[share_i];training_i++)
        {
            // GET TRAINING GRADIENT
            double * training_gradient;
            training_gradient = (double *) malloc (sizeof(double) * 4 * E_nrow);
            for(int ii=0;ii<E_nrow*4;ii++)
            {training_gradient[ii] = gradient[ii+training_i*4*E_nrow];}

            // FORWARD ALGORITHM TO GET P_NODE AND T_NODE
            double * results;
            results = forward_alg_omp(training_gradient, K, E, nlabel, node_degree, max_node_degree);
            double * P_node;
            double * T_node;
            P_node = (double *) malloc (sizeof(double) * K*nlabel*2*(max_node_degree+1));
            T_node = (double *) malloc (sizeof(double) * K*nlabel*2*(max_node_degree+1));
            for(int ii=0;ii<K*nlabel;ii++)
            {
                for(int jj=0;jj<2*(max_node_degree+1);jj++)
                {
                    P_node[ii+jj*K*nlabel] = results[ii+jj*K*nlabel*2];
                }
            }
            for(int ii=0;ii<K*nlabel;ii++)
            {
                for(int jj=0;jj<2*(max_node_degree+1);jj++)
                {
                    T_node[ii+jj*K*nlabel] = results[ii+K*nlabel+jj*K*nlabel*2];
                }
            }
            if(results){free(results);}

            // BACKWARD ALGORITHM TO GET MULTILABEL
            results = backward_alg_omp(P_node, T_node, K, E, nlabel, node_degree, max_node_degree);
            for(int ii=0;ii<K*nlabel;ii++)
            {
                Ymax[training_i+ii*mm] = results[ii];
            }
            for(int ii=0;ii<K;ii++)
            {
                YmaxVal[training_i+ii*mm] = results[ii+K*nlabel];
            }
            if(results){free(results);}
            
            // DESTROY POINTER SPACE
            if(T_node){free(T_node);}
            if(P_node){free(P_node);}
            if(training_gradient){free(training_gradient);}
        }         
    }
    free(start_pos);
    free(stop_pos);
    int tmpK=K;
    if(K>pow(2,nlabel)){tmpK=pow(2,nlabel);}
    //printf("%d\n",tmpK);
    for(int ii=0;ii<tmpK*mm;ii++)
    {
        YmaxVal[ii] = YmaxVal[ii]+min_gradient_val*(nlabel-1)-(nlabel);
    }   
   
    if(gradient){free(gradient);}
}

