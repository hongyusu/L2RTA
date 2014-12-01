
/* 
 * backward_alg_omp.c
 *
 * Ver 0.0
 *
 * March 2014
 *
 * Implemented with C OpenMP library of multiple processes.
 * Input:   score matrix, pointer matrix, K, edge list, number of labels, node degree, maximum node degree
 * Output:  multilabel, score of multilabel
 * Compile into MATLAB function with:
 *      part of compute_topk_omp.c
 *
 *
 * Note:
 *  1. memory check on 31.05.2014 to correct the problem of not free start and stop position
 */


#include "matrix.h"
#include "mex.h"
#include "backward_alg_omp.h"
#include "stdio.h"
#include "omp.h"


double * backward_alg_omp ( double * P_node, double * T_node, int K, double * E, int nlabel, double * in_node_degree, int max_node_degree )
{
    //b_print_mat(P_node,16,4);
    double * Ymax_single;
    double * YmaxVal_single;
    Ymax_single = (double *) malloc (sizeof(double) * K*nlabel);
    YmaxVal_single = (double *) malloc (sizeof(double) * K);
    // ROW AND COLUMN INDEX
    int P_node_nrow = K*nlabel;
    int P_node_ncol = 2*(max_node_degree+1);
    // NODE DEGREE
    double * node_degree;
    node_degree = (double *) malloc (sizeof(double)*nlabel);
    for(int ii=0;ii<nlabel;ii++)
    {node_degree[ii] = in_node_degree[ii];}
    node_degree[(int)(E[0]-1)] = node_degree[(int)(E[0]-1)] + 1; 
    
    
    
    // GET K BEST MULTILABEL WITH
    int para_i;
    int min_load = 8;
    int num_share = K/min_load;
    if(num_share<1){num_share = 1;}
    int * start_pos;
    int * stop_pos;
    start_pos = (int *) malloc (sizeof(int) * num_share);
    stop_pos = (int *) malloc (sizeof(int) * num_share);
    start_pos[0]=0;
    stop_pos[0]=min_load;
    for(int ii=1;ii<num_share;ii++)
    {
        start_pos[ii] = min_load*ii;
        stop_pos[ii] = min_load*(ii+1);
    }
    stop_pos[num_share-1]=K;
    
    // START PARALLEL REGION
    int num_cores = omp_get_num_procs();
    omp_set_dynamic(0);
    omp_set_num_threads(num_cores);
    #pragma omp parallel for shared(P_node,T_node,E,K,nlabel,node_degree,Ymax_single,YmaxVal_single) private(para_i)
    for(para_i=0; para_i<num_share; para_i++)
    {   
        
        //printf("id %d max %d num %d cpu %d\n", omp_get_thread_num(),omp_get_max_threads(),omp_get_num_threads(),omp_get_num_procs());
        for(int kk=start_pos[para_i]; kk<stop_pos[para_i]; kk++)
        {               
            //printf("%d %d %d\n",para_i,start_pos[para_i],stop_pos[para_i]);
            double * Y;
            double * Q_node;
            Y = (double *) malloc (nlabel*sizeof(double));
            Q_node = (double *) malloc (sizeof(double)*P_node_nrow*P_node_ncol);
            for(int ii=0;ii<P_node_nrow;ii++)
            {
                for(int jj=0;jj<P_node_ncol;jj++)
                {
                    Q_node[ii+jj*P_node_nrow] = P_node[ii+jj*P_node_nrow];
                }
            }
                    
            /* PICK UP THE Kth BEST VALUE */
            int par = (int)(E[0]-1);
            int chi;
            back_t_v2i * tmp_M;
            tmp_M = (struct back_v2i *) malloc((sizeof(back_t_v2i))*2*K);
            int ll=0;
            for(int ii=par*K;ii<(par+1)*K;ii++)
            {
                for(int jj=0;jj<2;jj++)
                {
                    tmp_M[ll].v = Q_node[ii+(jj*K*nlabel)];
                    tmp_M[ll].nrow = ii;
                    tmp_M[ll].ncol = jj;
                    ll ++;
                }
            }
            qsort(tmp_M, 2*K, sizeof(back_t_v2i), (void *)b_compare_structs);
            /* PICKING WILL BE FLAGED*/
            Q_node[(tmp_M[kk].nrow+par*K) + tmp_M[kk].ncol*K*nlabel]=PICK;
            YmaxVal_single[kk] = tmp_M[kk].v;
            free(tmp_M);
//             printf("-->%d\n",kk);
//             b_print_mat(Q_node,P_node_nrow,P_node_ncol);
//             b_print_mat(T_node,P_node_nrow,P_node_ncol);
//             printf("--|\n");

            /* EVERYTHING IS STANDARDIZE WE DO LOOP TRACE DOWN */
            par = -1;
            for(int ii=0;ii<nlabel-1;ii++)
            {
                //printf("+++|  %d-->%d:%d:%d\n",tid,training_i,kk,ii);
                if(par == (int)(E[ii]-1))
                {continue;}
                par = (int)(E[ii]-1);
                chi = (int)(E[ii + nlabel-1]-1);
                int col_pos;
                int row_pos;
                double n_chi;
                for(int jj=0;jj<K;jj++)
                {
                    for(int ll=0;ll<2;ll++)
                    {
                        if(Q_node[jj+par*K + ll*nlabel*K] == PICK)
                        {
                            row_pos = jj;
                            col_pos = ll;
                            break;
                        }
                    }
                }
                /* EMIT A LABEL */
                Y[par] = col_pos;
                n_chi = node_degree[par]-1;
                //printf("--> %d->%d %.2f\n", par, chi, n_chi);
                //printf("\t %d >> %d\n",par,col_pos);
                //printm(Q_node,P_node_nrow,P_node_ncol);
                int * cs = (int *) malloc (sizeof(int)*n_chi);
                int jj=0;
                while((int)(E[ii+jj]-1) == par)
                {
                    cs[jj] = (int)(E[ii+jj+nlabel-1]-1);
                    //printf("\t collect chi %d\n",cs[jj]);
                    jj++;
                    if(ii+jj>nlabel)
                    {break;}
                }
                /* LOOP THROUGHT CHILDREN */
                for(int jj=n_chi-1;jj>=0;jj--)
                {
                    //printf("+++|  %d-->%d:%d:%d:%d\n",tid,training_i,kk,ii,jj);
                    chi = cs[jj];
                    int chi_pos = n_chi - jj;
                    double index = T_node[(row_pos+par*K)+(chi_pos*2+col_pos)*nlabel*K] + col_pos*K;
                    //printf("\t(chi)%d (pos)%d %.2f\n",chi,chi_pos,index);
                    int c_col_pos, c_row_pos;
                    c_col_pos = (int)(index-1)/(int)(K);
                    c_row_pos = (int)(index-1)%(int)(K);
                    //printf("======%.2f %d %d\n",index,c_row_pos,c_col_pos);
                    double c_index;
                    c_index = T_node[(c_row_pos+chi*K)+(c_col_pos)*nlabel*K];
                    int cc_col_pos,cc_row_pos;
                    cc_row_pos = (int)(c_index-1)%(int)(K);
                    cc_col_pos = (int)(c_index-1)/(int)(K);
                    //printf("======%.2f %d %d\n",c_index,cc_row_pos,cc_col_pos);
                    /* UPDATE Q block for the child*/
                    Q_node[(cc_row_pos+chi*K)+(cc_col_pos)*nlabel*K] = PICK;
                    /* IF CURRENT CHILD IS A LEAVE EMIT A LABEL */
                    if(node_degree[chi]==1)
                    {
                        Y[chi]=cc_col_pos;
                        //printf("\t %d >> %d\n",chi,cc_col_pos);
                    }
                }
                free(cs);
            }
            //printf("+++||  %d-->%d\n",tid,training_i);
            //printf("--->\n");
            //b_print_mat(Y,1,10);
            for(int ii=0;ii<nlabel;ii++)
            {Ymax_single[kk*nlabel+ii] = Y[ii]*2-1;}

            free(Q_node);
            free(Y);
        }
    }
    free(stop_pos);
    free(start_pos);

    /* END OF PARALLEL REGION */
    //node_degree[(int)(E[0]-1)] = node_degree[(int)(E[0]-1)] - 1; 

    // SAVE RESULTS
    double * results;
    results = (double *) malloc (sizeof(double)*K*(nlabel+1));
    for(int ii=0;ii<K*nlabel;ii++)
    {
        results[ii] = Ymax_single[ii];
    }
    for(int ii=0;ii<K;ii++)
    {
        results[ii+K*nlabel] = YmaxVal_single[ii];
    }
    // DESTROY POINTER SPACE
    if(node_degree){free(node_degree);}
    if(YmaxVal_single){free(YmaxVal_single);}
    if(Ymax_single){free(Ymax_single);}
    
    return(results);
}



int b_compare_structs ( const void *a, const void *b )
{    
    back_t_v2i *struct_a = (back_t_v2i *) a;
    back_t_v2i *struct_b = (back_t_v2i *) b;
 
    if(struct_a->v < struct_b->v)
    {return 1;}
    else if(struct_a->v > struct_b->v)
    {return -1;}
    else
    //{return 0;}
    {
        if(struct_a->nrow > struct_b->nrow)
        {return 1;}
        else if(struct_a->nrow < struct_b->nrow)
        {return -1;}
        else
        {return 0;}
    };
}



void b_print_mat ( double * M, int nrow, int ncol )
{
    printf("#row: %d #ncol %d\n", nrow,ncol);
    for(int i=0; i<nrow; i++)
    {
        for(int j=0; j<ncol; j++)
        {
            printf("%.4f ", M[i+j*nrow]);
        }
        printf("\n");
    }
    printf("\n");
}
