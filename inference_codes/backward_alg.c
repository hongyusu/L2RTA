
#include "matrix.h"
#include "mex.h"
#include "backward_alg.h"
#include "stdio.h"
#include "omp.h"


/* Implemented with C OpenMP library for multiple process.
 *
 * compile with:
 *      mex backward_alg.c CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" CC="/usr/local/bin/gcc -std=c99"
 *
 * use in MATLAB:
 *      [Ymax_single, YmaxVal_single] = backward_alg(P_node, T_node, K, E, nlabel, node_degree)
 */

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    /* DEFINE INPUT AND OUTPUT */
    #define IN_P_node           prhs[0]
    #define IN_T_node           prhs[1]
    #define IN_K                prhs[2]
    #define IN_E                prhs[3]
    #define IN_nlabel           prhs[4]
    #define IN_node_degree      prhs[5]
    #define OUT_Ymax_single     plhs[0]
    #define OUT_YmaxVal_single  plhs[1]
    
    /* INPUT */
    double * P_node;
    double * T_node;
    mint K;
    double * E;
    mint nlabel;
    double * node_degree;
    /* OUTPUT */
    double * Ymax_single;
    double * YmaxVal_single;
    /* INITIALIZE INPUT */
    P_node = mxGetPr(IN_P_node);
    T_node = mxGetPr(IN_T_node);
    K = mxGetScalar(IN_K);
    E = mxGetPr(IN_E);
    nlabel = mxGetScalar(IN_nlabel);
    node_degree = mxGetPr(IN_node_degree);
    node_degree[(int)(E[0]-1)] = node_degree[(int)(E[0]-1)] + 1; 
    /* INITIALIZE OUTPUT */
    OUT_Ymax_single = mxCreateDoubleMatrix(1,K*nlabel,mxREAL);
    OUT_YmaxVal_single = mxCreateDoubleMatrix(1,K,mxREAL);
    Ymax_single = mxGetPr(OUT_Ymax_single);
    YmaxVal_single = mxGetPr(OUT_YmaxVal_single);
    /* OTHER */
    int P_node_nrow = mxGetM(IN_P_node);
    int P_node_ncol = mxGetN(IN_P_node);
    //printm(T_node,P_node_nrow,P_node_ncol);
    
    
    //printf("P_node add %p\nQ_node %p\nnode_degree %p\n", P_node,T_node,E,node_degree);
    //printf("Ymax_single add %p\nYmax_Val_single add %p\n", Ymax_single,YmaxVal_single);
    
    /* GET K BEST */
    /* start parallel region */
    mint kk;
    //#pragma omp parallel for shared(P_node,T_node,E,K,nlabel,node_degree,Ymax_single,YmaxVal_single) private(kk)
    //{
        for(kk=0; kk<K; kk++)
        {
             //mint nthreads = omp_get_thread_num();
             //printf("%d->%d\n",nthreads,kk);
            /* */
//             mxArray * M_Y;
//             mxArray * M_Q_node;
//             M_Y = mxCreateDoubleMatrix(1,nlabel,mxREAL);
//             M_Q_node = mxCreateDoubleMatrix(P_node_nrow,P_node_ncol,mxREAL);
//             M_Q_node = mxDuplicateArray(IN_P_node);
//             double * Y = mxGetPr(M_Y);
//             double * Q_node = mxGetPr(M_Q_node);

                        
            double * Y;
            double * Q_node;
            Y = (double *) malloc (nlabel*sizeof(double));
            Q_node = (double *) malloc (sizeof(double)*P_node_nrow*P_node_ncol);
            for(mint ii=0;ii<P_node_nrow;ii++)
            {
                for(mint jj=0;jj<P_node_ncol;jj++)
                {
                    Q_node[ii+jj*P_node_nrow] = P_node[ii+jj*P_node_nrow];
                }
            }
                    


            /* PICK UP THE Kth BEST VALUE */
            mint par = (int)(E[0]-1);
            mint chi;
            t_v2i * tmp_M;
            tmp_M = (struct v2i *) malloc((sizeof(t_v2i))*2*K);
            mint ll=0;
            for(mint ii=par*K;ii<(par+1)*K;ii++)
            {
                for(mint jj=0;jj<2;jj++)
                {
                    tmp_M[ll].v = Q_node[ii+(jj*K*nlabel)];
                    tmp_M[ll].nrow = ii;
                    tmp_M[ll].ncol = jj;
                    ll ++;
                }
            }
            qsort(tmp_M, 2*K, sizeof(t_v2i), (void *)compare_structs);
            /* PICKING WILL BE FLAGED*/
            Q_node[(tmp_M[kk].nrow+par*K) + tmp_M[kk].ncol*K*nlabel]=PICK;
            YmaxVal_single[kk] = tmp_M[kk].v;
            free(tmp_M);
//             printf("++-->%d\n",kk);
//             printm(Q_node,P_node_nrow,P_node_ncol);
//             printm(T_node,P_node_nrow,P_node_ncol);
//             printf("++--|\n");

            /* EVERYTHING IS STANDARDIZE WE DO LOOP TRACE DOWN */
            par = -1;
            for(mint ii=0;ii<nlabel-1;ii++)
            {
                if(par == (int)(E[ii]-1))
                {continue;}
                par = (int)(E[ii]-1);
                chi = (int)(E[ii + nlabel-1]-1);
                mint col_pos;
                mint row_pos;
                double n_chi;
                for(mint jj=0;jj<K;jj++)
                {
                    for(mint ll=0;ll<2;ll++)
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
                mint * cs = (mint *) malloc (sizeof(mint)*n_chi);
                mint jj=0;
                while((int)(E[ii+jj]-1) == par)
                {
                    cs[jj] = (int)(E[ii+jj+nlabel-1]-1);
                    //printf("\t collect chi %d\n",cs[jj]);
                    jj++;
                    if(ii+jj>nlabel)
                    {break;}
                }
                /* LOOP THROUGHT CHILDREN */
                for(mint jj=n_chi-1;jj>=0;jj--)
                {
                    chi = cs[jj];
                    mint chi_pos = n_chi - jj;
                    double index = T_node[(row_pos+par*K)+(chi_pos*2+col_pos)*nlabel*K] + col_pos*K;
                    //printf("\t(chi)%d (pos)%d %.2f\n",chi,chi_pos,index);
                    mint c_col_pos, c_row_pos;
                    c_col_pos = (mint)(index-1)/(mint)(K);
                    c_row_pos = (mint)(index-1)%(mint)(K);
                    //printf("======%.2f %d %d\n",index,c_row_pos,c_col_pos);
                    double c_index;
                    c_index = T_node[(c_row_pos+chi*K)+(c_col_pos)*nlabel*K];
                    mint cc_col_pos,cc_row_pos;
                    cc_row_pos = (mint)(c_index-1)%(mint)(K);
                    cc_col_pos = (mint)(c_index-1)/(mint)(K);
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
            //printf("++->\n");
            //printm(Y,1,10);
            for(mint ii=0;ii<nlabel;ii++)
            {Ymax_single[kk*nlabel+ii] = Y[ii]*2-1;}

            free(Q_node);
            free(Y);
//             if(M_Q_node)
//             {printf("\t%d-|%d M_Q_node %p\n",nthreads,kk,Q_node);}
            //mxDestroyArray(M_Q_node);
//             if(M_Y)
//             {printf("\t%d-|%d M_Y %p\n",nthreads,kk,Y);}
            //mxDestroyArray(M_Y);
        }
    //}

    /* END OF PARALLEL REGION */
    node_degree[(int)(E[0]-1)] = node_degree[(int)(E[0]-1)] - 1; 

    
//     printf("P_node add %p\nQ_node %p\nnode_degree %p\n", P_node,T_node,E,node_degree);
//     printf("Ymax_single add %p\nYmax_Val_single add %p\n", Ymax_single,YmaxVal_single);
//     printm(YmaxVal_single,1,K);
}



int compare_structs (const void *a, const void *b)
{    
    t_v2i *struct_a = (t_v2i *) a;
    t_v2i *struct_b = (t_v2i *) b;
 
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



void printm(double * M, mint nrow, mint ncol)
{
    printf("#row: %d #ncol %d\n", nrow,ncol);
    for(mint i=0; i<nrow; i++)
    {
        for(mint j=0; j<ncol; j++)
        {
            printf("%.4f ", M[i+j*nrow]);
        }
        printf("\n");
    }
    printf("\n");
}
