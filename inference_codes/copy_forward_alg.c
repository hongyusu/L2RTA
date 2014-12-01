
#include "matrix.h"
#include "mex.h"
#include "forward_alg.h"
#include "stdio.h"




/* The gateway function 
 * Input:
 *      gradient: positive gradient
 *      K:  kappa
 *      E:  edge list
 *      m:  number of examples
 *      l:  number of labels
 * Output:
 *      P_node:
 *      T_node:
 */
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    #define IN_gradient         prhs[0]
    #define IN_K                prhs[1]
    #define IN_E                prhs[2]
    #define IN_l                prhs[3]
    #define IN_node_degree      prhs[4]
    #define IN_max_node_degree  prhs[5]
    #define OUT_P_node          plhs[0]
    #define OUT_T_node          plhs[1]
    
    
    /* input */
    double * gradient;
    mint K;
    double * E;
    mint l;
    mint max_node_degree;
    double * node_degree;
    /* output */
    double * P_node;
    double * T_node;
    /* others */
    size_t size_E_1;
    mint gradient_nrow;
    mint gradient_ncol;
    
    /* get input data */
    gradient = mxGetPr(IN_gradient);
    gradient_nrow = mxGetM(IN_gradient);
    gradient_ncol = mxGetN(IN_gradient);
    /* printm(gradient,gradient_nrow,gradient_ncol); */
    K = mxGetScalar(IN_K);
    E = mxGetPr(IN_E);
    /* printm(E,mxGetM(IN_E),mxGetN(IN_E)); */
    l = mxGetScalar(IN_l);
    node_degree = mxGetPr(IN_node_degree);    
    max_node_degree = mxGetScalar(IN_max_node_degree);
    /* get output */
    OUT_P_node = mxCreateDoubleMatrix(K*l,2*(max_node_degree+1),mxREAL);
    OUT_T_node = mxCreateDoubleMatrix(K*l,2*(max_node_degree+1),mxREAL);
    P_node = mxGetPr(OUT_P_node);
    T_node = mxGetPr(OUT_T_node);

    mint p;
    mint c;
    /* iterate on each edge from leave to node to propagate message */
    for( mint i=l-2; i>=0; i-- )
    {
        /* printm(P_node,mxGetM(OUT_P_node),mxGetN(OUT_P_node)); */
        /* printm(T_node,mxGetM(OUT_P_node),mxGetN(OUT_P_node)); */
        /* if(i==0){break;} */
        /*  parent child index */
        p = E[i];
        c = E[i+(l-1)];
        /* printf("=============== %d (%d->%d)\n",i,p,c); */

        /*  update node score */
        mxArray * in_blk_array;
        in_blk_array = mxCreateDoubleMatrix(K,max_node_degree,mxREAL);
        double * in_blk = mxGetPr(in_blk_array);
        for(mint sign_pos=0;sign_pos<2;sign_pos++)
        {
            /*  assign value to inblock (-1) */
            for(mint ii=0;ii<mxGetM(in_blk_array);ii++)
            {
                mint start_col=3-1+sign_pos; /*  start from the 3rd column */
                for(mint jj=0;jj<mxGetN(in_blk_array);jj++)
                {
                    in_blk[ii+jj*(mxGetM(in_blk_array))]=P_node[((c-1)*K+ii)+(start_col+jj*2)*(K*l)];
                }
                start_col++;
            }
            /* printf("in block\n");printm(in_blk,K,max_node_degree); */
            /*  compute topk for inblock -1 */
            double * tmp_res = LinearMaxSum(in_blk_array,node_degree[c-1]);
            /* printf("out block\n");printm(tmp_res,K,max_node_degree+1); */
            /*  assign value to P_node T_node */
            for(mint ii=0;ii<mxGetM(in_blk_array);ii++)
            {
                /* printf("position %d %d\n",ii+(c-1)*K+sign_pos*(K*l),ii+K*max_node_degree); */
                P_node[ii+(c-1)*K+sign_pos*(K*l)] = tmp_res[ii+K*(max_node_degree)];
                mint start_col = 3-1+sign_pos;
                for(mint jj=0;jj<mxGetN(in_blk_array);jj++)
                {
                    /* printf("-->%d %d\n",ii+c*K+(start_col+jj*2)*(K*l),ii+jj*K); */
                    T_node[ii+(c-1)*K+(start_col+jj*2)*(K*l)] = tmp_res[ii+jj*K];
                }
                start_col++;
            }
        }
        /*  combine edge potential */
        mxArray * M_array;
        M_array = mxCreateDoubleMatrix(2*K,2,mxREAL);
        double * M = mxGetPr(M_array);
        for(mint ii=0;ii<2*K;ii++)
        {
            for(mint jj=0;jj<2;jj++)
            {
                if(P_node[(c-1)*K+ii%K+ii/K*K*l])
                {M[ii+jj*(K*2)] = P_node[(c-1)*K+ii%K+ii/K*K*l] + gradient[i*4+(ii/K)+jj*2];}
                else
                {M[ii+jj*(K*2)] = 0;}
                /* printf("%d %d %d %.3f\n",ii,jj,i*4+(ii/K)+jj*2,gradient[i*4+(ii/K)+jj*2]); */
                /* printf("%d %d %d %.3f\n",ii,jj,(c-1)*K+ii%K+ii/K*K*l,P_node[(c-1)*K+ii%K+ii/K*K*l]); */
                /* printf("%d, %d,%.3f\n",ii+jj*(K*2),ii%K+jj*2,P_node[c*K+ii%K+jj*K*l]+gradient[ii%K+jj*2]); */
            }
        }
        /* if(i==7){printm(M,mxGetM(M_array),mxGetN(M_array));} */
        /*  sort */
        int c_in_p=0; /*  should be at least 1 */
        for(mint j=i;j<mxGetM(IN_E);j++)
        {
            if(E[j] == p)
            {c_in_p++;}
            else
            {break;}
        }
        t_v2i * tmp_M;
        tmp_M = (struct v2i *) malloc((sizeof(t_v2i))*2*K);
        for(mint sign_pos=0;sign_pos<2;sign_pos++)
        {
            for(mint ii=0;ii<2*K;ii++)
            {
                tmp_M[ii].v = M[ii+sign_pos*2*K];
                /* printf("%d\n",ii+sign_pos*2*K); */
                tmp_M[ii].i = ii+1;
            }
            qsort(tmp_M, 2*K, sizeof(t_v2i), (void *)compare_structs);
            /*  put into correct block */
            for(mint ii=0;ii<K;ii++)
            {
                if(tmp_M[ii].v>0)
                {
                    T_node[(c-1)*K+ii+sign_pos*K*l] = tmp_M[ii].i;
                    P_node[(p-1)*K+ii+(c_in_p*2+sign_pos)*l*K] = tmp_M[ii].v;
                }
                else
                {
                    T_node[(c-1)*K+ii+sign_pos*K*l] = 0;
                    P_node[(p-1)*K+ii+(c_in_p*2+sign_pos)*l*K] = 0;
                }
            }
        } 
        
        free(tmp_M);
        mxDestroyArray(M_array);
        mxDestroyArray(in_blk_array);

    }
    /* one more iteration on root node */
    /*  update node score */
    c=p;
    mxArray * in_blk_array;
    in_blk_array = mxCreateDoubleMatrix(K,max_node_degree,mxREAL);
    double * in_blk = mxGetPr(in_blk_array);
    for(mint sign_pos=0;sign_pos<2;sign_pos++)
    {
        /*  assign value to inblock (-1) */
        for(mint ii=0;ii<mxGetM(in_blk_array);ii++)
        {
            mint start_col=3-1+sign_pos; /*  start from the 3rd column */
            for(mint jj=0;jj<mxGetN(in_blk_array);jj++)
            {
                in_blk[ii+jj*(mxGetM(in_blk_array))]=P_node[((c-1)*K+ii)+(start_col+jj*2)*(K*l)];
            }
            start_col++;
        }
        /* printf("in block\n");printm(in_blk,K,max_node_degree); */
        /*  compute topk for inblock -1 */
        double * tmp_res = LinearMaxSum(in_blk_array,node_degree[c-1]+1);
        /* printf("out block\n");printm(tmp_res,K,5); */
        /*  assign value to P_node T_node */
        for(mint ii=0;ii<mxGetM(in_blk_array);ii++)
        {
            /* printf("position %d %d\n",ii+(c-1)*K+sign_pos*(K*l),ii+K*max_node_degree); */
            P_node[ii+(c-1)*K+sign_pos*(K*l)] = tmp_res[ii+K*(max_node_degree)];
            mint start_col = 3-1+sign_pos;
            for(mint jj=0;jj<mxGetN(in_blk_array);jj++)
            {
                /* printf("-->%d %d\n",ii+c*K+(start_col+jj*2)*(K*l),ii+jj*K); */
                T_node[ii+(c-1)*K+(start_col+jj*2)*(K*l)] = tmp_res[ii+jj*K];
            }
            T_node[ii+(c-1)*K+(start_col-2)*(K*l)] = ii+1+(start_col-2)*K;
            start_col++;
        }
    }

    mxDestroyArray(in_blk_array);
   /* printm(P_node,mxGetM(OUT_P_node),mxGetN(OUT_P_node)); */
   /* printm(T_node,mxGetM(OUT_P_node),mxGetN(OUT_P_node)); */
        
}



double * LinearMaxSum(mxArray * M_array, mint current_node_degree)
{
    /* printf("******\n"); */
    /*  input */
    mint M_nrow = mxGetM(M_array);
    mint M_ncol = mxGetN(M_array);
    /* printf("%d %d\n", M_nrow, M_ncol); */
    /* printf("current node degree:%d\tM_nrow:%d\tM_ncol:%d\n",current_node_degree,M_nrow,M_ncol); */
    double * M = mxGetPr(M_array);
    /* output */
    mxArray * res_array;
    res_array = mxCreateDoubleMatrix(M_nrow,M_ncol+1,mxREAL);
    double * res = mxGetPr(res_array);

    /*no child*/
    if(current_node_degree-1==0)
    {
        res[0+M_ncol*M_nrow]=1;
        /* printf("--%d %d \n",mxGetM(res_array),mxGetN(res_array)); */
        /* printm(res,M_nrow,M_ncol+1); */
        return res;
    }
    
    /* printf("M\n");printm(M,M_nrow,M_ncol); */
    
    /*  INITIALIZE TMP_M WITH FIRST COLUMN OF M */
    t_v2is * tmp_M;
    tmp_M = (struct v2is *) malloc((sizeof(t_v2is))*M_nrow);
    for(mint ii=0;ii<M_nrow;ii++)
    {
        if(M[ii]>0)
        {
            tmp_M[ii].v=M[ii];
            tmp_M[ii].i=(mint *)malloc(sizeof(mint)*(current_node_degree-1));
            tmp_M[ii].i[0]=ii+1;
        }
        else
        {
            tmp_M[ii].v=0;
            tmp_M[ii].i=(mint *)malloc(sizeof(mint)*(current_node_degree-1));
            tmp_M[ii].i[0]=0;
        }
    }
    /* for(mint jj=0;jj<M_nrow;jj++){printf("tmp_M %.4f %d %d %d\n",tmp_M[jj].v,tmp_M[jj].i[0],tmp_M[jj].i[1],tmp_M[jj].i[2]);} */
    /*  TWO COLUMN AT A TIME */
    for(mint ii=1;ii<(current_node_degree-1);ii++)
    {
        /* printf("%.3f", M[1+ii*M_nrow]); */
        t_v2is * tmp_M_long;
        tmp_M_long = (struct v2is *) malloc((sizeof(t_v2is))*M_nrow*M_nrow);    
        
        for(mint jj=0;jj<M_nrow;jj++)
        {
            /*  */
            if(tmp_M[jj].v > 0)
            {
                for(mint kk=0;kk<M_nrow;kk++)
                {
                    if(M[kk+ii*M_nrow]>0)
                    {
                        tmp_M_long[jj*M_nrow+kk].v = tmp_M[jj].v+M[kk+ii*M_nrow];
                        tmp_M_long[jj*M_nrow+kk].i = (mint *)malloc(sizeof(mint)*(current_node_degree-1));    
                        for(mint ll=0;ll<ii;ll++)
                        {tmp_M_long[jj*M_nrow+kk].i[ll] = tmp_M[jj].i[ll];}
                        tmp_M_long[jj*M_nrow+kk].i[ii]=kk+1;
                    }
                    else
                    {
                        tmp_M_long[jj*M_nrow+kk].v = 0;
                        tmp_M_long[jj*M_nrow+kk].i = (mint *)malloc(sizeof(mint)*(current_node_degree-1));
                        for(mint ll=0;ll<current_node_degree-1;ll++)
                        {
                            tmp_M_long[jj*M_nrow+kk].i[ll] = 0;
                        }
                    }
                }
            }
            /*  */
            else
            {
                for(mint kk=0;kk<M_nrow;kk++)
                {
                    tmp_M_long[jj*M_nrow+kk].v = 0;
                    tmp_M_long[jj*M_nrow+kk].i = (mint *)malloc(sizeof(mint)*(current_node_degree-1));
                    for(mint ll=0;ll<current_node_degree-1;ll++)
                    {
                        tmp_M_long[jj*M_nrow+kk].i[ll] = 0;
                    }
                }
            }
        }
        /* for(mint jj=0;jj<M_nrow*M_nrow;jj++){printf("tmp_M_long %.4f %d %d %d\n",tmp_M_long[jj].v,tmp_M_long[jj].i[0],tmp_M_long[jj].i[1],tmp_M_long[jj].i[2]);} */
        /*  update tmp_M */
        /* for(mint jj=0;jj<M_nrow*M_nrow;jj++){printf("before %d %d ->%.3f %d %d %d\n", ii,jj,tmp_M_long[jj].v, tmp_M_long[jj].i[0], tmp_M_long[jj].i[1],tmp_M_long[jj].i[2]);}  */
        qsort(tmp_M_long, M_nrow*M_nrow, sizeof(t_v2is), (void *)compare_structs_is);
        /* for(mint jj=0;jj<M_nrow*M_nrow;jj++){printf("after %d %d ->%.3f %d %d %d\n", ii,jj,tmp_M_long[jj].v, tmp_M_long[jj].i[0], tmp_M_long[jj].i[1],tmp_M_long[jj].i[2]);}  */
        
        for(mint jj=0;jj<M_nrow;jj++)
        {
            tmp_M[jj].v = tmp_M_long[jj].v;
            /*  TODO  */
            for(mint kk=0;kk<ii+1;kk++)
            {tmp_M[jj].i[kk] = tmp_M_long[jj].i[kk];}
            /* printf("%d tmp_M %.4f %d %d %d\n",ii,tmp_M[jj].v,tmp_M[jj].i[0],tmp_M[jj].i[1],tmp_M[jj].i[2]); */
        }
        
        for(mint ii=0; ii<M_nrow*M_nrow; ii++)
        {free(tmp_M_long[ii].i);}
        free(tmp_M_long);
    }
    /* for(mint jj=0;jj<M_nrow;jj++){printf("tmp_M %.4f %d %d %d\n",tmp_M[jj].v,tmp_M[jj].i[0],tmp_M[jj].i[1],tmp_M[jj].i[2]);} */
    
    
    
    
    /*  collect results */
    for(mint ii=0;ii<M_nrow;ii++)
    {
        if(tmp_M[ii].v>0)
        {
            res[ii+M_nrow*M_ncol]=tmp_M[ii].v+1;
        }
        else
        {
            res[ii+M_nrow*M_ncol]=tmp_M[ii].v;
        }
        /* printf("-----%d %.3f\n",ii+M_nrow*M_ncol,tmp_M[ii].v+1); */
        for(mint jj=0;jj<current_node_degree-1;jj++)
        {res[ii+M_nrow*jj]=tmp_M[ii].i[jj];}
    }
    
    /* destroy */
    for(mint ii=0; ii<M_nrow; ii++)
    {free(tmp_M[ii].i);}
    free(tmp_M);

    /*  return results */
    return res;
    
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
    {return 0;};
}
int compare_structs_is (const void *a, const void *b)
{    
    t_v2is *struct_a = (t_v2is *) a;
    t_v2is *struct_b = (t_v2is *) b;
 
    if(struct_a->v < struct_b->v)
    {return 1;}
    else if(struct_a->v > struct_b->v)
    {return -1;}
    else
    {return 0;};
}

int coo2ind(mint x, mint y, mint len)
{
    return x+y*len;
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
