


/* 
 * forward_alg_omp.c
 *
 * Ver 0.0
 *
 * March 2014
 *
 * Implemented with C OpenMP library of multiple processes.
 * Input:   gradient, K, edge list, number of labels, node degree, maximum node degree
 * Output:  score matrix, pointer matrix
 * Compile into MATLAB function with:
 *      part of compute_topk_omp.c
 *
 * Note:
 *  1. memory lead check on 31.05.2014
 *
 */


#include "matrix.h"
#include "mex.h"
#include "forward_alg_omp.h"
#include "stdio.h"


double * forward_alg_omp ( double * gradient, int K, double * E, int l, double * node_degree, int max_node_degree )
{
    /* output */
    double * P_node;
    double * T_node;
    P_node = (double *) calloc (sizeof(double), K*l*2*(max_node_degree+1));
    T_node = (double *) calloc (sizeof(double), K*l*2*(max_node_degree+1));
    
    /* get output */
    int p;
    int c;
    /* iterate on each edge from leave to node to propagate message */
	int i;
    for( i=l-2; i>=0; i-- )
    {
        /* printm(P_node,mxGetM(OUT_P_node),mxGetN(OUT_P_node)); */
        /* printm(T_node,mxGetM(OUT_P_node),mxGetN(OUT_P_node)); */
        /* if(i==0){break;} */
        /*  parent child index */
        p = E[i];
        c = E[i+(l-1)];
        /* printf("=============== %d (%d->%d)\n",i,p,c); */

        /*  update node score */
        double * in_blk;
        in_blk = (double *) malloc (sizeof(double) * K*max_node_degree);
/*         mxArray * in_blk_array; */
/*         in_blk_array = mxCreateDoubleMatrix(K,max_node_degree,mxREAL); */
/*         double * in_blk = mxGetPr(in_blk_array); */
        /*printm(P_node,20,2*(max_node_degree+1)); */
		int sign_pos;
        for(sign_pos=0;sign_pos<2;sign_pos++)
        {
            /*  assign value to inblock (-1) */
			int ii,jj;
            for(ii=0;ii<K;ii++)
            {
                int start_col=3-1+sign_pos; /*  start from the 3rd column */
                for(jj=0;jj<max_node_degree;jj++)
                {
                    in_blk[ii+jj*(K)]=P_node[((c-1)*K+ii)+(start_col+jj*2)*(K*l)];
                }
                start_col++;
            }
            /*printf("in block\n");printm(in_blk,K,max_node_degree); 
            /*  compute topk for inblock -1 */
            /*printm(in_blk,K,max_node_degree);
            double * tmp_res = LinearMaxSum(in_blk,K,max_node_degree,node_degree[c-1]);
            /*printf("out block\n");printm(tmp_res,K,max_node_degree+1); 
            /*  assign value to P_node T_node */
            for(ii=0;ii<K;ii++)
            {
                /* printf("position %d %d\n",ii+(c-1)*K+sign_pos*(K*l),ii+K*max_node_degree); */
                P_node[ii+(c-1)*K+sign_pos*(K*l)] = tmp_res[ii+K*(max_node_degree)];
                int start_col = 3-1+sign_pos;
                for(jj=0;jj<max_node_degree;jj++)
                {
                    /* printf("-->%d %d\n",ii+c*K+(start_col+jj*2)*(K*l),ii+jj*K); */
                    T_node[ii+(c-1)*K+(start_col+jj*2)*(K*l)] = tmp_res[ii+jj*K];
                }
                start_col++;
            }
            free(tmp_res);
        }
        /*  combine edge potential */
        double * M;
        M = (double *) malloc (sizeof(double) * 2*K*2);
/*         mxArray * M_array; */
/*         M_array = mxCreateDoubleMatrix(2*K,2,mxREAL); */
/*         double * M = mxGetPr(M_array); */
		int ii,jj;
        for(ii=0;ii<2*K;ii++)
        {
            for(jj=0;jj<2;jj++)
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
		int j;
        for(j=i;j<l-1;j++)
        {
            if(E[j] == p)
            {c_in_p++;}
            else
            {break;}
        }
        t_v2i * tmp_M;
        tmp_M = (struct v2i *) malloc((sizeof(t_v2i))*2*K);
		int sign_pos,ii;
        for(sign_pos=0;sign_pos<2;sign_pos++)
        {
            for(ii=0;ii<2*K;ii++)
            {
                tmp_M[ii].v = M[ii+sign_pos*2*K];
                /* printf("%d\n",ii+sign_pos*2*K); */
                tmp_M[ii].i = ii+1;
            }
            qsort(tmp_M, 2*K, sizeof(t_v2i), (void *)compare_structs);
            /*  put into correct block */
            for(ii=0;ii<K;ii++)
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
        free(M);
        free(in_blk);
        

    }
    /* one more iteration on root node */
    /*  update node score */
    c=p;
    double * in_blk;
    in_blk = (double *) malloc (sizeof(double) * K * max_node_degree);
/*     mxArray * in_blk_array; */
/*     in_blk_array = mxCreateDoubleMatrix(K,max_node_degree,mxREAL); */
/*     double * in_blk = mxGetPr(in_blk_array); */
	int sign_pos,ii,jj;
    for(int sign_pos=0;sign_pos<2;sign_pos++)
    {
        /*  assign value to inblock (-1) */
        for(ii=0;ii<K;ii++)
        {
            int start_col=3-1+sign_pos; /*  start from the 3rd column */
            for(jj=0;jj<max_node_degree;jj++)
            {
                in_blk[ii+jj*(K)]=P_node[((c-1)*K+ii)+(start_col+jj*2)*(K*l)];
            }
            start_col++;
        }
        /*printm(P_node,20,2*(max_node_degree+1)); */
        /*printf("--in block\n");printm(in_blk,K,max_node_degree);  */
        /*  compute topk for inblock -1  */
        double * tmp_res = LinearMaxSum(in_blk, K, max_node_degree,node_degree[c-1]+1);
        /*printf("--out block\n");printm(tmp_res,K,5);  */
        /*  assign value to P_node T_node */
        for(ii=0;ii<K;ii++)
        {
            /* printf("position %d %d\n",ii+(c-1)*K+sign_pos*(K*l),ii+K*max_node_degree); */
            P_node[ii+(c-1)*K+sign_pos*(K*l)] = tmp_res[ii+K*(max_node_degree)];
            int start_col = 3-1+sign_pos;
            for(jj=0;jj<max_node_degree;jj++)
            {
                /* printf("-->%d %d\n",ii+c*K+(start_col+jj*2)*(K*l),ii+jj*K); */
                T_node[ii+(c-1)*K+(start_col+jj*2)*(K*l)] = tmp_res[ii+jj*K];
            }
            T_node[ii+(c-1)*K+(start_col-2)*(K*l)] = ii+1+(start_col-2)*K;
            start_col++;
        }
        free(tmp_res);
    }
    free(in_blk);
    
    /* SEND BACK RESULTS */
    double * results;
    results = (double *) malloc (sizeof(double) * 2 * K*l*2*(max_node_degree+1)*2);
    for(int ii=0;ii<K*l;ii++)
    {
        for(int jj=0;jj<2*(max_node_degree+1);jj++)
        {
            results[ii+jj*2*K*l] = P_node[ii+jj*K*l];
        }
    }
    for(int ii=0;ii<K*l;ii++)
    {
        for(int jj=0;jj<2*(max_node_degree+1);jj++)
        {
            results[ii+K*l+jj*2*K*l] = T_node[ii+jj*K*l];
        }
    }
    /*printm(P_node,16,4); */
    /*printm(T_node,16,4); */
    free(T_node);
    free(P_node);
    return(results);
}


double * LinearMaxSum(double * M, int M_nrow, int M_ncol, int current_node_degree)
{
    double * res;
    res = (double *) calloc (sizeof(double), M_nrow * (M_ncol+1));

    /* NO CHILDREN */
    if(current_node_degree-1==0)
    {
        res[0+M_ncol*M_nrow]=1;
        return res;
    }
    
    /*if(current_node_degree>3){printf("M\n");printm(M,M_nrow,M_ncol); }  */
    
    /*  INITIALIZE TMP_M WITH FIRST COLUMN OF M */
    t_v2is * tmp_M;
    tmp_M = (struct v2is *) malloc((sizeof(t_v2is))*M_nrow);
	int ii;
    for(ii=0;ii<M_nrow;ii++)
    {
        tmp_M[ii].v=0;
        tmp_M[ii].i=(int *)malloc(sizeof(int)*(current_node_degree-1));
        tmp_M[ii].i[0]=0;
        if(M[ii]>0)
        {
            tmp_M[ii].v=M[ii];
            tmp_M[ii].i[0]=ii+1;
        }
    }
    
    /*if(current_node_degree>3){for(int jj=0;jj<M_nrow;jj++){printf("INIT tmp_M %.4f %d %d %d\n",tmp_M[jj].v,tmp_M[jj].i[0],tmp_M[jj].i[1],tmp_M[jj].i[2]);} } */
    
    /*  PROCESSING FROM 2nd COLUMN */
    for(ii=1;ii<(current_node_degree-1);ii++)
    {
        /* VARIABLE TO STORE THE RESULTS */
        t_v2is * tmp_M_long;
        tmp_M_long = (struct v2is *) malloc((sizeof(t_v2is))*M_nrow);
        
        /* FIRST ELEMENT IS ALWAYS 1,1 */
        struct type_heap_array * heap_array;
        struct type_heap_array * heap_array_pt;
        struct type_heap_array * heap_array_element;
        heap_array = (struct type_heap_array *) malloc (sizeof(struct type_heap_array));
        heap_array->v = tmp_M[0].v;
        heap_array->x = 0;
        heap_array->y = -1;
        heap_array->next = NULL;
        if(M[ii*M_nrow]>0)
        {
            heap_array->v += M[ii*M_nrow];
            heap_array->y=0;
        }
        /* POP, ADD, REMOVE */
        int n_element = 0;
        while(n_element<M_nrow)
        {
            /* IF EMPTY */
            if(!heap_array)
            {
                tmp_M_long[n_element].v=0;
                tmp_M_long[n_element].i = (int *)malloc(sizeof(int)*(current_node_degree-1)); 
				int jj;
                for(jj=0;jj<ii;jj++)
                {tmp_M_long[n_element].i[jj] = 0;}
                tmp_M_long[n_element].i[ii] = 0;
                n_element++;
                continue;
            }
            /* NOT EMPTY */
            /* POP UP FIRST ELEMENT */
            tmp_M_long[n_element].v = heap_array->v;
            tmp_M_long[n_element].i = (int *)malloc(sizeof(int)*(current_node_degree-1)); 
			int jj;
            for(jj=0;jj<ii;jj++)
            {tmp_M_long[n_element].i[jj] = tmp_M[heap_array->x].i[jj];}
            tmp_M_long[n_element].i[ii] = heap_array->y+1;
            
            /*if(current_node_degree>3){printf("\t--pop %.2f %d %d\n",heap_array->v,heap_array->x+1,heap_array->y+1);} */
                
            /* ADD TWO NEW ELEMENTS */
            for(jj=0;jj<2;jj++)
            {
                int new_x,new_y;
                new_x = heap_array->x;
                new_y = heap_array->y;
                if(jj==0)
                {new_y += 1;}
                if(jj==1)
                {new_x += 1;}
                /* OUT OF RANGE */
                if(new_x>=M_nrow || new_y>=M_nrow)
                {
                    /*printf("\t-----range: \n"); */
                    continue;
                }
                /* NON-EXIST */
                if(tmp_M[new_x].v<=0 || M[new_y+ii*M_nrow]<=0)
                {
                    /*printf("\t-----exist: %d %d %.2f %.2f \n",new_x+1,new_y+1,tmp_M[new_x].v,M[new_y+ii*M_nrow]); */
                    continue;
                }
                /* GET NEW PAIR */
                heap_array_element = (struct type_heap_array *) malloc (sizeof(struct type_heap_array));
                heap_array_element->v = tmp_M[new_x].v + M[new_y+ii*M_nrow];
                heap_array_element->x = new_x;
                heap_array_element->y = new_y;
                heap_array_element->next = NULL;
                /*if(current_node_degree>3){printf("\t-----add %.2f %d %d\n",heap_array_element->v,heap_array_element->x+1,heap_array_element->y+1);} */
                /* PUT PAIR INTO HEAP ARRAY */
                heap_array_pt = heap_array;
                int overlap=0;
                while(heap_array_pt->next)
                {
                    if((heap_array_pt->next)->v < heap_array_element->v)
                    {break;}
                    if(heap_array_pt->x==heap_array_element->x && heap_array_pt->y==heap_array_element->y)
                    {
                        overlap=1;
                        break;
                    }
                    heap_array_pt = heap_array_pt->next;
                }
                if(heap_array_pt->x==heap_array_element->x && heap_array_pt->y==heap_array_element->y)
                {overlap=1;}
                if(overlap){free(heap_array_element);continue;}
                heap_array_element->next = heap_array_pt->next;
                heap_array_pt->next = heap_array_element;
/*                 if(heap_array_pt->next) */
/*                 { */
/*                     heap_array_element->next=(struct type_heap_array *) malloc (sizeof(struct type_heap_array)); */
/*                     heap_array_element->next = heap_array_pt->next; */
/*                 } */
/*                 else */
/*                 { */
/*                     heap_array_pt->next=(struct type_heap_array *) malloc (sizeof(struct type_heap_array)); */
/*                 } */
/*                 heap_array_pt->next = heap_array_element; */
            }
            /* destroy the first element */
            heap_array_pt = heap_array;
            heap_array = heap_array->next;
            heap_array_pt->next=NULL;
            free(heap_array_pt);
            n_element++;
        }
        /* DESTROY THE REST HEAP ARRAY */
        while(heap_array)
        {
            heap_array_pt = heap_array;
            heap_array = heap_array->next;
            heap_array_pt->next=NULL;
            free(heap_array_pt);
        }
        /* SAVE RESULTS */
		int jj;
        for(jj=0;jj<M_nrow;jj++)
        {
            tmp_M[jj].v = tmp_M_long[jj].v;
            /* COULD USE DEEP COPY */
            for(int kk=0;kk<ii+1;kk++)
            {tmp_M[jj].i[kk] = tmp_M_long[jj].i[kk];}
            /* printf("%d tmp_M %.4f %d %d %d\n",ii,tmp_M[jj].v,tmp_M[jj].i[0],tmp_M[jj].i[1],tmp_M[jj].i[2]); */
        }
        /* DESTROY tmp_M_long */
		int ii;
        for(ii=0; ii<M_nrow; ii++)
        {free(tmp_M_long[ii].i);}
        free(tmp_M_long);
        
    }
    
    /*for(int jj=0;jj<M_nrow;jj++){printf("res:tmp_M %.4f %d %d %d\n",tmp_M[jj].v,tmp_M[jj].i[0],tmp_M[jj].i[1],tmp_M[jj].i[2]);}  */
    /*  COLLECT RESULTS */
	int ii;
    for(ii=0;ii<M_nrow;ii++)
    {
        if(tmp_M[ii].v>0)
        {res[ii+M_nrow*M_ncol]=tmp_M[ii].v+1;}
        else
        {res[ii+M_nrow*M_ncol]=tmp_M[ii].v;}
		int jj;
        for(jj=0;jj<current_node_degree-1;jj++)
        {res[ii+M_nrow*jj]=tmp_M[ii].i[jj];}
    }
    /* DESTROY TMP_M */
    for(ii=0; ii<M_nrow; ii++)
    {free(tmp_M[ii].i);}
    free(tmp_M);
    /*  RETURN RESULT */
/*     printf("--->%d\n",current_node_degree); */
/*     printm(M,M_nrow,M_ncol); */
/*     printm(res,M_nrow,M_ncol+1); */
    return res;
    /*printf("--->stop\n"); */
}


int compare_structs ( const void *a, const void *b )
{    
    t_v2i *struct_a = (t_v2i *) a;
    t_v2i *struct_b = (t_v2i *) b;
 
    if(struct_a->v < struct_b->v)
    {return 1;}
    else if(struct_a->v > struct_b->v)
    {return -1;}
    else
    /*{return 0;}; */
    {
        if(struct_a->i > struct_b->i)
        {return 1;}
        else if(struct_a->i < struct_b->i)
        {return -1;}
        else
        {return 0;}
    }
}

int compare_structs_is ( const void *a, const void *b )
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



void printm ( double * M, int nrow, int ncol )
{
    printf("#row: %d #ncol %d\n", nrow,ncol);
	int i,j;
    for(i=0; i<nrow; i++)
    {
        for(j=0; j<ncol; j++)
        {
            printf("%.4f ", M[i+j*nrow]);
        }
        printf("\n");
    }
    printf("\n");
}
