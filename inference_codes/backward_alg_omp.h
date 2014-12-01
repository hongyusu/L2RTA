

#define PICK -100

typedef struct back_v2i
{
    double v;
    int nrow;
    int ncol;
} back_t_v2i;


double *    backward_alg_omp  ( double * P_node, double * T_node, int K, double * E, int nlabel, double * in_node_degree, int max_node_degree );
int         b_compare_structs   ( const void *a, const void *b );
void        b_print_mat         ( double * M, int nrow, int ncol );

