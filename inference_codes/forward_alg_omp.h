




typedef struct v2i
{
    double v;
    int i;
} t_v2i;
    
typedef struct v2is
{
    double v;
    int *i;
} t_v2is;


struct type_heap_array
{
   double v;
   int x;
   int y;
   struct type_heap_array * next;
};

void        printm              ( double * M, int nrow, int ncol );
double *    LinearMaxSum        ( double * M, int M_nrow, int M_ncol, int current_node_degree );
int         compare_structs     ( const void *a, const void *b );
int         compare_structs_is  ( const void *a, const void *b );
double *    forward_alg_omp     ( double * gradient, int K, double * E, int l, double * node_degree, int max_node_degree );