


#define mint mwSize

typedef struct v2i
{
    double v;
    mint i;
} t_v2i;
    
typedef struct v2is
{
    double v;
    mint *i;
} t_v2is;


struct type_heap_array
{
   double v;
   mint x;
   mint y;
   struct type_heap_array * next;
};

void printm(double * M, mint nrow, mint ncol);
int coo2ind(mwSize x, mwSize, mwSize len);
double * LinearMaxSum(mxArray * M, mint current_node_degree);
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] );
int compare_structs (const void *a, const void *b);
int compare_structs_is (const void *a, const void *b);