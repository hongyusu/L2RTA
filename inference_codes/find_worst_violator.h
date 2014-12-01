




struct type_element_list
{
   double id;
   double val;
   struct type_element_list * next;
};

struct type_arr2id_list
{
   double * arr;
   double id;
   struct type_arr2id_list * next;
};

void    printm          ( double * M, int nrow, int ncol );
void    mexFunction     ( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] );
double  Y2Fy            ( double *Y, double * E, double * gradient, double nlabel );
int     sortcompare     (const void * a, const void * b);
