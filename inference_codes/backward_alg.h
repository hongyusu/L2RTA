
#define mint mwSize
#define PICK -100

typedef struct v2i
{
    double v;
    mint nrow;
    mint ncol;
} t_v2i;


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] );
int compare_structs (const void *a, const void *b);


void printm(double * M, mint nrow, mint ncol);

