/**********************************************************************************
**	Filename 		: mpi_fox.cpp
**	Authors 		: Manu Kaul and Ahmad Bijairimi
**  Last Modified	: Monday, 26 Apr 2010
**
**  Description		: Parallel Matrix-Matrix Multiplication using Fox's Algorithm
**
**********************************************************************************/
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cmath>

#include <mpi.h>

using namespace std;
/* Constants */
const int 		SEED 				= 171;
const int 		SEED_MAIN 			= 30269;
const double 	SEED_A				= 30268.0;
const double 	SEED_B				= 30240.0;
const int 		MAX_MATRIX_PRINT	= 16;

/* Global Matrix */
double *matrixA, *matrixB, *matrixC;
int 	num_proc;				// Number of processors
int 	n 			= 0;		// Order of Matrices
int 	n_sub 		= 0;		// Order of Sub-matrices
int 	main_rank 	= 0;
double 	start_t 	= 0;
double 	end_t 		= 0;

/* Local Matrix */
double *subA, *subB, *subC;

/* Grid Related Variables */
int num_dim			= 2;					// Number of Grid Dimensions
int dimensions[2] 	= {0,0};				// 2-D Dimensions Array (x,y)
int coordinates[2]	= {0,0};				// (x,y) coordinates for grid topology
int dim_wrapping[2]	= {0,1};				// To do a cyclic shifting in the y dimension
int reorder 		= 1;					// Let virtual topology of grid be optimized to
// the physical topology by MPI for performance.
MPI_Comm grid_comm;							// Communicator for total grid
MPI_Comm row_comm;							// Communicator for Rows (x) in grid
MPI_Comm col_comm;							// Communicator for Columns (y) in grid

/* Function Declarations */
void makeGrid();
void distributeData( int, int, int, int, double *, double * );
void foxMultiply( double, int, double *, double *, double * );
void submatrix_multiply( double *, double *, double * );
void collectData( double, int, double *);

int main(int argc, char* argv[])
{
    int rc;

    /* Startup MPI */
    if( (rc = MPI_Init(&argc, &argv)) != MPI_SUCCESS ) {
        cout << "Error starting MPI program, terminating" << endl;
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    /* Process the command line arguments */
    for(int i = 1; i < argc; ++i ) {
        if (strcmp(argv[i], "-o") == 0 or strcmp(argv[i], "--order") == 0) {
            if( i+1 == argc) {
                printf( "Invalid %s " , argv[i]);
                printf( " Parameter: No order of matrices specified\n");
                MPI_Abort(MPI_COMM_WORLD, rc);
            }
            /* Copy command line argument into matrix order variable n
               which will then automatically be available to all processes */
            n = atoi(argv[++i]);
        }
    }
    if (n == 0)
        MPI_Abort(MPI_COMM_WORLD, rc);

    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &main_rank);

    /* Setup the 2-D Grid */
    makeGrid();
    /* Distribute Data */
    double p = num_proc;
    n_sub = n/sqrt(p);		/* We assume that number of procs is a perfect square and
								the n*n matrices are evenly divisble by sqrt(p)
							*/
    int sublen = n_sub * n_sub;
    int len = n * n;

    /* Allocate space for sub-matrix */
    subA = (double *)calloc( sublen, sizeof(double));
    subB = (double *)calloc( sublen, sizeof(double));

    subC = (double *)calloc( sublen, sizeof(double));
    /* Cache Optimization Ensuring that all items are "touched" in C submatrix */
    for(int i =0; i< sublen; i++ ) subC[i] = 0;

    /* Block Data Distribution */
    if( main_rank == 0) start_t = MPI_Wtime();
    distributeData( main_rank, n_sub, sublen, p, subA, subB );
    /* Fox Multiplication */
    foxMultiply( p, main_rank, subA, subB, subC);

    if( main_rank == 0 )
        matrixC = (double *)calloc( len, sizeof(double));

    /* Gather C matrix back for output */
    collectData( p, main_rank, subC );

    /* Print out the Result Matrix only for order <= 16! */
    if( main_rank == 0 and n <= MAX_MATRIX_PRINT ) {
        printf("\n\n");
        printf("*******************   C  Matrix ********************************** \n\n");
        for( int j = 0; j < len; j++ ) {
            printf("%f ", matrixC[j]);
            if( j>0 and (j+1) % n == 0 ) printf("\n");
        }
    }
    if( main_rank == 0) {
        end_t = MPI_Wtime();
        printf("###### Time spent by rank %d in total ---> %0.6f secs \n", main_rank, end_t - start_t);
    }

    /* Free the Resources */
    free(subA);
    free(subB);
    free(subC);
    if(main_rank == 0) free (matrixC);

    /* Free the MPI Variables */
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&grid_comm);
    MPI_Finalize();

    return 0;
}

/*********************************************************/
void makeGrid ()
{
    int my_rank;				// Rank

    /* Create a virtual 2D-grid topology */
    MPI_Dims_create( num_proc, num_dim, dimensions);
    MPI_Cart_create( MPI_COMM_WORLD, num_dim, dimensions, dim_wrapping, reorder, &grid_comm);
    MPI_Comm_rank( grid_comm, &my_rank);    /* Note: use all_grid because we want the rank in grid */

    /* Create a communicator for each row */
    MPI_Cart_coords( grid_comm, my_rank, num_dim, coordinates);
    MPI_Comm_split(  grid_comm, coordinates[0], coordinates[1], &row_comm);	/* x is the color and y the key */

    /* Create a communicator for each column */
    MPI_Comm_split( grid_comm, coordinates[1], coordinates[0], &col_comm);	/* y is the color and x the key */

    return;
}

/*********************************************************/
void distributeData (int in_rank, int n_sub, int sublen, int p, double *submatrixA, double *submatrixB )
{

    MPI_Datatype blocktype;
    MPI_Status 	 status;
    int seed = SEED;
    int len = n * n;	/* Total number of elements needed for matrix of order n */
    int pos[2] = {0,0};	/* Position for (x,y) coordinates */
    int grid_rank;

    MPI_Request req_send_A, req_send_B, req_recv_A, req_recv_B;

    /* If rank is zero then allocate and make random arrays for A, B */
    if( in_rank == 0 ) {
        /* Allocate (n x n) for two main matrices */
        matrixA = (double *)calloc( len, sizeof(double));
        matrixB = (double *)calloc( len, sizeof(double));

        /* Populate them both with random values */
        for( int i = 0; i < len; i++ ) {
            seed = ( SEED * seed ) % SEED_MAIN;
            matrixA[i] = seed / SEED_A;
            matrixB[i] = seed / SEED_B;
        }
        /* Only print out matrix A and B when order <= 16 ... to not print out very large matrices
           but still allow for testing accuracy with smaller matrices */
        if( n <= MAX_MATRIX_PRINT ) {
            /* Print out the random array for A and B and test if we received this in the grid ok or not */
            printf("*******************   A  Matrix ********************************** \n\n");
            for( int j = 0; j < len; j++ ) {
                printf("%f ", matrixA[j]);
                if( j>0 and (j+1) % n == 0 ) printf("\n");
            }
            printf("\n\n");
            printf("*******************   B  Matrix ********************************** \n\n");
            for( int j = 0; j < len; j++ ) {
                printf("%f ", matrixB[j]);
                if( j>0 and (j+1) % n == 0 ) printf("\n");
            }
        }
        /* Create a new MPI Vector Type to divide the array into blocks to transfer */
        int count		= n_sub;
        int blocklen	= n_sub;
        int stride		= n;
        MPI_Type_vector( count, blocklen, stride, MPI_DOUBLE, &blocktype);
        MPI_Type_commit( &blocktype );
        int block_starter = 0;

        /* In the 2-d grid looping through each grid cell! */
        for(int i =0; i < dimensions[0]; i++ ) {
            for(int j =0; j < dimensions[1]; j++ ) {
                pos[0]=i;
                pos[1]=j;
                /* Get the Rank of the Process to send to! */
                MPI_Cart_rank(grid_comm, pos, &grid_rank);
                /* Bit of manipulation to decide what section of array to send */
                MPI_Isend( matrixA + block_starter, 1, blocktype, grid_rank, 111, grid_comm, &req_send_A );
                MPI_Isend( matrixB + block_starter, 1, blocktype, grid_rank, 222, grid_comm, &req_send_B );

                if( j == dimensions[1]-1 )	// we reached the last block in the row!
                    block_starter = (i + 1) * n * n_sub;
                else
                    block_starter += n_sub;
            }
        }
    }			/* End of if */
    /* All other processes including process 0 will recieve their portions of main matrix into
       their sub matrix */

    /* Receive the block into an array of doubles */
    MPI_Irecv( submatrixA, sublen, MPI_DOUBLE, 0, 111, grid_comm, &req_recv_A );
    MPI_Irecv( submatrixB, sublen, MPI_DOUBLE, 0, 222, grid_comm, &req_recv_B );
    /* Receive Wait */
    MPI_Wait( &req_recv_A, MPI_STATUS_IGNORE );
    MPI_Wait( &req_recv_B, MPI_STATUS_IGNORE );

    if( in_rank == 0 ) {
        free (matrixA);
        free (matrixB);
    }
}
/*********************************************************/
void foxMultiply( double p, int in_rank, double *submatrixA, double *submatrixB, double *submatrixC)
{
    int src;			// source of B block when shifting upwards
    int dst;			// destination of B block when shifting upwards
    int x;				// x-cordinate = row in grid
    int y;				// y-cordinate = col in grid
    int root = 0;
    int len = n_sub * n_sub;
    int grid_order = sqrt(p);	// Order of the 2-D grid : Need this to shift A blocks in row!
    bool ON_DIAGONAL = false;
    MPI_Request req_send_B, req_recv_B;

    double start_t =0, end_t = 0, start_b = 0, end_b = 0;
    /* Get the grid co-ordinates based on processor's rank */
    MPI_Cart_coords( grid_comm, in_rank, 2, coordinates);
    x = coordinates[0];
    y = coordinates[1];

    /* Pre-compute the address for the circular shifting upwards of B matrix's blocks */
    src = (x + 1) % grid_order;
    dst = (x - 1 + grid_order ) % grid_order;		// Done in case we end up with a -ve number for x-1

    /* Local storage for submatrix A which we will broadcast around and
     	storage for submatrix B which we move upwards */
    double *temp_subA = (double *)calloc( len, sizeof(double));
    double *temp_subB = (double *)calloc( len, sizeof(double));

    MPI_Barrier(grid_comm);
    /* K steps to implement Fox's Algorithm.. done in sqrt(p) steps */
    for(int k = 0; k < grid_order; k++ ) {
        /* Phase 1. Broadcast the diagonal blocks at each step calculating the new diagonal */
        root = (x + k) % grid_order;				// Compute the root of the broadcast
        /* When on diagonal --> Broadcast OUTWARDS to all other procs in row */
        if( root == y ) {
            /* Out from root to all others in row */
            MPI_Bcast( submatrixA, len, MPI_DOUBLE, root, row_comm );
            ON_DIAGONAL = true;
        } else {
            /* Have to receive the broadcast INCOMING from the root */
            MPI_Bcast( temp_subA, len, MPI_DOUBLE, root, row_comm );
            ON_DIAGONAL = false;
        }
        MPI_Isend( submatrixB, len, MPI_DOUBLE, dst, 333, col_comm, &req_send_B );
        MPI_Irecv( temp_subB, len, MPI_DOUBLE, src, 333, col_comm, &req_recv_B );

        /* Phase 2. Multiply the sub matrices to get C */
        ( ON_DIAGONAL )? 	submatrix_multiply( submatrixA, submatrixB, submatrixC ) :
        submatrix_multiply( temp_subA,  submatrixB, submatrixC );

        /* Phase 3. Shift cyclically upwards matrix B */
        MPI_Wait( &req_send_B, MPI_STATUS_IGNORE );
        MPI_Wait( &req_recv_B, MPI_STATUS_IGNORE );

        /* Swap the Pointers now so that in the next step we are sending over temp_subB */
        double *tmp;
        tmp 		= submatrixB;
        submatrixB 	= temp_subB;
        temp_subB 	= tmp;

    }	/* End of stages for loop */
    free(temp_subA);
    free(temp_subB);
}		/* End of Fox Multiply */

/*********************************************************/
void collectData( double p, int in_rank, double *subC )
{
    MPI_Datatype blocktype;
    MPI_Request req_send_C, req_recv_C;
    int x=0, y=0;
    int len = n_sub * n_sub;

    /* Create a new MPI Vector Type to divide the array into blocks to transfer */
    int count		= n_sub;
    int blocklen	= n_sub;
    int stride		= n;
    int block_starter = 0;
    MPI_Type_vector( count, blocklen, stride, MPI_DOUBLE, &blocktype);
    MPI_Type_commit( &blocktype );

    /* Send your submatrix C to rank = 0 processor */
    MPI_Isend( subC, len, MPI_DOUBLE, 0, 444, grid_comm, &req_send_C );

    if( in_rank == 0) {
        /* Go through each possible Processor */
        for(int i=0; i < p; i++) {
            /* Probe and see if the incoming source rank is the one we expect */
            MPI_Probe( i, 444, grid_comm, MPI_STATUS_IGNORE );

            /* Get the Cartesian Co-ordinates for the grid rank */
            MPI_Cart_coords( grid_comm, i, 2, coordinates);
            x = coordinates[0];
            y = coordinates[1];

            /* Recieve in the right location in C array now */
            MPI_Recv( &matrixC[ block_starter ], 1, blocktype, i, 444, grid_comm, MPI_STATUS_IGNORE );

            /* Block starter Magic */
            if( y == dimensions[1]-1 )
                block_starter = (x + 1) * n * n_sub;
            else
                block_starter += n_sub;
        }
    }
    /* Wait on the send */
    MPI_Wait( &req_send_C, MPI_STATUS_IGNORE );
}

/*********************************************************/
void submatrix_multiply( double *A /* in */, double *B /* in */, double *C /* out */ )
{
    int i,j,k;
    int C_idx = 0, B_idx = 0;

    for(i=0; i< n_sub; i++) {
        C_idx = i * n_sub;
        for(k=0; k < n_sub; k++) {
            B_idx = k * n_sub;
            for(j=0; j < n_sub; j++)
                C[C_idx + j] += ( A[ C_idx + k ] * B[ B_idx + j ] );
        }
    }
}

