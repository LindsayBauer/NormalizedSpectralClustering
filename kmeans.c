#define PY_SSIZE_T_CLEAN
#include <Python.h>

/*
This module receives initialized centroids and vectors from kmeans_pp.py and executes the 
K-means algorithm in order to assign each vector to one of the centroids. 
*/

double distance(int d, double *p, double *q);
void swapElements(int *arr1, int *arr2, int n);
void swapDimensions(double *arr1, double *arr2, int n);

/* Distance function. Input: two vectors, Output: a double of the distance between them. */
double distance(int d, double *p, double *q){
    double sum = 0;
    int i;
    for (i = 0; i < d; i++){
        sum += (*(p + i) - *(q + i)) * (*(p + i) - *(q + i));
    }
    return sum;
}

/* Function to swap the elements of two arrays of ints */
void swapElements(int *arr1, int *arr2, int n){
    int i, temp;
    temp = 0;

    for (i = 0; i < n; i++){
        temp = arr1[i];
        arr1[i] = arr2[i];
        arr2[i] = temp;
    }
}

/* Function to swap elements of two arrays of doubles */
void swapDimensions(double *arr1, double *arr2, int dimension_num){
    int i;
    double temp;
    temp = 0.0;

    for (i = 0; i < dimension_num; i++){
        temp = arr1[i];
        arr1[i] = arr2[i];
        arr2[i] = temp;
    }
}

/* Function to terminate the program due to inadequate memory allocation. */
void terminate(){
    printf("Memory could not be allocated");
    exit(0);
}

/* Initialize centroids array- k pointers to the k initial centroids passed from the python code
     (each pointer points to an array of doubles of length d)
     Insert the centroids that were passed to the method */
void initialize_centroids(double **centroids, int K, int d, double **input_vectors_c, int *initial_centroids_indexes_c) {
    int t, s;
    for (t = 0; t < K; t++){
        centroids[t] = (double *)malloc(d * sizeof(double));
        for (s = 0; s < d; s++){
            centroids[t][s] = input_vectors_c[initial_centroids_indexes_c[t]][s];
        }
    }
}

/* Initialize clusters array- N pointers to ints, each index i holds the index j of the cluster that the vector in index i belongs to.
 Initialize the indexes of the initial centroids to be their indexes, and the rest to be -1 */
void initialize_clusters(int *clusters, int K, int N, int *initial_centroids_indexes_c){
    int s, t;
    for (t = 0; t < N; t++){
        clusters[t] = -1;
    }
    for (s = 0; s < K; s++){
        clusters[initial_centroids_indexes_c[s]] = s;
    }
}

void initialize_cluster_sums_and_sizes(double **curr_cluster_sums, int K, int d, int *curr_cluster_sizes){
    /* Initialize the cluster sums and sizes to be 0 */
    int s, t;
    for (s = 0; s < K; s++){
        curr_cluster_sums[s] = (double *)malloc(d * sizeof(double));
        if (curr_cluster_sums[s] == NULL) { terminate();}
        for (t = 0; t < d; t++){
            curr_cluster_sums[s][t] = 0;
        }
        curr_cluster_sizes[s] = 0;
    }
}

/* For vector in index i in input_vectors - find the closest centroid (j), change the curr_clusters[i]
    to be j, add 1 to the curr_cluster_sizes[j], add input_vectors[i] to curr_cluster_sums[j] one by one */
void find_and_update_closest_cluster(int N, int K, int d, double **input_vectors_c,
    double **centroids, int *curr_clusters, int *curr_cluster_sizes, double **curr_cluster_sums){
    int i, j;
    double dist;
    for (i = 0; i < N; i++){
        int closest_cluster = 0;
        double smallest_dist = -1.0;
        for (j = 0; j < K; j++){
            dist = distance(d, input_vectors_c[i], centroids[j]);
            if (smallest_dist < 0 || dist < smallest_dist){
                closest_cluster = j;
                smallest_dist = dist;
            }
        }
        curr_clusters[i] = closest_cluster;
        curr_cluster_sizes[closest_cluster] += 1;
        /* Add to total sum of cluster closest_cluster */
        for (j = 0; j < d; j++){
            curr_cluster_sums[closest_cluster][j] += input_vectors_c[i][j];
        }
    }
}

/* Check if any changes were made (if the curr_clusters are the same as the clusters) */
int clusters_are_the_same(int *clusters, int *curr_clusters, int N){
    int the_same = 1;
    int i;
    for (i = 0; i < N; i++){
        if (curr_clusters[i] != clusters[i]){
            the_same = 0;
            break;
        }
    }
    return the_same;
}

/* Free up memory allocated for centroids and clusters data */
void free_memory(double **curr_cluster_sums, double **centroids, int *curr_cluster_sizes, int *curr_clusters, int K){
    int i;
    for (i = 0; i < K; i++){
        free(curr_cluster_sums[i]);
        free(centroids[i]);
    }
    free(curr_cluster_sums);
    free(centroids);
    free(curr_cluster_sizes);
    free(curr_clusters);
}

static int* kmeans(int K, int N, int d, int MAX_ITER, double **input_vectors_c, int *initial_centroids_indexes_c){
    int i, j, num_of_iterations, the_same;
    int *clusters;
    double **centroids;

    centroids = (double **)malloc(K * sizeof(double *));
    if (centroids == NULL) { terminate();}
    initialize_centroids(centroids, K, d, input_vectors_c, initial_centroids_indexes_c); 
    clusters = (int *)malloc(N * sizeof(int));
    if (clusters == NULL) { terminate();}
    initialize_clusters(clusters, K, N, initial_centroids_indexes_c);

    /* Iterations */
    num_of_iterations = 0;
    while (num_of_iterations < MAX_ITER){
        int *curr_clusters;         /* Index i holds the index of the cluster that vector i belongs to */
        int *curr_cluster_sizes;    /* Number of vectors in each cluster */
        double **curr_cluster_sums; /* Sum of vectors in each cluster */

        curr_clusters = (int *)malloc(N * sizeof(int));
        if (curr_clusters == NULL) { terminate();}
        curr_cluster_sizes = (int *)malloc(K * sizeof(int));
        if (curr_cluster_sizes == NULL) { terminate();}
        curr_cluster_sums = (double **)malloc(K * sizeof(double *));
        if (curr_cluster_sums == NULL) { terminate();}

        initialize_cluster_sums_and_sizes(curr_cluster_sums, K, d, curr_cluster_sizes);
        find_and_update_closest_cluster(N,K,d,input_vectors_c,centroids,curr_clusters,curr_cluster_sizes,curr_cluster_sums);

        /* Divide the curr_cluster_sums by the curr_cluster_sizes, making curr_cluster_sums into the current centroids */
        for (i = 0; i < K; i++){
            for (j = 0; j < d; j++){
                curr_cluster_sums[i][j] = curr_cluster_sums[i][j] / curr_cluster_sizes[i];
            }
        }

        the_same = clusters_are_the_same(curr_clusters, clusters, N); 
        if (the_same == 1 || num_of_iterations == MAX_ITER - 1){
            free_memory(curr_cluster_sums, centroids, curr_cluster_sizes, curr_clusters, K);
            return clusters;
        }

        else{
            swapElements(clusters, curr_clusters, N);
            for (i = 0; i < K; i++){
                /* curr_cluster_sums are currently the centroids */
                swapDimensions(centroids[i], curr_cluster_sums[i], d); 
            }
            for (i = 0; i < K; i++){
                free(curr_cluster_sums[i]);
            }
            free(curr_cluster_sums);
            free(curr_cluster_sizes);
            free(curr_clusters);
            num_of_iterations++;
        }
    }
    return clusters;
}

static PyObject *kmeans_capi(PyObject *self, PyObject *args){
    int K, N, d, MAX_ITER, i, j;
    PyObject *input_vectors_py, *initial_centroids_indexes_py, *centroid_index, *vect, *dim;
    double **input_vectors_c;         /* The vectors read from Input File */
    int *initial_centroids_indexes_c; /* Indexes of K Initial Centroids */

    /* This parses the Python arguments into four int (i) variables named K,N,d, MAX_ITER*/
    if (!PyArg_ParseTuple(args, "iiiiOO;wrong input parameters!", &K, &N, &d, &MAX_ITER, &input_vectors_py, &initial_centroids_indexes_py)){
        return NULL; /* NULL is an invalid PyObject* in CAPI, so it is used to signal an error */
    }

    if (!PyList_Check(input_vectors_py)){
        return NULL;
    }

    /* Parses input_vectors_py from a PyObject to a list of lists of doubles */
    input_vectors_c = (double **)malloc(N * sizeof(double *));
    for (i = 0; i < N; i++){
        input_vectors_c[i] = (double *)malloc(d * sizeof(double));
    }

    for (i = 0; i < N; i++){
        vect = PyList_GetItem(input_vectors_py, i);
        for (j = 0; j < d; j++){
            dim = PyList_GetItem(vect, j);
            input_vectors_c[i][j] = PyFloat_AsDouble(dim);
        }
    }

    /* Parse the initial centroids indexes from a PyObject to an array of longs */
    initial_centroids_indexes_c = (int *)malloc(K * sizeof(int));
    for (i = 0; i < K; i++){
        centroid_index = PyList_GetItem(initial_centroids_indexes_py, i);
        initial_centroids_indexes_c[i] = PyLong_AsLong(centroid_index);
    }

    int *N_sized_cluster_membership = kmeans(K, N, d, MAX_ITER, input_vectors_c, initial_centroids_indexes_c);
    PyObject *py_cluster_num_of_observation = PyList_New(N);
    for (j = 0; j < N; j++){
        PyObject *py_int = Py_BuildValue("i", N_sized_cluster_membership[j]);
        PyList_SetItem(py_cluster_num_of_observation, j, py_int);
    }

    free(N_sized_cluster_membership);
    for (j = 0; j < N; j++){
        free(input_vectors_c[j]);
    }
    free(input_vectors_c);
    free(initial_centroids_indexes_c);
    return py_cluster_num_of_observation;
}

/*
 * This array tells Python what methods this module has.
 * We will use it in the next structure
 */
static PyMethodDef capiMethods[] = {
    {"Ckmeans",                                                                              /* the Python method name that will be used */
     (PyCFunction)kmeans_capi,                                                               /* the C-function that implements the Python function and returns static PyObject*  */
     METH_VARARGS,                                                                           /* flags indicating parametersaccepted for this function */
     PyDoc_STR("Calculates the kmeans and prints the centroids and the initial centroids")}, /*  The docstring for the function */
    {NULL, NULL, 0, NULL}                                                                    /* The last entry must be all NULL as shown to act as a
                                 sentinel. Python looks for this entry to know that all
                                 of the functions for the module have been defined. */
};

/* This initiates the module using the above definitions. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp", /* name of module */
    NULL,         /* module documentation, may be NULL */
    -1,           /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    capiMethods   /* the PyMethodDef array from before containing the methods of the extension */
};

/*
 * The PyModuleDef structure, in turn, must be passed to the interpreter in the module’s initialization function.
 * The initialization function must be named PyInit_name(), where name is the name of the module and should match
 * what we wrote in struct PyModuleDef.
 * This should be the only non-static item defined in the module file
 */
PyMODINIT_FUNC
PyInit_mykmeanssp(void){
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m){
        return NULL;
    }
    return m;
}
