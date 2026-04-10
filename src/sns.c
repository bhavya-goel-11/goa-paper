#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- 1. Macro-Driven CEC Wrapper ---
#if CEC_YEAR == 2014
    void cec14_test_func(double *x, double *f, int nx, int mx, int func_num);
    double evaluate(double *solution, int dim, int func_num) {
        double fitness = 0.0;
        cec14_test_func(solution, &fitness, dim, 1, func_num);
        return fitness;
    }
    #define NUM_FUNCS 30
#elif CEC_YEAR == 2017
    void cec17_test_func(double *x, double *f, int nx, int mx, int func_num);
    double evaluate(double *solution, int dim, int func_num) {
        double fitness = 0.0;
        cec17_test_func(solution, &fitness, dim, 1, func_num);
        return fitness;
    }
    #define NUM_FUNCS 30
#elif CEC_YEAR == 2020
    void cec20_test_func(double *x, double *f, int nx, int mx, int func_num);
    double evaluate(double *solution, int dim, int func_num) {
        double fitness = 0.0;
        cec20_test_func(solution, &fitness, dim, 1, func_num);
        return fitness;
    }
    #define NUM_FUNCS 10
#elif CEC_YEAR == 2022
    void cec22_test_func(double *x, double *f, int nx, int mx, int func_num);
    double evaluate(double *solution, int dim, int func_num) {
        double fitness = 0.0;
        cec22_test_func(solution, &fitness, dim, 1, func_num);
        return fitness;
    }
    #define NUM_FUNCS 12
#elif CEC_YEAR == 9999
    void eng_test_func(double *x, double *f, int nx, int mx, int func_num);
    void get_eng_bounds(int func_num, int *dim, double *lb, double *ub);
    double evaluate(double *solution, int dim, int func_num) {
        double fitness = 0.0;
        eng_test_func(solution, &fitness, dim, 1, func_num);
        return fitness;
    }
    #define NUM_FUNCS 5
#else
    #error "CRITICAL: You must define a valid CEC_YEAR during compilation"
#endif

// --- 2. Shared Math Utilities ---
double rand_01() { return (double)rand() / (double)RAND_MAX; }

// Function to shuffle an array (used for Mood 3 Disputation group selection)
void shuffle(int *array, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

// --- 3. Social Network Search (SNS) Algorithm ---
double run_sns(int pop_size, int dim, int max_iter, double *LB, double *UB, int func_num, double *convergence_history) {
    
    double **x = (double **)malloc(pop_size * sizeof(double *));
    double *fitness = (double *)malloc(pop_size * sizeof(double));
    double *gBest_pos = (double *)malloc(dim * sizeof(double));
    double gBest_fit = INFINITY;

    // Initialization
    for (int i = 0; i < pop_size; i++) {
        x[i] = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) {
            x[i][j] = LB[j] + rand_01() * (UB[j] - LB[j]);
        }
        fitness[i] = evaluate(x[i], dim, func_num);
        
        if (fitness[i] < gBest_fit) {
            gBest_fit = fitness[i];
            for (int j = 0; j < dim; j++) gBest_pos[j] = x[i][j];
        }
    }

    // Pre-allocate arrays for Mood selections
    double *nx = (double *)malloc(dim * sizeof(double));
    int *indices = (int *)malloc(pop_size * sizeof(int));
    double *M = (double *)malloc(dim * sizeof(double));

    // Main SNS Loop
    for (int iter = 1; iter <= max_iter; iter++) {
        
        for (int i = 0; i < pop_size; i++) {
            
            // Randomly select Mood (1 to 4)
            int mood = (rand() % 4) + 1;
            
            // Select random user 'j' (not equal to 'i')
            int j;
            do { j = rand() % pop_size; } while (j == i);
            
            if (mood == 1) { // Mood 1: Imitation
                for (int d = 0; d < dim; d++) {
                    double r = x[j][d] - x[i][d];
                    double R = rand_01() * r;
                    nx[d] = x[j][d] + (1.0 - 2.0 * rand_01()) * R;
                }
            } 
            else if (mood == 2) { // Mood 2: Conversation
                // Select random user 'k' (not equal to i or j)
                int k;
                do { k = rand() % pop_size; } while (k == i || k == j);
                
                double sign_val = (fitness[i] > fitness[j]) ? 1.0 : ((fitness[i] < fitness[j]) ? -1.0 : 0.0);
                
                for (int d = 0; d < dim; d++) {
                    double D = sign_val * (x[j][d] - x[i][d]);
                    nx[d] = x[k][d] + rand_01() * D;
                }
            } 
            else if (mood == 3) { // Mood 3: Disputation
                for (int idx = 0; idx < pop_size; idx++) indices[idx] = idx;
                shuffle(indices, pop_size);
                int group_size = (rand() % pop_size) + 1;
                
                // Calculate Mean (M) of the random group
                for (int d = 0; d < dim; d++) M[d] = 0.0;
                for (int g = 0; g < group_size; g++) {
                    for (int d = 0; d < dim; d++) {
                        M[d] += x[indices[g]][d];
                    }
                }
                for (int d = 0; d < dim; d++) M[d] /= (double)group_size;
                
                int AF = (rand() % 2) + 1; // 1 or 2
                
                for (int d = 0; d < dim; d++) {
                    nx[d] = x[i][d] + rand_01() * (M[d] - (double)AF * x[i][d]);
                }
            } 
            else { // Mood 4: Innovation
                for (int d = 0; d < dim; d++) nx[d] = x[i][d]; // Copy current user
                
                int select = rand() % dim; // Pick one dimension
                double t = rand_01();
                double event = LB[select] + rand_01() * (UB[select] - LB[select]);
                
                nx[select] = t * event + (1.0 - t) * x[j][select];
            }
            
            // Array Boundary Clamping
            for (int d = 0; d < dim; d++) {
                if (nx[d] > UB[d]) nx[d] = UB[d];
                if (nx[d] < LB[d]) nx[d] = LB[d];
            }
            
            // Evaluate and Update
            double n_fit = evaluate(nx, dim, func_num);
            if (n_fit < fitness[i]) {
                fitness[i] = n_fit;
                for (int d = 0; d < dim; d++) x[i][d] = nx[d];
                
                if (n_fit < gBest_fit) {
                    gBest_fit = n_fit;
                    for (int d = 0; d < dim; d++) gBest_pos[d] = nx[d];
                }
            }
        }
        convergence_history[iter - 1] = gBest_fit;
    }
    
    // Free Memory
    for (int i = 0; i < pop_size; i++) free(x[i]);
    free(x); free(fitness); free(gBest_pos);
    free(nx); free(indices); free(M);
    
    return gBest_fit;
}

// --- 4. The Isolated Test Harness ---
int main() {
    srand(time(NULL)); 
    
    char algo_name[] = "sns"; 
    int pop_size = 50, max_iter = 1200, num_runs = 50; 
    
    printf("--- STARTING BENCHMARK: %s | SUITE: %d ---\n", algo_name, CEC_YEAR);
    
    char sum_path[200], raw_path[200], conv_path[200];
    if (CEC_YEAR == 9999) {
        sprintf(sum_path, "../results/%s/engineering/summary.csv", algo_name);
        sprintf(raw_path, "../results/%s/engineering/raw.csv", algo_name);
        sprintf(conv_path, "../results/%s/engineering/convergence.csv", algo_name);
    } else {
        sprintf(sum_path, "../results/%s/cec%d/summary.csv", algo_name, CEC_YEAR);
        sprintf(raw_path, "../results/%s/cec%d/raw.csv", algo_name, CEC_YEAR);
        sprintf(conv_path, "../results/%s/cec%d/convergence.csv", algo_name, CEC_YEAR);
    }
    
    FILE *fp_summary = fopen(sum_path, "w");
    FILE *fp_raw = fopen(raw_path, "w");
    FILE *fp_conv = fopen(conv_path, "w");
    
    if (!fp_summary || !fp_raw || !fp_conv) {
        printf("ERROR: Could not open folders! Did you run 'mkdir -p ../results/%s/...'?\n", algo_name);
        return 1;
    }
    
    fprintf(fp_summary, "Function,Best,Worst,Mean,StdDev\n");
    
    for (int func_num = 1; func_num <= NUM_FUNCS; func_num++) {
        if (CEC_YEAR == 2017 && func_num == 2) continue; 
        
        printf("  -> Optimizing Function F%d...\n", func_num);
        
        // --- DYNAMIC BOUNDARY & DIMENSION ALLOCATION ---
        int dim; 
        double LB[100], UB[100]; 
        
        #if CEC_YEAR == 9999
            get_eng_bounds(func_num, &dim, LB, UB);
        #else
            #if CEC_YEAR == 2020 || CEC_YEAR == 2022
                dim = 20; 
            #else
                dim = 30; 
            #endif
            for(int i = 0; i < dim; i++) { LB[i] = -100.0; UB[i] = 100.0; }
        #endif
        
        double raw_results[50]; 
        double mean_history[1200] = {0};
        double sum = 0.0, best = INFINITY, worst = -INFINITY;
        
        fprintf(fp_raw, "F%d", func_num);
        fprintf(fp_conv, "F%d", func_num);
        
        for (int run = 0; run < num_runs; run++) {
            double run_history[1200] = {0};
            
            // Calling SNS
            double fit = run_sns(pop_size, dim, max_iter, LB, UB, func_num, run_history);
            
            raw_results[run] = fit;
            sum += fit;
            if (fit < best) best = fit;
            if (fit > worst) worst = fit;
            
            fprintf(fp_raw, ",%.5e", fit); 
            for(int i = 0; i < max_iter; i++) mean_history[i] += run_history[i];
        }
        fprintf(fp_raw, "\n");
        
        for(int i = 0; i < max_iter; i++) {
            mean_history[i] /= num_runs;
            fprintf(fp_conv, ",%.5e", mean_history[i]);
        }
        fprintf(fp_conv, "\n");
        
        double mean = sum / num_runs, variance = 0.0;
        for (int run = 0; run < num_runs; run++) variance += (raw_results[run] - mean) * (raw_results[run] - mean);
        double stddev = sqrt(variance / num_runs);
        
        fprintf(fp_summary, "F%d,%.5e,%.5e,%.5e,%.5e\n", func_num, best, worst, mean, stddev);
    }
    
    fclose(fp_summary); fclose(fp_raw); fclose(fp_conv);
    printf("Suite Completed for %s.\n", algo_name);
    
    return 0;
}