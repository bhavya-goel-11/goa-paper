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
    #error "CRITICAL: You must define a valid CEC_YEAR during compilation (e.g., -D CEC_YEAR=2014)"
#endif

// --- 2. Shared Math Utilities ---
double rand_01() { return (double)rand() / (double)RAND_MAX; }

// --- 3. GWO Algorithm ---
double run_gwo(int pop_size, int dim, int max_iter, double *LB, double *UB, int func_num, double *convergence_history) {
    
    // Memory Allocation for Wolf Pack
    double **wolves = (double **)malloc(pop_size * sizeof(double *));
    double *fitness = (double *)malloc(pop_size * sizeof(double));
    
    // Top 3 Leaders
    double *alpha_pos = (double *)malloc(dim * sizeof(double));
    double *beta_pos = (double *)malloc(dim * sizeof(double));
    double *delta_pos = (double *)malloc(dim * sizeof(double));
    double alpha_score = INFINITY;
    double beta_score = INFINITY;
    double delta_score = INFINITY;

    // Initialization
    for (int i = 0; i < pop_size; i++) {
        wolves[i] = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) {
            wolves[i][j] = LB[j] + rand_01() * (UB[j] - LB[j]);
        }
        
        fitness[i] = evaluate(wolves[i], dim, func_num);
        
        // Initial Leader Assignment
        if (fitness[i] < alpha_score) {
            delta_score = beta_score; for (int j=0; j<dim; j++) delta_pos[j] = beta_pos[j];
            beta_score = alpha_score; for (int j=0; j<dim; j++) beta_pos[j] = alpha_pos[j];
            alpha_score = fitness[i]; for (int j=0; j<dim; j++) alpha_pos[j] = wolves[i][j];
        } 
        else if (fitness[i] < beta_score && fitness[i] > alpha_score) {
            delta_score = beta_score; for (int j=0; j<dim; j++) delta_pos[j] = beta_pos[j];
            beta_score = fitness[i];  for (int j=0; j<dim; j++) beta_pos[j] = wolves[i][j];
        } 
        else if (fitness[i] < delta_score && fitness[i] > beta_score && fitness[i] > alpha_score) {
            delta_score = fitness[i]; for (int j=0; j<dim; j++) delta_pos[j] = wolves[i][j];
        }
    }

    // Main GWO Loop
    for (int iter = 1; iter <= max_iter; iter++) {
        
        // Parameter 'a' decreases linearly from 2 to 0
        double a = 2.0 - (double)iter * (2.0 / max_iter);
        
        for (int i = 0; i < pop_size; i++) {
            for (int j = 0; j < dim; j++) {
                
                // Update against Alpha
                double r1 = rand_01(), r2 = rand_01();
                double A1 = 2.0 * a * r1 - a;
                double C1 = 2.0 * r2;
                double D_alpha = fabs(C1 * alpha_pos[j] - wolves[i][j]);
                double X1 = alpha_pos[j] - A1 * D_alpha;
                
                // Update against Beta
                r1 = rand_01(); r2 = rand_01();
                double A2 = 2.0 * a * r1 - a;
                double C2 = 2.0 * r2;
                double D_beta = fabs(C2 * beta_pos[j] - wolves[i][j]);
                double X2 = beta_pos[j] - A2 * D_beta;
                
                // Update against Delta
                r1 = rand_01(); r2 = rand_01();
                double A3 = 2.0 * a * r1 - a;
                double C3 = 2.0 * r2;
                double D_delta = fabs(C3 * delta_pos[j] - wolves[i][j]);
                double X3 = delta_pos[j] - A3 * D_delta;
                
                // Average the 3 vectors to find new position
                wolves[i][j] = (X1 + X2 + X3) / 3.0;
                
                // Array Boundary Clamping
                if (wolves[i][j] > UB[j]) wolves[i][j] = UB[j];
                if (wolves[i][j] < LB[j]) wolves[i][j] = LB[j];
            }
            
            // Evaluate New Position
            fitness[i] = evaluate(wolves[i], dim, func_num);
            
            // Update Leaders
            if (fitness[i] < alpha_score) {
                delta_score = beta_score; for (int j=0; j<dim; j++) delta_pos[j] = beta_pos[j];
                beta_score = alpha_score; for (int j=0; j<dim; j++) beta_pos[j] = alpha_pos[j];
                alpha_score = fitness[i]; for (int j=0; j<dim; j++) alpha_pos[j] = wolves[i][j];
            } 
            else if (fitness[i] < beta_score && fitness[i] > alpha_score) {
                delta_score = beta_score; for (int j=0; j<dim; j++) delta_pos[j] = beta_pos[j];
                beta_score = fitness[i];  for (int j=0; j<dim; j++) beta_pos[j] = wolves[i][j];
            } 
            else if (fitness[i] < delta_score && fitness[i] > beta_score && fitness[i] > alpha_score) {
                delta_score = fitness[i]; for (int j=0; j<dim; j++) delta_pos[j] = wolves[i][j];
            }
        }
        // Alpha is always the best solution found so far
        convergence_history[iter - 1] = alpha_score;
    }
    
    // Free Memory
    for (int i = 0; i < pop_size; i++) free(wolves[i]);
    free(wolves); free(fitness); 
    free(alpha_pos); free(beta_pos); free(delta_pos);
    
    return alpha_score;
}

// --- 4. The Isolated Test Harness ---
int main() {
    srand(time(NULL)); 
    
    char algo_name[] = "gwo"; 
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
            
            // Calling GWO
            double fit = run_gwo(pop_size, dim, max_iter, LB, UB, func_num, run_history);
            
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