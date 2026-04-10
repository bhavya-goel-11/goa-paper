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

// --- 3. WOA Algorithm ---
double run_woa(int pop_size, int dim, int max_iter, double *LB, double *UB, int func_num, double *convergence_history) {
    
    // Memory Allocation for Whale Population
    double **whales = (double **)malloc(pop_size * sizeof(double *));
    double *fitness = (double *)malloc(pop_size * sizeof(double));
    
    // Best Whale (The "Prey")
    double *best_pos = (double *)malloc(dim * sizeof(double));
    double best_score = INFINITY;

    // Initialization
    for (int i = 0; i < pop_size; i++) {
        whales[i] = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) {
            whales[i][j] = LB[j] + rand_01() * (UB[j] - LB[j]);
        }
        
        fitness[i] = evaluate(whales[i], dim, func_num);
        
        // Initial Best Assignment
        if (fitness[i] < best_score) {
            best_score = fitness[i];
            for (int j = 0; j < dim; j++) best_pos[j] = whales[i][j];
        } 
    }

    // Main WOA Loop
    for (int iter = 1; iter <= max_iter; iter++) {
        
        // 'a' decreases linearly from 2 to 0
        double a = 2.0 - (double)iter * (2.0 / max_iter);
        
        for (int i = 0; i < pop_size; i++) {
            
            double p = rand_01(); // Probability switch
            
            for (int j = 0; j < dim; j++) {
                
                double r1 = rand_01();
                double r2 = rand_01();
                
                double A = 2.0 * a * r1 - a;
                double C = 2.0 * r2;
                
                double b = 1.0; // Logarithmic spiral shape constant
                double l = (rand_01() * 2.0 - 1.0); // Random number in [-1, 1]
                
                if (p < 0.5) {
                    if (fabs(A) < 1.0) {
                        // Shrinking Encircling Mechanism
                        double D = fabs(C * best_pos[j] - whales[i][j]);
                        whales[i][j] = best_pos[j] - A * D;
                    } else {
                        // Search for Prey (Exploration)
                        int rand_whale_index = rand() % pop_size;
                        double D = fabs(C * whales[rand_whale_index][j] - whales[i][j]);
                        whales[i][j] = whales[rand_whale_index][j] - A * D;
                    }
                } else {
                    // Spiral Bubble-net Attacking
                    double distance2Leader = fabs(best_pos[j] - whales[i][j]);
                    whales[i][j] = distance2Leader * exp(b * l) * cos(2.0 * M_PI * l) + best_pos[j];
                }
                
                // Array Boundary Clamping
                if (whales[i][j] > UB[j]) whales[i][j] = UB[j];
                if (whales[i][j] < LB[j]) whales[i][j] = LB[j];
            }
            
            // Evaluate New Position
            fitness[i] = evaluate(whales[i], dim, func_num);
            
            // Update Best Whale
            if (fitness[i] < best_score) {
                best_score = fitness[i];
                for (int j = 0; j < dim; j++) best_pos[j] = whales[i][j];
            } 
        }
        convergence_history[iter - 1] = best_score;
    }
    
    // Free Memory
    for (int i = 0; i < pop_size; i++) free(whales[i]);
    free(whales); free(fitness); free(best_pos);
    
    return best_score;
}

// --- 4. The Isolated Test Harness ---
int main() {
    srand(time(NULL)); 
    
    char algo_name[] = "woa"; 
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
            
            // Calling WOA
            double fit = run_woa(pop_size, dim, max_iter, LB, UB, func_num, run_history);
            
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