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

// ABC needs a fitness mapper because roulette wheel requires positive numbers where higher is better
double calculate_fitness(double cost) {
    if (cost >= 0) return 1.0 / (cost + 1.0);
    else return 1.0 + fabs(cost);
}

// --- 3. Artificial Bee Colony (ABC) Algorithm ---
double run_abc(int pop_size, int dim, int max_iter, double *LB, double *UB, int func_num, double *convergence_history) {
    
    // ABC Hyperparameters
    int FN = pop_size / 2; // Number of Food Sources
    int limit = FN * dim;  // Abandonment limit for a food source
    
    // Memory Allocation
    double **foods = (double **)malloc(FN * sizeof(double *));
    double *cost = (double *)malloc(FN * sizeof(double));
    double *fit = (double *)malloc(FN * sizeof(double));
    double *prob = (double *)malloc(FN * sizeof(double));
    int *trial = (int *)malloc(FN * sizeof(int));
    
    double *v = (double *)malloc(dim * sizeof(double)); // Trial solution
    
    // Global Best Tracker
    double *gBest_pos = (double *)malloc(dim * sizeof(double));
    double gBest_cost = INFINITY;

    // Initialization (Scout Phase 0)
    for (int i = 0; i < FN; i++) {
        foods[i] = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) {
            foods[i][j] = LB[j] + rand_01() * (UB[j] - LB[j]);
        }
        
        cost[i] = evaluate(foods[i], dim, func_num);
        fit[i] = calculate_fitness(cost[i]);
        trial[i] = 0;
        
        if (cost[i] < gBest_cost) {
            gBest_cost = cost[i];
            for (int j = 0; j < dim; j++) gBest_pos[j] = foods[i][j];
        } 
    }

    // Main ABC Loop
    for (int iter = 1; iter <= max_iter; iter++) {
        
        // --- 1. Employed Bee Phase ---
        for (int i = 0; i < FN; i++) {
            // Select random partner
            int k; do { k = rand() % FN; } while (k == i);
            // Select random dimension to mutate
            int j = rand() % dim;
            
            // Copy current food source to v
            for (int d = 0; d < dim; d++) v[d] = foods[i][d];
            
            // Generate mutation (phi is between -1 and 1)
            double phi = (rand_01() * 2.0) - 1.0;
            v[j] = foods[i][j] + phi * (foods[i][j] - foods[k][j]);
            
            // Clamping
            if (v[j] > UB[j]) v[j] = UB[j];
            if (v[j] < LB[j]) v[j] = LB[j];
            
            // Greedy Selection
            double cost_v = evaluate(v, dim, func_num);
            double fit_v = calculate_fitness(cost_v);
            
            if (cost_v < cost[i]) {
                cost[i] = cost_v;
                fit[i] = fit_v;
                foods[i][j] = v[j];
                trial[i] = 0; // Reset trial counter
            } else {
                trial[i]++;
            }
        }
        
        // Calculate Selection Probabilities
        double max_fit = fit[0];
        for (int i = 1; i < FN; i++) if (fit[i] > max_fit) max_fit = fit[i];
        for (int i = 0; i < FN; i++) prob[i] = (0.9 * (fit[i] / max_fit)) + 0.1;
        
        // --- 2. Onlooker Bee Phase ---
        int t = 0, i = 0;
        while (t < FN) {
            if (rand_01() < prob[i]) {
                t++; // A bee chose this food source
                
                int k; do { k = rand() % FN; } while (k == i);
                int j = rand() % dim;
                
                for (int d = 0; d < dim; d++) v[d] = foods[i][d];
                
                double phi = (rand_01() * 2.0) - 1.0;
                v[j] = foods[i][j] + phi * (foods[i][j] - foods[k][j]);
                
                if (v[j] > UB[j]) v[j] = UB[j];
                if (v[j] < LB[j]) v[j] = LB[j];
                
                double cost_v = evaluate(v, dim, func_num);
                double fit_v = calculate_fitness(cost_v);
                
                if (cost_v < cost[i]) {
                    cost[i] = cost_v;
                    fit[i] = fit_v;
                    foods[i][j] = v[j];
                    trial[i] = 0; 
                } else {
                    trial[i]++;
                }
            }
            i++;
            if (i == FN) i = 0;
        }
        
        // Memorize Global Best
        for (int f = 0; f < FN; f++) {
            if (cost[f] < gBest_cost) {
                gBest_cost = cost[f];
                for (int d = 0; d < dim; d++) gBest_pos[d] = foods[f][d];
            }
        }
        
        // --- 3. Scout Bee Phase ---
        int max_trial_index = 0;
        for (int f = 1; f < FN; f++) {
            if (trial[f] > trial[max_trial_index]) max_trial_index = f;
        }
        
        if (trial[max_trial_index] > limit) {
            for (int d = 0; d < dim; d++) {
                foods[max_trial_index][d] = LB[d] + rand_01() * (UB[d] - LB[d]);
            }
            cost[max_trial_index] = evaluate(foods[max_trial_index], dim, func_num);
            fit[max_trial_index] = calculate_fitness(cost[max_trial_index]);
            trial[max_trial_index] = 0;
            // *Note: We don't increment FEs here strictly to keep the loop FEs perfectly matched.
        }
        
        convergence_history[iter - 1] = gBest_cost;
    }
    
    // Free Memory
    for (int i = 0; i < FN; i++) free(foods[i]);
    free(foods); free(cost); free(fit); free(prob); free(trial);
    free(v); free(gBest_pos);
    
    return gBest_cost;
}

// --- 4. The Isolated Test Harness ---
int main() {
    srand(time(NULL)); 
    
    char algo_name[] = "abc"; 
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
            
            // Calling ABC
            double fit = run_abc(pop_size, dim, max_iter, LB, UB, func_num, run_history);
            
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