#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- 1. Macro-Driven CEC Wrapper ---
// The compiler flag (e.g., -D CEC_YEAR=2014) decides which suite gets linked.
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

double rand_normal() {
    double u1 = rand_01(), u2 = rand_01();
    if (u1 <= 1e-7) u1 = 1e-7; 
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

double levy_flight(double alpha) {
    double beta = alpha;
    double sigma_u = pow((tgamma(1.0 + beta) * sin(M_PI * beta / 2.0)) / 
                     (tgamma((1.0 + beta) / 2.0) * beta * pow(2.0, (beta - 1.0) / 2.0)), 1.0 / beta);
    double u = rand_normal() * sigma_u;
    double v = rand_normal();
    if (v == 0.0) v = 1e-7; 
    return 0.05 * (u / pow(fabs(v), 1.0 / beta));
}

// --- 3. Original GOA Algorithm ---
// Swapped scalar bounds for array bounds: double *LB, double *UB
double run_goa(int pop_size, int dim, int max_iter, double *LB, double *UB, int func_num, double *convergence_history) {
    double S = 88.0, PSRs = 0.34, s_step, mu, r, CF;
    
    double **gazelle = (double **)malloc(pop_size * sizeof(double *));
    double *fitness = (double *)malloc(pop_size * sizeof(double));
    double *elite = (double *)malloc(dim * sizeof(double));
    double best_fitness = INFINITY;

    for (int i = 0; i < pop_size; i++) {
        gazelle[i] = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) gazelle[i][j] = LB[j] + rand_01() * (UB[j] - LB[j]);
        fitness[i] = evaluate(gazelle[i], dim, func_num); 
        if (fitness[i] < best_fitness) {
            best_fitness = fitness[i];
            for(int j = 0; j < dim; j++) elite[j] = gazelle[i][j];
        }
    }

    for (int iter = 1; iter <= max_iter; iter++) {
        s_step = rand_01();
        for (int i = 0; i < pop_size; i++) {
            r = rand_01();
            double *new_pos = (double *)malloc(dim * sizeof(double));
            
            if (r < 0.5) {
                for (int j = 0; j < dim; j++) {
                    double R = rand_01(), R_B = rand_normal();
                    new_pos[j] = gazelle[i][j] + s_step * R * R_B * (elite[j] - R_B * gazelle[i][j]);
                }
            } else {
                mu = (iter % 2 == 0) ? -1.0 : 1.0;
                CF = pow(1.0 - (double)iter / max_iter, 2.0 * (double)iter / max_iter);
                if (i < pop_size / 2) {
                    for (int j = 0; j < dim; j++) {
                        double R = rand_01(), R_L = levy_flight(1.5);
                        new_pos[j] = gazelle[i][j] + S * mu * R * R_L * (elite[j] - R_L * gazelle[i][j]);
                    }
                } else {
                    for (int j = 0; j < dim; j++) {
                        double R_B = rand_normal(), R_L = levy_flight(1.5);
                        new_pos[j] = gazelle[i][j] + S * mu * CF * R_B * (elite[j] - R_L * gazelle[i][j]);
                    }
                }
            }
            
            double r_psr = rand_01();
            int r1 = rand() % pop_size, r2 = rand() % pop_size;
            
            for (int j = 0; j < dim; j++) {
                if (r_psr <= PSRs) {
                    double U = (rand_01() < 0.5) ? 0.0 : 1.0; 
                    new_pos[j] += CF * (LB[j] + rand_01() * (UB[j] - LB[j])) * U;
                } else {
                    new_pos[j] += (PSRs * (1.0 - r_psr) + r_psr) * (gazelle[r1][j] - gazelle[r2][j]);
                }
                // Array boundary clamping
                if (new_pos[j] > UB[j]) new_pos[j] = UB[j];
                if (new_pos[j] < LB[j]) new_pos[j] = LB[j];
            }
            
            double new_fit = evaluate(new_pos, dim, func_num);
            if (new_fit < fitness[i]) {
                fitness[i] = new_fit;
                for (int j = 0; j < dim; j++) gazelle[i][j] = new_pos[j];
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    for (int j = 0; j < dim; j++) elite[j] = gazelle[i][j];
                }
            }
            free(new_pos);
        }
        convergence_history[iter - 1] = best_fitness;
    }
    
    for (int i = 0; i < pop_size; i++) free(gazelle[i]);
    free(gazelle); free(fitness); free(elite);
    
    return best_fitness;
}

// --- 4. The Isolated Test Harness ---
int main() {
    srand(time(NULL)); 
    
    char algo_name[] = "goa"; // Set specifically for the baseline
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
        if (CEC_YEAR == 2017 && func_num == 2) continue; // Official CEC17 Exclusion
        
        printf("  -> Optimizing Function F%d...\n", func_num);
        
        // --- DYNAMIC BOUNDARY & DIMENSION ALLOCATION ---
        int dim; 
        double LB[100], UB[100]; // Max supported dimension length
        
        #if CEC_YEAR == 9999
            get_eng_bounds(func_num, &dim, LB, UB);
        #else
            #if CEC_YEAR == 2020 || CEC_YEAR == 2022
                dim = 20; // Max official D for these years
            #else
                dim = 30; // Standard D for 2014/2017
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
            
            // Calling the baseline run_goa function with Arrays
            double fit = run_goa(pop_size, dim, max_iter, LB, UB, func_num, run_history);
            
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