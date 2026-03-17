#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- 1. CEC17 Declaration & Wrapper ---
void cec17_test_func(double *x, double *f, int nx, int mx, int func_num);

double evaluate_cec17(double *solution, int dim, int func_num) {
    double fitness_value = 0.0;
    cec17_test_func(solution, &fitness_value, dim, 1, func_num);
    return fitness_value;
}

// --- 2. Math Utilities ---
double rand_01() {
    return (double)rand() / (double)RAND_MAX;
}

double rand_normal() {
    double u1 = rand_01();
    double u2 = rand_01();
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

double logistic_map(double current_c) {
    return 4.0 * current_c * (1.0 - current_c);
}

// --- 3. CGOA1 Algorithm (Chaotic Updates) ---
double run_cgoa1(int pop_size, int dim, int max_iter, double LB, double UB, int func_num) {
    double S = 88.0; 
    double PSRs = 0.34;
    double s_step, mu, r, CF;
    
    // Initialize Chaotic variable
    double C = 0.15 + rand_01() * 0.1; 
    
    // Memory Allocation
    double **gazelle = (double **)malloc(pop_size * sizeof(double *));
    double *fitness = (double *)malloc(pop_size * sizeof(double));
    double *elite = (double *)malloc(dim * sizeof(double));
    double best_fitness = INFINITY;

    // Standard Initialization
    for (int i = 0; i < pop_size; i++) {
        gazelle[i] = (double *)malloc(dim * sizeof(double));
        for (int j = 0; j < dim; j++) {
            gazelle[i][j] = LB + rand_01() * (UB - LB);
        }
        fitness[i] = evaluate_cec17(gazelle[i], dim, func_num); 
        
        if (fitness[i] < best_fitness) {
            best_fitness = fitness[i];
            for(int j = 0; j < dim; j++) elite[j] = gazelle[i][j];
        }
    }

    // Optimization Loop
    for (int iter = 1; iter <= max_iter; iter++) {
        s_step = rand_01();
        
        // Update chaotic variable for this iteration
        C = logistic_map(C);
        
        for (int i = 0; i < pop_size; i++) {
            r = rand_01();
            double *new_pos = (double *)malloc(dim * sizeof(double));
            
            if (r < 0.5) {
                // Exploitation (R replaced by C)
                for (int j = 0; j < dim; j++) {
                    double R_B = rand_normal();
                    new_pos[j] = gazelle[i][j] + s_step * C * R_B * (elite[j] - R_B * gazelle[i][j]);
                }
            } else {
                // Exploration
                mu = (iter % 2 == 0) ? -1.0 : 1.0;
                CF = pow(1.0 - (double)iter / max_iter, 2.0 * (double)iter / max_iter);
                
                if (i < pop_size / 2) {
                    // Gazelle Evasion (R replaced by C)
                    for (int j = 0; j < dim; j++) {
                        double R_L = levy_flight(1.5);
                        new_pos[j] = gazelle[i][j] + S * mu * C * R_L * (elite[j] - R_L * gazelle[i][j]);
                    }
                } else {
                    // Predator Chasing
                    for (int j = 0; j < dim; j++) {
                        double R_B = rand_normal();
                        double R_L = levy_flight(1.5);
                        new_pos[j] = gazelle[i][j] + S * mu * CF * R_B * (elite[j] - R_L * gazelle[i][j]);
                    }
                }
            }
            
            // Apply PSRs Effect (Standard R kept here for bounds)
            double r_psr = rand_01();
            int r1 = rand() % pop_size;
            int r2 = rand() % pop_size;
            
            for (int j = 0; j < dim; j++) {
                if (r_psr <= PSRs) {
                    double U = (rand_01() < 0.5) ? 0.0 : 1.0; 
                    double R = rand_01();
                    new_pos[j] = new_pos[j] + CF * (LB + R * (UB - LB)) * U;
                } else {
                    new_pos[j] = new_pos[j] + (PSRs * (1.0 - r_psr) + r_psr) * (gazelle[r1][j] - gazelle[r2][j]);
                }
                
                // Boundary Check
                if (new_pos[j] > UB) new_pos[j] = UB;
                if (new_pos[j] < LB) new_pos[j] = LB;
            }
            
            // Fitness Evaluation
            double new_fit = evaluate_cec17(new_pos, dim, func_num);
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
    }
    
    // Free Memory
    for (int i = 0; i < pop_size; i++) free(gazelle[i]);
    free(gazelle);
    free(fitness);
    free(elite);
    
    return best_fitness;
}

// --- 4. The CEC17 Test Harness ---
int main() {
    srand(time(NULL)); 
    
    int pop_size = 50;  
    int dim = 30;       
    int max_iter = 1000; 
    double LB = -100.0; 
    double UB = 100.0;  
    int num_runs = 30;  
    
    FILE *fp = fopen("../results/cgoa1_cec17_results.csv", "w");
    if (fp == NULL) {
        printf("Error opening file! Check your directory permissions.\n");
        return 1;
    }
    fprintf(fp, "Function,Best,Worst,Mean,StdDev\n");
    
    printf("Starting Mass Benchmark Test for CGOA1 (Chaotic Updates)...\n\n");
    
    for (int func_num = 1; func_num <= 30; func_num++) {
        if (func_num == 2) continue; 
        
        printf("Optimizing Function F%d...\n", func_num);
        
        double results[30]; 
        double sum = 0.0, best = INFINITY, worst = -INFINITY;
        
        for (int run = 0; run < num_runs; run++) {
            double fit = run_cgoa1(pop_size, dim, max_iter, LB, UB, func_num);
            results[run] = fit;
            sum += fit;
            if (fit < best) best = fit;
            if (fit > worst) worst = fit;
        }
        
        double mean = sum / num_runs;
        double variance = 0.0;
        for (int run = 0; run < num_runs; run++) {
            variance += (results[run] - mean) * (results[run] - mean);
        }
        double stddev = sqrt(variance / num_runs);
        
        fprintf(fp, "F%d,%.5e,%.5e,%.5e,%.5e\n", func_num, best, worst, mean, stddev);
        printf("   -> Mean: %.5e | StdDev: %.5e\n", mean, stddev);
    }
    
    fclose(fp);
    printf("\nAll 30 runs completed successfully! Data saved to 'results/cgoa1_cec17_results.csv'.\n");
    
    return 0;
}