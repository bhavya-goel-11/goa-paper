#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>

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

double logistic_map(double current_c) {
    return 4.0 * current_c * (1.0 - current_c);
}

double clamp_value(double x, double lb, double ub) {
    if (x < lb) return lb;
    if (x > ub) return ub;
    return x;
}

double reflect_or_clamp(double x, double lb, double ub) {
    if (x < lb) {
        x = lb + (lb - x);
        if (x > ub) x = lb;
    } else if (x > ub) {
        x = ub - (x - ub);
        if (x < lb) x = ub;
    }
    return clamp_value(x, lb, ub);
}

double population_diversity(double **gazelle, int pop_size, int dim, double *LB, double *UB) {
    double *mean = (double *)calloc(dim, sizeof(double));
    if (!mean) return 0.0;

    for (int i = 0; i < pop_size; i++) {
        for (int j = 0; j < dim; j++) mean[j] += gazelle[i][j];
    }
    for (int j = 0; j < dim; j++) mean[j] /= (double)pop_size;

    double div = 0.0;
    for (int i = 0; i < pop_size; i++) {
        for (int j = 0; j < dim; j++) {
            double range = UB[j] - LB[j];
            if (range < 1e-12) range = 1.0;
            div += fabs(gazelle[i][j] - mean[j]) / range;
        }
    }

    free(mean);
    return div / (double)(pop_size * dim);
}

int get_worst_index(double *fitness, int pop_size) {
    int idx = 0;
    for (int i = 1; i < pop_size; i++) {
        if (fitness[i] > fitness[idx]) idx = i;
    }
    return idx;
}

int tournament_best_index(double *fitness, int pop_size) {
    int a = rand() % pop_size;
    int b = rand() % pop_size;
    int c = rand() % pop_size;
    int best = a;
    if (fitness[b] < fitness[best]) best = b;
    if (fitness[c] < fitness[best]) best = c;
    return best;
}

void sort_indices_by_fitness(double *fitness, int *indices, int pop_size) {
    for (int i = 0; i < pop_size; i++) indices[i] = i;
    for (int i = 1; i < pop_size; i++) {
        int key = indices[i];
        double key_fit = fitness[key];
        int j = i - 1;
        while (j >= 0 && fitness[indices[j]] > key_fit) {
            indices[j + 1] = indices[j];
            j--;
        }
        indices[j + 1] = key;
    }
}

// --- 3. CGOA1 Algorithm (Chaotic Updates) ---
// Swapped scalar bounds for array bounds: double *LB, double *UB
double run_cgoa1(int pop_size, int dim, int max_iter, double *LB, double *UB, int func_num, double *convergence_history) {
    double S = 88.0, PSRs = 0.34, s_step, mu, r, CF;
    double C = 0.15 + rand_01() * 0.1; 
    double prev_best = INFINITY;
    int no_improve_count = 0;
    
    double **gazelle = (double **)malloc(pop_size * sizeof(double *));
    double *fitness = (double *)malloc(pop_size * sizeof(double));
    double *elite = (double *)malloc(dim * sizeof(double));
    double *new_pos = (double *)malloc(dim * sizeof(double));
    double *opp_pos = (double *)malloc(dim * sizeof(double));
    double *trial_pos = (double *)malloc(dim * sizeof(double));
    int *rank_idx = (int *)malloc(pop_size * sizeof(int));
    double best_fitness = INFINITY;

    if (!gazelle || !fitness || !elite || !new_pos || !opp_pos || !trial_pos || !rank_idx) {
        free(gazelle); free(fitness); free(elite); free(new_pos); free(opp_pos); free(trial_pos); free(rank_idx);
        return INFINITY;
    }

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
        CF = pow(1.0 - (double)iter / max_iter, 2.0 * (double)iter / max_iter);
        s_step = rand_01();
        C = logistic_map(C); // Chaotic Update
        if (C <= 1e-8 || C >= 1.0 - 1e-8) C = 0.2 + 0.6 * rand_01();

        double div = population_diversity(gazelle, pop_size, dim, LB, UB);
        double progress = (double)iter / (double)max_iter;
        double adaptive_psr = PSRs + 0.18 * (1.0 - div) + 0.10 * progress;
        if (adaptive_psr < 0.15) adaptive_psr = 0.15;
        if (adaptive_psr > 0.70) adaptive_psr = 0.70;

        double adaptive_S = S * (0.55 + 0.9 * (1.0 - progress));
        if (div < 0.06) adaptive_S *= 1.30;
        if (no_improve_count > 12) adaptive_S *= 1.20;

        sort_indices_by_fitness(fitness, rank_idx, pop_size);
        int pbest_pool = (int)(0.08 * pop_size + progress * 0.22 * pop_size);
        if (pbest_pool < 2) pbest_pool = 2;
        if (pbest_pool > pop_size / 2) pbest_pool = pop_size / 2;

        double de_prob = 0.12 + 0.22 * progress;
        if (div < 0.05) de_prob += 0.12;
        if (no_improve_count > 10) de_prob += 0.12;
        if (de_prob > 0.70) de_prob = 0.70;
        
        for (int i = 0; i < pop_size; i++) {
            r = rand_01();
            int leader_idx = tournament_best_index(fitness, pop_size);
            double *leader = (rand_01() < 0.75) ? elite : gazelle[leader_idx];
            
            if (r < 0.5) {
                for (int j = 0; j < dim; j++) {
                    double R_B = rand_normal();
                    new_pos[j] = gazelle[i][j] + s_step * C * R_B * (leader[j] - R_B * gazelle[i][j]);
                }
            } else {
                mu = (iter % 2 == 0) ? -1.0 : 1.0;
                if (i < pop_size / 2) {
                    for (int j = 0; j < dim; j++) {
                        double R_L = levy_flight(1.5);
                        new_pos[j] = gazelle[i][j] + adaptive_S * mu * C * R_L * (leader[j] - R_L * gazelle[i][j]);
                    }
                } else {
                    for (int j = 0; j < dim; j++) {
                        double R_B = rand_normal(), R_L = levy_flight(1.5);
                        new_pos[j] = gazelle[i][j] + adaptive_S * mu * CF * R_B * (leader[j] - R_L * gazelle[i][j]);
                    }
                }
            }
            
            double r_psr = rand_01();
            int r1 = rand() % pop_size, r2 = rand() % pop_size;
            
            for (int j = 0; j < dim; j++) {
                if (r_psr <= adaptive_psr) {
                    double U = (rand_01() < 0.5) ? 0.0 : 1.0; 
                    new_pos[j] += CF * (LB[j] + rand_01() * (UB[j] - LB[j])) * U;
                } else {
                    new_pos[j] += (adaptive_psr * (1.0 - r_psr) + r_psr) * (gazelle[r1][j] - gazelle[r2][j]);
                }

                if (div < 0.05 && rand_01() < 0.15) {
                    new_pos[j] += 0.08 * levy_flight(1.5) * (UB[j] - LB[j]);
                }

                new_pos[j] = reflect_or_clamp(new_pos[j], LB[j], UB[j]);
            }
            
            double new_fit = evaluate(new_pos, dim, func_num);

            // Opposition candidate helps escaping local basins when movement stagnates.
            for (int j = 0; j < dim; j++) {
                double opp_noise = 0.03 * rand_normal() * (UB[j] - LB[j]);
                opp_pos[j] = clamp_value(LB[j] + UB[j] - new_pos[j] + opp_noise, LB[j], UB[j]);
            }
            double opp_fit = evaluate(opp_pos, dim, func_num);
            if (opp_fit < new_fit) {
                new_fit = opp_fit;
                for (int j = 0; j < dim; j++) new_pos[j] = opp_pos[j];
            }

            if (new_fit < fitness[i]) {
                fitness[i] = new_fit;
                for (int j = 0; j < dim; j++) gazelle[i][j] = new_pos[j];
                if (fitness[i] < best_fitness) {
                    best_fitness = fitness[i];
                    for (int j = 0; j < dim; j++) elite[j] = gazelle[i][j];
                }
            }

            if (rand_01() < de_prob) {
                int pbest_idx = rank_idx[rand() % pbest_pool];

                int r1, r2;
                do { r1 = rand() % pop_size; } while (r1 == i || r1 == pbest_idx);
                do { r2 = rand() % pop_size; } while (r2 == i || r2 == pbest_idx || r2 == r1);

                double F = 0.45 + 0.45 * rand_01();
                if (no_improve_count > 10) F += 0.10 * rand_01();
                if (F > 0.98) F = 0.98;

                double CR = 0.55 + 0.40 * rand_01();
                int j_rand = rand() % dim;

                for (int j = 0; j < dim; j++) {
                    double mutant = gazelle[i][j]
                                  + F * (gazelle[pbest_idx][j] - gazelle[i][j])
                                  + F * (gazelle[r1][j] - gazelle[r2][j]);

                    if (rand_01() <= CR || j == j_rand) {
                        trial_pos[j] = reflect_or_clamp(mutant, LB[j], UB[j]);
                    } else {
                        trial_pos[j] = gazelle[i][j];
                    }
                }

                double trial_fit = evaluate(trial_pos, dim, func_num);
                if (trial_fit < fitness[i]) {
                    fitness[i] = trial_fit;
                    for (int j = 0; j < dim; j++) gazelle[i][j] = trial_pos[j];
                    if (trial_fit < best_fitness) {
                        best_fitness = trial_fit;
                        for (int j = 0; j < dim; j++) elite[j] = gazelle[i][j];
                    }
                }
            }
        }

        if (best_fitness < prev_best - 1e-14) {
            prev_best = best_fitness;
            no_improve_count = 0;
        } else {
            no_improve_count++;
        }

        if (no_improve_count >= 25) {
            int restart_count = pop_size / 5;
            if (restart_count < 1) restart_count = 1;

            for (int k = 0; k < restart_count; k++) {
                int worst = get_worst_index(fitness, pop_size);
                for (int j = 0; j < dim; j++) {
                    double range = UB[j] - LB[j];
                    double local_step = (2.0 * rand_01() - 1.0) * (0.25 + 0.25 * rand_01()) * range;
                    double candidate = elite[j] + local_step;
                    if (rand_01() < 0.35) candidate = LB[j] + rand_01() * range;
                    gazelle[worst][j] = reflect_or_clamp(candidate, LB[j], UB[j]);
                }
                fitness[worst] = evaluate(gazelle[worst], dim, func_num);
                if (fitness[worst] < best_fitness) {
                    best_fitness = fitness[worst];
                    for (int j = 0; j < dim; j++) elite[j] = gazelle[worst][j];
                }
            }

            no_improve_count = 8;
        }

        convergence_history[iter - 1] = best_fitness;
    }
    
    for (int i = 0; i < pop_size; i++) free(gazelle[i]);
    free(gazelle); free(fitness); free(elite); free(new_pos); free(opp_pos); free(trial_pos); free(rank_idx);
    
    return best_fitness;
}

// --- 4. The Isolated Test Harness ---
int main() {
    srand(time(NULL)); 
    
    char algo_name[] = "cgoa1"; // Set specifically for the chaotic variant
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
            
            // Calling the chaotic run_cgoa1 function with Arrays
            double fit = run_cgoa1(pop_size, dim, max_iter, LB, UB, func_num, run_history);
            
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