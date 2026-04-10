/*
  Engineering Design Problems Suite 
  Extracted from standard literature (SNS Algorithm Base)
  1. Tension/Compression Spring Design (D=3)
  2. Pressure Vessel Design (D=4)
  3. Welded Beam Design (D=4)
  4. Speed Reducer Design (D=7)
  5. Three-Bar Truss Design (D=2)
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define PENALTY_FACTOR 1.0e15 // Massive penalty for constraint violations
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// --- Helper: Static Penalty Calculator ---
double apply_penalty(double g_value) {
    if (g_value > 0.0) {
        return PENALTY_FACTOR * g_value * g_value; 
    }
    return 0.0;
}

// =======================================================
// F1: Tension/Compression Spring Design (D = 3)
// =======================================================
double eng_spring(double *x) {
    double x1 = x[0]; // d
    double x2 = x[1]; // D
    double x3 = x[2]; // N
    
    double obj = pow(x1, 2) * x2 * (x3 + 2.0); 
    
    double g1 = 1.0 - (pow(x2, 3) * x3) / (71785.0 * pow(x1, 4));
    double g2 = (4.0 * pow(x2, 2) - x1 * x2) / (12566.0 * (x2 * pow(x1, 3) - pow(x1, 4))) + 1.0 / (5108.0 * pow(x1, 2)) - 1.0;
    double g3 = 1.0 - (140.45 * x1) / (pow(x2, 2) * x3);
    double g4 = (x1 + x2) / 1.5 - 1.0;
    
    return obj + apply_penalty(g1) + apply_penalty(g2) + apply_penalty(g3) + apply_penalty(g4);
}

// =======================================================
// F2: Pressure Vessel Design (D = 4)
// =======================================================
double eng_pressure_vessel(double *x) {
    // Variables 1 and 2 are integer multiples of 0.0625 (Quantization from MATLAB)
    double x1 = 0.0625 * round(x[0] / 0.0625); 
    double x2 = 0.0625 * round(x[1] / 0.0625); 
    double x3 = x[2]; 
    double x4 = x[3]; 
    
    double obj = 0.6224 * x1 * x3 * x4 + 1.7781 * x2 * pow(x3, 2) + 3.1661 * pow(x1, 2) * x4 + 19.84 * pow(x1, 2) * x3;
    
    double g1 = -x1 + 0.0193 * x3;
    double g2 = -x2 + 0.00954 * x3;
    double g3 = -M_PI * pow(x3, 2) * x4 - (4.0 / 3.0) * M_PI * pow(x3, 3) + 1296000.0;
    double g4 = x4 - 240.0;
    
    return obj + apply_penalty(g1) + apply_penalty(g2) + apply_penalty(g3) + apply_penalty(g4);
}

// =======================================================
// F3: Welded Beam Design (D = 4)
// =======================================================
double eng_welded_beam(double *x) {
    double x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3];
    
    double p = 6000.0, l = 14.0, e = 30e6, g = 12e6;
    double deltamax = 0.25, tomax = 13600.0, sigmamax = 30000.0;
    
    double m = p * (l + x2 / 2.0);
    double r = sqrt(pow(x2 / 2.0, 2) + pow((x1 + x3) / 2.0, 2));
    double j = 2.0 * (sqrt(2.0) * x1 * x2 * (pow(x2, 2) / 12.0 + pow((x1 + x3) / 2.0, 2)));
    double toprim = p / (sqrt(2.0) * x1 * x2);
    double tozegond = m * r / j;
    double to = sqrt(pow(toprim, 2) + 2.0 * toprim * tozegond * x2 / (2.0 * r) + pow(tozegond, 2));
    
    double sigma = 6.0 * p * l / (x4 * pow(x3, 2));
    double delta = (4.0 * p * pow(l, 3)) / (e * pow(x3, 3) * x4);
    double pc = ((4.013 * e * sqrt(pow(x3, 2) * pow(x4, 6) / 36.0)) / pow(l, 2)) * (1.0 - 0.5 * (x3 / l) * sqrt(e / (4.0 * g)));

    double obj = 1.10471 * pow(x1, 2) * x2 + 0.04811 * x3 * x4 * (14.0 + x2);
    
    double g1 = to - tomax;
    double g2 = sigma - sigmamax;
    double g3 = x1 - x4;
    double g4 = 1.10471 * pow(x1, 2) + 0.04811 * x3 * x4 * (14.0 + x2) - 5.0;
    double g5 = 0.125 - x1;
    double g6 = delta - deltamax;
    double g7 = p - pc;
    
    return obj + apply_penalty(g1) + apply_penalty(g2) + apply_penalty(g3) + apply_penalty(g4) + apply_penalty(g5) + apply_penalty(g6) + apply_penalty(g7);
}

// =======================================================
// F4: Speed Reducer Design (D = 7)
// =======================================================
double eng_speed_reducer(double *x) {
    double x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3], x5 = x[4], x6 = x[5], x7 = x[6];
    
    double obj = 0.7854 * x1 * pow(x2, 2) * (3.3333 * pow(x3, 2) + 14.9334 * x3 - 43.0934) 
                 - 1.508 * x1 * (pow(x6, 2) + pow(x7, 2)) 
                 + 7.477 * (pow(x6, 3) + pow(x7, 3)) 
                 + 0.7854 * (x4 * pow(x6, 2) + x5 * pow(x7, 2));
                 
    double g1 = -x1 * pow(x2, 2) * x3 + 27.0;
    double g2 = -x1 * pow(x2, 2) * pow(x3, 2) + 397.5;
    double g3 = -x2 * pow(x6, 4) * x3 * pow(x4, -3) + 1.93;
    double g4 = -x2 * pow(x7, 4) * x3 / pow(x5, 3) + 1.93;
    double g5 = 10.0 * pow(x6, -3) * sqrt(16.91e6 + pow(745.0 * x4 / (x2 * x3), 2)) - 1100.0;
    double g6 = 10.0 * pow(x7, -3) * sqrt(157.5e6 + pow(745.0 * x5 / (x2 * x3), 2)) - 850.0;
    double g7 = x2 * x3 - 40.0;
    double g8 = -x1 / x2 + 5.0;
    double g9 = x1 / x2 - 12.0;
    double g10 = 1.5 * x6 - x4 + 1.9;
    double g11 = 1.1 * x7 - x5 + 1.9;
    
    return obj + apply_penalty(g1) + apply_penalty(g2) + apply_penalty(g3) + apply_penalty(g4) + 
           apply_penalty(g5) + apply_penalty(g6) + apply_penalty(g7) + apply_penalty(g8) + 
           apply_penalty(g9) + apply_penalty(g10) + apply_penalty(g11);
}

// =======================================================
// F5: Three-Bar Truss Design (D = 2)
// =======================================================
double eng_three_bar_truss(double *x) {
    double x1 = x[0], x2 = x[1];
    
    double obj = (2.0 * sqrt(2.0) * x1 + x2) * 100.0;
    
    double g1 = (sqrt(2.0) * x1 + x2) / (sqrt(2.0) * pow(x1, 2) + 2.0 * x1 * x2) * 2.0 - 2.0;
    double g2 = x2 / (sqrt(2.0) * pow(x1, 2) + 2.0 * x1 * x2) * 2.0 - 2.0;
    double g3 = 1.0 / (sqrt(2.0) * x2 + x1) * 2.0 - 2.0;
    
    return obj + apply_penalty(g1) + apply_penalty(g2) + apply_penalty(g3);
}

// =======================================================
// MASTER WRAPPERS (Matches your cgoa1.c architecture)
// =======================================================
void eng_test_func(double *x, double *f, int nx, int mx, int func_num) {
    if (func_num == 1)      *f = eng_spring(x);
    else if (func_num == 2) *f = eng_pressure_vessel(x);
    else if (func_num == 3) *f = eng_welded_beam(x);
    else if (func_num == 4) *f = eng_speed_reducer(x);
    else if (func_num == 5) *f = eng_three_bar_truss(x);
    else *f = 1e10; 
}

// Injects the problem-specific bounds and dimensions
void get_eng_bounds(int func_num, int *dim, double *lb, double *ub) {
    if (func_num == 1) { // Spring
        *dim = 3; 
        double l[] = {0.05, 0.25, 2.0}; double u[] = {2.0, 1.3, 15.0};
        for(int i=0; i<3; i++){ lb[i] = l[i]; ub[i] = u[i]; }
    } else if (func_num == 2) { // Pressure Vessel
        *dim = 4; 
        double l[] = {0.51, 0.51, 10.0, 10.0}; double u[] = {99.49, 99.49, 200.0, 200.0};
        for(int i=0; i<4; i++){ lb[i] = l[i]; ub[i] = u[i]; }
    } else if (func_num == 3) { // Welded Beam
        *dim = 4; 
        double l[] = {0.1, 0.1, 0.1, 0.1}; double u[] = {2.0, 10.0, 10.0, 2.0};
        for(int i=0; i<4; i++){ lb[i] = l[i]; ub[i] = u[i]; }
    } else if (func_num == 4) { // Speed Reducer
        *dim = 7; 
        double l[] = {2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0}; double u[] = {3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5};
        for(int i=0; i<7; i++){ lb[i] = l[i]; ub[i] = u[i]; }
    } else if (func_num == 5) { // Three-Bar Truss
        *dim = 2; 
        double l[] = {0.0, 0.0}; double u[] = {1.0, 1.0};
        for(int i=0; i<2; i++){ lb[i] = l[i]; ub[i] = u[i]; }
    }
}