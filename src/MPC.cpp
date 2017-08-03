#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// TODO: Set the timestep length and duration
size_t N = 30;
double dt = 0.05;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// The reference velocity when both CTE and orientation error are 0
double ref_v = 45;  // ~100 mph

// index book-keeping for solver vector
size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t epsi_start = cte_start + N;
size_t delta_start = epsi_start + N;
size_t a_start = delta_start + N - 1;

class FG_eval {
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
  FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    // TODO: implement MPC
    // `fg` a vector of the cost constraints, `vars` is a vector of variable values (state & actuators)
    // NOTE: You'll probably go back and forth between this function and
    // the Solver function below.

    // first elemnt of fg is cost
    fg[0] = 0;

    // CTE, orientation error and ref-speed error 
    for (size_t t = 0; t < N; t++) {
      fg[0] += 1. * CppAD::pow(vars[cte_start + t], 2); 
      fg[0] += 40e1 * CppAD::pow(vars[epsi_start + t], 2);
      fg[0] += 8e-4 * CppAD::pow(vars[v_start + t] - ref_v, 2);
    }

    // avoid sharp turns or huge acceleration 
    for (size_t t = 0; t < N - 1; t++) {
      fg[0] += 1e1 * CppAD::pow(vars[delta_start + t], 2); 
      fg[0] += 0. * CppAD::pow(vars[a_start + t], 2);
    }

    // sequential actuations should be similar.
    for (size_t t = 0; t < N - 2; t++) {
      fg[0] += 6e4 * CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2) ;
      fg[0] += 0. * CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2);
    }

    // initialize to initial state :
    // Note : 1 added to each index because index-0 is cost
    fg[1 + x_start] = vars[x_start];
    fg[1 + y_start] = vars[y_start];
    fg[1 + psi_start] = vars[psi_start];
    fg[1 + v_start] = vars[v_start];
    fg[1 + cte_start] = vars[cte_start];
    fg[1 + epsi_start] = vars[epsi_start];

    // other constraints :
    for (size_t t = 1; t < N; t++) {
      // The state at time t+1 .
      AD<double> x1 = vars[x_start + t];
      AD<double> y1 = vars[y_start + t];
      AD<double> psi1 = vars[psi_start + t];
      AD<double> v1 = vars[v_start + t];
      AD<double> cte1 = vars[cte_start + t];
      AD<double> epsi1 = vars[epsi_start + t];

      // The state at time t.
      AD<double> x0 = vars[x_start + t - 1];
      AD<double> y0 = vars[y_start + t - 1];
      AD<double> psi0 = vars[psi_start + t - 1];
      AD<double> v0 = vars[v_start + t - 1];
      AD<double> cte0 = vars[cte_start + t - 1];
      AD<double> epsi0 = vars[epsi_start + t - 1];

      // Only consider the actuation at time t.
      AD<double> delta0 = vars[delta_start + t - 1];
      AD<double> a0 = vars[a_start + t - 1];

      AD<double> f0 = coeffs[0];
      AD<double> f0_der = 0.;
      for (int i = 1; i < coeffs.size(); ++i){
        f0 += coeffs[i] * CppAD::pow(x0,i);
        f0_der += i * coeffs[i] * CppAD::pow(x0,i-1);
      }

      AD<double> corr0 = 1./CppAD::sqrt(1.+CppAD::pow(f0_der,2)); 
      AD<double> psides0 = CppAD::atan(f0_der);

      // using vehicle model :
      fg[1 + x_start + t] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[1 + y_start + t] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[1 + psi_start + t] = psi1 - (psi0 - v0 * delta0 / Lf * dt);
      fg[1 + v_start + t] = v1 - (v0 + a0 * dt);
      fg[1 + cte_start + t] =
          cte1 - ((f0 - y0) * corr0 + (v0 * CppAD::sin(epsi0) * dt));
      fg[1 + epsi_start + t] =
          epsi1 - ((psi0 - psides0) - v0 * delta0 / Lf * dt);
    }
  }
};

//
// MPC class definition implementation.
//
MPC::MPC(double steer, double acc){
  int_latency = 0.; 
  actuations = {steer, acc};
}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  size_t i;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  // TODO: Set the number of model variables (includes both states and inputs).
  // For example: If the state is a 4 element vector, the actuators is a 2
  // element vector and there are 10 timesteps. The number of variables is:
  //
  // 4 * 10 + 2 * 9
  size_t n_vars = 6 * N + 2 * (N - 1);
  // TODO: Set the number of constraints
  size_t n_constraints = 6 * N;

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (i = 0; i < n_vars; i++) {
    vars[i] = 0.;
  }
  // Set the initial variable values
  i = 0;
  vars[x_start]    = state[i++];
  vars[y_start]    = state[i++];
  vars[psi_start]  = state[i++];
  vars[v_start]    = state[i++];
  vars[cte_start]  = state[i++];
  vars[epsi_start] = state[i++];

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);
  // TODO: Set lower and upper limits for variables.
  // Set all non-actuators upper and lowerlimits
  // to the max negative and positive values.
  for (i = 0; i < v_start; i++) {
    vars_lowerbound[i] = -1.0e3;
    vars_upperbound[i] = 1.0e3;
  }

  for (i = v_start; i < cte_start; i++) {
    vars_lowerbound[i] = 0.; // min : 0 mph
    vars_upperbound[i] = 50; // max : 100 mph
  }

  for (i = cte_start; i < delta_start; i++) {
    vars_lowerbound[i] = -1.0e3;
    vars_upperbound[i] = 1.0e3;
  }

  // The upper and lower limits of delta are set to -25 and 25
  // degrees (values in radians).
  for (i = delta_start; i < a_start; i++) {
    vars_lowerbound[i] = -0.436332;
    vars_upperbound[i] = 0.436332;
  }

  // Acceleration/decceleration upper and lower limits.
  for (i = a_start; i < n_vars; i++) {
    vars_lowerbound[i] = -1. ; 
    vars_upperbound[i] = 1. ; 
  }

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }
  i = 0;
  constraints_lowerbound[x_start]    = state[i++]; 
  constraints_lowerbound[y_start]    = state[i++];
  constraints_lowerbound[psi_start]  = state[i++];
  constraints_lowerbound[v_start]    = state[i++];
  constraints_lowerbound[cte_start]  = state[i++];
  constraints_lowerbound[epsi_start] = state[i++]; 

  i = 0;
  constraints_upperbound[x_start]    = state[i++]; 
  constraints_upperbound[y_start]    = state[i++];
  constraints_upperbound[psi_start]  = state[i++];
  constraints_upperbound[v_start]    = state[i++];
  constraints_upperbound[cte_start]  = state[i++];
  constraints_upperbound[epsi_start] = state[i++]; 

  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  // TODO: Return the first actuator values. The variables can be accessed with
  // `solution.x[i]`.
  //
  int N_pred = 7; // Number of predicted points to display
  actuations.clear();
  x_pred.clear();
  y_pred.clear();
  actuations.push_back(solution.x[delta_start]);
  actuations.push_back(solution.x[a_start]);

  // This is for debugging purposes :
  // remaining elements of output are x,y predictions (N_pred pairs)
  for (size_t i = 1; i < N; i += (N-1)/N_pred){
    x_pred.push_back(solution.x[x_start+i]);
    y_pred.push_back(solution.x[y_start+i]);
  }

  return actuations;
}


// function to run simulation using vehicle model
vector<double> MPC::run_sim(vector<double> state, double ext_latency)
{
  double px = state[0];
  double py = state[1];
  double psi = state[2];
  double v = state[3];
  double delta = this->actuations[0]; 
  double a = this->actuations[1]; 

  // net latency
  double time = ext_latency + this->int_latency;

  double t = 0;
  double delta_t = 0.02;
  while (t < time){
    px += v * cos(psi) * delta_t;
    py += v * sin(psi) * delta_t;
    v += a * delta_t;
    psi -= v * delta / Lf * delta_t;
    t += delta_t; 
    if (time-t < delta_t) delta_t = time - t;
  }
  return {px, py, psi, v};
}
