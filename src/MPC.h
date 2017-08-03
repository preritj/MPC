#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

using namespace std;

class MPC {
 public:
  // first actuations 
  vector<double> actuations;
  // x and y predictions 
  vector<double> x_pred, y_pred;
  // internal latency
  double int_latency;

  MPC(double steer, double throttle);
  virtual ~MPC();

  // Solve the model given an initial state and polynomial coefficients.
  // Returns a vector of first actuations
  vector<double> Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);

  // Runs simulation and returns updates state
  vector<double> run_sim(vector<double> state, double time);
};

#endif /* MPC_H */
