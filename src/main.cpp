#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"
#include <ctime>

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

// transforms position vectors from global coordinates to car frame
vector<vector<double>> tf_to_car
                    (vector<double> ptsx, vector<double> ptsy, 
                     double px, double py, double psi)
{
  vector<double> ptsx_tf;
  vector<double> ptsy_tf;
  for (size_t i = 0; i < ptsx.size(); ++i){
    double x =  (ptsx[i] - px) * cos(psi) 
               +(ptsy[i] - py) * sin(psi); 
    double y = -(ptsx[i] - px) * sin(psi) 
               +(ptsy[i] - py) * cos(psi); 
    ptsx_tf.push_back(x);
    ptsy_tf.push_back(y);
  }
  return {ptsx_tf, ptsy_tf};
} 

const int latency = 100; // 100 ms latency

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc(0.,0.); // initialize with steer=0 and throttle=1

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {

          std::clock_t begin = clock();

          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];
          vector<double> ptsy = j[1]["ptsy"];
          double px = j[1]["x"];
          double py = j[1]["y"];
          double psi = j[1]["psi"];
          double v = j[1]["speed"]; // in mph
          v *= 0.44704;

          // update state to account for latency
          double time = latency * 0.001; 
          vector<double> state_ {px, py, psi, v};
          state_ = mpc.run_sim(state_, time);
          px = state_[0];
          py = state_[1];
          psi = state_[2];
          v = state_[3];

          // transform landmarks to car frame
          auto pts_tf = tf_to_car(ptsx, ptsy, px, py, psi);
          vector<double> ptsx_tf = pts_tf[0];
          vector<double> ptsy_tf = pts_tf[1];

          Eigen::VectorXd ptsx_ = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>
                                  (ptsx_tf.data(), ptsx.size());
          Eigen::VectorXd ptsy_ = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>
                                  (ptsy_tf.data(), ptsy.size());

          // fit polynomial curve in car frame
          int n_max = ptsx.size()-1;
          auto coeffs = polyfit(ptsx_, ptsy_, std::min(3,n_max));

          double cte = coeffs[0]/sqrt(1.+pow(coeffs[1],2));
          double epsi =  - atan(coeffs[1]);

          Eigen::VectorXd state(6);
          state << 0., 0., 0., v, cte, epsi;

          auto actuations = mpc.Solve(state, coeffs);

          /*
          * TODO: Calculate steering angle and throttle using MPC.
          *
          * Both are in between [-1, 1].
          *
          */
          double steer_value = actuations[0];
          double throttle_value = actuations[1]; // acceleration
          //double throttle_value = 0.3;

          json msgJson;
          // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
          // Otherwise the values will be in between [-deg2rad(25), deg2rad(25)] instead of [-1, 1].
          msgJson["steering_angle"] = steer_value/deg2rad(25);
          msgJson["throttle"] = throttle_value;

          //Display the MPC predicted trajectory 
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;

          mpc_x_vals = mpc.x_pred;
          mpc_y_vals = mpc.y_pred;

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line

          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          //Display the waypoints/reference line
          vector<double> next_x_vals;
          vector<double> next_y_vals;

          for (double x = 2.; x < 100.; x += 5.){
            next_x_vals.push_back(x);
            next_y_vals.push_back(polyeval(coeffs, x));
          }

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;


          std::clock_t end = clock();
          double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
          // set internal latency 
          mpc.int_latency = elapsed_secs;
          std::cout << "Internal latency : " << mpc.int_latency << std::endl;

          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
