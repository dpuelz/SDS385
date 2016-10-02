//[[Rcpp::depends(RcppArmadillo)]]
//[[Rcpp::depends(RcppEigen)]]
//[[Rcpp::plugins(cpp11)]]
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <RcppEigen.h>
#include <cmath>
#include <stdlib.h>
#include <iostream>

// additional header file from Kevin
#include "usertypes.hpp"
#include "sgd_utility.hpp"
#include "tinydir.h"

using namespace Rcpp; 
using namespace Eigen;
using namespace std;

/* Some other useful functions for speed (from Kevin) */

union cast_single{ uint32_t asInt; float asFloat; };
static inline float invSqrt( const float& number )
{ 
  cast_single caster;
  constexpr float threehalfs = 1.5F;
  float x2 = number * 0.5F;
  caster.asFloat  = number;
  caster.asInt  = 0x5f3759df - ( caster.asInt >> 1 );               // what the fuck?
  float y  = caster.asFloat;
  y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
  y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed
  
  return y;
}

/* The stochastic gradient descent function */

//[[Rcpp::export]]
List davesgd(VectorXf res, SparseMatrix<float,RowMajor,int> X, VectorXf B0, float lambda, float masterStepSize)
{

    // Compile-time constants--baked into code
    constexpr float adagradEpsilon = 1e-6;
    constexpr float m = 1.0;
    int nPred = X.cols();
    int nSamp = X.rows();
    
    // cout << nPred << endl;
    // cout << nSamp << endl;

    VectorXf agWeights = VectorXf::Constant(nPred, 1e-3);
    VectorXf objTracker = VectorXf::Zero(nSamp);
    float betaNormSquared = B0.norm() * B0.norm();
    float nllAvg = 0;
    constexpr float nllWt = 0.01; //Term for weighting the NLL exponential decay

    // // Last updated term
    vector<int> lastUpdate = vector<int>(nPred);
    uint64_t cc = 0; //Iteration counter

    // The big loop!!
    for(int i = 0; i < nSamp; i++)
    {
      // cout << i << endl;
      
      BetaVec Xsamp =X.row(i);
      float XB = Xsamp.dot(B0);
      float w = 1 / (1 + exp(-XB));
      float y = res(i);
      float logitDelta = y - m * w;

      nllAvg = (1-nllWt) * nllAvg + nllWt * (m * log(w + 1e-6) * (y-m) * log(1 - w + 1e-6));
      objTracker(cc) = -nllAvg;
      
      // Inner loop to update the betas!
      for(BetaVec::InnerIterator it(Xsamp); it; ++it)
      {
        int j = it.index();
      
        // Deferred L2 updates, see comment above this for-loop
        float skip = cc - lastUpdate[j];
        float l2Penalty = (lambda * skip) * B0(j);
        lastUpdate[j] = cc;

        // Calculate gradient(j), this element of the gradient
        float elem_gradient = -logitDelta * it.value() - l2Penalty;

        // Update weights for Adagrad
        agWeights(j) += elem_gradient * elem_gradient;

        // Calculate the scaling factor using fast-inverse-square-root
        float h = invSqrt(agWeights(j) + adagradEpsilon);

        float scaleFactor = masterStepSize * h;
        float totalDelta = scaleFactor * elem_gradient;
        B0(j) -= totalDelta; //Update this element

        // Update beta norm squared with (a+b)^2 = a^2 + 2ab + b^2
        betaNormSquared += 2 * totalDelta * B0(j) + totalDelta * totalDelta;
      }
      cc++;
    }

    // Apply any ridge-regression penalties that we have not yet evaluated
    for(int j = 0; j < nPred; j++)
    {
      float skip = cc - lastUpdate[j];
      float l2Delta = lambda * skip * B0(j);
      float h = invSqrt(agWeights(j) + adagradEpsilon);
      float scaleFactor = masterStepSize * h;
      float totalDelta = scaleFactor * l2Delta;
      B0(j) -= totalDelta;
    }
  return List::create(Named("Likelihood") = objTracker);
}

//[[Rcpp::export]]
List davesgdCV(VectorXf res, SparseMatrix<float,RowMajor,int> X, VectorXf B0, float lambda, float masterStepSize, int train)
{
  
  // Compile-time constants--baked into code
  constexpr float adagradEpsilon = 1e-6;
  constexpr float m = 1.0;
  int totalrows = X.rows();
  
  // Training and test X!
  SparseMatrix<float,RowMajor,int> Xtrain = X.topRows(train);
  SparseMatrix<float,RowMajor,int> Xtest = X.bottomRows(totalrows-train);
  int nPred = Xtrain.cols();
  int nSamp = Xtrain.rows();
  int nSamptest = Xtest.rows();
  
  VectorXf agWeights = VectorXf::Constant(nPred, 1e-3);
  VectorXf objTracker = VectorXf::Zero(nSamp);
  float betaNormSquared = B0.norm() * B0.norm();
  float nllAvg = 0;
  constexpr float nllWt = 0.01; //Term for weighting the NLL exponential decay
  
  // // Last updated term
  vector<int> lastUpdate = vector<int>(nPred);
  uint64_t cc = 0; //Iteration counter
  
  // The big loop to train!!
  for(int i = 0; i < nSamp; i++)
  {
    // cout << i << endl;
    
    BetaVec Xsamp =Xtrain.row(i);
    float XB = Xsamp.dot(B0);
    float w = 1 / (1 + exp(-XB));
    float y = res(i);
    float logitDelta = y - m * w;
    
    nllAvg = (1-nllWt) * nllAvg + nllWt * (m * log(w + 1e-6) * (y-m) * log(1 - w + 1e-6));
    objTracker(cc) = -nllAvg;
    
    // Inner loop to update the betas!
    for(BetaVec::InnerIterator it(Xsamp); it; ++it)
    {
      int j = it.index();
      
      // Deferred L2 updates, see comment above this for-loop
      float skip = cc - lastUpdate[j];
      float l2Penalty = (lambda * skip) * B0(j);
      lastUpdate[j] = cc;
      
      // Calculate gradient(j), this element of the gradient
      float elem_gradient = -logitDelta * it.value() - l2Penalty;
      
      // Update weights for Adagrad
      agWeights(j) += elem_gradient * elem_gradient;
      
      // Calculate the scaling factor using fast-inverse-square-root
      float h = invSqrt(agWeights(j) + adagradEpsilon);
      
      float scaleFactor = masterStepSize * h;
      float totalDelta = scaleFactor * elem_gradient;
      B0(j) -= totalDelta; //Update this element
      
      // Update beta norm squared with (a+b)^2 = a^2 + 2ab + b^2
      betaNormSquared += 2 * totalDelta * B0(j) + totalDelta * totalDelta;
    }
    cc++;
  }
  
  // Apply any ridge-regression penalties that we have not yet evaluated
  for(int j = 0; j < nPred; j++)
  {
    float skip = cc - lastUpdate[j];
    float l2Delta = lambda * skip * B0(j);
    float h = invSqrt(agWeights(j) + adagradEpsilon);
    float scaleFactor = masterStepSize * h;
    float totalDelta = scaleFactor * l2Delta;
    B0(j) -= totalDelta;
  }
  
  // Now, the big loop to test
  float tally = 0;
  int yhat;
  for(int i = 0; i < nSamptest; i++)
  {
    BetaVec Xsamp = Xtest.row(i);
    float XB = Xsamp.dot(B0);
    float w = 1 / (1 + exp(-XB));
    float y = res(i);
    
    if(w < 0.5)
    {
      yhat = 0;
    }
    else
    {
      yhat = 1;
    }
    tally += abs(y-yhat); 
  }
  
  float classrate = 1 - (tally/nSamptest);
    
  return List::create(Named("Likelihood") = objTracker,Named("Classrate") = classrate);
}










