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

using namespace Rcpp;
using namespace Eigen;
using namespace std;

/* Some other useful functions for speed (from Kevin) */


template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

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
List davesgdCV(VectorXf res, SparseMatrix<float,RowMajor,int> X, VectorXf B0, float lambda, float masterStepSize, int train)
{
  // Compile-time constants--baked into code
  constexpr float m = 1.0;
  int totalrows = X.rows();
  double w_hat = (res.sum() + 1.0) / (totalrows + 2.0);
  double alpha = log(w_hat/(1.0-w_hat));
  float delta,weight,mu;
  float g0squared = 0;

  // Training and test X!
  SparseMatrix<float,RowMajor,int> Xtrain = X.topRows(train);
  SparseMatrix<float,RowMajor,int> Xtest = X.bottomRows(totalrows-train);
  int nPred = Xtrain.cols();
  int nSamp = Xtrain.rows();
  int nSamptest = Xtest.rows();

  VectorXf agWeights = VectorXf::Constant(nPred, 1e-3);
  VectorXf objTracker = VectorXf::Zero(nSamp);
  float nllAvg = 0;
  constexpr float nllWt = 0.01; //Term for weighting the NLL exponential decay
  
  // Initialize Gsquared and beta
  VectorXf Beta(nPred);
  VectorXf Gsquared(nPred);
  for(int j=0; j<nPred; j++) 
  {
    Gsquared(j) = 1e-3;
    Beta(j) = B0(j);
  }

  // // Last updated term
  vector<int> lastUpdate = vector<int>(nPred);
  uint64_t cc = 0; //Iteration counter

  // The big loop to train!!
  for(int i = 0; i < nSamp; i++)
  {
    BetaVec Xsamp =Xtrain.row(i);
    float XB = alpha + Xsamp.dot(Beta);
    float w = 1 / (1 + exp(-XB));
    float y = res(i);
    float logitDelta = y - m * w;

    nllAvg = (1-nllWt) * nllAvg + nllWt * (m * log(1 + exp(XB)) - y*XB);
    objTracker(cc) = -nllAvg;
    
    // updating the intercept here
    delta = logitDelta;
    g0squared += delta*delta;
    alpha += (masterStepSize/sqrt(g0squared))*delta;

    // Inner loop to update the betas!
    for(BetaVec::InnerIterator it(Xsamp); it; ++it)
    {
      
      // cout << 2 << endl;
      
      // the first non-zero feature!
      int j = it.index();
      
      // weighting for penalty
      weight = 1.0/(1.0 + fabs(Beta(j)));

      /* Updating beta is done in two steps
      ###########################################
        1. Lazy update the penalty portion only
        2. Gradient descent update 
      ###########################################
      */
      
      // STEP 1: Penalty only updates via *lazy updating*
      float skip = cc - lastUpdate[j];
      float h = invSqrt(Gsquared(j));
      float gammatilde = skip*masterStepSize*h;
      Beta(j) = sgn(Beta(j))*fmax(0.0, fabs(Beta(j)) - gammatilde*weight*lambda);
      
      lastUpdate[j] = cc;
      
      // STEP 2: Gradient descent update
      // Calculate gradient(j), this element of the gradient
      float elem_gradient = -logitDelta * it.value();

      // Update weights for Adagrad scaling
      Gsquared(j) += elem_gradient * elem_gradient;

      // Calculate the scaling factor using fast-inverse-square-root
      h = invSqrt(Gsquared(j));
      gammatilde = masterStepSize*h;
      mu = Beta(j) - gammatilde*elem_gradient;
      Beta(j) = sgn(mu)*fmax(0.0, fabs(mu) - gammatilde*weight*lambda);
    }
    cc++; // the global counter
  }

  // Apply any ridge-regression penalties that we have not yet evaluated
  for(int j = 0; j < nPred; j++)
  {
    double skip = cc - lastUpdate[j];
    float h = sqrt(Gsquared(j));
    float gammatilde = skip*masterStepSize*h;
    Beta(j) = sgn(Beta(j))*fmax(0.0, fabs(Beta(j)) - gammatilde*weight*lambda);
  }

  // Now, the big loop to test
  float tally = 0;
  float tally2 = 0;
  float tally3 = 0;
  float num1s = 0;
  float num0s = 0;
  int yhat;
  for(int i = 0; i < nSamptest; i++)
  {
    BetaVec Xsamp = Xtest.row(i);
    float XB = alpha + Xsamp.dot(Beta);
    float w = 1 / (1 + exp(-XB));
    float y = res(i);
    
    if(w < 0.5)
    {
      yhat = 0;
      if(y==0){ tally3 += 1; }
    }
    else
    {
      yhat = 1;
      if(y==1){ tally2 += 1; }
    }
    tally += abs(y-yhat);
    if(y==0){ num0s += 1; }
    else{ num1s +=1; }
  }
  float classrate = 1 - (tally/nSamptest);
  float sensi = (tally2/num1s); 
  float speci = (tally3/num0s); 
  return List::create(Named("Likelihood") = objTracker,Named("Classrate") = classrate,Named("Sensitivity") = sensi,Named("Specificity") = speci);
}










