// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
#include <RcppArmadillo.h>
#include <cmath>
#include <time.h>
#include <stdlib.h>
#include <iostream>
using namespace Rcpp; 
using namespace arma;
using namespace std;

inline double hs(arma::mat beta, double v);
inline double dgamma_cpp(double x, double a, double b); 
inline double dt_cpp(double x, int n);

// [[Rcpp::export]]
List treatment_new(int n, arma::mat YY, arma::mat YXd, arma::mat XdXd, arma::mat DD, arma::mat DX, arma::mat XX, arma::vec beta_hat, arma::vec gamma_hat, double alpha_hat, arma::vec delta, arma::mat X, arma::vec Y, arma::vec d, double s0 = 5.0, double tad = 0.2, double sy = 1.0, double sd = 1.0, int nsamps = 10000, int burn = 1000, int skip = 1)
{
	
	burn = burn + 1;

    int p = gamma_hat.n_elem;
    
    double syeps;
    double ssq;
    double ly;
    double thetaprop;
    double thetamin;
    double thetamax;
    double piprop;
    double picurr;
    double vy;
    double vd;
    double vprop;
    vy = 0.01;
    vd = 0.01;

	arma::mat U;
	arma::vec D;
	arma::mat V;
	svd(U,D,V,XdXd);
	arma::mat Linvy = U.cols(0,p) * diagmat(1/sqrt(D));


    svd(U,D,V,XX);
    arma::mat Linvd = U.cols(0,p-1) * diagmat(1/sqrt(D));
    
    arma::vec beta_tilde(p);
    
    
	//initialize
	arma::mat betasamps(p,nsamps);
	betasamps.fill(0.0);
	
    arma::mat gammasamps(p,nsamps);
    gammasamps.fill(0.0);
    
    
    arma::vec alphasamps(nsamps);
	alphasamps.fill(0.0);
    
	arma::vec vysamps(nsamps);
	vysamps.fill(0.0);


    arma::vec vdsamps(nsamps);
    vdsamps.fill(0.0);
    
  
    arma::vec sdsamps(nsamps);
    sdsamps.fill(0.0);
   
    arma::vec sysamps(nsamps);
    sysamps.fill(0.0);
    

    int loopcount = 0;
    arma::vec loops(nsamps);
    loops.fill(0);
    
    
    double u;
    
    arma::mat nu;
    nu.fill(0.0);
    
    arma::mat nuy;
    nu.fill(0.0);
    
    arma::mat nud;
    nu.fill(0.0);
    
    arma::vec epsy(p+1);
    epsy.fill(0.0);

    arma::vec epsd(p);
    epsd.fill(0.0);

    
    arma::uvec temp;
    arma::vec all_one;
    arma::mat delta_prop;
    arma::vec temp2;
    double priorcomp;
    double alpha_prop;
	
	clock_t t1, t2;
	
	t1 = clock();
	
    
    arma::vec beta = beta_hat + delta.rows(1,p);
    double alpha = as_scalar(alpha_hat + delta.rows(0,0));
    //double alpha = alpha_hat;
    arma::vec gamma = gamma_hat + delta.rows(p+1,2*p);
    
    arma::vec alpha_beta(p+1);
    
    beta_tilde = beta + alpha*gamma;
    
    for (int h = 0; h < nsamps; ++h)
	{
        
        for (int skiploop = 0; skiploop < skip; ++skiploop){
        
        loopcount = 0;
            
            
            
        epsy = rnorm(Linvy.n_cols);
		nuy = Linvy * epsy;
        
        epsd = rnorm(Linvd.n_cols);
        nud = Linvd * epsd;
            
            
        nu = join_cols(sy*nuy, sd*nud);
                
		u = runif(1,0,1)[0];
		
            priorcomp = hs(beta_tilde, vy) + hs(gamma,vd) + R::dnorm(alpha, 0.0, sd, 1);
	
        //    priorcomp = hs(beta, vy) + hs(gamma,vd);
            
		ly = priorcomp + log(u);

		thetaprop = runif(1,0,2*M_PI)[0];
		
		delta_prop = delta * cos(thetaprop) + nu * sin(thetaprop);
            
		thetamin = thetaprop - 2.0 * M_PI;
		
		thetamax = thetaprop;

            while (hs(beta_hat + delta_prop.rows(1,p) + as_scalar(alpha_hat + delta_prop.rows(0,0))*(gamma_hat + delta_prop.rows(p+1,2*p)), vy) + hs(gamma_hat + delta_prop.rows(p+1,2*p),vd) + R::dnorm(as_scalar(alpha_hat + delta_prop.rows(0,0)),0.0,sd,1) < ly)
		
          //   while (hs(beta_hat + delta_prop.rows(1,p), vy) + hs(gamma_hat + delta_prop.rows(p+1,2*p),vd) < ly)
            {
            loopcount += 1;
			if (thetaprop < 0)
			{
				thetamin = thetaprop;
			}else
			{thetamax = thetaprop;}

			thetaprop = runif(1,thetamin,thetamax)[0];
			
			delta_prop = delta * cos(thetaprop) + nu * sin(thetaprop);

		}

            
       
          //  alpha_prop = alpha_hat + sy*alpha_prop*as<double>(rnorm(1))*as_scalar(Linvy.submat(0,0,0,0));
          //  alpha_prop = alpha + 0.1*as<double>(rnorm(1));
            
         
           // piprop = hs(beta + (alpha_prop)*gamma, vy);
           // picurr = hs(beta + alpha*gamma, vy);
        
            
           // if (as_scalar(randu(1)) < exp(piprop-picurr))
           // {
           //     alpha = alpha_prop;
           //    delta.rows(0,0) = alpha_prop - alpha_hat;
           // }
          
            
            
            
            
            
            
            
        delta = delta_prop;
        beta = beta_hat + delta.rows(1,p);
            alpha = as_scalar(alpha_hat + delta.rows(0,0));

            gamma = gamma_hat + delta.rows(p+1,2*p);
            beta_tilde = beta + alpha*gamma;

           
		
            
            alpha_prop = as_scalar((trans(Y) - trans(X*beta_tilde))*(d-X*gamma));
            syeps = sqrt(as_scalar((trans(Y) - trans(X*beta_tilde))*(Y - X*beta_tilde)/n));
          alpha_prop = alpha_prop/as_scalar(trans(d-X*gamma)*(d-X*gamma));
            alpha_prop = alpha_prop + syeps*as<double>(rnorm(1))/sqrt(1/pow(sd*s0,2) + as_scalar(trans(d-X*gamma)*(d-X*gamma)));
            
         //   picurr = hs(beta + alpha*gamma, vy);
          //  piprop = hs(beta + alpha_prop*gamma, vy);
            
            
          //  if (as_scalar(randu(1)) < exp(piprop-picurr))
          //  {
           //     alpha = alpha_prop;
           // }
            
            alpha = alpha_prop;
            delta.rows(0,0) = alpha - alpha_hat;
            
            beta = beta_tilde - alpha*gamma;
            
            delta.rows(1,p) = beta - beta_hat;
            
            alpha_beta.subvec(0,0) = alpha;
            alpha_beta.subvec(1,p) = beta;
            
        
        vprop = exp(log(vd) + tad*as_scalar(randn(1)));
		
 		piprop = hs(gamma, vprop) + log(dt_cpp(vprop, 1));
 		picurr = hs(gamma, vd) + log(dt_cpp(vd, 1));
        
        if (as_scalar(randu(1)) < exp(piprop-picurr))
        {
            vd = vprop;
        }
            
            vprop = exp(log(vy) + tad*as_scalar(randn(1)));
            
            piprop = hs(beta + alpha*gamma, vprop) + log(dt_cpp(vprop, 1));
            picurr = hs(beta + alpha*gamma, vy) + log(dt_cpp(vy, 1));
            
            if (as_scalar(randu(1)) < exp(piprop-picurr))
            {
                vy = vprop;
            }
            
        
            
            
            
            
		ssq = as_scalar(YY) - 2 * as_scalar(YXd * alpha_beta) + as_scalar(trans(alpha_beta) * XdXd * alpha_beta);
		sy = 1.0/sqrt(rgamma(1,n/2.0, 2.0/ssq)[0]);
       
        ssq = as_scalar(DD) - 2 * as_scalar(DX * gamma) + as_scalar(trans(gamma) * XX * gamma) + pow(alpha,2.0);
        sd = 1.0/sqrt(rgamma(1,n/2.0, 2.0/ssq)[0]);
            
            
        }

        
        betasamps.col(h) = beta;
        gammasamps.col(h) = gamma;
        alphasamps(h) = alpha;
 
        loops(h) = loopcount;
        
        vysamps(h) = vy;
        vdsamps(h) = vd;
        sysamps(h) = sy;
        sdsamps(h) = sd;
        
        

	}
	
	t2 = clock();
	float time_elapse =  ((float)t2-(float)t1);
	time_elapse = time_elapse/CLOCKS_PER_SEC;
	
	
    return List::create(Named("beta") = betasamps, Named("alpha") = alphasamps, Named("loops") = loops, Named("sigy") = sysamps, Named("sigd") = sdsamps, Named("vy") = vysamps, Named("vd") = vdsamps);
}







inline double hs(arma::mat beta, double v)
{
  arma::vec beta2 = conv_to<vec>::from(beta);
  beta2 = beta2/v;

  arma::vec temp = log(log(1.0 + 4.0/(pow(beta2,2.0))));
  double ll;
    ll = sum(temp) - beta2.n_elem*log(v);
  return ll;
}

inline double dgamma_cpp(double x, double a, double b) 
{
    double c;
    c = pow(b,a)/tgamma(a)*pow(x,a-1.0)*exp(-b*x);
    return c;
}


inline double dt_cpp(double x, int n)
{
  double c;
  c = tgamma((n+1.0)/2)/tgamma(n/2.0)/sqrt(n*datum::pi) * pow((1.0+pow(x,2)/n),-(n+1)/2.0);
  return c;
}

 	