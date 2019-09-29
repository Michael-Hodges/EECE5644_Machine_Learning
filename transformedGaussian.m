function x = transformedGaussian(N,n,mu,Sigma)
% Generates N samples from a Gaussian pdf with vector length n mean mu covariance Sigma
z =  randn(n,N); % Generate N random samples of i.i.d. n-dimensional vectors vectors 
                 % with zero mean and identity covariance matrix
A = sqrtm(Sigma); % apply transformation by finding A that satisfies AA' = Sigma
x = A*z + repmat(mu,1,N);