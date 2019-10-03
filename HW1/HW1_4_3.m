mu1 = 0;
sig1 = 1;
mu2 = 1;
sig2 = 2;

a = (1/2)-(1/(2*sig2));
b = (mu2)/sig2;
c = -mu2/(2*sig2)-(1/2)*log(2*pi*sig2)+(1/2)*log(2*pi);
p = [a b c];
r = roots(p);

l1fun = @(x) (1/sqrt(2*pi*sig1))*exp(-(1/2)*(1/sig1)*(x-mu1).^2);
l2fun = @(x) (1/sqrt(2*pi*sig2))*exp(-(1/2)*(1/sig2)*(x-mu2).^2); 
%Calculate error when choosing class 1 when should choose class two within
%the decision region
error1 = integral(l2fun,r(1),r(2))/(integral(l2fun,r(1),r(2))+(integral(l1fun,r(1),r(2))));
%Calculate error when chossing class 2 when should choos class 1 which
%extends to and from -infinity and infinity outside of the class 1 decision
%bound
y = integral(l1fun,-Inf,r(1))+integral(l1fun,r(2),Inf);
q = integral(l1fun,-Inf,r(1))+integral(l2fun,-Inf,r(1))+integral(l1fun,r(2),Inf)+integral(l2fun,r(2),Inf);
error2 = y/q;

error = error1+error2;
disp(error);
