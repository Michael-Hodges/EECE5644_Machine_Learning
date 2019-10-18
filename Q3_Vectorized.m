
clear all, close all; clc;
N = 100; % Number of Samples
x = (-1+2*rand(1,N))'; % Generate N samples from [-1,1]
x_vec = [x.^3 x.^2 x ones(1,N)']';
w = ([1000 -200 -640 128]/1000)'; % true coefficients w real roots at (x+.8)(x-.2)(x-.8)
y = zeros(1,N)'; % used to store values of y generated from plot
nsig = .075;
noise = normrnd(0,nsig,N,1);
y = x_vec'*w+noise;

x_true = linspace(-1, 1);
figure(1); hold on;
scatter(x,y);
plot(x_true,(x_true+0.8).*(x_true-.2).*(x_true-.8));
legend('samples','true');
title('Plot of samples generated and true polynomial');
xlabel('x'); ylabel('y');

iterations = 101;
l2Map_error = zeros(1,iterations);
C=3;
val = logspace(-C,C,iterations);
w_map = zeros(1,4);
for z = 1:iterations
    Gamma = val(z);
    w_map = ((x_vec*y)/nsig)'/((x_vec*x_vec')/nsig + inv(Gamma^2*eye(4)));
    w_map = w_map';
    l2Map_error(z) = norm(w-w_map);

end
figure(2);
semilogx(val,l2Map_error);
legend('L_2 error');
title('L_2 error with respect to changing values of \gamma');
xlabel('\gamma'); ylabel('Error');