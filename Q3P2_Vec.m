
clear all, close all; clc;

N = 10; % Number of Samples
w = ([1000 -200 -640 128]/1000)'; % true coefficients w real roots at (x+.8)(x-.2)(x-.8)
y = zeros(1,N)'; % used to store values of y generated from plot
nsig = .075;

iterations = 101;
l2Map_error = zeros(1,iterations);
C=3;
val = logspace(-C,C,iterations);
w_map = zeros(1,4);
trials = 100;
stored = zeros(trials,iterations);

for l = 1:trials % number of trials to perform
    x = (-1+2*rand(1,N))'; % Generate N samples from [-1,1]
    x_vec = [x.^3 x.^2 x ones(1,N)']';
    noise = normrnd(0,nsig,N,1);
    y = x_vec'*w+noise;

    for z = 1:iterations
        Gamma = val(z);
        A = zeros(1,4);
        B = zeros(4);
        w_map = ((x_vec*y)/nsig)'/((x_vec*x_vec')/nsig + inv(Gamma^2*eye(4)));
        w_map = w_map';
        l2Map_error(z) = norm(w-w_map);
    end
    stored(l,:) = l2Map_error;
end
figure(1);
%semilogx(val,l2Map_error);
stored = stored.^2;
semilogx(val, median(stored));hold on;
semilogx(val, prctile(stored,75));
semilogx(val, prctile(stored, 25));
semilogx(val, min(stored));
semilogx(val, max(stored));
legend('median','75%', '25%', 'min', 'max');
xlabel('\gamma'); ylabel('L_2^2 error')
title('L_2^2 error as a function of \gamma')

% figure(2);
% boxplot(stored, val);
% set(gca,'xscale','log');

