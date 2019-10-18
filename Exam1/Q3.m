
clear all, close all; clc;
N = 10; % Number of Samples
x = (-1+2*rand(1,N))'; % Generate N samples from [-1,1]
x_vec = [x.^3 x.^2 x ones(1,N)']';
w = ([1000 -200 -640 128]/1000)'; % true coefficients w real roots at (x+.8)(x-.2)(x-.8)
y = zeros(1,N); % used to store values of y generated from plot
nsig = .075;
for i = 1:N
    y(i) = w'*x_vec(:,i)+normrnd(0,nsig);
end
y = y';
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
    A = zeros(1,4);
    B = zeros(4);
    for i = 1:N
        A = A+(y(i)*(x_vec(:,i)'))/nsig;
        B = B+(x_vec(:,i)*x_vec(:,i)')/nsig;
    end
    w_map = A/(B+inv(Gamma^2*eye(4)));
    w_map = w_map';
    l2Map_error(z) = norm(w-w_map);
%     x_test = linspace(-1,1)';
%     x_tVec = [x_test.^3 x_test.^2 x_test ones(1,length(x_test))']';
%     test_wMap = repmat(w_map,1,size(w_map,4));
%     
% 
%     plot(x_test,w_map(1)*x_test.^3 + w_map(2)*x_test.^2 + w_map(3)*x_test+w_map(4));
    %legend(['gamma = ' +num2str(gamma)]);
end
figure(2);
semilogx(val,l2Map_error);
legend('L_2 error');
title('L_2 error with respect to changing values of \gamma');
xlabel('\gamma'); ylabel('Error');
