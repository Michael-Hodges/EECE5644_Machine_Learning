% MAP with 2 classes
clear all, close all,

%------------------------------------------------------------------------%
%Part 1

n = 2; % number of feature dimensions
N = 400; % number of iid samples
mu(:,1) = [0;0]; mu(:,2) = [3;3];
Sigma(:,:,1) = [1 0;0 1]; Sigma(:,:,2) = [1 0;0 1];
p = [0.5,0.5]; % class priors for labels 0 and 1 respectively
label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % save up space
% Draw samples from each class pdf
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end

pxw1 = mvnpdf(x',mu(:,1)',Sigma(:,:,1)); pxw2 = mvnpdf(x',mu(:,2)',Sigma(:,:,2));
pw1 = p(1); pw2 = (2);
px = pw1*pxw1 + pw2*pxw2;
pw1x = pw1*pxw1./px;
pw2x = pw2*pxw2./px;

decision = pw2x' > pw1x'; %choose whichever class is more likely based on the posterior after using bayes rule

ind00 = find(decision==0 & label==0); %p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); %p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); %p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); %p11 = length(ind11)/Nc(2); % probability of true positive
disp('error MAP = ');
disp(length(ind10)+length(ind01));

figure(1),
subplot(2,1,1),
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Question 2 Part 1 Class Visualization Scatter Plot'),
xlabel('x_1'), ylabel('x_2'), 

subplot(2,1,2), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
legend('class 0 correct', 'class 0 incorrect', 'class 1 incorrect', 'class 1 correct')
title('Question 2 Part 1 Class Decision Scatter Plot'),
xlabel('x_1'), ylabel('x_2'), 
axis equal,

mu0hat = mean(x(:,label==0),2); S0hat = cov(x(:,label==0)'); %estimated mean and covariance from the sampled data
mu1hat = mean(x(:,label==1),2); S1hat = cov(x(:,label==1)');

S_b = (mu0hat-mu1hat)*(mu0hat-mu1hat)'; %Scatter 1 wrt scatter 2
S_w = S0hat+S1hat;

[V,D] = eig(inv(S_w)*S_b);
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector

y = w'*x;
w = sign(mean(y(find(label==1)))-mean(y(find(label==0))))*w; % ensures class1 falls on the + side of the axis
y = sign(mean(y(find(label==1)))-mean(y(find(label==0))))*y; % flip y accordingly
mu0LDA = mean(y(find(label==0))); sig0LDA = (cov(y(find(label==0))));
mu1LDA = mean(y(find(label==1))); sig1LDA = (cov(y(find(label==1))));
decisionLDA = normpdf(y,mu1LDA,sig1LDA)>normpdf(y,mu0LDA,sig0LDA);

ind00 = find(label==0 & decisionLDA==0);
ind01 = find(label==0 & decisionLDA==1);
ind10 = find(label==1 & decisionLDA==0);
ind11 = find(label==1 & decisionLDA==1);

figure(7);
subplot(2,1,1),
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Question 2 Part 1 Class Visualization Scatter Plot'),
xlabel('x_1'), ylabel('x_2'), 
subplot(2,1,2), cla,
plot(y(ind00),zeros(1,length(y(ind00))),'og'), hold on,
plot(y(ind01),zeros(1,length(y(ind01))),'or');
plot(y(ind11),zeros(1,length(y(ind11))),'+g');
plot(y(ind10),zeros(1,length(y(ind10))),'+r');
legend('class 0 correct', 'class 0 incorrect', 'class 1 correct', 'class 1 incorrect')
title('Fisher LDA');
axis equal,
disp('error LDA =');
disp(length(ind01)+length(ind10))

%------------------------------------------------------------------------%
%Part 2
n = 2; % number of feature dimensions
N = 400; % number of iid samples
mu(:,1) = [0;0]; mu(:,2) = [3;3];
Sigma(:,:,1) = [3 1;1 0.8]; Sigma(:,:,2) = [3 1;1 0.8];
p = [0.5,0.5]; % class priors for labels 0 and 1 respectively
label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % save up space
% Draw samples from each class pdf
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end

pxw1 = mvnpdf(x',mu(:,1)',Sigma(:,:,1)); pxw2 = mvnpdf(x',mu(:,2)',Sigma(:,:,2));
pw1 = p(1); pw2 = (2);
px = pw1*pxw1 + pw2*pxw2;
pw1x = pw1*pxw1./px;
pw2x = pw2*pxw2./px;

decision = pw2x' > pw1x'; %choose whichever class is more likely based on the posterior after using bayes rule

ind00 = find(decision==0 & label==0); 
ind10 = find(decision==1 & label==0); 
ind01 = find(decision==0 & label==1); 
ind11 = find(decision==1 & label==1); 

disp('error MAP = ');
disp(length(ind10)+length(ind01))

figure(2),
subplot(2,1,1),
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Question 2 Part 2 Class Visualization Scatter Plot'),
xlabel('x_1'), ylabel('x_2'), 

subplot(2,1,2), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
legend('class 0 correct', 'class 0 incorrect', 'class 1 incorrect', 'class 1 correct')
title('Question 2 Part 2 Class Decision Scatter Plot'),
xlabel('x_1'), ylabel('x_2'), 
axis equal,


mu0hat = mean(x(:,label==0),2); S0hat = cov(x(:,label==0)'); %estimated mean and covariance from the sampled data
mu1hat = mean(x(:,label==1),2); S1hat = cov(x(:,label==1)');

S_b = (mu0hat-mu1hat)*(mu0hat-mu1hat)'; %Scatter 1 wrt scatter 2
S_w = S0hat+S1hat;

[V,D] = eig(inv(S_w)*S_b);
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector

y = w'*x;
w = sign(mean(y(find(label==1)))-mean(y(find(label==0))))*w; % ensures class1 falls on the + side of the axis
y = sign(mean(y(find(label==1)))-mean(y(find(label==0))))*y; % flip y accordingly
mu0LDA = mean(y(find(label==0))); sig0LDA = (cov(y(find(label==0))));
mu1LDA = mean(y(find(label==1))); sig1LDA = (cov(y(find(label==1))));
decisionLDA = normpdf(y,mu1LDA,sig1LDA)>normpdf(y,mu0LDA,sig0LDA);

ind00 = find(label==0 & decisionLDA==0);
ind01 = find(label==0 & decisionLDA==1);
ind10 = find(label==1 & decisionLDA==0);
ind11 = find(label==1 & decisionLDA==1);

figure(8);
subplot(2,1,1),
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Question 2 Part 2 Class Visualization Scatter Plot'),
xlabel('x_1'), ylabel('x_2'), 
subplot(2,1,2), cla,
plot(y(ind00),zeros(1,length(y(ind00))),'og'), hold on,
plot(y(ind01),zeros(1,length(y(ind01))),'or');
plot(y(ind11),zeros(1,length(y(ind11))),'+g');
plot(y(ind10),zeros(1,length(y(ind10))),'+r');
legend('class 0 correct', 'class 0 incorrect', 'class 1 correct', 'class 1 incorrect')
title('Fisher LDA');
axis equal,
disp('error LDA =');
disp(length(ind01)+length(ind10))


%------------------------------------------------------------------------%
%Part 3
n = 2; % number of feature dimensions
N = 400; % number of iid samples
mu(:,1) = [0;0]; mu(:,2) = [2;2];
Sigma(:,:,1) = [2,0.5;0.5,1]; Sigma(:,:,2) = [2,-1.9;-1.9,5];
p = [0.5,0.5]; % class priors for labels 0 and 1 respectively
label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % save up space
% Draw samples from each class pdf
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end

pxw1 = mvnpdf(x',mu(:,1)',Sigma(:,:,1)); pxw2 = mvnpdf(x',mu(:,2)',Sigma(:,:,2));
pw1 = p(1); pw2 = (2);
px = pw1*pxw1 + pw2*pxw2;
pw1x = pw1*pxw1./px;
pw2x = pw2*pxw2./px;

decision = pw2x' > pw1x'; %choose whichever class is more likely based on the posterior after using bayes rule

ind00 = find(decision==0 & label==0); %p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); %p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); %p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); %p11 = length(ind11)/Nc(2); % probability of true positive

disp('error MAP = ');
disp(length(ind10)+length(ind01))

figure(3),
subplot(2,1,1),
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Question 2 Part 3 Class Visualization Scatter Plot'),
xlabel('x_1'), ylabel('x_2'), 

subplot(2,1,2) % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
legend('class 0 correct', 'class 0 incorrect', 'class 1 incorrect', 'class 1 correct')
title('Question 2 Part 3 Class Decision Scatter Plot'),
xlabel('x_1'), ylabel('x_2'), 
axis equal,

mu0hat = mean(x(:,label==0),2); S0hat = cov(x(:,label==0)'); %estimated mean and covariance from the sampled data
mu1hat = mean(x(:,label==1),2); S1hat = cov(x(:,label==1)');

S_b = (mu0hat-mu1hat)*(mu0hat-mu1hat)'; %Scatter 1 wrt scatter 2
S_w = S0hat+S1hat;

[V,D] = eig(inv(S_w)*S_b);
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector

y = w'*x;
w = sign(mean(y(find(label==1)))-mean(y(find(label==0))))*w; % ensures class1 falls on the + side of the axis
y = sign(mean(y(find(label==1)))-mean(y(find(label==0))))*y; % flip y accordingly
mu0LDA = mean(y(find(label==0))); sig0LDA = (cov(y(find(label==0))));
mu1LDA = mean(y(find(label==1))); sig1LDA = (cov(y(find(label==1))));
decisionLDA = normpdf(y,mu1LDA,sig1LDA)>normpdf(y,mu0LDA,sig0LDA);

ind00 = find(label==0 & decisionLDA==0);
ind01 = find(label==0 & decisionLDA==1);
ind10 = find(label==1 & decisionLDA==0);
ind11 = find(label==1 & decisionLDA==1);

figure(9);
subplot(2,1,1),
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Question 2 Part 3 Class Visualization Scatter Plot'),
xlabel('x_1'), ylabel('x_2'), 
subplot(2,1,2), cla,
plot(y(ind00),zeros(1,length(y(ind00))),'og'), hold on,
plot(y(ind01),zeros(1,length(y(ind01))),'or');
plot(y(ind11),zeros(1,length(y(ind11))),'+g');
plot(y(ind10),zeros(1,length(y(ind10))),'+r');
legend('class 0 correct', 'class 0 incorrect', 'class 1 correct', 'class 1 incorrect')
title('Fisher LDA');
axis equal,
disp('error LDA =');
disp(length(ind01)+length(ind10))
%------------------------------------------------------------------------%
%Part 4

n = 2; % number of feature dimensions
N = 400; % number of iid samples
mu(:,1) = [0;0]; mu(:,2) = [3;3];
Sigma(:,:,1) = [1 0;0 1]; Sigma(:,:,2) = [1 0;0 1];
p = [0.05,0.95]; % class priors for labels 0 and 1 respectively
label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % save up space
% Draw samples from each class pdf
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end

pxw1 = mvnpdf(x',mu(:,1)',Sigma(:,:,1)); pxw2 = mvnpdf(x',mu(:,2)',Sigma(:,:,2));
pw1 = p(1); pw2 = (2);
px = pw1*pxw1 + pw2*pxw2;
pw1x = pw1*pxw1./px;
pw2x = pw2*pxw2./px;

decision = pw2x' > pw1x'; %choose whichever class is more likely based on the posterior after using bayes rule

ind00 = find(decision==0 & label==0); %p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); %p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); %p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); %p11 = length(ind11)/Nc(2); % probability of true positive

disp('error MAP = ');
disp(length(ind10)+length(ind01))

figure(4)
subplot(2,1,1),
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Question 2 Part 4 Class Visualization Scatter Plot'),
xlabel('x_1'), ylabel('x_2'), 

subplot(2,1,2), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
legend('class 0 correct', 'class 0 incorrect', 'class 1 incorrect', 'class 1 correct')
title('Question 2 Part 4 Class Decision Scatter Plot'),
xlabel('x_1'), ylabel('x_2'), 
axis equal,

mu0hat = mean(x(:,label==0),2); S0hat = cov(x(:,label==0)'); %estimated mean and covariance from the sampled data
mu1hat = mean(x(:,label==1),2); S1hat = cov(x(:,label==1)');

S_b = (mu0hat-mu1hat)*(mu0hat-mu1hat)'; %Scatter 1 wrt scatter 2
S_w = S0hat+S1hat;

[V,D] = eig(inv(S_w)*S_b);
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector

y = w'*x;
w = sign(mean(y(find(label==1)))-mean(y(find(label==0))))*w; % ensures class1 falls on the + side of the axis
y = sign(mean(y(find(label==1)))-mean(y(find(label==0))))*y; % flip y accordingly
mu0LDA = mean(y(find(label==0))); sig0LDA = (cov(y(find(label==0))));
mu1LDA = mean(y(find(label==1))); sig1LDA = (cov(y(find(label==1))));
decisionLDA = normpdf(y,mu1LDA,sig1LDA)>normpdf(y,mu0LDA,sig0LDA);

ind00 = find(label==0 & decisionLDA==0);
ind01 = find(label==0 & decisionLDA==1);
ind10 = find(label==1 & decisionLDA==0);
ind11 = find(label==1 & decisionLDA==1);

figure(10);
subplot(2,1,1),
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Question 2 Part 4 Class Visualization Scatter Plot'),
xlabel('x_1'), ylabel('x_2'), 
subplot(2,1,2), cla,
plot(y(ind00),zeros(1,length(y(ind00))),'og'), hold on,
plot(y(ind01),zeros(1,length(y(ind01))),'or');
plot(y(ind11),zeros(1,length(y(ind11))),'+g');
plot(y(ind10),zeros(1,length(y(ind10))),'+r');
legend('class 0 correct', 'class 0 incorrect', 'class 1 correct', 'class 1 incorrect')
title('Fisher LDA');
axis equal,
disp('error LDA =');
disp(length(ind01)+length(ind10))
%------------------------------------------------------------------------%
%Part 5
n = 2; % number of feature dimensions
N = 400; % number of iid samples
mu(:,1) = [0;0]; mu(:,2) = [3;3];
Sigma(:,:,1) = [3 1;1 0.8]; Sigma(:,:,2) = [3 1;1 0.8];
p = [0.05,0.95]; % class priors for labels 0 and 1 respectively
label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % save up space
% Draw samples from each class pdf
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end

pxw1 = mvnpdf(x',mu(:,1)',Sigma(:,:,1)); pxw2 = mvnpdf(x',mu(:,2)',Sigma(:,:,2));
pw1 = p(1); pw2 = (2);
px = pw1*pxw1 + pw2*pxw2;
pw1x = pw1*pxw1./px;
pw2x = pw2*pxw2./px;

decision = pw2x' > pw1x'; %choose whichever class is more likely based on the posterior after using bayes rule

ind00 = find(decision==0 & label==0); %p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); %p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); %p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); %p11 = length(ind11)/Nc(2); % probability of true positive

disp('error MAP = ');
disp(length(ind10)+length(ind01))

figure(5)
subplot(2,1,1),
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Question 2 Part 5 Class Visualization Scatter Plot'),
xlabel('x_1'), ylabel('x_2'), 

subplot(2,1,2), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
legend('class 0 correct', 'class 0 incorrect', 'class 1 incorrect', 'class 1 correct')
title('Question 2 Part 5 Class Decision Scatter Plot'),
xlabel('x_1'), ylabel('x_2'), 
axis equal,

mu0hat = mean(x(:,label==0),2); S0hat = cov(x(:,label==0)'); %estimated mean and covariance from the sampled data
mu1hat = mean(x(:,label==1),2); S1hat = cov(x(:,label==1)');

S_b = (mu0hat-mu1hat)*(mu0hat-mu1hat)'; %Scatter 1 wrt scatter 2
S_w = S0hat+S1hat;

[V,D] = eig(inv(S_w)*S_b);
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector

y = w'*x;
w = sign(mean(y(find(label==1)))-mean(y(find(label==0))))*w; % ensures class1 falls on the + side of the axis
y = sign(mean(y(find(label==1)))-mean(y(find(label==0))))*y; % flip y accordingly

mu0LDA = mean(y(find(label==0))); sig0LDA = (cov(y(find(label==0))));
mu1LDA = mean(y(find(label==1))); sig1LDA = (cov(y(find(label==1))));
decisionLDA = normpdf(y,mu1LDA,sig1LDA)>normpdf(y,mu0LDA,sig0LDA);

ind00 = find(label==0 & decisionLDA==0);
ind01 = find(label==0 & decisionLDA==1);
ind10 = find(label==1 & decisionLDA==0);
ind11 = find(label==1 & decisionLDA==1);

figure(11);
subplot(2,1,1),
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Question 2 Part 5 Class Visualization Scatter Plot'),
xlabel('x_1'), ylabel('x_2'), 
subplot(2,1,2), cla,
plot(y(ind00),zeros(1,length(y(ind00))),'og'), hold on,
plot(y(ind01),zeros(1,length(y(ind01))),'or');
plot(y(ind11),zeros(1,length(y(ind11))),'+g');
plot(y(ind10),zeros(1,length(y(ind10))),'+r');
legend('class 0 correct', 'class 0 incorrect', 'class 1 correct', 'class 1 incorrect')
title('Fisher LDA');
axis equal,
disp('error LDA =');
disp(length(ind01)+length(ind10))
%------------------------------------------------------------------------%
%Part 6
n = 2; % number of feature dimensions
N = 400; % number of iid samples
mu(:,1) = [0;0]; mu(:,2) = [2;2];
Sigma(:,:,1) = [2,0.5;0.5,1]; Sigma(:,:,2) = [2,-1.9;-1.9,5];
p = [0.05,0.95]; % class priors for labels 0 and 1 respectively
label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % save up space
% Draw samples from each class pdf
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end

pxw1 = mvnpdf(x',mu(:,1)',Sigma(:,:,1)); pxw2 = mvnpdf(x',mu(:,2)',Sigma(:,:,2));
pw1 = p(1); pw2 = (2);
px = pw1*pxw1 + pw2*pxw2;
pw1x = pw1*pxw1./px;
pw2x = pw2*pxw2./px;

decision = pw2x' > pw1x'; %choose whichever class is more likely based on the posterior after using bayes rule

ind00 = find(decision==0 & label==0); %p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); %p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); %p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); %p11 = length(ind11)/Nc(2); % probability of true positive

disp('error MAP = ');
disp(length(ind10)+length(ind01))

figure(6)
subplot(2,1,1),
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Question 2 Part 6 Class Visualization Scatter Plot'),
xlabel('x_1'), ylabel('x_2'), 

subplot(2,1,2), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
legend('class 0 correct', 'class 0 incorrect', 'class 1 incorrect', 'class 1 correct')
title('Question 2 Part 6 Class Decision Scatter Plot'),
xlabel('x_1'), ylabel('x_2'), 
axis equal,

mu0hat = mean(x(:,label==0),2); S0hat = cov(x(:,label==0)'); %estimated mean and covariance from the sampled data
mu1hat = mean(x(:,label==1),2); S1hat = cov(x(:,label==1)');

S_b = (mu0hat-mu1hat)*(mu0hat-mu1hat)'; %Scatter 1 wrt scatter 2
S_w = S0hat+S1hat;

[V,D] = eig(inv(S_w)*S_b);
[~,ind] = sort(diag(D),'descend');
w = V(:,ind(1)); % Fisher LDA projection vector

y = w'*x;
w = sign(mean(y(find(label==1)))-mean(y(find(label==0))))*w; % ensures class1 falls on the + side of the axis
y = sign(mean(y(find(label==1)))-mean(y(find(label==0))))*y; % flip y accordingly
mu0LDA = mean(y(find(label==0))); sig0LDA = (cov(y(find(label==0))));
mu1LDA = mean(y(find(label==1))); sig1LDA = (cov(y(find(label==1))));
decisionLDA = normpdf(y,mu1LDA,sig1LDA)>normpdf(y,mu0LDA,sig0LDA);

ind00 = find(label==0 & decisionLDA==0);
ind01 = find(label==0 & decisionLDA==1);
ind10 = find(label==1 & decisionLDA==0);
ind11 = find(label==1 & decisionLDA==1);

figure(12);
subplot(2,1,1),
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Question 2 Part 6 Class Visualization Scatter Plot'),
xlabel('x_1'), ylabel('x_2'), 
subplot(2,1,2), cla,
plot(y(ind00),zeros(1,length(y(ind00))),'og'), hold on,
plot(y(ind01),zeros(1,length(y(ind01))),'or');
plot(y(ind11),zeros(1,length(y(ind11))),'+g');
plot(y(ind10),zeros(1,length(y(ind10))),'+r');
legend('class 0 correct', 'class 0 incorrect', 'class 1 correct', 'class 1 incorrect')
title('Fisher LDA');
axis equal,
disp('error LDA =');
disp(length(ind01)+length(ind10))