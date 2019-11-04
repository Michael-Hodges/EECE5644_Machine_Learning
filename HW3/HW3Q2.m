clear all; close all; clc
% Gaussian Mixture Model Parameters
% Means:
mu(:,1) = [3, 3]; % Q-
mu(:,2) = [0, 3]; %  Q+

% Covariances:
Sigma(:,:,1) = [1 .4; .4 1]; % Q-
Sigma(:,:,2) = [0.1 0; 0 2]; % Q+

% Priors:
classPriors = [0.3 0.7]; % True Priors
classPriors1 = [0.3 0.7];
assert(max(cumsum(classPriors)) == 1, 'Priors do not equal 1');
thr = [0,cumsum(classPriors1)];

N = 999; % Number of Samples to take
% Generate samples from each class
figure(1),clf, colorList = 'rbg';
subplot(2,2,1);
u = rand(1,N);
for l = 1:2
    indices = find(thr(l)<=u & u<thr(l+1)); % fixed using classPriors1 adding a small term to last prior if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    L(1,indices) = l*ones(1,length(indices));
    x(:,indices) = mvnrnd(mu(:,l),Sigma(:,:,l),length(indices))';
    plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(l)); axis equal, hold on,
    axis('equal');   
end
legend('class -', 'class +');
title('Original distribution');
xlabel('x1'), ylabel('x2');
assert(isempty(find(L==0, 1)),'some values unclassified');
x(3,:) = L;
disp("Number Samples N = " + N);
disp("Class1: " + num2str(length(find(L==1))));
disp("Class2: " + num2str(length(find(L==2))));

mu1hat = mean(x(find(L==1))); S1hat = cov(x(find(L==1)));
mu2hat = mean(x(find(L==1))); S2hat = cov(x(find(L==1)));

% Sb = (mu1hat-mu2hat)*(mu1hat-mu2hat)';
% Sw = S1hat + S2hat;
Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);

[V,D] = eig(inv(Sw)*Sb);
[~,ind] = sort(diag(D),'descend');
wLDA = V(:,ind(1)); % Fisher LDA projection vector

disp(wLDA)
yLDA = wLDA'*x(1:2,:); % All data projected on to the line spanned by wLDA
wLDA = sign(mean(yLDA(find(L==2)))-mean(yLDA(find(L==1))))*wLDA; % ensures class1 falls on the + side of the axis
yLDA = sign(mean(yLDA(find(L==2)))-mean(yLDA(find(L==1))))*yLDA; % flip yLDA accordingly
disp(wLDA)
% figure(2), clf,
% subplot(2,2,2);
% plot(yLDA(find(L==1)),zeros(1,length(find(L==1))),'o'), hold on,
% plot(yLDA(find(L==2)),zeros(1,length(find(L==2))),'+'), axis equal,
% legend('Class -','Class +'), 
% title('LDA projection of data and their true labels'),
% xlabel('x_1'), ylabel('x_2'), 
tau = 0;
decisionLDA = (yLDA >= -classPriors(1)/classPriors(2)); %classPriors(2)/classPriors(1)
decisionLDA = decisionLDA+1;

%Note: wLDA is w and classPriors(1)/classPriors(2) is b

ind00LDA = find(L==1 & decisionLDA==1);
ind01LDA = find(L==1 & decisionLDA==2);
ind10LDA = find(L==2 & decisionLDA==1);
ind11LDA = find(L==2 & decisionLDA==2);

disp((length(ind01LDA)+length(ind10LDA))/N);
% figure(3), clf,
% plot(yLDA(ind00),zeros(1,length(yLDA(ind00))),'og'), hold on,
% plot(yLDA(ind01),zeros(1,length(yLDA(ind01))),'or');
% plot(yLDA(ind11),zeros(1,length(yLDA(ind11))),'+g');
% plot(yLDA(ind10),zeros(1,length(yLDA(ind10))),'+r');
% legend('class - correct', 'class 0- incorrect', 'class + correct', 'class + incorrect')
% figure(4), clf,
subplot(2,2,2);
plot(x(1,ind00LDA),x(2,ind00LDA),'og'), hold on,
plot(x(1,ind11LDA),x(2,ind11LDA),'+g');
plot(x(1,ind01LDA),x(2,ind01LDA),'or');
plot(x(1,ind10LDA),x(2,ind10LDA),'+r');
title('Fisher LDA');
xlabel('x1'),ylabel('x2');
%hleg = legend('class - correct', 'class + correct', 'class - incorrect', 'class + incorrect');
%set(hleg,'fontsize',14)


%MAP 

lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * classPriors(1)/classPriors(2); %threshold
discriminantScore = log(mvnpdf(x(1:2,:)',mu(:,2)',Sigma(:,:,2)))-log(mvnpdf(x(1:2,:)',mu(:,1)',Sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));
decision = decision+1;

ind00MAP = find(decision'==1 & L==1); %p00 = length(ind00)/Nc(1); % probability of true negative
ind10MAP = find(decision'==2 & L==1); %p10 = length(ind10)/Nc(1); % probability of false positive
ind01MAP = find(decision'==1 & L==2); %p01 = length(ind01)/Nc(2); % probability of false negative
ind11MAP = find(decision'==2 & L==2); %p11 = length(ind11)/Nc(2); % probability of true positive
%p(error) = [p10,p01]*Nc'/N; % probability of error, empirically estimated

% figure(5), % class 0 circle, class 1 +, correct green, incorrect red
subplot(2,2,3);
plot(x(1,ind00MAP),x(2,ind00MAP),'og'); hold on,
plot(x(1,ind10MAP),x(2,ind10MAP),'or'); hold on,
plot(x(1,ind01MAP),x(2,ind01MAP),'+r'); hold on,
plot(x(1,ind11MAP),x(2,ind11MAP),'+g'); hold on,
title('MAP Classifier');
xlabel('x1'); ylabel('x2');
axis equal,

disp((length(ind01MAP)+length(ind10MAP))/N);

% Logarithmic classifier
% fun = @(w)for 1+exp(w(1)'*x(1:2,:)+w(2,1));
x(3,:) = 1;
w0 = [1 1 1];
%fun = @(w,x)sum(-classPriors(1)*log(1./(1+exp(w(1:2)*x(1:2,:)+w(3))))-classPriors(2)*log(1-1./(1+exp(w(1:2)*x(1:2,:)+w(3)))));
fun = @(w,x,L)sum( (-classPriors(L).*log(1./(1+exp(w*x)))) - ((1-classPriors(L)).*log(1-(1./(1+exp(w*x))))));
%fun = @(w,x)-sum(exp(w(1:2)*x(1:2,find(L==1))-w(3)))+sum(exp(w(1:2)*x(1:2,find(L==2))-w(3)));
f = @(w)fun(w,x,L);
w_l = fminsearch(f,w0);

logDscore = 1./(1+exp(w_l*x));
logDecision = (logDscore>=0.5);
logDecision = logDecision+1;

ind00log = find(logDecision==1 & L==1); %p00 = length(ind00)/Nc(1); % probability of true negative
ind10log = find(logDecision==2 & L==1); %p10 = length(ind10)/Nc(1); % probability of false positive
ind01log = find(logDecision==1 & L==2); %p01 = length(ind01)/Nc(2); % probability of false negative
ind11log = find(logDecision==2 & L==2); %p11 = length(ind11)/Nc(2); % probability of true positive

subplot(2,2,4);
plot(x(1,ind00log),x(2,ind00log),'og'); hold on,
plot(x(1,ind10log),x(2,ind10log),'or'); hold on,
plot(x(1,ind01log),x(2,ind01log),'+r'); hold on,
plot(x(1,ind11log),x(2,ind11log),'+g'); hold on,
title('Log Classifier');
xlabel('x1'); ylabel('x2');
axis equal,
