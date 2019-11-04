clear all; close all; clc
% Gaussian Mixture Model Parameters
% Means:
mu(:,1) = [2, 2];
mu(:,2) = [-2, 2];
mu(:,3) = [-2, -2];
mu(:,4) = [2, -2];

% Covariances:
Sigma(:,:,1) = [1 0; 0 1];
Sigma(:,:,2) = [0.1 0; 0 2];
Sigma(:,:,3) = [2 0; 0 0.1];
Sigma(:,:,4) = [1 -0.7; -0.7 1];

% Priors:
classPriors = [0.2 0.3 0.1 0.4]; % True Priors
classPriors1 = [0.25 0.25 0.25 0.251];
assert(max(cumsum(classPriors)) == 1, 'Priors do not equal 1');
thr = [0,cumsum(classPriors1)];

disp("True Means");
disp(mu);
disp("True Covariances:");
disp(Sigma);
disp("True Priors:");
disp(classPriors);
k = 10; % Perform k fold validation

%Generate Data:
% figure(1),clf, colorList = 'rbgy';
for i = 1:4
    N = 10^i; % Number of samples to generate for each set(10,100,1000,10000)
    u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
    for l = 1:4
        indices = find(thr(l)<=u & u<thr(l+1)); % fixed using classPriors1 adding a small term to last prior if u happens to be precisely 1, that sample will get omitted - needs to be fixed
        L(1,indices) = l*ones(1,length(indices));
        x(:,indices) = mvnrnd(mu(:,l),Sigma(:,:,l),length(indices))';
%         subplot(2,2,i);
%         plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(l)); axis equal, hold on,
%         axis([-10 10 -10 10]);   
    end
%     hold off;
    assert(isempty(find(L==0, 1)),'some values unclassified');
    x(3,:) = L;
    disp("Number Samples N = " + 10^i);
    disp("Class1: " + num2str(length(find(L==1))));
    disp("Class2: " + num2str(length(find(L==2))));
    disp("Class3: " + num2str(length(find(L==3))));
    disp("Class4: " + num2str(length(find(L==4))));


        % Loop through number of models to try 1-6
    regWeight = 1e-10; % regularization parameter for covariance estimates
    for l = 1:6 % number of models to loop through
        disp("performing: " + k + "-fold cross validation N=" + 10^i + ", Models =" + l);
        for q = 1:k % k fold cross validation
            disp(q+"th fold");
            dVal = x(:,1+(N/k)*(q-1):N/k*q);
%             disp("All " + num2str(x));
%             disp("Validation: " + num2str(dVal));
            dTrain = x;
            dTrain(:,1+(N/k)*(q-1):N/k*q) = [];
%             disp("Train: " + num2str(dTrain));
            % Set up initial GMM for l number of Gaussians
            EMdTrain = dTrain; % take train and set equal to new set to work with
            shuffledIndices = randperm(length(EMdTrain(1,:))); % Pick l random samples to be initial mean
            EM_mu = EMdTrain(1:2,shuffledIndices(1:l)); EM_mu_new = EM_mu; % starting means
%             [~,EMdTrain(3,:)] = min(pdist2(EM_mu',EMdTrain(1:2,:)'),[],1); % Set labels to nearest centroid
%             EM_Sigma = zeros(2,2,l); EM_Sigma_new = EM_Sigma;
%             EM_Prior = zeros(1,l); EM_Prior_new = EM_Prior;
            % initialize Priors
%             EM_Prior = ones(1,l)./l;
%             for b = 1:l
%                 EM_Prior(b) = length(find(EMdTrain(3,:)==b))/length(EMdTrain(3,:));
%             end
            % Initialize Covariances
%             for b = 1:l
%                 EM_Sigma(:,:,b) = cov(EMdTrain(1:2,find(EMdTrain(3,:)==b))') + regWeight*eye(2,2);% Generate covariances with given labels
%             end
%             epsilon = 1e-4; % convergence factor
%             Converged=0;
%             t = 0;
            % Try two initializations
            [EMlog1, EM_Prior1, EM_mu1, EM_Sigma1] = EMforGMM_Edited(length(EMdTrain(3,:)), EMdTrain(1:2,:), l);
            [EMlog2, EM_Prior2, EM_mu2, EM_Sigma2] = EMforGMM_Edited(length(EMdTrain(3,:)), EMdTrain(1:2,:), l);
            if EMlog1>EMlog2
                EM_Prior = EM_Prior1;
                EM_mu = EM_mu1;
                EM_Sigma = EM_Sigma1;
            else 
                EM_Prior = EM_Prior2;
                EM_mu = EM_mu2;
                EM_Sigma = EM_Sigma2;                
            end
            loglikelihood(l,q,i) = sum(log(evalGMM(dVal(1:2,:),EM_Prior,EM_mu,EM_Sigma)));  
        end
        disp(loglikelihood(:,:,i));
        %Store error here for each model
        
    end
    figure(2); hold on;
    subplot(2,2,i);
    means = mean(loglikelihood(:,:,i),2);
    plot(1:6,means');  
    xlabel("Gaussian Model Numbers");
    ylabel("log likelihood");
    title("Log Likelihood with N = " + 10^i);
    axis([1 6 -inf inf]);
    rangex1 = [min(x(1,:)),max(x(1,:))];
    rangex2 = [min(x(2,:)),max(x(2,:))];
    [x1Grid,x2Grid,zGMM] = contourGMM(EM_Prior,EM_mu,EM_Sigma,rangex1,rangex2);
    figure(3); 
    subplot(2,2,i); hold on;
    plot(x(1,find(L==1)),x(2,find(L==1)),'.r');
    plot(x(1,find(L==2)),x(2,find(L==2)),'.b');
    plot(x(1,find(L==3)),x(2,find(L==3)),'.g');
    plot(x(1,find(L==4)),x(2,find(L==4)),'.y');
%     contour(x1Grid,x2Grid,zGMM);    
    title("Number of samples: " + 10^i);
    xlabel("x1");
    ylabel("x2");
    axis('equal');
end

function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end

function [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2)
x1Grid = linspace(floor(rangex1(1)),ceil(rangex1(2)),101);
x2Grid = linspace(floor(rangex2(1)),ceil(rangex2(2)),91);
[h,v] = meshgrid(x1Grid,x2Grid);
GMM = evalGMM([h(:)';v(:)'],alpha, mu, Sigma);
zGMM = reshape(GMM,91,101);
end
