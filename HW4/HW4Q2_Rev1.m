% mu(:,1) = [-1;0]; mu(:,2) = [1;0]; 
% Sigma(:,:,1) = [2 0;0 1]; Sigma(:,:,2) = [1 0;0 4];
% p = [0.35,0.65]; % class priors for labels 0 and 1 respectively
% % Generate samples
% label = rand(1,N) >= p(1); l = 2*(label-0.5);
% Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
% x = zeros(n,N); % reserve space
% % Draw samples from each class pdf
% for lbl = 0:1
%     x(:,label==lbl) = randGaussian(Nc(lbl+1),mu(:,lbl+1),Sigma(:,:,lbl+1));
% end

close all, clear all,
N=1000; n = 2; K=10;
m = 2; % size of feature vector
% Class -1 setup
mu = zeros(1,m)';
Sigma = eye(m);

% Class +1 setup
m = zeros(m,m);
m(:,1) = [2,3]';
m(:,2) = [-pi, pi]';

classPriors = [0.35 0.65];
classPriors1 = [0.35 0.651];
assert(max(cumsum(classPriors)) == 1, 'Priors do not equal 1');
thr = [0,cumsum(classPriors1)];

%Generate Data:
% Defined above N = 1000; % Number of samples to generate for each set(10,100,1000,10000)
u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
figure(1),clf, colorList = 'rbgy', hold on;
for l = 1:2
    indices = find(thr(l)<=u & u<thr(l+1)); % fixed using classPriors1 adding a small term to last prior if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    L(1,indices) = l*ones(1,length(indices));
    if l == 1
        x(:,indices) = mvnrnd(mu(:,l),Sigma(:,:,l),length(indices))';
        plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(l)); axis equal;
    end
    if l == 2
        x(:,indices) = [(m(1,1) + (m(2,1)-m(1,1)).*rand(1,length(indices)))', (m(1,2)+ (m(2,2)-m(1,2))*rand(1,length(indices)))']';%r = a + (b-a).*rand(N,1)
        x(:,indices) = [(x(1,indices).*cos(x(2,indices)))',(x(1,indices).*sin(x(2,indices)))']';
        plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(l)), axis equal;
        %plot(x(1,indices).*cos(x(2,indices)),x(1,indices).*sin(x(2,indices)),'.','MarkerFaceColor',colorList(l)); axis equal;
    end
    
%     axis([-10 10 -10 10]);   
end
xlabel('x1'), ylabel('x2'), legend('Class "-"','Class "+"'), title('Data Original Distribution')

l = 2*(L-1.5);


% Train a Linear kernel SVM with cross-validation
% to select hyperparameters that minimize probability 
% of error (i.e. maximize accuracy; 0-1 loss scenario)
dummy = ceil(linspace(0,N,K+1));
for k = 1:K, indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; end,
CList = 10.^linspace(-3,7,11);
for CCounter = 1:length(CList)
    [CCounter,length(CList)],
    C = CList(CCounter);
    for k = 1:K
        indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
        xValidate = x(:,indValidate); % Using folk k as validation set
        lValidate = l(indValidate);
        if k == 1
            indTrain = [indPartitionLimits(k,2)+1:N];
        elseif k == K
            indTrain = [1:indPartitionLimits(k,1)-1];
        else
            indTrain = [indPartitionLimits(k-1,2)+1:indPartitionLimits(k+1,1)-1];
        end
        % using all other folds as training set
        xTrain = x(:,indTrain); lTrain = l(indTrain);
        SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','linear');
        dValidate = SVMk.predict(xValidate')'; % Labels of validation data using the trained SVM
        indCORRECT = find(lValidate.*dValidate == 1); 
        Ncorrect(k)=length(indCORRECT);
    end 
    PCorrect(CCounter)= sum(Ncorrect)/N; 
end 
disp(strcat('Minimum error linear CV',num2str(min(Ncorrect))));
figure(2), subplot(1,2,1),
plot(log10(CList),PCorrect,'.',log10(CList),PCorrect,'-'),
xlabel('log_{10} C'),ylabel('K-fold Validation Accuracy Estimate'),
title('Linear-SVM Cross-Val Accuracy Estimate'), %axis equal,
[dummy,indi] = max(PCorrect(:)); [indBestC, indBestSigma] = ind2sub(size(PCorrect),indi);
CBest= CList(indBestC); 
SVMBest = fitcsvm(x',l','BoxConstraint',CBest,'KernelFunction','linear');
d = SVMBest.predict(x')'; % Labels of training data using the trained SVM
indINCORRECT = find(l.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(l.*d == 1); % Find training samples that are correctly classified by the trained SVM
figure(2), subplot(1,2,2), 
plot(x(1,indCORRECT),x(2,indCORRECT),'g.'), hold on,
plot(x(1,indINCORRECT),x(2,indINCORRECT),'r.'), axis equal,
title('Training Data (RED: Incorrectly Classified)'),
disp('Cross-Fold Validation Gaussian Error');
pTrainingError = length(indINCORRECT)/N, % Empirical estimate of training error probability
Nx = 1001; Ny = 990; xGrid = linspace(-10,10,Nx); yGrid = linspace(-10,10,Ny);
[h,v] = meshgrid(xGrid,yGrid); dGrid = SVMBest.predict([h(:),v(:)]); zGrid = reshape(dGrid,Ny,Nx);
figure(2), subplot(1,2,2), contour(xGrid,yGrid,zGrid,0); xlabel('x1'), ylabel('x2'), axis equal,
CtrueLinear = CList(indBestC);

% Train a Gaussian kernel SVM with cross-validation
% to select hyperparameters that minimize probability 
% of error (i.e. maximize accuracy; 0-1 loss scenario)
dummy = ceil(linspace(0,N,K+1));
for k = 1:K, indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; end,
CList = 10.^linspace(-1,9,23); sigmaList = 10.^linspace(-2,3,23);
for sigmaCounter = 1:length(sigmaList)
    [sigmaCounter,length(sigmaList)],
    sigma = sigmaList(sigmaCounter);
    for CCounter = 1:length(CList)
        C = CList(CCounter);
        for k = 1:K
            indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
            xValidate = x(:,indValidate); % Using folk k as validation set
            lValidate = l(indValidate);
            if k == 1
                indTrain = [indPartitionLimits(k,2)+1:N];
            elseif k == K
                indTrain = [1:indPartitionLimits(k,1)-1];
            else
                indTrain = [indPartitionLimits(k-1,2)+1:indPartitionLimits(k+1,1)-1];
            end
            % using all other folds as training set
            xTrain = x(:,indTrain); lTrain = l(indTrain);
            SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','gaussian','KernelScale',sigma);
            dValidate = SVMk.predict(xValidate')'; % Labels of validation data using the trained SVM
            indCORRECT = find(lValidate.*dValidate == 1); 
            Ncorrect(k)=length(indCORRECT);
        end 
        PCorrect(CCounter,sigmaCounter)= sum(Ncorrect)/N;
    end 
end
disp(strcat('Minimum error Gaussian CV', num2str(min(Ncorrect))));

figure(3), subplot(1,2,1),
contour(log10(CList),log10(sigmaList),PCorrect',50); xlabel('log_{10} C'), ylabel('log_{10} sigma'),
title('Gaussian-SVM Cross-Val Accuracy Estimate'), axis equal,
[dummy,indi] = max(PCorrect(:)); [indBestC, indBestSigma] = ind2sub(size(PCorrect),indi);
CBest= CList(indBestC); sigmaBest= sigmaList(indBestSigma); 
SVMBest = fitcsvm(x',l','BoxConstraint',CBest,'KernelFunction','gaussian','KernelScale',sigmaBest);
d = SVMBest.predict(x')'; % Labels of training data using the trained SVM
indINCORRECT = find(l.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(l.*d == 1); % Find training samples that are correctly classified by the trained SVM
figure(3), subplot(1,2,2), 
plot(x(1,indCORRECT),x(2,indCORRECT),'g.'), hold on,
plot(x(1,indINCORRECT),x(2,indINCORRECT),'r.'), axis equal,
title('Training Data (RED: Incorrectly Classified)'),
disp('Cross-Fold Validation Gaussian Error');
pTrainingError = length(indINCORRECT)/N, % Empirical estimate of training error probability
Nx = 10000; Ny = 9900; xGrid = linspace(-10,10,Nx); yGrid = linspace(-10,10,Ny);
[h,v] = meshgrid(xGrid,yGrid); dGrid = SVMBest.predict([h(:),v(:)]); zGrid = reshape(dGrid,Ny,Nx);
figure(3), subplot(1,2,2), contour(xGrid,yGrid,zGrid,0); xlabel('x1'), ylabel('x2'), axis equal,
CTrue_Gaussian = CList(indBestC);
SigmaTrue_Guassian = sigmaList(indBestSigma);



%Generate new Data:
% Defined above N = 1000; % Number of samples to generate for each set(10,100,1000,10000)
u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
figure(4),clf, colorList = 'rbgy', hold on;
for l = 1:2
    indices = find(thr(l)<=u & u<thr(l+1)); % fixed using classPriors1 adding a small term to last prior if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    L(1,indices) = l*ones(1,length(indices));
    if l == 1
        x(:,indices) = mvnrnd(mu(:,l),Sigma(:,:,l),length(indices))';
        plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(l)); axis equal;
    end
    if l == 2
        x(:,indices) = [(m(1,1) + (m(2,1)-m(1,1)).*rand(1,length(indices)))', (m(1,2)+ (m(2,2)-m(1,2))*rand(1,length(indices)))']';%r = a + (b-a).*rand(N,1)
        x(:,indices) = [(x(1,indices).*cos(x(2,indices)))',(x(1,indices).*sin(x(2,indices)))']';
        plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(l)), axis equal;
        %plot(x(1,indices).*cos(x(2,indices)),x(1,indices).*sin(x(2,indices)),'.','MarkerFaceColor',colorList(l)); axis equal;
    end
    
%     axis([-10 10 -10 10]);   
end
xlabel('x1'), ylabel('x2'), legend('Class "-"','Class "+"'), title('Additional Data Original Distribution')

l = 2*(L-1.5);

SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',CtrueLinear,'KernelFunction','linear');
testValidate = SVMk.predict(x')'; % Labels of validation data using the trained SVM
indINCORRECT = find(l.*testValidate == -1); % Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(l.*testValidate == 1); 
Ncorrect(k)=length(indCORRECT);
figure(5);
plot(x(1,indCORRECT),x(2,indCORRECT),'g.'), hold on,
plot(x(1,indINCORRECT),x(2,indINCORRECT),'r.'), axis equal,
title('Training Data (RED: Incorrectly Classified)'),
disp('Linear SVM Error New Data');
pTrainingError = length(indINCORRECT)/N, % Empirical estimate of training error probability


SVMk = fitcsvm(x',l,'BoxConstraint',CTrue_Gaussian,'KernelFunction','gaussian','KernelScale',SigmaTrue_Guassian);
dValidate = SVMk.predict(x')'; % Labels of validation data using the trained SVM
indINCORRECT = find(l.*dValidate == -1); % Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(l.*dValidate == 1); 
Ncorrect(k)=length(indCORRECT);
figure(6);
plot(x(1,indCORRECT),x(2,indCORRECT),'g.'), hold on,
plot(x(1,indINCORRECT),x(2,indINCORRECT),'r.'), axis equal,
title('Training Data (RED: Incorrectly Classified)'),
disp('Gaussian SVM Error New Data');
pTrainingError = length(indINCORRECT)/N, % Empirical estimate of training error probability
