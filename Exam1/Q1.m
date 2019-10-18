m(:,1) = [-1;0]; Sigma(:,:,1) = 0.1*[10 -4;-4,5]; % mean and covariance of data pdf conditioned on label 3
m(:,2) = [1;0]; Sigma(:,:,2) = 0.1*[5 0;0,2]; % mean and covariance of data pdf conditioned on label 2
m(:,3) = [0;1]; Sigma(:,:,3) = 0.1*eye(2); % mean and covariance of data pdf conditioned on label 1
classPriors = [0.15,0.35,0.5]; classPriors1 = [0.15 0.35,0.51]; thr = [0,cumsum(classPriors1)];
N = 10000; u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
figure(1),clf, colorList = 'rbg';
for l = 1:3
    indices = find(thr(l)<=u & u<thr(l+1)); % fixed using classPriors1 adding a small term to last prior if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    L(1,indices) = l*ones(1,length(indices));
    x(:,indices) = mvnrnd(m(:,l),Sigma(:,:,l),length(indices))';
    figure(1), plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(l)); axis equal, hold on,
end
title('Original Class Distribution', 'fontsize', 16), xlabel('x_1'), ylabel('x_2');
disp("CLass 1: " + length(find(L==1))); disp("CLass 2: " +length(find(L==2))); disp("Class 3: " +length(find(L==3)));
%for l = 1:3
 %   indices = find(mvnpdf(x',m(:,l)',Sigma(:,:,l))*classPriors(l)>mvnpdf(x',m(:,mod(l+1,3))',Sigma(:,:,l+1))*classPriors(l+1) & mvnpdf(x',m(:,l)',Sigma(:,:,l))*classPriors(l)>mvnpdf(x',m(:,l+2)',Sigma(:,:,l+2))*classPriors(l+2));
%end
decision = zeros(1,N);
indices =  find(mvnpdf(x',m(:,1)',Sigma(:,:,1))*classPriors(1)>mvnpdf(x',m(:,2)',Sigma(:,:,2))*classPriors(2) & mvnpdf(x',m(:,1)',Sigma(:,:,1))*classPriors(1)>mvnpdf(x',m(:,3)',Sigma(:,:,3))*classPriors(3));
decision(1,indices) = 1*ones(1,length(indices));
indices =  find(mvnpdf(x',m(:,2)',Sigma(:,:,2))*classPriors(2)>mvnpdf(x',m(:,1)',Sigma(:,:,1))*classPriors(1) & mvnpdf(x',m(:,2)',Sigma(:,:,2))*classPriors(2)>mvnpdf(x',m(:,3)',Sigma(:,:,3))*classPriors(3));
decision(1,indices) = 2*ones(1,length(indices));
indices =  find(mvnpdf(x',m(:,3)',Sigma(:,:,3))*classPriors(3)>mvnpdf(x',m(:,1)',Sigma(:,:,1))*classPriors(1) & mvnpdf(x',m(:,3)',Sigma(:,:,3))*classPriors(3)>mvnpdf(x',m(:,2)',Sigma(:,:,2))*classPriors(2));
decision(1,indices) = 3*ones(1,length(indices));
confusion = zeros(3);
errors = 0;
for i = 1:3
    for j =1:3
        confusion(i,j) = length(find(decision==i & L==j));
        if i ~= j
            errors = errors+length(find(decision==i & L==j));
        end
    end
end


disp("Confusion Matrix for Rows as decision and columns as true Labels: " );
disp(confusion);
disp("Total Errors: "+errors);
disp("Probability of error: " + (errors/N)*100 + "%");



figure(2); hold on;
plot(x(1,find(L==1 & decision ==1)), x(2,find(L==1 & decision==1)), '.g');
plot(x(1,find(L==2 & decision ==2)), x(2,find(L==2 & decision==2)), 'xr');
plot(x(1,find(L==3 & decision ==3)), x(2,find(L==3 & decision==3)), '+b');

plot(x(1,find(L==1 & decision ==2)), x(2,find(L==1 & decision==2)), 'xg');
plot(x(1,find(L==1 & decision ==3)), x(2,find(L==1 & decision==3)), '+g');

plot(x(1,find(L==2 & decision ==1)), x(2,find(L==2 & decision==1)), '.r');
plot(x(1,find(L==2 & decision ==3)), x(2,find(L==2 & decision==3)), '+r');

plot(x(1,find(L==3 & decision ==1)), x(2,find(L==3 & decision==1)), '.b');
plot(x(1,find(L==3 & decision ==2)), x(2,find(L==3 & decision==2)), 'xb');

legend('Class 1 Decision 1', 'Class 2 Decision 2','Class 3 Decision 3', ...
'Class 1 Decision 2', 'Class 1 Decision 3', ... 
'Class 2 Decision 1',  'Class 2 Decision 3',...
'Class 3 Decision 1', 'Class 3 Decision 2')
title('Minimum Error decision for 3 class problem', 'fontsize', 16),
xlabel('x_1'), ylabel('x_2')
%plot(x(1,find(L==1 & decision ==1)), x(2,find(L==1 & decision==1)), '.');
figure(3);
hold on;
plot(x(1,find(L==1 & decision ==1)), x(2,find(L==1 & decision==1)), '.g');
plot(x(1,find(L==2 & decision ==2)), x(2,find(L==2 & decision==2)), 'xr');
plot(x(1,find(L==3 & decision ==3)), x(2,find(L==3 & decision==3)), '+b');
xlabel('x_1'), ylabel('x_2')
title('Minimum Error decision for 3 class problem', 'fontsize', 16),
title('Accurate Classifications only', 'fontsize', 16)
legend('Class 1 Decision 1', 'Class 2 Decision 2','Class 3 Decision 3')
