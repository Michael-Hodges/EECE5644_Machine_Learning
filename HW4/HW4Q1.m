%clear all; close all;
% Import Image
I1 = imread('colorBird.jpg');
I2 = imread('colorPlane.jpg');
% Get width and height of image
Iwidth = length(I1(1,:,1));
Iheight = length(I1(:,1,1));

% Cast image to double for scaling
I1 = cast(I1, 'double'); 
Ivec = [reshape(I1(:,:,1),[],1),reshape(I1(:,:,2),[],1),reshape(I1(:,:,3),[],1)]; % Reshape image into an nx3 matrix of the form [R G B]
Ivec = Ivec./256; % Divide by max of uint8 (256) to normalize vector
indices = find(Ivec(:,1) == Ivec(:,1));
% [Vertical distance from corner, Horizontal distance from corner]
% Vertical and horizontal distance distance starts at 1
locFeat = [ceil(indices./(Iwidth)), indices-Iwidth*(ceil(indices./(Iwidth))-1)];
% Normalize distances
locFeat1 = [locFeat(:,1)./max(locFeat(:,1)), locFeat(:,2)./max(locFeat(:,2))];

% Develop Feauture Vecotrs in X as follows [Vertical Distance, Horizontal
% Distance, R, G, B]
X = [locFeat1, Ivec];

colors(1,:) = [256 0 0];
colors(2,:) = [0 256 0];
colors(3,:) = [0 0 256];
colors(4,:) = [256 256 0];
colors(5,:) = [0 256 256];
segment = zeros(Iheight,Iwidth,3);
figure(1);
for k  = 2:5
    [idx, C] = kmeans(X,k,'MaxIter',10000,'Replicates',5);
    idxReshape = reshape(idx,[],Iwidth);
    segment = zeros(Iheight,Iwidth,3);
    for j = 1:k
        tmp = idxReshape==j;
        tmp = repmat(tmp,[1 1 3]);
        segment(:,:,1) = segment(:,:,1) + colors(j,1).*tmp(:,:,1);
        segment(:,:,2) = segment(:,:,2) + colors(j,2).*tmp(:,:,2);
        segment(:,:,3) = segment(:,:,3) + colors(j,3).*tmp(:,:,3);
    end
    subplot(2,2,k-1);
    segment = cast(segment,'uint8');
    image(segment);
    xlabel('X Pixels'), ylabel('Y Pixels'), title(strcat('k-means with k = ', num2str(k)));
end

figure(2);
for k  = 2:5
    [~, EMpriors, EMmu, EMsigma] = EMforGMM_Edited(length(X(:,1)),length(X(1,:)),X',k);
    idx = evalTopGMM(X, EMpriors, EMmu, EMsigma);
    idxReshape = reshape(idx,[],Iwidth);
    segment = zeros(Iheight,Iwidth,3);
    for j = 1:k
        tmp = idxReshape==j;
        tmp = repmat(tmp,[1 1 3]);
        segment(:,:,1) = segment(:,:,1) + colors(j,1).*tmp(:,:,1);
        segment(:,:,2) = segment(:,:,2) + colors(j,2).*tmp(:,:,2);
        segment(:,:,3) = segment(:,:,3) + colors(j,3).*tmp(:,:,3);
    end
    subplot(2,2,k-1);
    segment = cast(segment,'uint8');
    image(segment);
    xlabel('X Pixels'), ylabel('Y Pixels'), title(strcat('GMM with  Model# = ', num2str(k)));
end


%Begin Plane here
Iwidth = length(I2(1,:,1));
Iheight = length(I2(:,1,1));

% Cast image to double for scaling
I2 = cast(I2, 'double'); 
Ivec = [reshape(I2(:,:,1),[],1),reshape(I2(:,:,2),[],1),reshape(I2(:,:,3),[],1)]; % Reshape image into an nx3 matrix of the form [R G B]
Ivec = Ivec./256; % Divide by max of uint8 (256) to normalize vector
indices = find(Ivec(:,1) == Ivec(:,1));
% [Vertical distance from corner, Horizontal distance from corner]
% Vertical and horizontal distance distance starts at 1
locFeat = [ceil(indices./(Iwidth)), indices-Iwidth*(ceil(indices./(Iwidth))-1)];
% Normalize distances
locFeat1 = [locFeat(:,1)./max(locFeat(:,1)), locFeat(:,2)./max(locFeat(:,2))];

% Develop Feauture Vecotrs in X as follows [Vertical Distance, Horizontal
% Distance, R, G, B]
X = [locFeat1, Ivec];
segment = zeros(Iheight,Iwidth,3);
figure(3);
for k  = 2:5
    [idx, C] = kmeans(X,k,'MaxIter',10000,'Replicates',5);
    idxReshape = reshape(idx,[],Iwidth);
    segment = zeros(Iheight,Iwidth,3);
    for j = 1:k
        tmp = idxReshape==j;
        tmp = repmat(tmp,[1 1 3]);
        segment(:,:,1) = segment(:,:,1) + colors(j,1).*tmp(:,:,1);
        segment(:,:,2) = segment(:,:,2) + colors(j,2).*tmp(:,:,2);
        segment(:,:,3) = segment(:,:,3) + colors(j,3).*tmp(:,:,3);
    end
    subplot(2,2,k-1);
    segment = cast(segment,'uint8');
    image(segment);
    xlabel('X Pixels'), ylabel('Y Pixels'), title(strcat('k-means with k = ', num2str(k)));
end

figure(4);
for k  = 2:5
    [~, EMpriors, EMmu, EMsigma] = EMforGMM_Edited(length(X(:,1)),length(X(1,:)),X',k);
    idx = evalTopGMM(X, EMpriors, EMmu, EMsigma);
    idxReshape = reshape(idx,[],Iwidth);
    segment = zeros(Iheight,Iwidth,3);
    for j = 1:k
        tmp = idxReshape==j;
        tmp = repmat(tmp,[1 1 3]);
        segment(:,:,1) = segment(:,:,1) + colors(j,1).*tmp(:,:,1);
        segment(:,:,2) = segment(:,:,2) + colors(j,2).*tmp(:,:,2);
        segment(:,:,3) = segment(:,:,3) + colors(j,3).*tmp(:,:,3);
    end
    subplot(2,2,k-1);
    segment = cast(segment,'uint8');
    image(segment);
    xlabel('X Pixels'), ylabel('Y Pixels'), title(strcat('GMM with  Model# = ', num2str(k)));
end


function model = evalTopGMM(x, alpha,mu,sigma)
tGMM = zeros(size(x,1),length(alpha));
for m = 1:length(alpha)
    tGMM(:,m) = alpha(m)*mvnpdf(x,mu(:,m)',sigma(:,:,m));
end
[row, col] = find(tGMM == max(tGMM, [], 2));
model(row) = col;
model = model';
% for m = 1:length(alpha)
%     tGMM(:,m
% end
end

%%%
function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end
%%%
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
