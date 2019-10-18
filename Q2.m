clear all; close all; clc;

%set tru object location within the unit circle
r_true = [0.5 -.2];
%values to iterate over when doing contour plot
x = linspace(-2,2,200);

[X, Y] = meshgrid(x);
nsig = 0.1; xsig = 0.25; ysig = 0.25;
%set landmark locations
k(:,:,1) = [1 0; 0 0; 0 0; 0 0];%set landmark on unit circle
k(:,:,2) = [1 0; -1 0; 0 0; 0 0];
k(:,:,3) = [1 0; cos(2*pi/3) sin(2*pi/3); cos(4*pi/3) sin(4*pi/3); 0 0];
k(:,:,4) = [1 0; 0 1; -1 0; 0 -1];
figure(1);
sgtitle('Contour Plots')
r = zeros(1,4);
for i = 1:4
    for j = 1:i
        tmp = -1; %set initial condition of loop to be negative.
                   %Loop continues to take samples until we get a positive
                   %range measurement
        while tmp<0
            tmp = norm(r_true(1,:)'-k(j,:,i)')+normrnd(0,nsig);
        end
        r(j) = tmp;
    end
    z = 0;
    for j =1:i
        z = z+(((r(j)-sqrt(((X-k(j,1,i)).^2) + ((Y-k(j,2,i)).^2))).^2)/nsig);%-(1/2).*[X Y];%.*inv([nsig 0; 0 nsig]).*[X Y]';
        %z = z-sqrt(((X-k(i,1,j)).^2) + ((Y-k(i,2,j)).^2));%-(1/2).*[X Y];%.*inv([nsig 0; 0 nsig]).*[X Y]';
    end
        z = z+(X.^2/xsig + Y.^2/ysig);
        subplot(2,2,i);hold on;
        [C,h] = contour(X,Y,z,[50 40 30 20 10 5]);
        clabel(C,h);
        plot(k(1:j,1,j),k(1:j,2,j),'ob','markersize',10,'MarkerFaceColor',[0.5,0.5,0.5]);
        plot(r_true(:,1),r_true(:,2),'+b','markersize',10,'MarkerFaceColor',[0.5,0.5,0.5]);
        title([num2str(i), ' landmark']);
        loc_string = num2str(r_true);
        legend('Contour Heights [50 40 30 20 10 5]','landmarks',['object - (' num2str(r_true(1)) ', ' num2str(r_true(2)) ')' ])
        xlabel('X'); ylabel('Y');
        axis([-2 2 -2 2]);
end


% figure(1); hold on;
% plot(r_true(:,1),r_true(:,2),'+');
% plot(k1(:,1),k1(:,2),'o');
% scatter(k2(:,1),k2(:,2),'o');
% scatter(k3(:,1),k3(:,2)'o');
% %scatter(k4(:,1),k4(:,2),'o');
% grid on;
