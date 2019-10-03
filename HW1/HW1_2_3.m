clear all, close all,

% set values for log-likelihood function
a1 = 0;
b1 = 1;
a2 = 1;
b2 = 2;
x = -10:10;

%log-likelihood ratio function
l=(abs(x-a2)/b2)-(abs(x-a1)/b1);

figure(1),
plot(x,l);
title("Log Likely Ratio",'FontSize',20);
xlabel("x",'FontSize',20);
ylabel("l(x)",'FontSize',20,'Rotation',0,'HorizontalAlignment','right');