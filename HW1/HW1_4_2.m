clear all, close all,

mu2 = 1;
sig2 = 2;
x= -5:.1:5;
l1cond = evalGaussian(x,0,1); %PDF of class conditional l=1 P(x|l=1)
l2cond = evalGaussian(x,mu2,sig2); %PDF of class conditionl l=2 P(x|l=2)
l1post = l1cond*(1/2);%PDF of class posterior l=1 P(l=1|x)
l2post = l2cond*(1/2);%PDF of class posterior l=2 P(l=2|x)
dERM = -(1/2)*log(2*pi*sig2)-((x-mu2).^2)/(2*sig2)+(1/2)*log(2*pi)+(x.^2)/2; %decision boundary that minimize risk
dERM2 = log(evalGaussian(x,mu2,sig2))-log(evalGaussian(x,0,1));
z = 4;
a = (1/2)-(1/(2*sig2));
b = (mu2)/sig2;
c = -mu2/(2*sig2)-(1/2)*log(2*pi*sig2)+(1/2)*log(2*pi);
p = [a b c];
r = roots(p);
%disp(r);
figure(1);
plot(x,l1cond,'color','r');
hold on;
plot(x,l2cond,'color','b');
plot(x,dERM,'color','g');
%vline can be found at the following link: 
%https://www.mathworks.com/matlabcentral/fileexchange/1039-hline-and-vline
vline(r(1),'k','');
vline(r(2),'k','');
patch([r(1) r(1) r(2) r(2)],[-1 1 1 -1],'red');
alpha(0.25);
axis([-5 5 -1 1]);
%plot(x,dERM2,'color','blue');
legend('P(x|l=1)','P(x|l=2)','Derm: Class 1 if Derm<0');
title("Decision Boundary with Class-Conditionals",'FontSize',20);
xlabel("x",'FontSize',12);
ylabel("f(x) for Derm(x) and PDF(x|L=l) l=1,2",'FontSize',12,'Rotation',90);

figure(2);
hold on;
plot(x,l1post,'color','r');
plot(x,l2post,'color','b');
plot(x,dERM,'color','g');
legend('P(l=1|x)','P(l=2|x)','Derm: Class 1 if Derm<0');
vline(r(1),'k','');
vline(r(2),'k','');
patch([r(1) r(1) r(2) r(2)],[-1 1 1 -1],'red');
alpha(0.25);
axis([-5 5 -1 1]);
title("Decision Boundary with Class-Posteriors",'FontSize',20);
xlabel("x",'FontSize',12);
ylabel("f(x) for Derm(x) and PDF(L=l|x) l=1,2",'FontSize',12,'Rotation',90);
