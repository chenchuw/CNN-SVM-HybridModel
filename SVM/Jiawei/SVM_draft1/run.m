clc
clear

X = [1 2; 
     1 2; 
     3 4; 
     7 8; 
     7 8];
Y = [1; 
     1; 
     2; 
     3; 
     3];

% SVMModels = svm_multi(X,Y);
% t = [1 2;3 4;7 8];
% y_pred=svm_predict(SVMModels,t);
% disp(y_pred)

load("mnist.mat");
X=Xtr';
Y=ytr';
SVMModels = svm_multi(X,Y);
t = Xte;
y_pred=svm_predict(SVMModels,t);
disp(y_pred)