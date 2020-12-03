% Add toolbox path 
path(path,'./toolbox/toolbox_general');
path(path,'./toolbox/toolbox_signal');


disp('Task1');
%...........................Task1: Sparse Regularization Principle  ............................
n =256;
% input image cameraman
f0 =double (imread('cameraman.png')) ;

% input image lena
%f0=double(imread('lena'));

f0 =f0(1:n,1:n) ;
clf;

% Show the cropped image
%imshow(f0,[]) ;

%............1.2 Damage the image...............
% What we need here is to change iterations in the optimization function
% and change the Phi parameter and see the results and discuss it in the
% report. 

%rho = 0.9 ; rho = 0.8 ; rho = 1; 
rho = 0.8;
Lambda = rand (n,n)>rho;

% Linear Operator(Phi) which will be applied on the image 
Phi = @(f)f.*Lambda;

% damaged image y. it is done using the linear operator Phi
y = Phi(f0);

% Show the damaged image
imshow(y,[]);

%....................1.2 Inpainting Processing.............

% L1 minimization,Soft Thresholding
SoftThresh=@(x,T)x.*max(0,1-T./max(abs(x),1e-10)) ;

% Set Wavelet parameters
Jmax = log2(n)-1;
Jmin = Jmax-3;
options.ti=0; % use orthogonality .
Psi = @(a)perform_wavelet_transf(a,Jmin,-1,options);
PsiS = @(f)perform_wavelet_transf(f,Jmin,+1,options );

SoftThreshPsi=@(f,T)Psi(SoftThresh(PsiS(f),T)) ;

% Show thresholded image
%imshow(SoftThreshPsi(y,.1),[]);

%%
% with different threshold values
niter = 10000;

%niter=500;

T = linspace(0,0.6,niter);

x=y;

for i=1:niter 
    x=SoftThreshPsi(x,T(i));
    x=x.*not(Lambda)+y;
end

% reconstructed image
imshow(x,[]);

%plot(T,vsnr);
%xlabel('T');
%ylabel('SNR');

% increase the reconstruction result quality using Soft Thresholding, try
%T = linspace(-1,1,1000);

% Plotting SNR 
%plot(T,SoftThresh(T,.5));
%%
