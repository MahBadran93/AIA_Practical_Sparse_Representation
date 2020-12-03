
% Add toolbox path 
path(path,'./toolbox/toolbox_general');
path(path,'./toolbox/toolbox_signal');
%.......................................................................................................
%.............................Task2: Primal-Dual Total Variation...........


disp('Task2');
%...........................Task1: Sparse Regularization Principle  ............................
n =256;
% input image cameraman and lena
%f0 =double (imread('cameraman.png')) ;
f0=double(load_image('lena'));

f0 =f0(1:n,1:n);
clf;

% Show the cropped image
imshow(f0,[]);

%............1.2 Damage the image...............
% What we need here is to change iterations in the optimization function
% and change the Phi parameter and see the results and discuss it in the
% report. 

%rho = 0.9 ; rho = 0.8 ; rho = 1; 
rho = 0.4;
Lambda = rand (n,n)>rho;

% Linear Operator(Phi) which will be applied on the image 
Phi = @(f)f.*Lambda;

% damaged image y. it is done using the linear operator Phi
y = Phi(f0);

% Show the damaged image
imshow(y,[]);

%#####################################################################################################

%......2.3: Inpainting using Primal-dual Total Variation Regularization scheme
% implement the primal-dual total variation algorithm
% where iH is the indicator function, we setup these parameters 
K=@(f)grad(f);
KS=@(u)-div(u);
Amplitude=@(u)sqrt(sum(u.^2 ,3));
% L1 minimisation 
F=@(u)sum(sum(Amplitude(u)));

% proximal operator
ProxF=@(u,lambda)max(0,1-lambda./repmat(Amplitude(u),[1 1 2])).*u;

% compute the proximal operator of the dual function F
ProxFS=@(y,sigma)y-sigma*ProxF(y/sigma,1/sigma);

% Compute the projection on H (iH is projecter on H) 
ProxG=@(f,tau)f+Phi(y-Phi(f));

% Set the parameters for the algorithm 
L = 8;
sigma = 10;
tau = .9/(L*sigma);
theta = 1;
f = y ;
g = K(y)*0;
f1 = f;

% Iterate, here we create a loop and then iterate(we choose number of
% iteration and analyse the different results)
iterationNum = 10000;
for i = 1:iterationNum
    fold=f;
    g = ProxFS(g+sigma*K(f1),sigma);
    f = ProxG(f-tau*KS(g),tau);
    f1 = f + theta*(f-fold);
    % ..........................
    E(i) = F(K(f));
    C(i) = snr(f0,f);
end
% Show The restorted Image

figure,imshow(f,[])
% clf;
figure
h = plot(E);
set(h, 'CameraManWidth', 2);
axis('tight');


%%
%........................Task3: Image Denoising via Sparse Representation

% 3.1.2 Apply Dictionary Learning for denozing(remove additive noise) 

% Compute the dictionary and add a gaussian noise on the input image
w = 10 ; % Width w of the patches.
n = w*w; % Dimension n=wxw of the data to be sparse coded.
p = 2*n ; % Number of atoms p in the dictionary.
m = 20*p ; % Number m of patches used for the training.
k = 4 ; % Target sparsity
%
sd = .06 ; % Gaussian n o i s e standard d e v i a t i o n
n0 = 256; % Image s i z e
f0 =  imresize(imread('lena.bmp'),[n0 n0]);
f = f0 + uint8(sd*randn(n0));
% Show Noisy Image
%imshow(f,[]);






