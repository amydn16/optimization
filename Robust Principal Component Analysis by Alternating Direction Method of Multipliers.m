function [L,S] = RPCA_ADMM(X)

%% ADMM

[n1,n2] = size(X); % dimensions of X
beta = 1; % penalty parameter
lambda = 1/sqrt(max(n1,n2)); % parameter lambda in problem

L = zeros(n1,n2); % initialize matrix L
S = zeros(n1,n2); % initialize matrix S
Y = zeros(n1,n2); % initialize multiplier matrix y

%tol = 1e-2; % maximum tolerance
tol = 1e-3; % maximum tolerance
maxit = 1e4; % maximum number of iterations

iter_list = []; % list to hold number of outer loop iterations
pres_list = []; % list to hold primal residual
obj_list = []; % list to hold objective value
iter = 1; % initialize number of iterations

pres = norm(L+S-X, 'fro'); % compute primal residual

fprintf('%s %s %s\n', 'Iterations', 'Primal residual', 'Objective value');

while max(0,abs(pres)) > tol && iter < maxit

    [u, s, v] = svd(X-(Y/beta)-S,0); % compute reduced SVD of X - Y/beta - S
    % apply soft-thresholding operator to s
    [ns,~] = size(s); 
    for i = 1:ns
        s(i,i) = max(0, s(i,i) - 1/beta);
    end
    L = u*s*v'; % update L
    
    % compute proximal mapping of 1-norm 
    for i = 1:n2 % loop over columns of X - Y/beta - L
        for j = 1:n1 % loop over rows
            z = X(j,i) - (L(j,i) + (Y(j,i)/beta)); 
            S(j,i) = sign(z)*max(0, abs(z) - lambda/beta);
        end
    end
    
    Y = Y + beta*(L+S-X); % update multiplier
    pres = norm(L+S-X, 'fro'); % update primal residual
    
    iter_list = [iter_list; iter]; % record number of iterations
    pres_list = [pres_list; pres]; % record primal residual
    obj_list = [obj_list; norm(eig(s),1) + lambda*norm(S,1)]; % record objective value
    
    fprintf('%d %5.4e %5.4e\n', iter, pres, obj_list(iter));
    
    iter = iter + 1; % update number of iterations
end

% plot objective value and constraint violation
figure(1) % constraint violation
semilogy(pres_list, 'r', 'LineWidth',2);
xlabel('Number of iterations'); ylabel('Constraint violation (primal residual)'); set(gca,'FontSize',12);
figure(2) % objective value
semilogy(obj_list, 'b', 'LineWidth', 2);
xlabel('Number of iterations'); ylabel('Objective value'); set(gca,'FontSize',12);
end
