function [x,runtime,outputs] = expectCSA(H,c,Q,a,b,x,lb,ub,m,N,params)

t0 = tic; % start cpu timer

outputs = []; % array to hold outputs
outputs.x = []; % array to hold x at each epoch
outputs.f = []; % holds objective function value at each iteration
outputs.favg = []; % holds objective function value at each epoch
outputs.viol = []; % holds average constraint violation at each iteration
outputs.violmax = []; % holds maximum constraint violation at each iteration
outputs.violavg = []; % holds constraint violation at each epoch
outputs.violmaxavg = []; % holds maximum constraint violation at each epoch

maxepoch = params.maxepoch; % maximum number of epochs
maxiter = params.maxiter; % maximum number of iterations per epoch
batchsize = params.batchsize; % size of each minibatch

alpha = params.alpha0/(params.K^0.5); % update step size for x
eta = 1/(params.batchsize^0.5); % tolerance for constraint violations

s = 1; % minimum number of iterations done before adding x to xfinal

for anepoch = 1:maxepoch
    
    xfinal = 0; % initialize x outputted after each epoch

    for aniter = 0:maxiter
        
        f = getf(H,c,x,N); % compute f at x
        outputs.f = [outputs.f f]; % update outputs.f
        
        % randomly select batchsize many integers from [1,m]
        jks = randsample(m,batchsize);
        
        h1 = 0; % initialize h1 as zero
        h2 = 0; % initialize h2 as zero
        
        violbatch = 0; % initialize constraint violation for minibatch
        violmax = 0; % initialize maximum constraint violation for minibatch
        
        for i = 1:batchsize
            jk = jks(i); % get ith element of jks

            aviol = getfconstr(Q,a,b,x,jk); % compute jkth violation at x
            violbatch = violbatch + aviol; % update violbatch
            if (aviol >= violmax) % jkth violation larger than violmax
                violmax = aviol; % update violmax
            end

            % compute stochastic subgradient of f_0jk at x, update h
            h1 = h1 + getdfj(H,c,x,jk); 
            % compute stochastic subgradient of f_jk at x, update h
            h2 = h2 + getdfconstr(Q,a,b,x,jk); 
        end
        
        violbatch = violbatch/batchsize; % average violbatch
        outputs.viol = [outputs.viol violbatch]; % update outputs.viol
        outputs.violmax = [outputs.violmax violmax]; % update outputs.violmax
        
        if violbatch <= eta % eta is tolerance for violbatch
            h = h1/batchsize; % use f_0 for stochastic subgradients
            
            % suitable number of iterations have been done: update xfinal
            if (aniter >= s) && (aniter <= maxiter) 
                xfinal = xfinal + x;
            end
            
        else % use f_j for stochastic subgradients
            h = h2/batchsize; 
        end
        
        fprintf('Epochs: %d, Iterations: %d, f = %8.4e, avg constr viol = %8.4e, max constr viol = %8.4e\n',...
            anepoch, aniter, f, violbatch, violmax);
        
        if (aniter == maxiter)
            break % terminate once algorithm has done maxiter many iterations
        else % execute updates

        x = x - alpha*h; % update x
        x = min(ub, max(lb, x)); % projection mapping of x into X
        end
    end
    
    xfinal = xfinal/maxiter; % average xfinal
    x = xfinal; % update x
    outputs.x = [outputs.x xfinal]; % update outputs.x
    
    f = getf(H,c,x,N); % update f
    outputs.favg = [outputs.favg f]; % update outputs.favg
    
    violmax = 0; % initialize maximum constraint violation after an epoch
    violtot = 0; % initialize total constraint violation after an epoch
    
    for i = 1:m
        aviol = getfconstr(Q,a,b,x,i); % compute ith violation at x
        violtot = violtot + aviol; % update violtot    
        if (aviol >= violmax) % jkth violation larger than violmax
            violmax = aviol; % update violmax
        end
    end

    violtot = violtot/m; % average violtot
    outputs.violavg = [outputs.violavg violtot]; % update outputs.violavg
    outputs.violmaxavg = [outputs.violmaxavg violmax]; % update outputs.violmax
    
    fprintf('Epochs: %d, f = %8.4e, avg constr viol = %8.4e, max constr viol = %8.4e\n',...
        anepoch, f, violtot, violmax); 
    
    runtime = toc(t0); % stop recording time, return
end
end

%% functions to call within algorithm

% objective function f0 = (sum^{N}_{i=1} ||H_i x - c_i||^2)/2N
function f = getf(H,c,x,N) 
result = 0; % initialize result as zero
for i = 1:N
    result = result + norm(H{i}*x - c{i})^2; % add ||H_i x - c_i||^2
end
f = result/(2*N); % average result, halve it, return
end

% gradient of f_0j: grad(f_0j) = H_i'*(H_i x - c_i)
function dfj = getdfj(H,c,x,j)
dfj = H{j}'*(H{j}*x - c{j}); % compute grad(f_0j), return
end

% jth constraint f_j = [x'*Q_j*x + a_j'*x - b_j]_+
function fconstr = getfconstr(Q,a,b,x,j)
fconstr = max(0, 0.5*x'*Q{j}*x + a{j}'*x - b(j)); % return value at x
end

% gradient of jth constraint
function dfconstr = getdfconstr(Q,a,b,x,j)
fconstr = getfconstr(Q,a,b,x,j); % get jth constraint violation at x
if fconstr == 0 % no violation
    dfconstr = zeros(size(a{1})); % gradient is zero vector
else % violation > 0
    dfconstr = Q{j}*x + a{j}; % gradient of f_j when f_j > 0
end
end
