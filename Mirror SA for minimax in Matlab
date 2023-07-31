%% Mirror SA for minimax problem
function [x,runtime,outputs] = minimaxMSA(H,c,Q,a,b,x,z,lbx,ubx,m,N,params)

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

d = norm(ubx - lbx); % approximate maximum distance attained in X
Mstar = 2; % approximate square root of E[gradx(F)], E[gradz(F)]
gamma = (2*params.theta*d)/((5*Mstar*maxepoch*maxiter)^0.5); 

for anepoch = 1:maxepoch
    
    xfinal = 0; % initialize x outputted after each epoch
    zfinal = 0; % initialize z outputted after each epoch

    for aniter = 0:maxiter
        
        xfinal = xfinal + x; % update xfinal
        zfinal = zfinal + z; % update zfinal
        
        f = getf(H,c,x,N); % compute f at x
        outputs.f = [outputs.f f]; % update outputs.f
        
        % randomly select batchsize many integers from [1,m]
        jks = randsample(m,batchsize);
        
        hx = 0; % initialize hx as zero 
        hz = 0; % initialize hz as zero
        
        violbatch = 0; % initialize constraint violation for minibatch
        violmax = 0; % initialize maximum constraint violation for minibatch
        
        for i = 1:batchsize
            jk = jks(i); % get ith element of jks
            
            % compute gradx(F_jk) at x, update hx
            hx = hx + getdFx(H,c,Q,a,x,z,jk);
            % compute gradz(F_jk) at x, update hz
            hz = hz - getdFzj(Q,a,b,x,jk); 

            aviol = getviol(Q,a,b,x,jk); % compute jkth violation at x
            violbatch = violbatch + aviol; % update violbatch
            if (aviol >= violmax) % jkth violation larger than violmax
                violmax = aviol; % update violmax
            end
        end
        
        hx = hx/batchsize; % average hx
        hz = hz/batchsize; % average hz
        
        violbatch = violbatch/batchsize; % average violbatch
        outputs.viol = [outputs.viol violbatch]; % update outputs.viol
        outputs.violmax = [outputs.violmax violmax]; % update outputs.violmax
        
        fprintf('Epochs: %d, Iterations: %d, f = %8.4e, avg constr viol = %8.4e, max constr viol = %8.4e\n',...
            anepoch, aniter, f, violbatch, violmax);
        
        if (aniter == maxiter)
            break % terminate once algorithm has done maxiter many iterations
        else % execute updates
            
            x = x - gamma*hx; % update x 
            x = min(ubx, max(lbx, x)); % projection mapping of x into X 
            z = z - gamma*hz*ones(size(z)); % update z
        end
    end
   
    xfinal = xfinal/maxiter; % compute xfinal at end of epoch
    x = xfinal; % update x
    outputs.x = [outputs.x xfinal]; % update outputs.x
    
    zfinal = zfinal/maxiter; % compute zfinal at end of epoch
    z = zfinal; % update z
    
    f = getf(H,c,x,N); % update z
    outputs.favg = [outputs.favg f]; % update outputs.favg
    
    violmax = 0; % initialize maximum constraint violation after an epoch
    violtot = 0; % initialize total constraint violation after an epoch
    
    for i = 1:m
        aviol = getviol(Q,a,b,x,i); % compute ith violation
        violtot = violtot + aviol; % update violtot       
        if (aviol >= violmax) % ith violation larger than violmax
            violmax = aviol; % update violmax
        end
    end
    
    violtot = violtot/m; % average violtot
    outputs.violavg = [outputs.violavg violtot]; % update outputs.violavg
    outputs.violmaxavg = [outputs.violmaxavg violmax]; % update outputs.violavgmax
    
    fprintf('Epochs: %d, f = %8.4e, avg constr viol = %8.4e, max constr viol = %8.4e\n',...
        anepoch, f, violtot, violmax);
    
    runtime = toc(t0); % stop recording time, return
end
end

%% Functions to call within algorithm

% objective function f0 = (sum^{N}_{i=1} ||H_i*x - c_i||^2)/2N
function f = getf(H,c,x,N) 
result = 0; % initialize result as 0
for i = 1:N
    result = result + norm(H{i}*x - c{i})^2; % add ||H_i x - c_i||^2
end
f = result/(2*N); % take average of result, halve it, return
end

% jth constraint f_j = 0.5*x'*Q_j*x + a_j'*x - b_j, j >= 1
function fconstr = getfconstr(Q,a,b,x,j)
fconstr = 0.5*x'*Q{j}*x + a{j}'*x - b(j); % compute f_j, return
end

% jth constraint violation f_j = 0.5*x'*Q_j*x + a_j'*x - b_j =< 0
function viol = getviol(Q,a,b,x,j)
result = getfconstr(Q,a,b,x,j); % get f_j
if (result <= 0)
    viol = 0; % f_j =< 0 implies no violation
else
    viol = result; % nonzero violation if f_j > 0
end
end

% F_j = ||H_j*x - c_j||^2)/2 + z_j*(0.5*x'*Q_j*x + a_j'*x - b_j), j >= 1
% get gradient of F_j wrt x: grad_x(F_j) = H_j'*(H_j*x - c_j) + z_j*(Q_j*x + a_j)
function dFx = getdFx(H,c,Q,a,x,z,j)
dFx = H{j}'*(H{j}*x - c{j}) + z(j)*(Q{j}*x + a{j}); % compute dFx, return
end

% get gradient of F_j wrt z_j: grad_z_j(F_j) = 0.5*x'*Q_j*x + a_j'*x - b_j
function dFzj = getdFzj(Q,a,b,x,j)
dFzj = getfconstr(Q,a,b,x,j); % it equals f_j; compute, return
end
