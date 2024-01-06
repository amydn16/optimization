function [Jres,samples,runtime] = Qfactor_sim(N,n,m)
clc;

Jres = cell(n,N+1); % cell to hold results for J at each state

for k = N:-1:0 % loop through stages k = 0,..., N-1 and last (Nth) stage
    
    if k == N % at the end of the horizon
        for x = 1:n % loop through states in state space
            bp = base_pol(x); % get base policy
            w = disturb(bp); % get w
            J = cost_fun(k,x,bp,w,N); % get cost-to-go
            Jres{x,k+1} = J; % store J at state x
        end
    
    elseif k == N-1 % at last stage
        for x = 1:n % loop through states in state space
            
            if x == 1 % at first state
                
                u = 'R'; % can only turn right
                w = disturb(u); % get w        
                g = cost_fun(k,x,u,w,N); % get cost
                
                f = state_trans(x,w); % state at (k+1)th stage, transitioned from kth stage
                J = Jres{f,k+2}; % get J at (k+1)th stage and state f
                Jres{x,k+1} = g + J; % store J at kth stage and state x     

            elseif x == n % at last state
                
                u = 'L'; % can only turn left
                w = disturb(u); % get w      
                g = cost_fun(k,x,u,w,N); % get cost
                
                f = state_trans(x,w); % state at (k+1)th stage, transitioned from kth stage
                J = Jres{f,k+2}; % get J at (k+1)th stage and state f
                Jres{x,k+1} = g + J; % store J at kth stage and state x
                
            else % at any other state x
                
                u = 'R'; % try turning right
                w = disturb(u); % get w
                gr = cost_fun(k,x,u,w,N); % get cost
                
                u = 'L'; % try turning left
                w = disturb(u); % get w
                gl = cost_fun(k,x,u,w,N); % get cost
                
                bp = base_pol(x); % get base policy for J
                w = disturb(bp); % get w based on base policy
                f = state_trans(x,w); % state at (k+1)th stage, transitioned from kth stage
                
                J = Jres{f,k+2}; % get J at (k+1)th stage and state f
                Jres{x,k+1} = min(gr,gl) + J; % store J at kth stage and state x
            end
        end
        
    else % at any other kth stage
        for x = 1:n % loop through states in state space
            
            if x == 1 % at first state
                
                u = 'R'; % can only turn right
                w = disturb(u); % get w        
                g = cost_fun(k,x,u,w,N); % get cost
                
                f = state_trans(x,w); % state at (k+1)th stage, transitioned from kth stage
                J = Jres{f,k+2}; % get J at (k+1)th stage and state f
                Jres{x,k+1} = g + J; % store J at kth stage and state x
                
            elseif x == n % at last state
                
                u = 'L'; % can only turn left
                w = disturb(u); % get w
                g = cost_fun(k,x,u,w,N); % get cost
                
                f = state_trans(x,w); % state at (k+1)th stage, transitioned from kth stage
                J = Jres{f,k+2}; % get J at (k+1)th stage and state f
                Jres{x,k+1} = g + J; % store J at kth stage and state x
                
            else % any other kth state
                
                u = 'R'; % try turning right
                w = disturb(u); % get w
                gr = cost_fun(k,x,u,w,N); % get cost
                
                u = 'L'; % try turning left
                w = disturb(u); % get w
                gl = cost_fun(k,x,u,w,N); % get cost
                
                bp = base_pol(x); % get base policy for J
                w = disturb(bp); % get w based on base policy
                f = state_trans(x,w); % state at (k+1)th stage, transitioned from kth stage
                
                J = Jres{f,k+2}; % get J at (k+1)th stage and state x = f
                Jres{x,k+1} = min(gr,gl) + J; % compute and store J at kth stage                 
            end
        end
    end
end

% simulation-implemented stochastic rollout
statesl = []; % initialize array to hold trajectories with initial state 'L'
statesr = []; % initialize array to hold trajectories with initial state 'R'
consl = []; % initialize array to hold control sequences with initial state 'L'
consr = []; % initialize array to hold control sequences with initial state 'R'
gsl = []; % initialize array to hold costs over trajectories with initial state 'L'
gsr = []; % initialize array to hold costs over trajectories with initial state 'R'

% comment out this section or next depending on whether non-adaptive or
% adaptive sampling is done

%% perform non-adaptive sampling

t0 = tic;

% generate sample trajectories
for x = 1:n % loop through inner states
    for i = 1:m % loop through number of samples
        
        if x ~= 1 && x ~= n
            [states,conts,gs] = getpath(x,'L',0,N,n); % get trajectories with initial state 'L'
            statesl = [statesl, states]; % update statesl
            consl = [consl, conts]; % update consl
            gsl = [gsl, gs]; % update gsl
        
            [states,conts,gs] = getpath(x,'R',0,N,n); % get trajectories with initial state 'R'
            statesr = [statesr, states]; % update statesr
            consr = [consr, conts]; % update consr
            gsr = [gsr, gs]; % update gsr
            
        elseif x == 1
            [states,conts,gs] = getpath(x,'R',0,N,n); % get trajectories with initial state 'R'
            statesr = [statesr, states]; % update statesr
            consr = [consr, conts]; % update consr
            gsr = [gsr, gs]; % update gsr
            
        elseif x == n
            [states,conts,gs] = getpath(x,'L',0,N,n); % get trajectories with initial state 'L'
            statesl = [statesl, states]; % update statesl
            consl = [consl, conts]; % update consl
            gsl = [gsl, gs]; % update gsl
        end
    end
end

samples = cell(n,2); % initialize cell to hold results of sampling
for x = 1:n % loop through inner states
    Qresl = []; % initialize array to hold Q-values with u = 'L'
    Qresr = []; % initialize array to hold Q-values with u = 'R'
    
    for i = 1:(n-1)*m % loop through all sample trajectories generated
        for j = 1:N % loop through all stages in each trajectory
         
            if statesl(j,i) == x
                
                if x ~= 1 && x ~= n
                    if consl(j,i) == 'L' % store Q-value in Qresl
                        Qresl = [Qresl; gsl(j,i) + Jres{x,j}]; 
                    elseif consl(j,i) == 'R' % store Q-value in Qresr
                        Qresr = [Qresr; gsl(j,i) + Jres{x,j}];
                    end
                    
                elseif x == 1
                    Qresr = [Qresr; gsl(j,i) + Jres{x,j}];
                    
                elseif x == n
                    Qresl = [Qresl; gsl(j,i) + Jres{x,j}];
                end
            end
            
            if statesr(j,i) == x 
                
                if x ~= 1 && x ~= n
                    if consr(j,i) == 'L' % store Q-value in Qresl
                        Qresl = [Qresl; gsr(j,i) + Jres{x,j}]; 
                    elseif consr(j,i) == 'R' % store Q-value in Qresr
                        Qresr = [Qresr; gsr(j,i) + Jres{x,j}];
                    end
                    
                elseif x == 1
                    Qresr = [Qresr; gsr(j,i) + Jres{x,j}];
                    
                elseif x == n
                    Qresl = [Qresl; gsr(j,i) + Jres{x,j}];
                end
            end
        end
    end
    
    samples{x,1} = min(Qresl); % take the minimum of Q-values with u = L
    samples{x,2} = min(Qresr); % take the minimum of Q-values with u = R
end

runtime = toc(t0);

% %% perform adaptive sampling 
% 
% t0 = tic;
% 
% % generate sample trajectories 
% for x = 1:n % loop through inner states
%     for i = 1:m % loop through number of samples
%         
%         if x ~= 1 && x ~= n
%             [states,conts,gs] = getpath(x,'L',0,N,n); % get trajectories with initial state 'L'
%             statesl = [statesl, states]; % update statesl
%             consl = [consl, conts]; % update consl
%             gsl = [gsl, gs]; % update gsl
%         
%             [states,conts,gs] = getpath(x,'R',0,N,n); % get trajectories with initial state 'R'
%             statesr = [statesr, states]; % update statesr
%             consr = [consr, conts]; % update consr
%             gsr = [gsr, gs]; % update gsr
%             
%         elseif x == 1
%             [states,conts,gs] = getpath(x,'R',0,N,n); % get trajectories with initial state 'R'
%             statesr = [statesr, states]; % update statesr
%             consr = [consr, conts]; % update consr
%             gsr = [gsr, gs]; % update gsr
%             
%         elseif x == n
%             [states,conts,gs] = getpath(x,'L',0,N,n); % get trajectories with initial state 'L'
%             statesl = [statesl, states]; % update statesl
%             consl = [consl, conts]; % update consl
%             gsl = [gsl, gs]; % update gsl
%         end
%     end
% end
% 
% samples = cell(n,4); % initialize cell to hold results of sampling
% for x = 1:n % loop through inner states
%     Qresl = []; % initialize array to hold Q-values with u = 'L'
%     Qresr = []; % initialize array to hold Q-values with u = 'R'
%     
%     for i = 1:(n-1)*m % loop through all sample trajectories generated
%         for j = 1:N % loop through all stages in each trajectory
%          
%             if statesl(j,i) == x
%                 
%                 if x == 1
%                     Qresr = [Qresr; gsl(j,i) + Jres{x,j}];
%                     
%                 elseif x == n && (gsl(j,i) + Jres{x,j} <= min(samples{2,1},samples{2,3})) 
%                     Qresl = [Qresl; gsl(j,i) + Jres{x,j}];
%                     
%                 elseif x == 2 
%                     if consl(j,i) == 'L'
%                         Qresl = [Qresl; gsl(j,i) + Jres{x,j}]; % store Q-value in Qresl
%                     elseif consl(j,i) == 'R' 
%                         Qresr = [Qresr; gsl(j,i) + Jres{x,j}]; % store Q-value in Qresr
%                     end
%                     
%                 elseif x ~= 1 && x ~= 2 && x ~= n
%                     if consl(j,i) == 'L' && (gsl(j,i) + Jres{x,j} <= min(samples{2,1},samples{2,3})) 
%                         Qresl = [Qresl; gsl(j,i) + Jres{x,j}]; % store Q-value in Qresl
%                     elseif consl(j,i) == 'R' && (gsl(j,i) + Jres{x,j} <= min(samples{2,1},samples{2,3}))
%                         Qresr = [Qresr; gsl(j,i) + Jres{x,j}]; % store Q-value in Qresr
%                     end
%                 end
%             end
%             
%             if statesr(j,i) == x 
%                 
%                 if x == 1
%                     Qresr = [Qresr; gsr(j,i) + Jres{x,j}];
%                     
%                 elseif x == n && (gsr(j,i) + Jres{x,j} <= min(samples{2,1},samples{2,3})) 
%                     Qresl = [Qresl; gsr(j,i) + Jres{x,j}];
%                     
%                 elseif x == 2 
%                     if consr(j,i) == 'L'
%                         Qresl = [Qresl; gsr(j,i) + Jres{x,j}]; % store Q-value in Qresl
%                     elseif consr(j,i) == 'R' 
%                         Qresr = [Qresr; gsr(j,i) + Jres{x,j}]; % store Q-value in Qresr
%                     end
%                     
%                 elseif x ~= 1 && x ~= 2 && x ~= n
%                     if consr(j,i) == 'L' && (gsr(j,i) + Jres{x,j} <= min(samples{2,1},samples{2,3})) 
%                         Qresl = [Qresl; gsr(j,i) + Jres{x,j}]; % store Q-value in Qresl
%                     elseif consr(j,i) == 'R' && (gsr(j,i) + Jres{x,j} <= min(samples{2,1},samples{2,3}))
%                         Qresr = [Qresr; gsr(j,i) + Jres{x,j}]; % store Q-value in Qresr
%                     end
%                 end
%             end
%         end
%     end
%     
%     samples{x,1} = mean(Qresl); % average all Q-values with u = L
%     samples{x,2} = length(Qresl); % number of Q-values with u = L
%     samples{x,3} = mean(Qresr); % average all Q-values with u = R
%     samples{x,4} = length(Qresr); % number of Q-values with u = R
% end
% 
% runtime = toc(t0);

end


% distribution of disturbance wk depending on uk
function w = disturb(u)
distL = [0 -1 -2 -3 -1 -2 -3 -1 -2 -1]; % distribution of w when u = L
distR = [0 1 2 3 0 1 2 1 2 2]; % distribution of w when u = R

if u == 'L'
    w = randsample(distL,1); % randomly select w from distL
    
elseif u == 'R'
    w = randsample(distR,1); % randomly select w from distR
end
end

% state transition (in xk) model for system
function f = state_trans(x,w)
f = min(10, max(1, x+w));
end

% generate sample state-control trajectories given a state and control
function [states,conts,gs] = getpath(x,u,kinit,N,n)

for k = kinit:(N-1) % loop from initial to last stage 
    
    if k == kinit % at initial stage
        states = x; % first state in path is x
        conts = u; % first control in conts is u
        
        w = disturb(conts); % get w
        f = state_trans(states,w); % get state at next stage
        gs = cost_fun(k,states,conts,w,N); % compute cost
        
    else % at any other stage, get state and control from those at previous stage
        states = [states;f]; % update states
        
        if states(end) == 1
            conts = [conts;'R']; % can only turn right
            w = disturb('R'); % get w
            f = state_trans(1,w); % get state at next stage
            gs = [gs; cost_fun(k,1,'R',w,N)]; % compute cost and update gs
            
        elseif states(end) == n
            conts = [conts;'L']; % can only turn left
            w = disturb('L'); % get w
            f = state_trans(n,w); % get state at next stage
            gs = [gs; cost_fun(k,n,'L',w,N)]; % compute cost and update gs
            
        elseif states(end) == x
            conts = [conts;u]; % update conts
            w = disturb(u); % get w
            f = state_trans(x,w); % get state at next stage
            gs = [gs; cost_fun(k,x,u,w,N)]; % compute cost and update gs

        else
            conts = [conts;randsample(['R','L'],1)]; % randomly choose control
            w = disturb(conts(end)); % get w
            f = state_trans(states(end),w); % get state at next stage
            gs = [gs; cost_fun(k,states(end),conts(end),w,N)]; % compute cost and update gs
        end
    end
end
end

% base policy depends on state xk in state space
function bp = base_pol(x)
if x <= 5
    bp = 'R';
    
elseif x >= 6
    bp = 'L';
end
end

% cost function g_k at kth stage, where k = 0,..., N-1
function g = cost_fun(k,x,u,w,N)

% check feasibility
if round(x) ~= x || x < 1 || x > 10
    error('state variable is not feasible');
end

if x == 1 && u ~= 'R'
    error('control variable is not feasible at x = 1');
end

if x == 10 && u ~= 'L'
    error('control variable is not feasible at x = 10');
end

if u == 'L' && (round(w) ~= w || w > 0 || w < -3)
    error('disturbance is not feasible at u = L')
end

if u == 'R' && (round(w) ~= w || w < 0 || w > 3)
    error('disturbance is not feasible at u = R')
end

if k == N
    g = (x-5)^2 / 100;
else
    switch u
        case 'R'
            g = abs(w) / (k+10);
        case 'L'
            g = abs(w) / ( abs(k - 20) + 1) * (-1)^(k+x);
        otherwise
            error('control variable is not feasible');
    end
end
end
