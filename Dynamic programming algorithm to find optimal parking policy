function outputs = dp_parking(N,C,p)
clc; clf;

outputs = zeros(N,4); % initialize array to hold results

% step through k stages, where k = 0,..., N-1, in reverse
for k = (N-1):-1:0
    cost = getcost(N,k,C); % get cost to park at current spot
    
    if (k == N-1) % at last spot: choose between spot or garage
        JO = getcost(N,k+1,C); % if spot is occupied, park in garage
        JF = min(cost,C); % if spot is free, choose whichever is cheaper
        
    else % at kth spot, where k >= 1 and k < N-1
        JOinit = JO; % store "previous" JO
        JFinit = JF; % store "previous" JF
        
        JO = p*JFinit + (1-p)*JOinit; % JO at kth spot
        JF = min(cost, p*JFinit + (1-p)*JOinit); % JF at kth spot
    end
    
    outputs(k+1,:) = [k;JO;JF;cost]; % store results
end

figure(1)
hold on
plot(outputs(:,1),outputs(:,2),'ro','linewidth',2)
plot(outputs(:,1),outputs(:,3),'b+','linewidth',2)
plot(outputs(:,1),outputs(:,4),'g.','linewidth',2)
hold off
xlabel('Stage'); ylabel('Optimal expected cost-to-go or cost to park'); 
legend('J(O)', 'J(F)','c(k)'); set(gca,'FontSize',12);
end


% function to compute cost to park at kth spot
function cost = getcost(N,k,C)
if (k <= N-1) % parking in the N parking spots
    cost = N - k; % cost of 0th to (n-1)th parking spot
elseif (k == N) % reaches the parking garage
    cost = C; % cost of garage
end
end
