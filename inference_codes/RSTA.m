


%% 
%
% Random spanning tree approximations assumes a model built with a complete graph as output graph. As learning/inference on a complete graph is
% known to be difficult, the algorithm construct a set of predictors each with a random spanning tree as the output graph. With max-margin assumption,
% if there exists a classifier achieving a margin on a complete graph, there will be a collection of random tree predictors achieving a similar margin.
% 
% PARAMETERS:
%   paramsIn:   input parameters
%   dataIn:     input data e.g., kernel and label matrices for training and testing
%   rtn:        return value
%   ts_err:     test error
%
%
% USAGE:
%   This function is called by a wrapper run_RSTA()
%
function [rtn, ts_err] = RSTA(paramsIn, dataIn,nm)

    %% Definition of global variables
    global loss_list;           % losses associated with different edge labels
    global mu_list;             % marginal dual varibles: these are the parameters to be learned
    global E_list;              % edges of the Markov network e_i = [E(i,1),E(i,2)];
    global ind_edge_val_list;	% ind_edge_val{u} = [Ye == u] 
    global Ye_list;             % Denotes the edge-labelings 1 <-- [-1,-1], 2 <-- [-1,+1], 3 <-- [+1,-1], 4 <-- [+1,+1]
    global Kx_tr;               % X-kernel, assume to be positive semidefinite and normalized (Kx_tr(i,i) = 1)
    global Kx_ts;
    global Y_tr;                % Y-data: assumed to be class labels encoded {-1,+1}
    global Y_ts;
    global params;              % parameters use by the learning algorithm
    global m;                   % number of training instances
    global l;                   % number of labels
    global primal_ub;
    global profile;
    global obj;
    global delta_obj_list;
    global obj_list;
    global opt_round;
    global Rmu_list;
    global Smu_list;
    global T_size;              % number of trees
    global norm_const_linear;
    global norm_const_quadratic_list;
    global Kxx_mu_x_list;
    global kappa;               % K best
    global Yspos_list;  
    global iter;                % the indicator for the number of iteration
    global duality_gap_on_trees;
    global val_list;
    global kappa_list;
    global Yipos_list;
    global GmaxG0_list;
    global GoodUpdate_list;
    
    rand('twister', 0);
    
    global previous;
    previous=[];

    params=paramsIn;
    if params.l_norm == 1
        l1norm = 1;
    else
        l1norm = 0;
    end
    Kx_tr       = dataIn.Kx_tr;
    Kx_ts       = dataIn.Kx_ts;
    Y_tr        = dataIn.Y_tr;
    Y_ts        = dataIn.Y_ts;      
    E_list      = dataIn.Elist;     % a list of random spanning trees in terms of edges
    l           = size(Y_tr,2);     % the number of microlabels
    m           = size(Kx_tr,1);    % the number of examples
    T_size      = size(E_list,1);   % the number of random spanning trees
    loss_list   = cell(T_size, 1);  % a list of losses on the collection of trees
    Ye_list     = cell(T_size, 1);   
    ind_edge_val_list           = cell(T_size, 1);
    Kxx_mu_x_list               = cell(T_size, 1);
    duality_gap_on_trees        = ones(1,T_size)*1e10;          % relative duality gap on individual spanning tree
    norm_const_linear           = 1/(T_size)/size(E_list{1},1); % The linear term will be normalized by the total number of edges
    norm_const_quadratic_list   = zeros(1,T_size)+1/(T_size);   % The quadratic term is normalized by 1
    mu_list = cell(T_size);         % a list of solutions in terms of marginalized dual variables on the collection of trees
   
    
    if T_size <= 1
        kappa_INIT  = 2;
        kappa_MIN   = 2;
        kappa_MAX   = 2;
    else
        kappa_INIT  = min(params.maxkappa,2^l);
        kappa_MIN   = min(params.maxkappa,2^l); 
        kappa_MAX   = min(params.maxkappa,2^l);
    end
    
    
    Yspos_list = ones(1,m)*(params.maxkappa);
    kappa = kappa_INIT;
    
    loss_scaling_factor = 100;
    
    for t=1:T_size
        [loss_list{t},Ye_list{t},ind_edge_val_list{t}] = compute_loss_vector(Y_tr,t,params.mlloss);
        loss_list{t} = loss_list{t} / loss_scaling_factor;
        mu_list{t}          = zeros(4*size(E_list{1},1),m);
        Kxx_mu_x_list{t}    = zeros(4*size(E_list{1},1),m);
    end

    iter = 0; 
    
    
    
    val_list        = zeros(1,m);
    Yipos_list      = ones(1,m)*(params.maxkappa+1);
    kappa_list      = zeros(1,m);
    GmaxG0_list     = zeros(1,m);
    GoodUpdate_list = zeros(1,m);
    obj_list        = zeros(1,T_size);
    
    
    %% Initialization
    optimizer_init;
    profile_init;
    
    initialize_global_consensus_graph();
    
    
    global loss_global
    loss_global = loss_global / loss_scaling_factor;
    
    
    

    %% Optimization
    print_message('Conditional gradient descend ...',0);
    primal_ub = Inf;    % primal upper bound
    opt_round = 0;      % the number of optimization

    % Compute dualit gap
	compute_duality_gap;

    % compute the profiling statistics before optimization
    profile_update_tr;
       
    
    %% Iteration over examples untile convergece ?
    % parameters
    
    prev_obj = 0;
    
    nflip=Inf;
    params.verbosity = 2;
    progress_made = 1;
    profile.n_err_microlabel_prev=profile.n_err_microlabel;

    best_n_err_microlabel=Inf;
    best_iter = iter;
    best_kappa = kappa;
    best_mu_list=mu_list;
    best_Kxx_mu_x_list=Kxx_mu_x_list;
    best_Rmu_list=Rmu_list;
    best_Smu_list=Smu_list;
    best_norm_const_quadratic_list = norm_const_quadratic_list;
    
    
    
    %% Iterate over training examples until convergence 
    while(opt_round < params.maxiter && ...
            primal_ub - obj >= params.epsilon*obj)
%     while (primal_ub - obj >= params.epsilon*obj && ... % satisfy duality gap
%             progress_made == 1 && ...                   % make progress
%             nflip > 0 && ...                            % number of flips
%             opt_round < params.maxiter ...              % within iteration limitation
%             )
        opt_round = opt_round + 1;
        
        % update lambda / quadratic term
        
        if iter>0 && l1norm==1
            for t=1:T_size
                Kmu_tmp = compute_Kmu(Kx_tr,mu_list{t},E_list{t},ind_edge_val_list{t});
                Kmu_tmp = reshape(Kmu_tmp,1,size(Kmu_tmp,1)*size(Kmu_tmp,2));
                mu_tmp = reshape(mu_list{t},1,size(mu_list{t},1)*size(mu_list{t},2));
                norm_const_quadratic_list(t) = sqrt(Kmu_tmp*mu_tmp'*norm_const_quadratic_list(t)/2);
            end
            norm_const_quadratic_list = norm_const_quadratic_list /  sum(norm_const_quadratic_list);    
        end
        
        %% iterate over examples 
        iter = iter +1;   
        Yspos_list = ones(1,m);
        %Yipos_list = ones(1,m);
        val_list = zeros(1,m);

        if iter <= 30
            Yipos_list = ones(1,m)*(params.maxkappa+1);
        end
%        for xi = randsample(1:m,m,true,Yipos_list/sum(Yipos_list))
%        for xi = selected_samples
%         for xi = randsample(1:m,m)
        for xi = 1:m
            print_message(sprintf('Start descend on example %d initial k %d',xi,kappa),3)
            kappa_decrease_flag(xi)=0;
            if nm == 0
            [delta_obj_list] = conditional_gradient_descent(xi,kappa);    % optimize on single example
            else
            [delta_obj_list] = conditional_gradient_descent_with_Newton(xi,kappa);    % optimize on single example
            end
                 
%                 kappa0=kappa;
%                 while ( Yspos_list(xi)==0 ) && kappa0 < params.maxkappa 
%                     kappa0=kappa0*2;
%                     [delta_obj_list,Yspos_list(xi)] = conditional_gradient_descent(xi,kappa0);    % optimize on single example
%                 end
            kappa_list(xi)=kappa;
            obj_list = obj_list + delta_obj_list;
            obj = obj + sum(delta_obj_list);
            % update kappa
            if Yspos_list(xi)==0
                kappa = min(kappa*2,kappa_MAX);
            else
                kappa = max(ceil(kappa/2),kappa_MIN);
            end
        end
        
        
        if mod(iter, params.profileiter)==0
            progress_made = (obj >= prev_obj);  
            prev_obj = obj;
            
            % Compute duality gap and update profile for all training examples.
            compute_duality_gap;            
            profile_update_tr;          
            % Update flip number.
            if profile.n_err_microlabel > profile.n_err_microlabel_prev
                nflip = nflip - 1;
            end
            % update current best solution
            if profile.n_err_microlabel < best_n_err_microlabel
                best_n_err_microlabel = profile.n_err_microlabel;
                best_iter   = iter;
                best_kappa  = kappa;
                best_mu_list        = mu_list;
                best_Kxx_mu_x_list  = Kxx_mu_x_list;
                best_Rmu_list       = Rmu_list;
                best_Smu_list       = Smu_list;
                best_norm_const_quadratic_list = norm_const_quadratic_list;
            end
        end
        
        
    end
    
    

    %% last optimization iteration
    if paramsIn.extra_iter
        iter = best_iter+1;
        kappa = best_kappa;
        mu_list = best_mu_list;
        Kxx_mu_x_list = best_Kxx_mu_x_list;
        Rmu_list = best_Rmu_list;
        Smu_list = best_Smu_list;
        norm_const_quadratic_list = best_norm_const_quadratic_list;
        for xi=1:m

                [~,~] = conditional_gradient_descent(xi,kappa);    % optimize on single example

            profile_update_tr;
            if profile.n_err_microlabel < best_n_err_microlabel
                best_n_err_microlabel=profile.n_err_microlabel;
                best_iter = iter;
                best_kappa = kappa;
                best_mu_list=mu_list;
                best_Kxx_mu_x_list=Kxx_mu_x_list;
                best_Rmu_list=Rmu_list;
                best_Smu_list=Smu_list;
            end
        end
    end
    
    %% final prediction
    iter = best_iter+1;
    kappa = best_kappa;
    mu_list = best_mu_list;
    Kxx_mu_x_list = best_Kxx_mu_x_list;
    Rmu_list = best_Rmu_list;
    Smu_list = best_Smu_list;
    norm_const_quadratic_list = best_norm_const_quadratic_list;
    profile_update_ts;
    
    

    rtn = 0;
    ts_err = 0;
end


%% Compute variables on global consensus graph
% mu_ts --> mu_T --> inverse --> ind_forwards --> mu_global --> ind_backwards --> inverse --> mu_T --> mu_ts
function initialize_global_consensus_graph
    global E_list;
    global E_global;
    global inverse_flag;
    global ind_forwards;
    global ind_backwards;
    global loss_global;
    global Y_tr;
    global Ye_global;
    global ind_edge_val_global;
    global m;
    global Rmu_global;
    global Smu_global;
    
    edge_set        = cell2mat(E_list);
    inverse_edges   = 1:size(edge_set,1);
    inverse_edges   = inverse_edges(edge_set(:,1)>edge_set(:,2));
    edge_set(inverse_edges,:) = edge_set(inverse_edges,[2,1]);
    tmp             = 1:(size(edge_set,1)*4);
    inverse_flag    = tmp;
    inverse_flag((inverse_edges-1)*4+2) = tmp((inverse_edges-1)*4+3);
    inverse_flag((inverse_edges-1)*4+3) = tmp((inverse_edges-1)*4+2);
    [~, ind_forwards, ind_backwards] = unique(edge_set,'rows');
    E_global = edge_set(ind_forwards,:);
    tmp = 1:4;
    ind_forwards    = (ind_forwards-1)*4;
    ind_forwards    = reshape(ind_forwards(:,ones(1,4))'+tmp(ones(size(ind_forwards,1),1),:)',size(ind_forwards,1)*4,1);
    ind_backwards   = (ind_backwards-1)*4;
    ind_backwards   = reshape(ind_backwards(:,ones(1,4))'+tmp(ones(size(ind_backwards,1),1),:)',size(ind_backwards,1)*4,1);
    

    % Compute the loss vector for the global consensus graph.
    loss_global = zeros(4, m*size(E_global,1));
    Te1_global  = Y_tr(:,E_global(:,1))';
    Te2_global  = Y_tr(:,E_global(:,2))';
    u = 0;
    for u_1 = [-1, 1]
        for u_2 = [-1, 1]
            u = u + 1;
            loss_global(u,:) = reshape((Te1_global ~= u_1)+(Te2_global ~= u_2),m*size(E_global,1),1);
        end
    end     
    loss_global = reshape(loss_global,4*size(E_global,1),m);
    
    % Compute the vector of Ye and ind_edge_val of the global consensus graph
    Ye_global = reshape(loss_global==0,4,size(E_global,1)*m);
    ind_edge_val_global = cell(4,1);
    for u=1:4
        ind_edge_val_global{u} = sparse(reshape(Ye_global(u,:)~=0,size(E_global,1),m));
    end
    
    % initialize global Rmu and global Smu
    for u=1:4
        Smu_global{u} = zeros(size(E_global,1),m);
        Rmu_global{u} = zeros(size(E_global,1),m);
    end
    
end

%% Compose global variables based on the local variables
function [mu_global] = compose_mu_global_from_local
    global mu_list;
    global inverse_flag;
    global ind_forwards;
    
    % Form a matrix --> inverse --> select
    mu_global = cell2mat(mu_list);
    mu_global = mu_global(inverse_flag,:);
    mu_global = mu_global(ind_forwards,:);
    
end

%% Compose a set of local variables based on global variable
function [dmu_set] = compose_mu_local_from_global(dmu_global)
    global ind_backwards;
    global inverse_flag;
    global T_size;
    
    % complete --> inverse --> transform
    dmu_set = dmu_global(ind_backwards,:);
    dmu_set = dmu_set(inverse_flag,:);
    dmu_set = reshape(dmu_set,size(dmu_set,1)/T_size,T_size);
    
end

%% Compute marginal dual variables on global consensus graph mu_global for all marginal dual variables mu over all examples
% Input:    x is the x'th training example
% Output:   mu_global
%           E_global
%           ind_backwards:  reverse mapping from global to the collection of local
%           inverse_flag:   positions that corresponds to the change of edge directions
function [mu_global,E_global,ind_backwards,inverse_flag] = compose_global_from_local(x)
    

    global E_list;
    global mu_list;
    global T_size;
    global l;
    global m;
    
    % Pool all edges and corresponding marginal dual variables together.
    % TODO: initialize Emu
    Emu = [];
    for t=1:T_size
        E   = E_list{t};
        % make a new mu
        mu = reshape(mu_list{t}, 4, (l-1)*m);
        % TODO: initialize newMu
        newMu = [];
        for u=1:4
            newMu = [ newMu, reshape(mu(u,:), l-1, m) ];
        end
        Emu = [Emu;[E,newMu]];
    end
    % Clear the repeating rows.
    inverse_flag = Emu(:,1)>Emu(:,2);
    Emu(inverse_flag,:) = Emu(inverse_flag,[2,1, 3:(m+2), (2*m+3):(2*m+m+2), (m+3):(m+m+2), (3*m+3):(3*m+m+2)]);
    [~, ind_forwards, ind_backwards] = unique(Emu(:,1:2),'rows');
    Emu = Emu(ind_forwards,:);
    E_global    = Emu(:,1:2);
    newMu       = Emu(:,3:size(Emu,2));
    mu_global   = [];
    for u=1:4
        mu_global = [mu_global; reshape(newMu(:,((u-1)*m+1):(u*m)), 1, size(E_global,1)*m)];
    end
    
    mu_global = reshape(mu_global, 4*size(E_global,1), m);
    
    if sum(sum(isnan(mu_global)))>0
        [x,sum(sum(isnan(mu_global)))]
        asdfasd
    end
end

%% Decompose the global mu into a set of mu defined on a collection of spanning trees
% Input:    mu_global
%           E_global
%           ind_backwards
%           inverse_flag
% Output:   mu0_list
function [mu0_list] = decompose_local_from_global(mu_global, E_global, ind_backwards, inverse_flag)
    
    global T_size;
    Emu = [E_global, reshape(mu_global,4,size(E_global,1))'];
    Emu = Emu(ind_backwards,:);
    Emu(inverse_flag,:) = Emu(inverse_flag,[2,1,3,5,4,6]);
    mu0_list = cell(T_size,1);
    for t=1:T_size
        mu0_list{t} = Emu(((t-1)*(size(Emu,1)/T_size)+1):(t*(size(Emu,1)/T_size)),3:6);
        mu0_list{t} = reshape(mu0_list{t}',size(mu0_list{t},1)*size(mu0_list{t},2),1);
    end
    
end



%% New function to compute <K^{delta}(i,:),mu>, which is the part of the gradient, the dimenson of Kmu is m*4*|E|
% Input:
%       Kx: part of the kernel matrix for current examples
%       mu: complete marginal dual variable
%       E:  complete set of edges
%       ind_edge_val: edge value indicator
%       x:  indicator for current set of examples
% 27/01/2015
function Kmu = compute_Kmu_matrix ( Kx, mu, E, ind_edge_val, x )


    numExample = size(x,2);   % number of example in this computation
    m = size(Kx,1);         % total number of examples
    numE = size(E,1);       % number of edges

    
    mu = reshape(mu, 4, numE * m);         % 4 by |E|*m
    sum_mu = reshape(sum(mu), numE, m);    % |E| by m
    
    term12 = zeros(1, numE * numExample);	% 1 by (|E|*m)
    Kmu = zeros(4, numE * numExample);      % 4 by (|E|*m)
    
    for u = 1:4
        edgeLabelIndicator = full(ind_edge_val{u});     % |E| by m 
        real_mu_u = reshape(mu(u,:),numE,m);   % |E| by m
        H_u = sum_mu.*edgeLabelIndicator - real_mu_u;   % |E| by m
        Q_u = H_u * Kx; % |E| by m   
        term12 = term12 + reshape(Q_u.*edgeLabelIndicator(:,x), 1, numE * numExample);   % 1 by |E|*m
        Kmu(u,:) = reshape(-Q_u, 1, numE*numExample);
    end
    
    for u = 1:4
        Kmu(u,:) = Kmu(u,:) + term12;
    end
    
    Kmu = reshape(Kmu, 4*numE, numExample);
end


%% Complete part of gradient for everything
% 05/01/2015
function Kmu = compute_Kmu(Kx,mu,E,ind_edge_val)


    m_oup = size(Kx,2);
    m = size(Kx,1);
    mp = size(mu,2);
    
    mu = reshape(mu,4,size(E,1)*mp);
    Smu = reshape(sum(mu),size(E,1),mp);
    term12 =zeros(1,size(E,1)*m_oup);
    Kmu = zeros(4,size(E,1)*m_oup);
    
    
    for u = 1:4
        IndEVu = full(ind_edge_val{u});    
        Rmu_u = reshape(mu(u,:),size(E,1),mp);
        H_u = Smu.*IndEVu;
        H_u = H_u - Rmu_u;
        
        Q_u = H_u*Kx;
        term12 = term12 + reshape(Q_u.*IndEVu,1,m_oup*size(E,1));
        Kmu(u,:) = reshape(-Q_u,1,m_oup*size(E,1));
    end
    for u = 1:4
        Kmu(u,:) = Kmu(u,:) + term12;
    end
    
    %mu = reshape(mu,mu_siz);
end


%% Compute the part of the gradient of current example x, gradient is l-ku, this function will compute ku, which is a vector of 4*|E| dimension
% update 08/01/2015
% Input:
%   x,Kx,t
% Output
%   gradient for current example x
function Kmu_x = compute_Kmu_x(x,Kx,E,ind_edge_val,Rmu,Smu)

    % local
    term12 = zeros(1,size(E,1));
    term34 = zeros(4,size(E,1));
    
    for u = 1:4
        Ind_te_u = full(ind_edge_val{u}(:,x));
        H_u = Smu{u}*Kx-Rmu{u}*Kx;
        term12(1,Ind_te_u) = H_u(Ind_te_u)';
        term34(u,:) = -H_u';
    end
    
    Kmu_x = reshape(term12(ones(4,1),:) + term34,4*size(E,1),1);
    
end


%% Compute relative duality gap.
%       03/12/2014
%       30/01/2014
%
function compute_duality_gap

    %% global parameters
    global duality_gap_on_trees;
    global T_size;
    global Kx_tr;
    global loss_list;
    global E_list;
    global Y_tr;
    global params;
    global mu_list;
    global ind_edge_val_list;
    global primal_ub;
    global obj;
    global kappa;
    global norm_const_linear;
    global norm_const_quadratic_list;
    global iter;
    global m;
    global l;
    global node_degree_list;
    
    Y           = Y_tr;                                     % the true multilabel
    Ypred       = zeros(size(Y));                           % the predicted best multilabel
    Y_kappa     = zeros(size(Y,1)*T_size, size(Y,2)*kappa); % Holder for the k-best multilabels
    Y_kappa_val = zeros(size(Y,1)*T_size, kappa);           % Holder for the value achieved by the k-best multilabels
    
    %% Compute K best multilabel from a collection of random spanning trees
    Kmu_list_local      = cell(1,T_size);
    gradient_list_local = cell(1,T_size);    
    for t = 1:T_size
        % obtain or compute variables locally on each spanning tree
        loss    = loss_list{t};
        E       = E_list{t};
        mu      = mu_list{t};
        ind_edge_val = ind_edge_val_list{t};
        loss    = reshape(loss,4,size(E,1)*m);
        Kmu_list_local{t} = compute_Kmu(Kx_tr, mu, E, ind_edge_val);
        Kmu_list_local{t} = reshape(Kmu_list_local{t}, 4, size(E,1)*m);
        Kmu = Kmu_list_local{t};
        gradient_list_local{t}  = norm_const_linear*loss - norm_const_quadratic_list(t)*Kmu;
        gradient = gradient_list_local{t};
        node_degree = node_degree_list{t};
        in_gradient = reshape(gradient,numel(gradient),1);
        % Compute K best multilabel
        [Y_tmp,Y_tmp_val] = compute_topk_omp(in_gradient, kappa, E, node_degree);
        % Save the result
        Y_kappa(((t-1)*size(Y,1)+1):(t*size(Y,1)),:)        = Y_tmp;
        Y_kappa_val(((t-1)*size(Y,1)+1):(t*size(Y,1)),:)    = Y_tmp_val;
    end
    
    %% Get the worst violator from the K best predictions of each example
    for i = 1:size(Y,1)
        % Assign default value to the worst violating multilabel when the gradient descent has not started.
        if iter == 0
            Ypred(i,:) = -1*ones(1,size(Y_tr,2));
        else
            IN_E = zeros((l-1)*2,T_size);
            for t = 1:T_size
                IN_E(:,t) = reshape(E_list{t},(l-1)*2,1);
            end
            IN_gradient = zeros(4*(l-1),T_size);
            for t=1:T_size
                IN_gradient(:,t) = reshape(gradient_list_local{t}(:,((i-1)*(l-1)+1):(i*(l-1))),4*(l-1),1);
            end
            Y_kappa((i:size(Y_tr,1):size(Y_kappa,1)),:) = (Y_kappa((i:size(Y_tr,1):size(Y_kappa,1)),:)+1)/2;
            % Compute the worst violator
            [Ypred(i,:),~,~,~] = ...
                find_worst_violator_new(...
                Y_kappa((i:size(Y_tr,1):size(Y_kappa,1)),:),...
                Y_kappa_val((i:size(Y_tr,1):size(Y_kappa_val,1)),:)...
                ,[],IN_E,IN_gradient);
            Ypred(i,:) = Ypred(i,:)*2-1;
        end
    end

    clear Y_kappa;
    clear Y_kappa_val;
    
    %% Compute duality gap for all training examples on each individual random spanning tree
    % TODO: duality gap computation is based on G0 and Gmax
    % Define variable to save the results
    dgap = zeros(1,T_size);
    % Iterate over a collection of random spanning trees
    for t = 1:T_size
        % compute or collect variables locally on each spanning tree
        loss    = loss_list{t};
        E       = E_list{t};
        mu      = mu_list{t};
        %Kmu = Kmu_list_local{t};
        gradient = gradient_list_local{t};
        loss	= reshape(loss,size(loss,1)*size(loss,2),1);
        mu      = reshape(mu,size(loss,1)*size(loss,2),1);
        %Kmu = reshape(Kmu,size(loss,1)*size(loss,2),1);
        % compute current maxima on function
        %Gcur = norm_const_linear*mu'*loss - 1/2*norm_const_quadratic_list(t)*mu'*Kmu;
        % compute current maxima on gradient
        Gcur = reshape(gradient,1,size(gradient,1)*size(gradient,2))*reshape(mu,1,size(mu,1)*size(mu,2))';
        % compute best possible objective along Y* and gradient.
        Gmax = compute_Gmax(gradient,Ypred,E);
        % the difference is estimated as duality gap
        dgap(t) = params.C*sum(Gmax)-Gcur;
    end

    %% Compute primal upper bound, which is obj+duality gap
    dgap = max(0,dgap);
    %duality_gap_on_trees = min(dgap,duality_gap_on_trees);
    duality_gap_on_trees = dgap;
    primal_ub = obj + sum(dgap);
    
end



%%
% Perform conditional gradient descend on individual training example,
% to update corresponding marginal dual variables of that training example on a collection of random spanning trees
%
% working in progress on 16/05/2014
% working in progress on 03/12/2014
% working in progress on 29/01/2015
%
% PARAMETERS: 
%   x:      index of the current example under optimization
%   kappa:  number of best multilabel computed from each individual random spanning tree
%   delta_obj_list:     difference in terms of objective value on each random spanning tree
%
%
function [delta_obj_list] = conditional_gradient_descent(x, kappa)

    %% Definition of the parameters
    global loss_list;
    global loss;
    global Ye_list;
    global Ye;
    global E_list;
    global E;
    global mu_list;
    global mu;
    global ind_edge_val_list;
    global ind_edge_val;
    global Rmu_list;
    global Smu_list;
    global Kxx_mu_x_list;
    global norm_const_linear;
    global norm_const_quadratic_list;
    global l;
    global Kx_tr;
    global Y_tr;
    global T_size;
    global params;
    global val_list;
    global Yipos_list;
    global GmaxG0_list;
    global GoodUpdate_list;
    global node_degree_list;
    
    
    %% Compute K best multilabels from a collection of random spanning trees.
    % Define variables to store results.
    Y_kappa     = zeros(T_size, kappa*l);
    Y_kappa_val = zeros(T_size, kappa);
    gradient_list_local = cell(1, T_size);
    Kmu_x_list_local    = cell(1, T_size);
    % Iterate over a collection of random spanning trees and compute the K best multilabels on each spanning tree by Dynamic Programming.
    for t=1:T_size
        % Variables located on the spanning tree T_t and example x.
        loss    = loss_list{t}(:,x);
        Ye      = Ye_list{t}(:,x);
        ind_edge_val = ind_edge_val_list{t};
        mu      = mu_list{t}(:,x);
        E       = E_list{t};
        Rmu     = Rmu_list{t};
        Smu     = Smu_list{t};    
        % Compute some necessary quantities for the spanning tree T_t.
        % Compute Kmu_x = K_x*mu
        Kmu_x_list_local{t} = compute_Kmu_x(x,Kx_tr(:,x),E,ind_edge_val,Rmu,Smu);
        Kmu_x = Kmu_x_list_local{t};
        % current gradient    
        gradient_list_local{t} =  norm_const_linear*loss - norm_const_quadratic_list(t)*Kmu_x;
        gradient = gradient_list_local{t};
        % Compute top K-best multilabels
        [Ymax,YmaxVal] = compute_topk_omp(gradient,kappa,E,node_degree_list{t});
        % Save results
        Y_kappa(t,:)        = Ymax;
        Y_kappa_val(t,:)    = YmaxVal;
    end
    
    
    %% Compute the worst violating multilabel from the K best list.
    IN_E = zeros((l-1)*2,(l-1));
    for t=1:T_size
        IN_E(:,t) = reshape(E_list{t},(l-1)*2,1);
    end
    IN_gradient = zeros((l-1)*4,(l-1));
    for t=1:T_size
        IN_gradient(:,t) = gradient_list_local{t};
    end
    % change label from -1/+1 to 0/+1
    Y_kappa = (Y_kappa+1)/2;
    Yi = (Y_tr(x,:)+1)/2;
    
    % find the worst violating multilabel from the K best list
    %   Ymax:         best multilabel
    %   Ymax_val:     
    %   Yi_pos:       position of the true multilabel in the K best list
    [Ymax, Ymax_val, ~, Yi_pos] = find_worst_violator_new(Y_kappa,Y_kappa_val,Yi,IN_E,IN_gradient);
    % save results to global variables
    val_list(x) = Ymax_val;
    Yipos_list(x) = Yi_pos;
    % change label back from 0/+1 to -1/+1
    Ymax = Ymax*2-1;
     

    %% If the worst violator is the correct label, exit without update current marginal dual variable of current example.
    if sum(Ymax~=Y_tr(x,:))==0 %|| ( ( (kappa_decrease_flag==0) && kappa < params.maxkappa) && iter~=1 )
        delta_obj_list = zeros(1,T_size);
        return;
    end
    
    
    %% Otherwise we need line serach to find optimal step size to the saddle point.
    mu_d_list   = mu_list;
    nomi        = zeros(1,T_size);
    denomi      = zeros(1,T_size);
    kxx_mu_0    = cell(1,T_size);
    Gmax        = zeros(1,T_size);
    G0          = zeros(1,T_size);
    Kmu_d_list  = cell(1,T_size);
    for t=1:T_size
        % Obtain variables located for tree t and example x.
        loss = loss_list{t}(:,x);
        Ye  = Ye_list{t}(:,x);
        ind_edge_val = ind_edge_val_list{t};
        mu  = mu_list{t}(:,x);
        E   = E_list{t};
        Rmu = Rmu_list{t};
        Smu = Smu_list{t};
        Kmu_x       = Kmu_x_list_local{t};
        gradient    = gradient_list_local{t};
        
        % Compute Gmax, which is the best objective value along the gradient.
        Gmax(t) = compute_Gmax(gradient,Ymax,E);
        Gmax(t) = Gmax(t)*params.C;
        % Compute G0, which is current objective value along the gradient.
        G0(t) = -mu'*gradient;
        % Compute mu_0, which is the best point along the descent direction.
        Umax_e = 1+2*(Ymax(:,E(:,1))>0) + (Ymax(:,E(:,2)) >0);
        mu_0 = zeros(size(mu));
        for u = 1:4
            mu_0(4*(1:(l-1))-4 + u) = params.C*(Umax_e == u);
        end
        % compute Kmu_0
        if sum(mu_0) > 0
            smu_1_te = sum(reshape(mu_0.*Ye,4,size(E,1)),1);
            smu_1_te = reshape(smu_1_te(ones(4,1),:),length(mu),1);
            kxx_mu_0{t} = ~Ye*params.C+mu_0-smu_1_te;
        else
            kxx_mu_0{t} = zeros(size(mu));
        end
        Kmu_0 = Kmu_x + kxx_mu_0{t} - Kxx_mu_x_list{t}(:,x);

        mu_d    = mu_0 - mu;
        Kmu_d   = Kmu_0 - Kmu_x;      
        Kmu_d_list{t}   = Kmu_d;
        mu_d_list{t}    = mu_d;
        
        nomi(t)     = mu_d' * gradient;
        denomi(t)   = norm_const_quadratic_list(t) * Kmu_d' * mu_d;
    end
    
    %% Decide whether to update the marginal dual variable on a collection of spanning trees by looking at the maximum objective along the gradient.
    % TODO: this can be very problemetic, as using global tau, the quality on individual random spanning tree can be very bad.
    if sum(Gmax)>=sum(G0)
        tau = min(sum(nomi)/sum(denomi),1);
    else
        tau=0;
    end
    tau = max(tau,0);
	GmaxG0_list(x) = sum(Gmax>=G0);
    GoodUpdate_list(x) = (tau>0);


    %% Update marginal dual variables based on the step size given by the line search on each individual random spanning tree.
    % TODO: as mentioned the update might not optimal for a step size given by tau
    delta_obj_list = zeros(1,T_size);
    for t=1:T_size
        % Obtain variables that are local on each random spanning tree.
        loss    = loss_list{t}(:,x);
        Ye      = Ye_list{t}(:,x);
        ind_edge_val = ind_edge_val_list{t};
        mu      = mu_list{t}(:,x);
        E       = E_list{t};
        gradient    =  gradient_list_local{t};
        mu_d    = mu_d_list{t};
        Kmu_d   = Kmu_d_list{t};
        
        % Compute the difference in the objective in individual random spanning tree.
        delta_obj_list(t) = gradient'*mu_d*tau - norm_const_quadratic_list(t)*tau^2/2*mu_d'*Kmu_d;
        %[gradient'*mu_d*tau, norm_const_quadratic_list(t)*tau^2/2*mu_d'*Kmu_d]
        mu = mu + tau*mu_d;
        Kxx_mu_x_list{t}(:,x) = (1-tau)*Kxx_mu_x_list{t}(:,x) + tau*kxx_mu_0{t};
        % Update Smu and Rmu
        mu = reshape(mu,4,size(E,1));
        for u = 1:4
            Smu_list{t}{u}(:,x) = (sum(mu)').*ind_edge_val{u}(:,x);
            Rmu_list{t}{u}(:,x) = mu(u,:)';
        end
        mu = reshape(mu,4*size(E,1),1);
        % Update marginal dual variable on individual spanning tree.
        mu_list{t}(:,x) = mu;
    end

    %%
    return;
end



%% 
% Conditional gradient optimization with Newton method to find the best update direction by a convex combination of multiple update directions.
%
%
%
function [delta_obj_list] = conditional_gradient_descent_with_Newton(x, kappa)

    %% Definition of the parameters
    global loss_list;
    global loss;
    global Ye_list;
    global Ye;
    global E_list;
    global E;
    global mu_list;
    global mu;
    global ind_edge_val_list;
    global ind_edge_val;
    global Rmu_list;
    global Smu_list;
    global norm_const_linear;
    global norm_const_quadratic_list;
    global l;
    global Kx_tr;
    global T_size;
    global params;
    global iter;
    global val_list;
    global Yipos_list;
    global Yspos_list;
    global GmaxG0_list;
    global GoodUpdate_list;
    global node_degree_list;
    global m;
    
    global inverse_flag;
    global ind_forwards;
    global ind_backwards;
    global E_global;
    global loss_global;
    global Ye_global;
    global ind_edge_val_global;
    global Smu_global;
    global Rmu_global;
    
    
    %% Compute K best multilabels from a collection of random spanning trees.
    % Define variables to save results.
    Y_kappa     = zeros(T_size, kappa*l);   % K best multilabel
    Y_kappa_val = zeros(T_size, kappa);     % Score of K best multilabel
    gradient_list_local = cell(1, T_size);  % Gradient vector locally on each random spanning tree
    Kmu_x_list_local    = cell(1, T_size);  % may not be necessary to have
    % Iterate over a collection of random spanning trees and compute the K best multilabels on each spanning tree by Dynamic Programming.
    for t=1:T_size
        % Variables located on the spanning tree T_t of the current example x.
        loss    = loss_list{t}(:,x);
        Ye      = Ye_list{t}(:,x);
        ind_edge_val = ind_edge_val_list{t};
        mu      = mu_list{t}(:,x);
        E       = E_list{t};
        Rmu     = Rmu_list{t};
        Smu     = Smu_list{t};    
        % compute Kmu_x = K_x*mu, which is a part of the gradient, of dimension 4*|E| by m
        Kmu_x = compute_Kmu_x(x,Kx_tr(:,x),E,ind_edge_val,Rmu,Smu); % this function can be merged with another function
        Kmu_x_list_local{t} = Kmu_x;
        % compute the gradient vector on the current spanning tree  
        gradient = norm_const_linear*loss - norm_const_quadratic_list(t)*Kmu_x;
        gradient_list_local{t} = gradient;
        % Compute the K-best multilabels
        [Ymax,YmaxVal] = compute_topk_omp(gradient,kappa,E,node_degree_list{t});
        % Save results, including predicted multilabel and the corresponding score on the current spanning tree
        Y_kappa(t,:)        = Ymax;
        Y_kappa_val(t,:)    = YmaxVal;
    end
    

    %% Compose current global marginal dual variable (mu) from local marginal dual variables {mu_t}_{t=1}^{T}
    mu_global = compose_mu_global_from_local;
    
    
    normalization_linear    = 1/size(E_global,1);
    normalization_quadratic = 1;
%     
%     normalization_linear    = 1/size(E_global,1);
%     normalization_quadratic = 1/size(E_global,1);
%     
    normalization_linear    = 1;
    normalization_quadratic = 1;
    
    %% convex combination of update directions, combination is given by lmd
    % For each update direction in terms of multilabels, compute the corresponding mu_0, and compute the different mu_0-mu
    dmu_set = zeros(size(mu_global,1),T_size);
%     Y_kappa = reshape(Y_kappa',l,kappa*T_size);
%     Y_kappa = Y_kappa';
    for t=1:T_size%*kappa
        Ymax    = Y_kappa(t,1:l);
        Umax_e  = 1+2*(Ymax(:,E_global(:,1))>0) + (Ymax(:,E_global(:,2)) >0);
        mu_0    = zeros(4*size(E_global,1),1);
        for u = 1:4
            mu_0(4*(1:size(E_global,1))-4 + u) = params.C*(Umax_e == u);
        end
        dmu_set(:,t) = mu_0-mu_global(:,x);
    end


    % compute Kmu_x
    Kmu_x_global = compute_Kmu_x(x,Kx_tr(:,x),E_global,ind_edge_val_global,Rmu_global,Smu_global);
    
    
    % compute the f'(x)
    f_prim = normalization_linear * loss_global(:,x) - normalization_quadratic * Kmu_x_global;
    
    % compute g = <f'(x),M>
    g_global = f_prim' * dmu_set;
    
    % Compute Q (in brute force)
    num_directions = size(dmu_set, 2);
    dmu = reshape(dmu_set,4,size(E_global,1)*num_directions);
    S_dmu = sum(dmu,1);
    term12 = zeros(1,size(E_global,1)*num_directions);
    K_dmu = zeros(4,size(E_global,1)*num_directions);
    
    for u=1:4
        IndEdgeVal = full(ind_edge_val_global{u}(:,x));
        IndEdgeVal = reshape(IndEdgeVal(:,ones(num_directions,1)),1,size(E_global,1)*num_directions);
        H_u = S_dmu.*IndEdgeVal - dmu(u,:);
        term12 = term12 + H_u.*IndEdgeVal;
        K_dmu(u,:) = -H_u;
    end
    
    for u=1:4
        K_dmu(u,:) = K_dmu(u,:) + term12;
    end
    
    %Q = reshape(dmu,4*size(E_global,1)*num_directions,1)' * reshape(K_dmu,4*size(E_global,1)*num_directions,1);
    
    Q = reshape(dmu,4*size(E_global,1),num_directions)' * reshape(K_dmu,4*size(E_global,1),num_directions);
    
    % Compute lambda
    lambda = g_global * pinv(Q);
    
    
    % Ensure lambda is feasible
    if sum(lambda)>1 || sum(lambda)<0
        delta_obj_list(1)=0;
        return
    end

    % compute dmu with a convex combination of multiple update directions
    dmu_global  = dmu_set * lambda';
    mu_global_i = reshape(mu_global(:,x) + dmu_global,4,size(E_global,1));
    
    for u=1:4
        Smu_global{u}(:,x) = sum(mu_global_i)'.*ind_edge_val_global{u}(:,x);
        Rmu_global{u}(:,x) = mu_global_i(u,:)';
    end
    
    
    
    % decompose global update into a set of local updates on individual trees, assuming the quantities are correctly computed
    %dmu_set = decompose_local_from_global ( dmu_global, E_global, ind_backwards, inverse_flag );
    dmu_set = compose_mu_local_from_global(dmu_global);
    
    
    
    
    %% Compute the difference in terms of the global objective
    % linear part of the objective
    delta_obj_first = f_prim' * dmu_global;

    % quadratic part
    dmu_global = reshape(dmu_global,4,size(E_global,1));
    smu = reshape(sum(dmu_global),size(E_global,1),1);
    term12 = zeros(1,size(E_global,1));
    Kxx_dmu = zeros(4,size(E_global,1));
    for u=1:4
        edgelabelindicator = full(ind_edge_val_global{u}(:,x));
        real_dmu_global = reshape(dmu_global(u,:),size(E_global,1),1);
        H_u = smu.*edgelabelindicator - real_dmu_global;
        term12 = term12 + reshape(H_u.*edgelabelindicator,1,size(E_global,1));
        Kxx_dmu(u,:) = reshape(-H_u,1,size(E_global,1));
    end
    for u=1:4
        Kxx_dmu(u,:) = Kxx_dmu(u,:) + term12;
    end
    Kxx_dmu = reshape(Kxx_dmu,4*size(E_global,1),1);
    delta_obj_second = -1/2 * normalization_quadratic *Kxx_dmu'*reshape(dmu_global,size(E_global,1)*4,1);
    % combine the first and the second term
    delta_obj_list(1) = delta_obj_first + delta_obj_second;
    %[delta_obj_first,delta_obj_second]

    
    %% Update the marginal dual variable on each individual random spanning tree
    for t=1:T_size
        ind_edge_val = ind_edge_val_list{t};
        mu = mu_list{t}(:,x);
        mu = mu + dmu_set(:,t);
        mu = reshape(mu, 4, size(E,1));
        for u = 1:4
            Smu_list{t}{u}(:,x) = (sum(mu)').*ind_edge_val{u}(:,x);
            Rmu_list{t}{u}(:,x) = mu(u,:)';
        end
        mu = reshape(mu, 4*(l-1),1);
        mu_list{t}(:,x) = mu;
        
       
    end
    
    return;
end




%% Compute the best objective value along the gradient
function [Gmax] = compute_Gmax(gradient, Ymax, E)

    m = size(Ymax,1);
    gradient    = reshape(gradient,4,size(E,1)*m);
    mu_max(1,:) = reshape(and(Ymax(:,E(:,1)) == -1,Ymax(:,E(:,2)) == -1)', 1, size(E,1)*m);
    mu_max(2,:) = reshape(and(Ymax(:,E(:,1)) == -1,Ymax(:,E(:,2)) == 1)', 1, size(E,1)*m);
    mu_max(3,:) = reshape(and(Ymax(:,E(:,1)) == 1,Ymax(:,E(:,2)) == -1)', 1, size(E,1)*m);
    mu_max(4,:) = reshape(and(Ymax(:,E(:,1)) == 1,Ymax(:,E(:,2)) == 1)', 1, size(E,1)*m);
    % Sum up the edge gradients that correspond to the edge labels
    Gmax = reshape(sum(gradient.*mu_max),size(E,1),m);
    Gmax = reshape(sum(Gmax,1),m,1);
    
end





%% Profile during test
function profile_update_ts
    global params;
    global profile;
    global E;
    global Ye;
    global Y_tr;
    global Kx_tr;
    global Y_ts;
    global Kx_ts;
    global mu;
    global obj;
    global primal_ub;
    global kappa;
    global norm_const_quadratic_list;
    m = size(Ye,2);
    tm = cputime;
    
    print_message(sprintf('tm: %d  iter: %d obj: %f mu: max %f min %f dgap: %f',...
    round(tm-profile.start_time),profile.iter,obj,max(max(mu)),min(min(mu)),primal_ub-obj),5,sprintf('/var/tmp/%s.log',params.filestem));

    if params.profiling
        profile.n_err_microlabel_prev = profile.n_err_microlabel;
        
        % Compute training error and statistics
        [Ypred_tr,~,Ys_positions_tr,Yi_positions_tr] = compute_error(Y_tr,Kx_tr,1);
        profile.microlabel_errors = sum(abs(Ypred_tr-Y_tr) >0,2);
        profile.n_err_microlabel = sum(profile.microlabel_errors);
        profile.p_err_microlabel = profile.n_err_microlabel/numel(Y_tr);
        profile.n_err = sum(profile.microlabel_errors > 0);
        profile.p_err = profile.n_err/length(profile.microlabel_errors);
        
        % Compute test error and statistics
        [Ypred_ts,~,Ys_positions_ts,Yi_positions_ts] = compute_error(Y_ts,Kx_ts,1);
        profile.microlabel_errors_ts = sum(abs(Ypred_ts-Y_ts) > 0,2);
        profile.n_err_microlabel_ts = sum(profile.microlabel_errors_ts);
        profile.p_err_microlabel_ts = profile.n_err_microlabel_ts/numel(Y_ts);
        profile.n_err_ts = sum(profile.microlabel_errors_ts > 0);
        profile.p_err_ts = profile.n_err_ts/length(profile.microlabel_errors_ts);
        
        % Print out message
        print_message(...
            sprintf('tm: %d 1_er_tr: %d (%3.2f) er_tr: %d (%3.2f) 1_er_ts: %d (%3.2f) er_ts: %d (%3.2f) Y*tr %3.2f%% %.2f Yitr %3.2f%% %.2f Y*ts %3.2f%% %.2f Yits %3.2f%% %.2f',...
            round(tm-profile.start_time),...
            profile.n_err,...
            profile.p_err*100,...
            profile.n_err_microlabel,...
            profile.p_err_microlabel*100,...
            round(profile.p_err_ts*size(Y_ts,1)),...
            profile.p_err_ts*100,sum(profile.microlabel_errors_ts),...
            sum(profile.microlabel_errors_ts)/numel(Y_ts)*100,...
            sum(Ys_positions_tr<=kappa)/size(Y_tr,1)*100,...
            mean(Ys_positions_tr),...
            sum(Yi_positions_tr<=kappa)/size(Y_tr,1)*100,...
            mean(Yi_positions_tr),...
            sum(Ys_positions_ts<=kappa)/size(Y_ts,1)*100,...
            mean(Ys_positions_ts),...
            sum(Yi_positions_ts<=kappa)/size(Y_ts,1)*100,...
            mean(Yi_positions_ts)),...
            0,sprintf('/var/tmp/%s.log',params.filestem));

        running_time = tm - profile.start_time;
        sfile = sprintf('/var/tmp/Ypred_%s.mat',params.filestem);
        save(sfile,'Ypred_tr','Ypred_ts','params','running_time','norm_const_quadratic_list');
        Ye = reshape(Ye,4*size(E,1),m);
    end
end

%% Profile during training
function profile_update_tr

    global params;
    global profile;
    global Y_tr;
    global Kx_tr;
    global obj;
    global kappa;
    global opt_round;
    global val_list;
    global kappa_list;
    global GmaxG0_list;
    global GoodUpdate_list;
    global T_size;
    global duality_gap_on_trees;
    global obj_list;

    if params.profiling
        profile.n_err_microlabel_prev = profile.n_err_microlabel;
        
        % compute training error
        [Ypred_tr,~,Ys_positions_tr,Yi_positions_tr] = compute_error(Y_tr,Kx_tr,1);  
        profile.microlabel_errors = sum(abs(Ypred_tr-Y_tr) >0,2);
        profile.n_err_microlabel = sum(profile.microlabel_errors);
        profile.p_err_microlabel = profile.n_err_microlabel/numel(Y_tr);
        profile.n_err = sum(profile.microlabel_errors > 0);
        profile.p_err = profile.n_err/length(profile.microlabel_errors);
        % Print out messages
        print_message(...
            sprintf('iter: %d 1_tr: %d %3.2f h_tr: %d %3.2f obj: %.2f gap: %.2f%% K: %d Y*: %3.2f%% %.2f Yi: %3.2f%% %.2f K: %.2f %.2f %d Mg: %.2f%% %.3f %.3f Update %.2f%% %.2f%%',...
            opt_round,...
            profile.n_err,...
            profile.p_err*100,...
            profile.n_err_microlabel,...
            profile.p_err_microlabel*100,...
            obj,...
            mean(duality_gap_on_trees./(obj_list+duality_gap_on_trees))*100,...
            kappa,...
            sum(Ys_positions_tr<=kappa)/size(Y_tr,1)*100,...
            mean(Ys_positions_tr),...
            sum(Yi_positions_tr<=kappa)/size(Y_tr,1)*100,...
            mean(Yi_positions_tr),...
            mean(kappa_list),...
            std(kappa_list),...
            max(kappa_list),...
            sum((val_list>0))/size(Y_tr,1)*100,...
            mean(val_list)*100,...
            std(val_list)*100,...
            mean(GmaxG0_list)/T_size*100,...
            sum(GoodUpdate_list)/size(Y_tr,1)*100),...
            0,sprintf('/var/tmp/%s.log',params.filestem));
    end
    
end

%% Compute training or test error
% PARAMETERS:
%       Ypred:  predictions
%       Ypred:  value of the predictions
%       Ys_positions:   positions of the predicted multilabel Y* that can be validated
%       Yi_positions:   positions of the true multilabel Yi
%
function [Ypred,YpredVal,Ys_positions,Yi_positions] = compute_error(Y,Kx,needPositions)

    % Collect global variables
    global T_size;
    global E_list;
    global Ye_list;
    global mu_list;
    global kappa;
    global iter;
    global l;
    Ypred = zeros(size(Y));
    YpredVal = zeros(size(Y,1),1);
    Y_kappa = zeros(size(Y,1)*T_size,size(Y,2)*kappa);
    Y_kappa_val = zeros(size(Y,1)*T_size,kappa);
    w_phi_e_local_list = cell(1,T_size);
    Ys_positions = ones(size(Y,1),1)*(kappa+1);
    Yi_positions = ones(size(Y,1),1)*(kappa+1);
    
    
    % Iteration over a collection of random spanning trees, and compute the K-best multilabel from each tree
    for t=1:T_size
        E   = E_list{t};
        Ye  = Ye_list{t};
        mu  = mu_list{t};
        w_phi_e = compute_w_phi_e(Kx,E,Ye,mu);
        
        
        l=size(E,1)+1;
        node_degree = zeros(1,l);
        for i=1:l
            node_degree(i) = sum(sum(E==i));
        end
        w_phi_e = reshape(w_phi_e,size(w_phi_e,1)*size(w_phi_e,2),1);
        w_phi_e_local_list{t} = w_phi_e;
        [Y_tmp,Y_tmp_val] = compute_topk_omp(w_phi_e,kappa,E,node_degree);
        %[Y_tmp,Y_tmp_val] = compute_topk(w_phi_e,kappa,E);

        
        Y_kappa(((t-1)*size(Y,1)+1):(t*size(Y,1)),:)        = Y_tmp;
        Y_kappa_val(((t-1)*size(Y,1)+1):(t*size(Y,1)),:)    = Y_tmp_val;
    end
    
    % Collect K-best multilabels and compute the best multilabel from the list
    for i=1:size(Y,1)
        % if it is in the initialization step, the prediction will be the default values
        if iter==0
            Ypred(i,:) = -1*ones(1,size(Y,2));
        else
            IN_E = zeros((l-1)*2,T_size);
            for t=1:T_size
                IN_E(:,t) = reshape(E_list{t},(l-1)*2,1);
            end
            IN_gradient = zeros(4*(l-1),T_size);
            for t=1:T_size
                IN_gradient(:,t) = w_phi_e_local_list{t}(((i-1)*4*(l-1)+1):i*4*(l-1),:);
            end
            Y_kappa((i:size(Y_kappa,1)/T_size:size(Y_kappa,1)),:) = (Y_kappa((i:size(Y_kappa,1)/T_size:size(Y_kappa,1)),:)+1)/2;

            if needPositions
                % Get the position of Yi
                [~,~,~,Yi_pos] = find_worst_violator_new(...
                    Y_kappa((i:size(Y_kappa,1)/T_size:size(Y_kappa,1)),:),...
                    Y_kappa_val((i:size(Y_kappa,1)/T_size:size(Y_kappa_val,1)),:),...
                    (Y(i,:)+1)/2,...
                    IN_E,...
                    IN_gradient);
                Yi_positions(i) = Yi_pos;
                % Get the position of Y*, and the best multilabel
                [Ypred(i,:),~,Ys_pos,~] = find_worst_violator_new(...
                    Y_kappa((i:size(Y_kappa,1)/T_size:size(Y_kappa,1)),:),...
                    Y_kappa_val((i:size(Y_kappa,1)/T_size:size(Y_kappa_val,1)),:),...
                    [],...
                    IN_E,...
                    IN_gradient);
                Ys_positions(i) = Ys_pos;
            else
                [Ypred(i,:),~,~,~] = find_worst_violator_new(...
                Y_kappa((i:size(Y_kappa,1)/T_size:size(Y_kappa,1)),:),...
                Y_kappa_val((i:size(Y_kappa,1)/T_size:size(Y_kappa_val,1)),:),...
                [],...
                IN_E,...
                IN_gradient);
            end
            
            Ypred(i,:) = Ypred(i,:)*2-1;
        end
    end
end


%% for testing
% Input: test kernel, tree index
% Output: gradient
function w_phi_e = compute_w_phi_e(Kx,E,Ye,mu)
    m = numel(mu)/size(E,1)/4;
    Ye = reshape(Ye,4,size(E,1)*m);   
    mu = reshape(mu,4,size(E,1)*m);
    m_oup = size(Kx,2);

    % compute gradient
    if isempty(find(mu,1))
        w_phi_e = zeros(4,size(E,1)*m_oup);
    else  
        w_phi_e = sum(mu);
        w_phi_e = w_phi_e(ones(4,1),:);
        w_phi_e = Ye.*w_phi_e;
        w_phi_e = w_phi_e-mu;
        w_phi_e = reshape(w_phi_e,4*size(E,1),m);
        w_phi_e = w_phi_e*Kx;
        w_phi_e = reshape(w_phi_e,4,size(E,1)*m_oup);
    end
    
    return;
end

%% Compute loss vector. there are two type of losses:
%   1. 1 loss, incorrect multilabel will always have loss=1, will encourage
%   sparsity of support vector.
%   2. scaled loss, incorrect multilabel will have loss propotion to its
%   error, will have more support vector.
function [loss,Ye,ind_edge_val] = compute_loss_vector(Y,t,scaling)
    % scaling: 0=do nothing, 1=rescale node loss by degree
    global params;
    global E_list;
    global m;
    ind_edge_val = cell(4,1);
    %print_message(sprintf('Computing loss vector for %d_th Tree.',t),0);
    E = E_list{t};
    loss = ones(4,m*size(E,1));
    Te1 = Y(:,E(:,1))'; % the label of edge tail
    Te2 = Y(:,E(:,2))'; % the label of edge head
    NodeDegree = ones(size(Y,2),1);
    if scaling ~= 0 % rescale to microlabels by dividing node loss among the adjacent edges
        for v = 1:size(Y,2)
            NodeDegree(v) = sum(E(:) == v);
        end
    end
    NodeDegree = repmat(NodeDegree,1,m);    
    u = 0;
    for u_1 = [-1, 1]
        for u_2 = [-1, 1]
            u = u + 1;
            loss(u,:) = reshape((Te1 ~= u_1).*NodeDegree(E(:,1),:)+(Te2 ~= u_2).*NodeDegree(E(:,2),:),m*size(E,1),1);
        end
    end     
    loss = reshape(loss,4*size(E,1),m);
      
    Ye = reshape(loss==0,4,size(E,1)*m);
    for u = 1:4
        ind_edge_val{u} = sparse(reshape(Ye(u,:)~=0,size(E,1),m));
        %full(ind_edge_val{u})
    end
    Ye = reshape(Ye,4*size(E,1),m);
    if params.losstype == 'r'   % uniform loss
        loss = loss*0+1; 
    end
    
    %loss = loss*0 + size(E,1);
    %loss = loss; % scaled loss
    %loss = loss + sqrt(size(E_list{1},1));
    
    return;
end

%% Initialize the profiling function
function profile_init

    global profile;
    profile.start_time          = cputime;
    profile.n_err               = 0;
    profile.p_err               = 0; 
    profile.n_err_microlabel    = 0; 
    profile.p_err_microlabel    = 0; 
    profile.n_err_microlabel_prev   = 0;
    profile.microlabel_errors   = [];
    profile.iter                = 0;
    profile.err_ts              = 0;
    
end

%% Initialize the optimization
function optimizer_init

    clear;
    clear iter;
    global T_size;
    global Rmu_list;
    global Smu_list;
    global obj;
    global delta_obj;
    global opt_round;
    global E_list;
    global m;
    global l;
    global node_degree_list;
    
    Rmu_list = cell(T_size,1);
    Smu_list = cell(T_size,1);
    
    for t=1:T_size
        Rmu_list{t} = cell(1,4);
        Smu_list{t} = cell(1,4);
        for u = 1:4
            Smu_list{t}{u} = zeros(size(E_list{t},1),m);
            Rmu_list{t}{u} = zeros(size(E_list{t},1),m);
        end
    end
    
    node_degree_list = cell(T_size,1);
    for t=1:T_size
	    node_degree = zeros(1,l);
        for i=1:l
            node_degree(i) = sum(sum(E_list{t}==i));
        end
        node_degree_list{t}=node_degree;
    end
    
    
    obj         = 0;
    delta_obj   = 0;
    opt_round   = 0;
    
end

%% Print out message
function print_message(msg,verbosity_level,filename)

    global params;
    if params.verbosity >= verbosity_level
        fprintf('\n%s: %s ',datestr(clock,13),msg);
        if nargin == 3
            fid = fopen(filename,'a');
            fprintf(fid,'%s: %s\n',datestr(clock,13),msg);
            fclose(fid);
        end
    end
    
end
