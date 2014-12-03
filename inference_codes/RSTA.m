


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
%   This function is called by run_RSTA(.)
%
%

function [rtn, ts_err] = RSTA(paramsIn, dataIn)

    %% Definition of global variables
    global loss_list;           % losses associated with different edge labelings
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
    global kappa_decrease_flags;  
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
    norm_const_linear           = 1/(T_size)/size(E_list{1},1); % normalization constant of the linear term of the objective function
    norm_const_quadratic_list   = zeros(1,T_size)+1/(T_size);   % normalization constant of the quadratic term of the objective function
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
    
    
    kappa_decrease_flags = zeros(1,m);
    kappa = kappa_INIT;
    
    
    for t=1:T_size
        [loss_list{t},Ye_list{t},ind_edge_val_list{t}] = compute_loss_vector(Y_tr,t,params.mlloss);
        mu_list{t}          = zeros(4*size(E_list{1},1),m);
        Kxx_mu_x_list{t}    = zeros(4*size(E_list{1},1),m);
    end

    iter = 0; 
    
    
    
    val_list        = zeros(1,m);
    Yipos_list      = zeros(1,m)*T_size;
    kappa_list      = zeros(1,m);
    GmaxG0_list     = zeros(1,m);
    GoodUpdate_list = zeros(1,m);
    obj_list        = zeros(1,T_size);
    
    
    %% Initialization
    optimizer_init;
    profile_init;

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
    
    
    Yipos_list = ones(1,m);
    
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
        kappa_decrease_flags = ones(1,m);
        %Yipos_list = ones(1,m);
        val_list = zeros(1,m);

        if iter <= 30
            Yipos_list = ones(1,m);
        end
        for xi = randsample(1:m,m,true,Yipos_list/sum(Yipos_list))
%        for xi = selected_samples
%         for xi = randsample(1:m,m)
%         for xi = 1:m
            print_message(sprintf('Start descend on example %d initial k %d',xi,kappa),3)

                kappa_decrease_flag(xi)=0;
                [delta_obj_list,kappa_decrease_flags(xi)] = conditional_gradient_descent(xi,kappa);    % optimize on single example
%                 
%                 kappa0=kappa;
%                 while ( kappa_decrease_flags(xi)==0 ) && kappa0 < params.maxkappa 
%                     kappa0=kappa0*2;
%                     [delta_obj_list,kappa_decrease_flags(xi)] = conditional_gradient_descent(xi,kappa0);    % optimize on single example
%                 end
                kappa_list(xi)=kappa;
            
            obj_list = obj_list + delta_obj_list;
            obj = obj + sum(delta_obj_list);
            % update kappa
            if kappa_decrease_flags(xi)==0
                kappa = min(kappa*2,kappa_MAX);
            else
                kappa = max(ceil(kappa/2),kappa_MIN);
            end
        end
        
        %[kappa_decrease_flags;Yipos_list]
        %GmaxG0_list
        %kappa_decrease_flags
        %Yipos_list
        %obj_list
        
        if mod(iter, params.profileiter)==0
            progress_made = (obj >= prev_obj);  
            prev_obj = obj;
            
                compute_duality_gap;
            
            profile_update_tr;          % profile update for training
            % update flip number
            if profile.n_err_microlabel > profile.n_err_microlabel_prev
                nflip = nflip - 1;
            end
            % update current best solution
            if profile.n_err_microlabel < best_n_err_microlabel || 1
                best_n_err_microlabel=profile.n_err_microlabel;
                best_iter = iter;
                best_kappa = kappa;
                best_mu_list=mu_list;
                best_Kxx_mu_x_list=Kxx_mu_x_list;
                best_Rmu_list=Rmu_list;
                best_Smu_list=Smu_list;
                best_norm_const_quadratic_list = norm_const_quadratic_list;
            end
            %mu_list{1}
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
    profile_update;
    
    

    rtn = 0;
    ts_err = 0;
end


%% Complete part of gradient for everything
function Kmu = compute_Kmu(Kx,mu,E,ind_edge_val)

    m_oup = size(Kx,2);
    m = size(Kx,1);
    if  0 %and(params.debugging, nargin == 2)
        for x = 1:m
           Kmu(:,x) = compute_Kmu_x(x,Kx(:,x));
        end
        Kmu = reshape(Kmu,4,size(E,1)*m);
    else
        mu = reshape(mu,4,size(E,1)*m);
        Smu = reshape(sum(mu),size(E,1),m);
        term12 =zeros(1,size(E,1)*m_oup);
        Kmu = zeros(4,size(E,1)*m_oup);
        for u = 1:4
            IndEVu = full(ind_edge_val{u});    
            Rmu_u = reshape(mu(u,:),size(E,1),m);
            H_u = Smu.*IndEVu;
            H_u = H_u - Rmu_u;
            Q_u = H_u*Kx;
            term12 = term12 + reshape(Q_u.*IndEVu,1,m_oup*size(E,1));
            Kmu(u,:) = reshape(-Q_u,1,m_oup*size(E,1));
        end
        for u = 1:4
            Kmu(u,:) = Kmu(u,:) + term12;
        end
    end
    %mu = reshape(mu,mu_siz);
    return;
end


%% Compute gradient on example x, gradient is l-uk, this function will compute uk
% Input:
%   x,Kx,t
% Output
%   gradient for current example x
function Kmu_x = compute_Kmu_x(x,Kx,E,ind_edge_val,Rmu,Smu)

    % local
    term12 = zeros(1,size(E,1));
    term34 = zeros(4,size(E,1));
    
    % main
    % For speeding up gradient computations: 
    % store sums of marginal dual variables, distributed by the
    % true edge values into Smu
    % store marginal dual variables, distributed by the
    % pseudo edge values into Rmu
   
    for u = 1:4
        Ind_te_u = full(ind_edge_val{u}(:,x));
        H_u = Smu{u}*Kx-Rmu{u}*Kx;
        term12(1,Ind_te_u) = H_u(Ind_te_u)';
        term34(u,:) = -H_u';
    end
    Kmu_x = reshape(term12(ones(4,1),:) + term34,4*size(E,1),1);
end


%% Function to compute the relative duality gap
function compute_duality_gap

    %% global parameters
    global obj_list;
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
    
    %% Get k best prediction from each random spanning tree
    
    % result holders
    Kmu_list_local      = cell(1,T_size);
    gradient_list_local = cell(1,T_size);
    
    % iteration over a collection of spanning trees
    for t = 1:T_size
        % retrieve or compute variables locally on each spanning tree
        loss = loss_list{t};
        E = E_list{t};
        mu = mu_list{t};
        ind_edge_val = ind_edge_val_list{t};
        loss = reshape(loss,4,size(E,1)*m);
        Kmu_list_local{t} = compute_Kmu(Kx_tr, mu, E, ind_edge_val);
        Kmu_list_local{t} = reshape(Kmu_list_local{t}, 4, size(E,1)*m);
        Kmu = Kmu_list_local{t};
        gradient_list_local{t} = norm_const_linear*loss - norm_const_quadratic_list(t)*Kmu;
        gradient = gradient_list_local{t};
        node_degree = node_degree_list{t};
        in_gradient = reshape(gradient,numel(gradient),1);
        
        % compute and save the K-best multilabels and their scores
        [Y_tmp,Y_tmp_val] = compute_topk_omp(in_gradient, kappa, E, node_degree);
        Y_kappa(((t-1)*size(Y,1)+1):(t*size(Y,1)),:)        = Y_tmp;
        Y_kappa_val(((t-1)*size(Y,1)+1):(t*size(Y,1)),:)    = Y_tmp_val;
    end
    
    %% Get the worst violator from the K best predictions of each example
    for i=1:size(Y,1)
        % if the optimization has not been started yet, give default value to the predictions
        if iter==0
            kappa_decrease_flag = 1;
            Ypred(i,:) = -1*ones(1,size(Y_tr,2));
        else
            IN_E = zeros((l-1)*2,T_size);
            for t=1:T_size
                IN_E(:,t) = reshape(E_list{t},(l-1)*2,1);
            end
            IN_gradient = zeros(4*(l-1),T_size);
            for t=1:T_size
                IN_gradient(:,t) = reshape(gradient_list_local{t}(:,((i-1)*(l-1)+1):(i*(l-1))),4*(l-1),1);
            end
            Y_kappa((i:size(Y_tr,1):size(Y_kappa,1)),:) = (Y_kappa((i:size(Y_tr,1):size(Y_kappa,1)),:)+1)/2;
            % Compute the worst violator
            [Ypred(i,:),~,kappa_decrease_flag] = ...
                find_worst_violator_new(...
                Y_kappa((i:size(Y_tr,1):size(Y_kappa,1)),:),...
                Y_kappa_val((i:size(Y_tr,1):size(Y_kappa_val,1)),:)...
                ,[],IN_E,IN_gradient);
            Ypred(i,:) = Ypred(i,:)*2-1;
        end
    end

    clear Y_kappa;
    clear Y_kappa_val;
    
    %% Compute duality gap of all training examples from each random spanning tree
    % Define variable to save the results
    dgap = zeros(1,T_size);
    % Iterate over a collection of random spanning trees
    for t=1:T_size
        % compute or collect variables locally on each spanning tree
        loss = loss_list{t};
        E = E_list{t};
        mu = mu_list{t};
        Kmu = Kmu_list_local{t};
        gradient = gradient_list_local{t};
        loss = reshape(loss,size(loss,1)*size(loss,2),1);
        mu = reshape(mu,size(loss,1)*size(loss,2),1);
        Kmu = reshape(Kmu,size(loss,1)*size(loss,2),1);
        % compute current maxima on function
        %Gcur = norm_const_linear*mu'*loss - 1/2*norm_const_quadratic_list(t)*mu'*Kmu;
        % compute current maxima on gradient
        Gcur = reshape(gradient,1,size(gradient,1)*size(gradient,2))*reshape(mu,1,size(mu,1)*size(mu,2))';
        % compute best possible objective along Y* and gradient.
        Gmax = compute_Gmax(gradient,Ypred,E);
        % the difference is estimated as duality gap
        dgap(t) = params.C*sum(Gmax)-Gcur;
    end
% %     % Comment out old duality gap computation
% %     for t=1:T_size
% %         %loss = loss_list{t};
% %         E = E_list{t};
% %         mu = mu_list{t};
% %         %ind_edge_val = ind_edge_val_list{t};
% %         %Kmu = Kmu_list_local{t};
% %         gradient = gradient_list_local{t};
% %         
% %         Gmax = compute_Gmax(gradient,Ypred,E);
% %         mu = reshape(mu,4,m*size(E,1));
% % 
% %         %[params.C*max(Gmax,0), sum(reshape(sum(gradient.*mu),size(E,1),m),1)']'
% %         duality_gap = params.C*max(Gmax,0) - sum(reshape(sum(gradient.*mu),size(E,1),m),1)';
% %         
% %         dgap(t) = sum(duality_gap);
% %     end

    %% Compute primal upper bound, which is obj+duality gap
    dgap = max(0,dgap);
    %duality_gap_on_trees = min(dgap,duality_gap_on_trees);
    duality_gap_on_trees = dgap;
    
    primal_ub = obj + sum(dgap);
    
end



%% Perform conditional gradient optimization on single training example, upadte corresponding marginal dual variable.
% Reviewed on 16/05/2014
% input: 
%   x:      the id of current training example
%   obj:    current objective
%   kappa:  current kappa
function [delta_obj_list,kappa_decrease_flag] = conditional_gradient_descent(x, kappa)
    %% Parameter definition
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
    global iter;
    
    global val_list;
    global Yipos_list;
    global GmaxG0_list;
    global GoodUpdate_list;
    
    
    %% Collect top-K prediction from each tree.
    % Define variables to collect results.
    Y_kappa = zeros(T_size,kappa*l);
    Y_kappa_val = zeros(T_size,kappa);
    gradient_list_local = cell(1,T_size);
    Kmu_x_list_local = cell(1,T_size);
    
    % Iterate over spanning trees.
    for t=1:T_size
        % Variables located for tree t and example x.
        loss = loss_list{t}(:,x);
        Ye = Ye_list{t}(:,x);
        ind_edge_val = ind_edge_val_list{t};
        mu = mu_list{t}(:,x);
        E = E_list{t};
        Rmu = Rmu_list{t};
        Smu = Smu_list{t};    
        % Compute some quantities for tree t.
        % Kmu_x = K_x*mu_x
        Kmu_x_list_local{t} = compute_Kmu_x(x,Kx_tr(:,x),E,ind_edge_val,Rmu,Smu);
        Kmu_x = Kmu_x_list_local{t};
        % current gradient    
        gradient_list_local{t} =  norm_const_linear*loss - norm_const_quadratic_list(t)*Kmu_x;
        gradient = gradient_list_local{t};
        
        % Find top K violator.
%         [Ymax,YmaxVal] = compute_topk(gradient,kappa,E);
        node_degree = zeros(1,l);
        for i=1:l
            node_degree(i) = sum(sum(E==i));
        end
        
        [Ymax,YmaxVal] = compute_topk_omp(gradient,kappa,E,node_degree);

        if iter==0
            adfs
        end
        if x==0
            disp('1. k best from all trees');
            reshape(mu,4,size(gradient,1)/4)  
            reshape(gradient,4,size(gradient,1)/4)  
            Y_tr(x,:)
            Ymax
            gradient
        end
        

        
        % Save results.
        Y_kappa(t,:) = Ymax;
        Y_kappa_val(t,:) = YmaxVal;
    end
    
    
    %% Get worst violating multilabel from K best list.
    if iter==0
        % TODO, the condition might not be satisfied
        % the first iteration: set default violator
        Ymax = ones(1,l)*(-1);
        kappa_decrease_flag=1;
    else
        IN_E = zeros(size(E_list{1},1)*2,size(E_list,1));
        for t=1:T_size
            IN_E(:,t) = reshape(E_list{t},size(E_list{1},1)*2,1);
        end
        IN_gradient = zeros(size(gradient_list_local{1},1),size(E_list,1));
        for t=1:T_size
            IN_gradient(:,t) = gradient_list_local{t};
        end
        Y_kappa = (Y_kappa+1)/2;
        Yi = (Y_tr(x,:)+1)/2;

        [Ymax, val, kappa_decrease_flag,Yi_pos] = find_worst_violator_new(Y_kappa,Y_kappa_val,Yi,IN_E,IN_gradient);
        val_list(x) = val;
        Yipos_list(x) = Yi_pos;
        Ymax = Ymax*2-1;
    end
     

    %% If the worst violator is the correct label, exit without update mu.
    if sum(Ymax~=Y_tr(x,:))==0 %|| ( ( (kappa_decrease_flag==0) && kappa < params.maxkappa) && iter~=1 )
        delta_obj_list = zeros(1,T_size);
        return;
    end
    
    
    %% Otherwise we need line serach to find optimal step size to the saddle point.
    % reviewed on 16/05/2014
    mu_d_list = mu_list;
    nomi=zeros(1,T_size);
    denomi=zeros(1,T_size);
    kxx_mu_0 = cell(1,T_size);
    Gmax = zeros(1,T_size);
    G0 = zeros(1,T_size);
    Kmu_d_list = cell(1,T_size);
    for t=1:T_size
        %% Obtain variables located for tree t and example x.
        loss = loss_list{t}(:,x);
        Ye = Ye_list{t}(:,x);
        ind_edge_val = ind_edge_val_list{t};
        mu = mu_list{t}(:,x);
        E = E_list{t};
        Rmu = Rmu_list{t};
        Smu = Smu_list{t};
        
        %% compute Gmax and G0
        Kmu_x = Kmu_x_list_local{t};
        gradient = gradient_list_local{t};
        % Compute Gmax, which is the best objective value along gradient.
        Gmax(t) = compute_Gmax(gradient,Ymax,E);
        Gmax(t) = Gmax(t)*params.C;
        % Compute G0, which is current objective value. 
        G0(t) = -mu'*gradient;
        %G0(t) = norm_const_linear*loss'*mu - 1/2*norm_const_quadratic_list(t)*Kmu_x'*mu;
        
        %% Compte mu_0, which is the best margin violator into update direction.
        Umax_e = 1+2*(Ymax(:,E(:,1))>0) + (Ymax(:,E(:,2)) >0);
        mu_0 = zeros(size(mu));
        for u = 1:4
            mu_0(4*(1:size(E,1))-4 + u) = params.C*(Umax_e == u);
        end
        
        %% compute Kmu_0
        if sum(mu_0) > 0
            smu_1_te = sum(reshape(mu_0.*Ye,4,size(E,1)),1);
            smu_1_te = reshape(smu_1_te(ones(4,1),:),length(mu),1);
            kxx_mu_0{t} = ~Ye*params.C+mu_0-smu_1_te;
        else
            kxx_mu_0{t} = zeros(size(mu));
        end
        
        Kmu_0 = Kmu_x + kxx_mu_0{t} - Kxx_mu_x_list{t}(:,x);

        mu_d = mu_0 - mu;
        Kmu_d = Kmu_0-Kmu_x;
              
        Kmu_d_list{t} = Kmu_d;
        mu_d_list{t} = mu_d;
        nomi(t) = mu_d'*gradient;
        denomi(t) = norm_const_quadratic_list(t) * Kmu_d' * mu_d;
        
            
        if x==0
            %reshape(mu_d,4,size(E,1))
            [nomi(t),denomi(t)]
            [nomi(t)/denomi(t)]
            if iter == 10
                asdfad
            end
        end

    end
    
    %% Decide whether to update or not.
    if sum(Gmax)>=sum(G0)
        tau = min(sum(nomi)/sum(denomi),1);
    else
        tau=0;
    end
    % Step size should not be negative.
    tau = max(tau,0);

	GmaxG0_list(x) = sum(Gmax>=G0);
    GoodUpdate_list(x) = (tau>0);
    
    
    %% update for each tree
    delta_obj_list = zeros(1,T_size);
    for t=1:T_size
        % variables located for tree t and example x
        loss = loss_list{t}(:,x);
        Ye = Ye_list{t}(:,x);
        ind_edge_val = ind_edge_val_list{t};
        mu = mu_list{t}(:,x);
        E = E_list{t};
        gradient =  gradient_list_local{t};
        mu_d = mu_d_list{t};
        Kmu_d = Kmu_d_list{t};
        %
        delta_obj_list(t) = gradient'*mu_d*tau - norm_const_quadratic_list(t)*tau^2/2*mu_d'*Kmu_d;
        mu = mu + tau*mu_d;
        Kxx_mu_x_list{t}(:,x) = (1-tau)*Kxx_mu_x_list{t}(:,x) + tau*kxx_mu_0{t};
        % update Smu Rmu
        mu = reshape(mu,4,size(E,1));
        for u = 1:4
            Smu_list{t}{u}(:,x) = (sum(mu)').*ind_edge_val{u}(:,x);
            Rmu_list{t}{u}(:,x) = mu(u,:)';
        end
        
        mu = reshape(mu,4*size(E,1),1);
        mu_list{t}(:,x) = mu;
    end
    
    %%
    return;
end
%% Perform conditional gradient optimization on single training example, upadte corresponding marginal dual variable.
% Reviewed on 16/05/2014
% input: 
%   x:      the id of current training example
%   obj:    current objective
%   kappa:  current kappa
function [delta_obj_list,kappa_decrease_flag] = conditional_gradient_descent_convex(x, kappa)
    %% Parameter definition
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
    global iter;
    
    global val_list;
    global Yipos_list;
    global GmaxG0_list;
    global GoodUpdate_list;
    
    
    %% Collect top-K prediction from each tree.
    % Define variables to collect results.
    Y_kappa = zeros(T_size,kappa*l);
    Y_kappa_val = zeros(T_size,kappa);
    gradient_list_local = cell(1,T_size);
    Kmu_x_list_local = cell(1,T_size);
    
    % Iterate over spanning trees.
    for t=1:T_size
        % Variables located for tree t and example x.
        loss = loss_list{t}(:,x);
        Ye = Ye_list{t}(:,x);
        ind_edge_val = ind_edge_val_list{t};
        mu = mu_list{t}(:,x);
        E = E_list{t};
        Rmu = Rmu_list{t};
        Smu = Smu_list{t};    
        % Compute some quantities for tree t.
        % Kmu_x = K_x*mu_x
        Kmu_x_list_local{t} = compute_Kmu_x(x,Kx_tr(:,x),E,ind_edge_val,Rmu,Smu);
        Kmu_x = Kmu_x_list_local{t};
        % current gradient    
        gradient_list_local{t} =  norm_const_linear*loss - norm_const_quadratic_list(t)*Kmu_x;
        gradient = gradient_list_local{t};
        
        % Find top K violator.
%         [Ymax,YmaxVal] = compute_topk(gradient,kappa,E);
        node_degree = zeros(1,l);
        for i=1:l
            node_degree(i) = sum(sum(E==i));
        end
        
        [Ymax,YmaxVal] = compute_topk_omp(gradient,kappa,E,node_degree);

        if iter==0
            adfs
        end
        if x==0
            disp('1. k best from all trees');
            reshape(mu,4,size(gradient,1)/4)  
            reshape(gradient,4,size(gradient,1)/4)  
            Y_tr(x,:)
            Ymax
            gradient
        end
        
        % Save results.
        Y_kappa(t,:) = Ymax;
        Y_kappa_val(t,:) = YmaxVal;
    end
    
    %% Compute update direction. Worst violating multilabels is defined as the first column, the number of trees define the number of multilabel used in convex direction finding.
    mu_0_T = cell(T_size,1);
    for t=1:T_size
        E=E_list{t};
        mu_0s = zeros(4*size(E,1),T_size);
        for i=1:T_size
            Ymax = Y_kappa(i,1:l);
            Ymax = Y_kappa(1,((i-1)*l+1):(i*l));
            Umax_e = 1+2*(Ymax(:,E(:,1))>0) + (Ymax(:,E(:,2)) >0);
            mu_0 = zeros(size(mu));
            for u = 1:4
                mu_0(4*(1:size(E,1))-4 + u) = params.C*(Umax_e == u);
            end
            mu_0s(:,i)=reshape(mu_0,4*size(E,1),1);
        end
        mu_0_T{t}=mu_0s;
    end
    

    kappa_decrease_flag = 1;
    
    %% Generalized line search to get lambda.
    % direction mu_0_T{t}:4|E|*n_direction
    % loss: 4|E|*1
    linear_term = zeros(T_size,1);
    quadratic_term = zeros(T_size,T_size);
    n_direction=size(mu_0_T{1},2);
    kxx_mu_0 = cell(1,T_size);
    % Iterate on each tree.
    for t=1:T_size
        % Comptue linear term;
        linear_term = linear_term + norm_const_linear*mu_0_T{t}'*loss;
        % Compute quadratic term
        Ye = Ye_list{t}(:,x);
        E = E_list{t};
        Kmu_x = Kmu_x_list_local{t};
        mu_0s = mu_0_T{t};
        % Compute Kmu_x
        Kmu_0 = zeros(4*size(E,1),n_direction);
        kxx_mu_0{t} = zeros(size(mu_0s,1),n_direction);
        for i=1:n_direction
            mu_0 = mu_0s(:,i);
            smu_1_te = sum(reshape(mu_0.*Ye,4,size(E,1)),1);
            smu_1_te = reshape(smu_1_te(ones(4,1),:),length(mu),1);
            kxx_mu_0{t}(:,i) = ~Ye*params.C+mu_0-smu_1_te;
%             Kmu_x
%             kxx_mu_0{t}(:,i)
%             Kxx_mu_x_list{t}(:,x)
            Kmu_0(:,i) = Kmu_x + kxx_mu_0{t}(:,i) - Kxx_mu_x_list{t}(:,x);
        end
        Kmu_0'
        mu_0_T{t}
        quadratic_term = quadratic_term + 0.5*norm_const_quadratic_list(t)*Kmu_0'*mu_0_T{t}
    end
    % Perform quadratic programming to solve for lambda.
    opts = optimoptions('quadprog',...
        'Diagnostics','off',...
        'Algorithm','active-set',...
        'Display','final');
    quadratic_term*2
    lambda = quadprog(quadratic_term*2,linear_term,...
        [],[],...
        ones(1,n_direction),1,... % equality constraint: sum of lambda is 1
        zeros(1,n_direction),ones(1,n_direction),... % inequality constraint: 0<=lambda<<1
        [],opts); 
    lambda
    
    if x==2
        sdafasd
    end
    %% Compute Gmax and G0.
    for t=1:T_size
        mu = mu_list{t}(:,x);
        mu_t = reshape(lambda'*mu_0_T{t}',4*size(E,1),1);
        gradient = gradient_list_local{t};
        Gmax(t) = mu_t'*gradient;
        G0(t) = -mu'*gradient;
    end
    
    
    
    GmaxG0_list(x) = sum(Gmax>=G0);
    GoodUpdate_list(x) = 1;
    
    
    
    %% Decide whether to update.
    for t=1:T_size
        delta_obj_list(t) = 0;
    end
    if sum(Gmax) >= sum(G0)
        for t=1:T_size
            % update mu
            mu = reshape(lambda'*mu_0_T{t}',4*size(E,1),1);
            delta_obj_list(t) = 0;% gradient'*mu_d*tau - norm_const_quadratic_list(t)*tau^2/2*mu_d'*Kmu_d;
            % update Kxx_mu_x_list
            Kxx_mu_x_list{t}(:,x) = lambda'*kxx_mu_0{t}';
            % update Smu Rmu
            mu = reshape(mu,4,size(E,1));
            for u = 1:4
                Smu_list{t}{u}(:,x) = (sum(mu)').*ind_edge_val{u}(:,x);
                Rmu_list{t}{u}(:,x) = mu(u,:)';
            end
            % save mu
            mu = reshape(mu,4*size(E,1),1);
            mu_list{t}(:,x) = mu;
        end
    end
    
    

    
    %%
    return;
end
% function [delta_obj_list,kappa_decrease_flag] = par_conditional_gradient_descent(x, kappa)
%     global loss_list;
%     global loss;
%     global Ye_list;
%     global Ye;
%     global E_list;
%     global E;
%     global mu_list;
%     global mu;
%     global ind_edge_val_list;
%     global ind_edge_val;
%     global Rmu_list;
%     global Smu_list;
%     global Kxx_mu_x_list;
%     global norm_const_linear;
%     global norm_const_quadratic_list;
%     global l;
%     global Kx_tr;
%     global Y_tr;
%     global T_size;
%     global params;
%     
%     Y_kappa = zeros(T_size,kappa*l);        % label prediction
%     Y_kappa_val = zeros(T_size,kappa);      % score    
%     gradient_list_local = cell(1,T_size);
%     Kmu_x_list_local = cell(1,T_size);
% 
%     %% collect top-K prediction from each tree
%     print_message(sprintf('Collect top-k prediction from each tree T-size %d', T_size),3)
%     parfor t=1:T_size
%         pause(0.000);
%         loss = loss_list{t}(:,x);
%         Ye = Ye_list{t}(:,x);
%         ind_edge_val = ind_edge_val_list{t};
%         mu = mu_list{t}(:,x);
%         E = E_list{t};
%         Rmu = Rmu_list{t};
%         Smu = Smu_list{t};
%         % compute the quantity for tree t
%         Kmu_x_list_local{t} = compute_Kmu_x(x,Kx_tr(:,x),E,ind_edge_val,Rmu,Smu); % Kmu_x = K_x*mu_x
%         Kmu_x = Kmu_x_list_local{t};
%         gradient_list_local{t} =  norm_const_linear*loss - norm_const_quadratic_list(t)*Kmu_x;    % current gradient    
%         gradient = gradient_list_local{t};
%         % find top k violator
%         [Ymax,YmaxVal] = compute_topk(gradient,kappa,E);
%         % save result
%         Y_kappa(t,:) = Ymax;
%         Y_kappa_val(t,:) = YmaxVal;
%     end
%     
%     %% get worst violator from top K
%     print_message(sprintf('Get worst violator'),3)
%     [Ymax, ~, kappa_decrease_flag] = find_worst_violator(Y_kappa,Y_kappa_val,[]);
%     
%     if ~kappa_decrease_flag
%         [Ymax,~] = majority_voting(Y_kappa,Y_kappa_val,l);
%     end
% 
%     %% if the worst violator is the correct label, exit without update mu
%     if sum(Ymax~=Y_tr(x,:))==0
%         delta_obj_list = zeros(1,T_size);
%         return;
%     end
% 
%     
% %% otherwise line serach
%     mu_d_list = mu_list;
%     nomi=zeros(1,T_size);
%     denomi=zeros(1,T_size);
%     kxx_mu_0 = cell(1,T_size);
%     Gmax = zeros(1,T_size);
%     G0 = zeros(1,T_size);
%     Kmu_d_list = cell(1,T_size);
%     parfor t=1:T_size
%         pause(0.000);
%         % variables located for tree t and example x
%         loss = loss_list{t}(:,x);
%         Ye = Ye_list{t}(:,x);
%         ind_edge_val = ind_edge_val_list{t};
%         mu = mu_list{t}(:,x);
%         E = E_list{t};
%         Rmu = Rmu_list{t};
%         Smu = Smu_list{t};
% 
%         %% 
%         Kmu_x = Kmu_x_list_local{t};
%         gradient = gradient_list_local{t};
%         Gmax(t) = compute_Gmax(gradient,Ymax,E);            % objective under best labeling
%         G0(t) = -mu'*gradient;                               % current objective
% 
%         %% best margin violator into update direction mu_0
%         Umax_e = 1+2*(Ymax(:,E(:,1))>0) + (Ymax(:,E(:,2)) >0);
%         mu_0 = zeros(size(mu));
%         for u = 1:4
%             mu_0(4*(1:size(E,1))-4 + u) = params.C*(Umax_e == u);
%         end
%         % compute Kmu_0
%         if sum(mu_0) > 0
%             smu_1_te = sum(reshape(mu_0.*Ye,4,size(E,1)),1);
%             smu_1_te = reshape(smu_1_te(ones(4,1),:),length(mu),1);
%             kxx_mu_0{t} = ~Ye*params.C+mu_0-smu_1_te;
%         else
%             kxx_mu_0{t} = zeros(size(mu));
%         end
% 
%         
%         Kmu_0 = Kmu_x + kxx_mu_0{t} - Kxx_mu_x_list{t}(:,x);
% 
%         mu_d = mu_0 - mu;
%         Kmu_d = Kmu_0-Kmu_x;
%         
%         Kmu_d_list{t} = Kmu_d;
%         mu_d_list{t} = mu_d;
%         nomi(t) = mu_d'*gradient;
%         denomi(t) = norm_const_quadratic_list(t)*Kmu_d' * mu_d;
% 
%         
%     end
%     
%     
%     % decide whether to update or not
%     if sum(Gmax)>=sum(G0) %&& sum(Gmax>G0)>T_size/2
%         tau = min(sum(nomi)/sum(denomi),1);
%     else
%         tau=0;
%     end
%     
%     %% update for each tree
%     delta_obj_list = zeros(1,T_size);
%     parfor t=1:T_size
%         % variables located for tree t and example x
%         loss = loss_list{t}(:,x);
%         Ye = Ye_list{t}(:,x);
%         ind_edge_val = ind_edge_val_list{t};
%         mu = mu_list{t}(:,x);
%         E = E_list{t};
%         gradient =  gradient_list_local{t};
%         mu_d = mu_d_list{t};
%         Kmu_d = Kmu_d_list{t};
%         % 
%         delta_obj_list(t) = gradient'*mu_d*tau - norm_const_quadratic_list(t)*tau^2/2*mu_d'*Kmu_d;
%         mu = mu + tau*mu_d;
%         Kxx_mu_x_list{t}(:,x) = (1-tau)*Kxx_mu_x_list{t}(:,x) + tau*kxx_mu_0{t};
%         % update Smu Rmu
%         mu = reshape(mu,4,size(E,1));
%         for u = 1:4
%             Smu_list{t}{u}(:,x) = (sum(mu)').*ind_edge_val{u}(:,x);
%             Rmu_list{t}{u}(:,x) = mu(u,:)';
%         end
%         
%         mu = reshape(mu,4*size(E,1),1);
%         mu_list{t}(:,x) = mu;
%     end
%      
%     return;
% end



%% Compute Gmax
function [Gmax] = compute_Gmax(gradient,Ymax,E)
    m = size(Ymax,1);
    
    gradient = reshape(gradient,4,size(E,1)*m);
    Umax(1,:) = reshape(and(Ymax(:,E(:,1)) == -1,Ymax(:,E(:,2)) == -1)',1,size(E,1)*m);
    Umax(2,:) = reshape(and(Ymax(:,E(:,1)) == -1,Ymax(:,E(:,2)) == 1)',1,size(E,1)*m);
    Umax(3,:) = reshape(and(Ymax(:,E(:,1)) == 1,Ymax(:,E(:,2)) == -1)',1,size(E,1)*m);
    Umax(4,:) = reshape(and(Ymax(:,E(:,1)) == 1,Ymax(:,E(:,2)) == 1)',1,size(E,1)*m);
    % sum up the corresponding edge-gradients
    Gmax = reshape(sum(gradient.*Umax),size(E,1),m);
    Gmax = reshape(sum(Gmax,1),m,1);
    
    
    return;
end





%% train profile and test profile update
function profile_update
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
    global norm_const_quadratic_list;
    m = size(Ye,2);
    tm = cputime;
    print_message(sprintf('tm: %d  iter: %d obj: %f mu: max %f min %f dgap: %f',...
    round(tm-profile.start_time),profile.iter,obj,max(max(mu)),min(min(mu)),primal_ub-obj),5,sprintf('/var/tmp/%s.log',params.filestem));
    if params.profiling
        profile.next_profile_tm = profile.next_profile_tm + params.profile_tm_interval;
        profile.n_err_microlabel_prev = profile.n_err_microlabel;

        %% train
        
            [Ypred_tr,~] = compute_error(Y_tr,Kx_tr);
        
        profile.microlabel_errors = sum(abs(Ypred_tr-Y_tr) >0,2);
        profile.n_err_microlabel = sum(profile.microlabel_errors);
        profile.p_err_microlabel = profile.n_err_microlabel/numel(Y_tr);
        profile.n_err = sum(profile.microlabel_errors > 0);
        profile.p_err = profile.n_err/length(profile.microlabel_errors);

        %% test
        
            [Ypred_ts,~] = compute_error(Y_ts,Kx_ts);
        
        profile.microlabel_errors_ts = sum(abs(Ypred_ts-Y_ts) > 0,2);
        profile.n_err_microlabel_ts = sum(profile.microlabel_errors_ts);
        profile.p_err_microlabel_ts = profile.n_err_microlabel_ts/numel(Y_ts);
        profile.n_err_ts = sum(profile.microlabel_errors_ts > 0);
        profile.p_err_ts = profile.n_err_ts/length(profile.microlabel_errors_ts);

        print_message(...
            sprintf('tm: %d 1_er_tr: %d (%3.2f) er_tr: %d (%3.2f) 1_er_ts: %d (%3.2f) er_ts: %d (%3.2f)',...
            round(tm-profile.start_time),...
            profile.n_err,...
            profile.p_err*100,...
            profile.n_err_microlabel,...
            profile.p_err_microlabel*100,...
            round(profile.p_err_ts*size(Y_ts,1)),...
            profile.p_err_ts*100,sum(profile.microlabel_errors_ts),...
            sum(profile.microlabel_errors_ts)/numel(Y_ts)*100),...
            0,sprintf('/var/tmp/%s.log',params.filestem));

        running_time = tm-profile.start_time;
        sfile = sprintf('/var/tmp/Ypred_%s.mat',params.filestem);
        save(sfile,'Ypred_tr','Ypred_ts','params','running_time','norm_const_quadratic_list');
        Ye = reshape(Ye,4*size(E,1),m);
    end
end

%% Profiling during training phase
function profile_update_tr

    global params;
    global profile;
    global Y_tr;
    global Kx_tr;
    global obj;
    global primal_ub;
    global kappa;
    global opt_round;
    global kappa_decrease_flags;
    global val_list;
    global kappa_list;
    global Yipos_list;
    global GmaxG0_list;
    global GoodUpdate_list;
    global T_size;
    global duality_gap_on_trees;
    global obj_list;

    tm = cputime;
    
    if params.profiling
        profile.next_profile_tm = profile.next_profile_tm + params.profile_tm_interval;
        profile.n_err_microlabel_prev = profile.n_err_microlabel;
        % compute training error
  
        % Computer error
        [Ypred_tr,~] = compute_error(Y_tr,Kx_tr);  
        profile.microlabel_errors = sum(abs(Ypred_tr-Y_tr) >0,2);
        profile.n_err_microlabel = sum(profile.microlabel_errors);
        profile.p_err_microlabel = profile.n_err_microlabel/numel(Y_tr);
        profile.n_err = sum(profile.microlabel_errors > 0);
        profile.p_err = profile.n_err/length(profile.microlabel_errors);
        
        % Print out messages
        print_message(...
            sprintf('t: %d iter: %d 1_er_tr: %d (%3.2f) er_tr: %d (%3.2f) K: %d Y*pos: %3.2f%% %.2f %.2f %d Yipos: %3.2f%% %.2f %.2f %.2f K: %.2f %.2f %d Mg: %.2f%% %.3f %.3f obj: %.2f gap: %.2f%% Update %.2f%% %.2f%% gap: %.2f%%',...
            round(tm-profile.start_time),...
            opt_round,...
            profile.n_err,...
            profile.p_err*100,...
            profile.n_err_microlabel,...
            profile.p_err_microlabel*100,...
            kappa,...
            sum(kappa_decrease_flags<=kappa)/size(Y_tr,1)*100,...
            mean(kappa_decrease_flags(kappa_decrease_flags>0)),...
            std(kappa_decrease_flags(kappa_decrease_flags>0)),...
            max(kappa_decrease_flags(kappa_decrease_flags>0)),...
            sum(Yipos_list<=kappa)/size(Y_tr,1)*100,...
            mean(Yipos_list(Yipos_list>0)),...
            std(Yipos_list(Yipos_list>0)),...
            min(Yipos_list(Yipos_list>0)),...
            mean(kappa_list),...
            std(kappa_list),...
            max(kappa_list),...
            sum((val_list>0))/size(Y_tr,1)*100,...
            mean(val_list)*100,...
            std(val_list)*100,...
            obj,...
            mean(duality_gap_on_trees./(obj_list+duality_gap_on_trees))*100,... %(primal_ub-obj)/primal_ub*100,...
            mean(GmaxG0_list)/T_size*100,...
            sum(GoodUpdate_list)/size(Y_tr,1)*100,...
            (primal_ub - obj)/(primal_ub)*100),...
            0,sprintf('/var/tmp/%s.log',params.filestem));
    end
    
end

%% test error
function [Ypred,YpredVal] = compute_error(Y,Kx) 
    %% global variable
    global T_size;
    global E_list;
    global Ye_list;
    global mu_list;
    global kappa;
    global iter;
    Ypred = zeros(size(Y));
    YpredVal = zeros(size(Y,1),1);
    Y_kappa = zeros(size(Y,1)*T_size,size(Y,2)*kappa);
    Y_kappa_val = zeros(size(Y,1)*T_size,kappa);
    w_phi_e_local_list = cell(1,T_size);
    %% compute 'k' best from each random spanning tree
    for t=1:T_size
        E = E_list{t};
        Ye = Ye_list{t};
        mu = mu_list{t};
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

        
        Y_kappa(((t-1)*size(Y,1)+1):(t*size(Y,1)),:) = Y_tmp;
        Y_kappa_val(((t-1)*size(Y,1)+1):(t*size(Y,1)),:) = Y_tmp_val;
    end
    
    %% compute top '1' for all tree
    MATLABPAR = 0;
    if MATLABPAR == 1
        input_labels = cell(1,size(Y,1));
        input_scores = cell(1,size(Y,1));
        for i=1:size(Y,1)
            input_labels{i} = Y_kappa((i:size(Y_kappa,1)/T_size:size(Y_kappa,1)),:);
            input_scores{i} = Y_kappa_val((i:size(Y_kappa,1)/T_size:size(Y_kappa_val,1)),:);
        end
        
        if matlabpool('size') == 0
            matlabpool open;
        end
        num_partition = matlabpool('size');
        ii=1;
        start_position{ii}=1;
        while start_position{ii} + ceil(size(Y,1)/num_partition) < size(Y,1)
            ii=ii+1;
            start_position{ii}=start_position{ii-1}+ceil(size(Y,1)/num_partition);
        end
        start_position{ii+1} = size(Y,1)+1;
        par_Ypred = cell(size(start_position,2)-1);
        par_YpredVal = cell(size(start_position,2)-1);
        parfor partition_i = 1:(size(start_position,2)-1)
            training_i_range = [start_position{partition_i}:(start_position{partition_i+1}-1)];
            i_Ypred = zeros(size(training_i_range,1),size(Y,2));
            i_YpredVal = zeros(size(training_i_range,1),1);
            for training_i=training_i_range
                if iter==0
                    kappa_decrease_flag = 1;
                    i_Ypred(training_i-training_i_range(1)+1,:) = -1*ones(1,size(Y,2));
                else
                    [i_Ypred(training_i-training_i_range(1)+1,:),...
                        i_YpredVal(training_i-training_i_range(1)+1,:),...
                        kappa_decrease_flag] = ...
                        find_worst_violator(input_labels{training_i},input_scores{training_i},[]);
                end
            end
            par_Ypred{partition_i} = i_Ypred;
            par_YpredVal{partition_i} = i_YpredVal;
        end
        for partition_i = 1:(size(start_position,2)-1)
            training_i_range = [start_position{partition_i}:(start_position{partition_i+1}-1)];
            Ypred(training_i_range,:) = par_Ypred{partition_i};
            YpredVal(training_i_range,:) = par_YpredVal{partition_i};
        end
    else
        for i=1:size(Y,1)
            if iter==0
                kappa_decrease_flag = 1;
                Ypred(i,:) = -1*ones(1,size(Y,2));
            else
                
%                 [Ypred(i,:),YpredVal(i,:),kappa_decrease_flag] = ...
%                     find_worst_violator(...
%                     Y_kappa((i:size(Y_kappa,1)/T_size:size(Y_kappa,1)),:),...
%                     Y_kappa_val((i:size(Y_kappa,1)/T_size:size(Y_kappa_val,1)),:),[]);
                
                
                IN_E = zeros(size(E_list{1},1)*2,size(E_list,1));
                for t=1:T_size
                    IN_E(:,t) = reshape(E_list{t},size(E_list{1},1)*2,1);
                end
                IN_gradient = zeros(4*size(E_list{1},1),size(E_list,1));
                for t=1:T_size
                    IN_gradient(:,t) = w_phi_e_local_list{t}(((i-1)*4*size(E_list{1},1)+1):i*4*size(E_list{1},1),:);
                end
                Y_kappa((i:size(Y_kappa,1)/T_size:size(Y_kappa,1)),:) = (Y_kappa((i:size(Y_kappa,1)/T_size:size(Y_kappa,1)),:)+1)/2;
                
                
          

                [Ypred(i,:),YpredVal(i,:),~,~] = ...
                    find_worst_violator_new(...
                    Y_kappa((i:size(Y_kappa,1)/T_size:size(Y_kappa,1)),:),...
                    Y_kappa_val((i:size(Y_kappa,1)/T_size:size(Y_kappa_val,1)),:),...
                    [],IN_E,IN_gradient);
                Ypred(i,:) = Ypred(i,:)*2-1;
                
            end
        end
        
%         parfor i=1:size(Y,1)
%             pause(0.000);
%             [Ypred(i,:),YpredVal(i,:),kappa_decrease_flag] = find_worst_violator(input_labels{i},input_scores{i},[]);
%             if ~kappa_decrease_flag
%                 [Ypred(i,:),YpredVal(i,:)] = majority_voting(...
%                 input_labels{i},...
%                 input_scores{i},...
%                 size(Y,2));
%             end
%         end        
    end

    
    return;
end
% function [Ypred,YpredVal] = par_compute_error(Y,Kx) 
%     %% global variable
%     global T_size;
%     global E_list;
%     global Ye_list;
%     global mu_list;
%     global kappa;
%     Ypred = zeros(size(Y));
%     YpredVal = zeros(size(Y,1),1);
%     Y_kappa = zeros(size(Y,1)*T_size,size(Y,2)*kappa);
%     Y_kappa_val = zeros(size(Y,1)*T_size,kappa); 
%     %% compute 'k' best from each random spanning tree
%     Y_tmp = cell(1,T_size);
%     Y_tmp_val = cell(1,T_size);
%     parfor t=1:T_size
%         pause(0.000);
%         E = E_list{t};
%         Ye = Ye_list{t};
%         mu = mu_list{t};
%         w_phi_e = compute_w_phi_e(Kx,E,Ye,mu);
%         [Y_tmp{t},Y_tmp_val{t}] = compute_topk(w_phi_e,kappa,E);
%     end
%     for t=1:T_size
%         Y_kappa(((t-1)*size(Y,1)+1):(t*size(Y,1)),:) = Y_tmp{t};
%         Y_kappa_val(((t-1)*size(Y,1)+1):(t*size(Y,1)),:) = Y_tmp_val{t};
%     end
%     clear Y_tmp;
%     clear Y_tmp_val;
%     %% compute top '1' for all tree
%     input_labels = cell(1,size(Y,1));
%     input_scores = cell(1,size(Y,1));
%     for i=1:size(Y,1)
%         input_labels{i} = Y_kappa((i:size(Y_kappa,1)/T_size:size(Y_kappa,1)),:);
%         input_scores{i} = Y_kappa_val((i:size(Y_kappa,1)/T_size:size(Y_kappa_val,1)),:);
%     end
%    
%     
% 
%     parfor i=1:size(Y,1)
%         pause(0.000);
%         [Ypred(i,:),YpredVal(i,:),kappa_decrease_flag] = find_worst_violator(input_labels{i},input_scores{i},[]);
%         if ~kappa_decrease_flag
%             [Ypred(i,:),YpredVal(i,:)] = majority_voting(...
%             input_labels{i},...
%             input_scores{i},...
%             size(Y,2));
%         end
%     end
%     
%     return;
% end

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
    profile.next_profile_tm     = profile.start_time;
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
