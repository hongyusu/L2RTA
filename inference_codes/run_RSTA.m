%%
%
% CURRENT PROBLEM:
%   the current code seems to work with current setting of feature, kernel, etc. need to imporve
%
%
% COMPILE WITH:
%   mex compute_topk_omp.c forward_alg_omp.c backward_alg_omp.c  CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" CC="/usr/local/bin/gcc -std=c99"
%   mex find_worst_violator_new.c
%
%
% PARAMETERS:
%   filename:       the prefix of the input file
%   graph_type:     spanning tree or random pairing graph
%   t:              the number of trees
%   isTest:         if 'TRUE' the algorithm will select a small portion of the training data for code sanity check, results are not saved.
%   kth_fold:       the kth fold of five fold Cross-Validation
%   l_norm:         '1'->l1 norm regularization, '2'->l2 norm regularization. Currently, only L2 regularization implemented
%   maxkappa:       the length of K best list
%   slack_c:        slack parameter C
%   loss_scaling_factor:    loss scaling parameter to scale the element of the loss function
%   newton_method:          boolean variable indicating whether to use newton methods to combine multiple update direction
%
%
% SUGGESTIONS:
%   loss_scaling_factor:
%           test loss_scaling_factor=1 which will use the original loss function
%           test loss_sclaing_factor=|E| which will downscale the original loss function by the number of edges in the graph/tree
%   slack_c:
%           samilar strategy used in SVM C parameter will apply here also
%
%
% EXAMPLE USAGE:
%   Use command:    run_RSTA('ArD10','tree','5','1','1','2','2','100','10','1')
%   This will run the algorithm:
%       on ArD10 dataset,
%       with random spanning tree as the output graph,
%       generating 5 random spanning tree,
%       in test mode which selects a small portion of the training data for sanity check,
%       running first fold of five fold Cross-Validation,
%       with l2 norm regularization on featuer weight parameters,
%       with K best multilabel list of depth 2,
%       with margin slack parameter C equals to 100.
%       with loss scaling parameter equals to 10, in which loss will be down scaled by a factor of 10
%       with newton method to combine multiple update directions
%
%
function run_RSTA (filename, graph_type, t, isTest, kth_fold, l_norm, maxkappa, slack_c, loss_scaling_factor, newton_method)

    % Process input parameters, input at least the name of the dataset, requires all 9 additional parameters, otherwise run on a default parameter setting.
    if nargin < 1
        disp('Not enough input parameters! Dataset should be given as minimum.')
        return
    end
    
    if nargin < 10
        disp('Not enough input parameters! Run in default parameter setting on the data.')
        graph_type  = 'tree';
        t           = '1';
        isTest      = '0';
        kth_fold    = '1';
        l_norm      = '2';
        maxkappa    = '2';
        slack_c     = '100';
        loss_scaling_factor = '10';
        netwon_method       = '1';
    end
    
    % Set the seed of the random number generator
    rand('twister', 0);
    
    %losstype = 'r';     % 1 loss, loss is evaluated over the whole output vector
    losstype = 's';     % scaled loss, losses are distributed on the edge of the output graph
    
    % Set the suffix of result files
    suffix = sprintf('%s_%s_%s_f%s_l%s_k%s_c%s_s%s_n%s_RSTA%s', filename,graph_type,t,kth_fold,l_norm,maxkappa,slack_c,loss_scaling_factor,newton_method,losstype);
    system(sprintf('rm /var/tmp/%s.log', suffix));
    system(sprintf('rm /var/tmp/Ypred_%s.mat', suffix));
    
    % Convert parameters from string to numerical
    t           = eval(t);
    isTest      = eval(isTest);
    kth_fold    = eval(kth_fold);
    l_norm      = eval(l_norm);
    maxkappa    = eval(maxkappa);
    slack_c     = eval(slack_c);
    loss_scaling_factor = eval(loss_scaling_factor);
    newton_method       = eval(newton_method);
    
    % Add search path
    addpath('../shared_scripts/');  
    
    % Get current hostname to run on computer cluster
    [~,comres] = system('hostname');
    
    % Read in X and Y matrix of data from location based on the current computing node
    if strcmp(comres(1:4),'melk') || strcmp(comres(1:4),'ukko') || strcmp(comres(1:4),'node')
        X = dlmread(sprintf('/cs/taatto/group/urenzyme/workspace/data/%s_features',filename));
        Y = dlmread(sprintf('/cs/taatto/group/urenzyme/workspace/data/%s_targets',filename));
    else
        X = dlmread(sprintf('../shared_scripts/test_data/%s_features',filename));
        Y = dlmread(sprintf('../shared_scripts/test_data/%s_targets',filename));
    end
    
    %% Process input data X and Y matrix
    % Select the examples with non-zero features. This makes sense for e.g., document classification problem.
    Xsum = sum(X,2);
    X = X(Xsum~=0,:);
    Y = Y(Xsum~=0,:);
    % Select the labels which is not constant over the selected examples. In other words, remove easy tasks which is all -1 or all +1.
    % The average performance will drop after the label selection due to the removal of the easy-to-classify labels.
    Yuniq=zeros(1,size(Y,2));
    for i=1:size(Y,2)
        if size(unique(Y(:,i)),1)>1
            Yuniq(i)=i;
        end
    end
    Y = Y(:,Yuniq(Yuniq~=0));
    
    %% Feature normalization (tf-idf for text data, scale and centralization for other numerical features).
    if or(strcmp(filename,'medical'),strcmp(filename,'enron')) 
        X = tfidf(X);
    elseif ~(strcmp(filename(1:2),'to'))
        X = (X-repmat(min(X),size(X,1),1))./repmat(max(X)-min(X),size(X,1),1);
    end
    
    %% Change Y from -1 to 0, now we have standard label of +1 and 0. Note we use +1/-1 in the algorithm
    Y(Y==-1)=0;
    %X=(X-repmat(mean(X),size(X,1),1));

    %% Get dot product kernels from normalized features or just read in precomputed kernel matrix.
    if or(strcmp(filename,'fpuni'),strcmp(filename,'cancer'))
        if strcmp(comres(1:4),'dave') | strcmp(comres(1:4),'ukko') | strcmp(comres(1:4),'node')
            K = dlmread(sprintf('/cs/taatto/group/urenzyme/workspace/data/%s_kernel',filename));
        else
            K = dlmread(sprintf('../shared_scripts/test_data/%s_kernel',filename));
        end
    else
        K = X * X';         % Dot product
        K = K ./ sqrt(diag(K)*diag(K)');    % Normalization of the kernel matrix to make sure points are in a unit sphere.
        % TODO: Center kernel
    end
    
    %% Stratified n fold cross validation index.
    nfold   = 5;
    Ind     = getCVIndex(Y,nfold);
    
    %% Select part of the data for code sanity check if 'isTest==1'.
    ntrain = 10000;
    ntrain = min(ntrain,size(Y,1));
    if isTest == 1
        X   = X(1:ntrain,:);
        Y   = Y(1:ntrain,:);
        K   = K(1:ntrain,1:ntrain);
        Ind = Ind(1:ntrain);
    end

%     %% Perform parameter selection.
%     % TODO: to be better implemented
%     % ues results from parameter selection, otherwise use fixed parameters
%     para_n=11;
%     parameters=zeros(para_n,10);
%     for i=1:para_n
%         try
%             load(sprintf('../parameters/%s_%s_1_f%d_l2_i%d_RSTAp.mat',filename,graph_type,kth_fold,i));
%             parameters(i,:) = perf;
%         catch err
%             parameters(i,:) = [i,10,zeros(1,8)];
%         end
%     end
%     parameters=sortrows(parameters,[3,2]);
%     mmcrf_c = parameters(para_n,2);
    
    % currently use following parameters
    mmcrf_c         = slack_c;      % margin slack parameter
    mmcrf_g         = -1e6;         % relative duality gap
    mmcrf_i         = 120;           % number of iteration
    mmcrf_maxkappa  = maxkappa;     % length of the K-best list
    
    % Print out all parameters
    fprintf('\n\tC:%d G:%.2f Iteration:%d MaxKappa:%d T:%d Loss_scaling:%d \n', mmcrf_c,mmcrf_g,mmcrf_i,mmcrf_maxkappa, t, loss_scaling_factor);
    
    %% Generate random graphs.
    rand('twister', 0); % Fix random seed to make sure random spanning trees generated from each run are of the same.
    Nrep    = t;             % number of random graph
    Nnode   = size(Y,2);
    Elist   = cell(Nrep,1);
    for i = 1:Nrep
        if strcmp(graph_type,'tree')
            E = randTreeGenerator(Nnode); % generate
        end
        if strcmp(graph_type,'pair')
            E = randPairGenerator(Nnode); % generate
        end
        E = [E,min(E,[],2),max(E,[],2)];E=E(:,3:4); % arrange head and tail
        E = sortrows(E,[1,2]);  % sort by head and tail
        Elist{i} = RootTree(E); % put into cell array
      
% Construct a collection of similar spanning trees         
%         if i~=1
%             Elist{i}=Elist{1};
%             pos = randsample(2:(size(Elist{1},1)),1);
%             Elist{i}(pos,1) = randsample(unique(Elist{1}(1:(pos-1),:)),1);
%             Elist{i}=RootTree(Elist{i});
%         end
    end
    
    
    %% Variable to keep results.
    Ypred       = zeros(size(Y));
    YpredVal    = zeros(size(Y));
    running_times = zeros(nfold,1);
    muList      = cell(nfold,1);

    
    %% Perform the experiment on the k'th fold of the 5 fold cross-validation
    % TODO: some of the variables are no longer necessary, to be removed
    for k=kth_fold
        paramsIn.profileiter    = 40;           % Profile the training every fix number of iterations
        paramsIn.losstype       = losstype;     % losstype
        paramsIn.mlloss         = 0;            % assign loss to microlabels(0) edges(1)
        paramsIn.profiling      = 1;            % profile (test during learning)
        paramsIn.epsilon        = mmcrf_g;      % stopping criterion: minimum relative duality gap
        paramsIn.C              = mmcrf_c;      % margin slack
        paramsIn.maxkappa       = mmcrf_maxkappa;
        paramsIn.max_CGD_iter   = 1;            % maximum number of conditional gradient iterations per example
        paramsIn.max_LBP_iter   = 3;            % number of Loopy belief propagation iterations
        paramsIn.tolerance      = 1E-10;        % numbers smaller than this are treated as zero
        paramsIn.maxiter        = mmcrf_i;      % maximum number of iterations in the outer loop
        paramsIn.verbosity      = 1;
        paramsIn.debugging      = 3;
        paramsIn.l_norm         = l_norm;
        if isTest
            paramsIn.extra_iter = 0;            % extra iteration through examples when optimization is over
        else
            paramsIn.extra_iter = 0;            % extra iteration through examples when optimization is over
        end
        paramsIn.filestem       = sprintf('%s',suffix);	% file name stem used for writing output
        paramsIn.loss_scaling_factor = loss_scaling_factor;
        paramsIn.newton_method  = newton_method;
        
        % nfold cross validation
        Itrain  = find(Ind ~= k);
        Itest   = find(Ind == k);
        %Itrain = [Itrain;Itest(1:ceil(numel(Itest)/5))];
        gKx_tr  = K(Itrain, Itrain);    % Training kernel
        gKx_ts  = K(Itest,  Itrain)';   % Test kernel
        gY_tr   = Y(Itrain,:);  gY_tr(gY_tr==0) = -1;   % Label of the training examples
        gY_ts   = Y(Itest,:);   gY_ts(gY_ts==0) = -1;   % Label of the test examples
        % set input data
        dataIn.Elist =  Elist;      % edge
        dataIn.Kx_tr =  gKx_tr;     % training kernel
        dataIn.Kx_ts =  gKx_ts;     % test kernel
        dataIn.Y_tr  =   gY_tr;      % training label
        dataIn.Y_ts  =   gY_ts;      % test label
        % running
        [rtn,~] = RSTA (paramsIn, dataIn);
        % save margin dual mu
        muList{k} = rtn;
        % collecting results
        load(sprintf('/var/tmp/Ypred_%s.mat', paramsIn.filestem));
        Ypred(Itest,:) = Ypred_ts;          % The prediction in binary value
        %YpredVal(Itest,:) = Ypred_ts_val;  % The prediction in real value
        running_times(k,1) = running_time;  % Running time of the algorithm on the Kth fold
    end

    
    %% Compute performance metrics, mainly for sanity check/display purpose.
    [acc,vecacc,pre,rec,f1,auc1,auc2] = get_performance(Y(Itest,:),(Ypred(Itest,:)==1),YpredVal(Itest));
    % print out the performance on screen
    perf = [acc,vecacc,pre,rec,f1,auc1,auc2,norm_const_quadratic_list]
    
    
    %% If current session is not a test run (a true run), save the results files, save the log files, and terminate the MATLAB session.
    if ~isTest
        %% need to save: Ypred, YpredVal, running_time, mu for current baselearner t,filename
        save(sprintf('../outputs/%s.mat', paramsIn.filestem), 'perf','Ypred', 'YpredVal', 'running_times', 'muList','norm_const_quadratic_list');
        system(sprintf('mv /var/tmp/%s.log ../outputs/', suffix));    
        exit
    end
end


%% 
% Construct a rooted tree, always start from node 1
function [E] = RootTree(E)
    
    clist       = [1];
    nclist      = [];
    workingE    = [E,ones(size(E,1),1)];
    newE        = [];
    while size(clist)~=0
        for j = clist
            for i = 1:size(E,1)
                if workingE(i,3) == 0
                    continue
                end
                if workingE(i,1) == j
                    nclist  = [nclist,workingE(i,2)];
                    newE    = [newE;[j,E(i,2)]];
                    workingE(i,3) = 0;
                end
                if workingE(i,2) == j
                    nclist  = [nclist,workingE(i,1)];
                    newE    = [newE;[j,E(i,1)]];
                    workingE(i,3) = 0;
                end
            end            
        end
        clist   = nclist;
        nclist  = [];
    end
    E = newE;
end



