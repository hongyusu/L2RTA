
%% compute topk labels
function [Ymax,YmaxVal,Gmax] = compute_topk(gradient,K,E)
    %tic
    if nargin < 3
        disp('Wrong input parameters!');
        return
    end
    
    %% 
    m = numel(gradient)/(4*size(E,1));
    nlabel = max(max(E));
    gradient = reshape(gradient,4,size(E,1)*m);
    min_gradient_val = min(min(gradient));
    gradient = gradient - min_gradient_val + 1e-5;
    node_degree = zeros(1,nlabel);
    for i=1:nlabel
        node_degree(i) = sum(sum(E==i));
    end
    Ymax = zeros(m,nlabel*K);
    YmaxVal = zeros(m,K);
    if numel(gradient(gradient~=1)) == 0
            Ymax = Ymax +1;
            return
    end
    training_gradients = cell(m);
    for training_i = 1:m
        training_gradients{training_i} = gradient(1:4,((training_i-1)*size(E,1)+1):(training_i*size(E,1)));
    end
    
    %% iteration throught examples
    MATLABPAR=0;
    if m>1 && MATLABPAR==1
        if matlabpool('size') == 0 % checking to see if my pool is already open
            matlabpool open;
        end
        num_partition = matlabpool('size');
        ii=1;
        start_position{ii}=1;
        while start_position{ii} + ceil(m/num_partition) <m
            ii=ii+1;
            start_position{ii}=start_position{ii-1}+ceil(m/num_partition);
        end
        start_position{ii+1} = m+1;
        par_Ymax = cell(size(start_position,2)-1);
        par_YmaxVal = cell(size(start_position,2)-1);
        parfor partition_i = 1:(size(start_position,2)-1)
            training_i_range = [start_position{partition_i}:(start_position{partition_i+1}-1)];
            i_Ymax = zeros(size(training_i_range,1),nlabel*K);
            i_YmaxVal = zeros(size(training_i_range,1),K);
            for training_i = training_i_range
                %[partition_i,training_i]
                training_gradient = training_gradients{training_i};
                [P_node,T_node] = forward_alg(training_gradient,K,E,nlabel,node_degree,max(max(node_degree)));
                [Ymax_single, YmaxVal_single] = backward_alg(P_node, T_node, K, E, nlabel, node_degree);
                i_Ymax(training_i-training_i_range(1)+1,:) = Ymax_single;
                i_YmaxVal(training_i-training_i_range(1)+1,:) = YmaxVal_single;
            end
            par_Ymax{partition_i} = i_Ymax;
            par_YmaxVal{partition_i} = i_YmaxVal;
        end
        for partition_i = 1:(size(start_position,2)-1)
            training_i_range = [start_position{partition_i}:(start_position{partition_i+1}-1)];
            Ymax(training_i_range,:) = par_Ymax{partition_i};
            YmaxVal(training_i_range,:) = par_YmaxVal{partition_i};
        end
        %matlabpool close;
        YmaxVal = YmaxVal + min_gradient_val*size(E,1);
    else
        for training_i = 1:m
            %% get training gradient
            training_gradient = training_gradients{training_i};
            %% forward algorithm to get P_node and T_node
            [P_node,T_node] = forward_alg(training_gradient,K,E,nlabel,node_degree,max(max(node_degree)));
            [Ymax_single, YmaxVal_single] = backward_alg(P_node, T_node, K, E, nlabel, node_degree);
            Ymax(training_i,:) = Ymax_single;
            YmaxVal(training_i,:) = YmaxVal_single;       
        end

        YmaxVal = YmaxVal + min_gradient_val*size(E,1);
    end    
    
 
        
    %%
    if nargout > 2
        % find out the max gradient for each example: pick out the edge labelings
        % consistent with Ymax
        Ymax_1 = Ymax(:,1:nlabel);
        Umax(1,:) = reshape(and(Ymax_1(:,E(:,1)) == -1,Ymax_1(:,E(:,2)) == -1)',1,size(E,1)*m);
        Umax(2,:) = reshape(and(Ymax_1(:,E(:,1)) == -1,Ymax_1(:,E(:,2)) == 1)',1,size(E,1)*m);
        Umax(3,:) = reshape(and(Ymax_1(:,E(:,1)) == 1,Ymax_1(:,E(:,2)) == -1)',1,size(E,1)*m);
        Umax(4,:) = reshape(and(Ymax_1(:,E(:,1)) == 1,Ymax_1(:,E(:,2)) == 1)',1,size(E,1)*m);
        % sum up the corresponding edge-gradients
        Gmax = reshape(sum(gradient.*Umax),size(E,1),m);
        Gmax = reshape(sum(Gmax,1),m,1);
    end
    
    %toc
    return
end