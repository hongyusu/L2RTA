

%% forward algorithm
%   Input:
%   Output: 
%       P_node: score matrix
%       T_node: pointer matrix
function [P_node,T_node] = forward_alg_matlab(training_gradient,K,E,l,node_degree,max_node_degree)
    nlabel=l;
    P_node = zeros(K*nlabel,2*(max_node_degree+1)); % score matrix
    T_node = zeros(K*nlabel,2*(max_node_degree+1)); % tracker matrix
    for i=size(E,1):-1:1
        if i==0
            break
        end

        p = E(i,1);
        c = E(i,2);
        %[i,p,c]
        % row block index for current edge (child, parent)
        row_block_chi_ind = ((c-1)*K+1):c*K;
        row_block_par_ind = ((p-1)*K+1):p*K;


        % update node score, calculate node top K list score P(v) + sum_{v'\in chi(v)}M_{v'->v}(v)
        col_block_ind = 3:2:size(P_node,2);
        [P_node(row_block_chi_ind,1),T_node(row_block_chi_ind,col_block_ind)] = LinearMaxSum(P_node(row_block_chi_ind,col_block_ind));
        [P_node(row_block_chi_ind,2),T_node(row_block_chi_ind,col_block_ind+1)] = LinearMaxSum(P_node(row_block_chi_ind,col_block_ind+1));

        % combine edge potential and send message to parent
        % S
        S_e = reshape(training_gradient(:,i),2,2);
        S_e = [repmat(S_e(1,:),K,1);repmat(S_e(2,:),K,1)];
        % M=S+P
        M = repmat(reshape(P_node(row_block_chi_ind,1:2),2*K,1),1,2);
        M = (M + S_e) .* (M & M);
        T = M;
        [u,v] = sort(M(:,1),'descend');
        M(:,1) = u;T(:,1) = v .* (u & u);
        [u,v] = sort(M(:,2),'descend');
        M(:,2) = u;T(:,2) = v .* (u & u);
        M = M(1:K,:);
        T = T(1:K,:);
        % put into correspond blocks
        j = sum(E(i:size(E,1),1) == p)+1;
        P_node(row_block_par_ind,(j-1)*2+1:j*2) = M;
        T_node(row_block_chi_ind,1:2) = T;
    end

    % one more iteration on root node
    row_block_chi_ind = ((p-1)*K+1):p*K;
    %disp([reshape(repmat(1:nlabel,K,1),nlabel*K,1),repmat([1:K]',nlabel,1),P_node])
    %disp([reshape(repmat(1:nlabel,K,1),nlabel*K,1),repmat([1:K]',nlabel,1),T_node])
    % update node score, calculate node top K list score P(v) + sum_{v'\in chi(v)}M_{v'->v}(v)
    col_block_ind = 3:2:size(P_node,2);
    [P_node(row_block_chi_ind,1),T_node(row_block_chi_ind,col_block_ind)] = LinearMaxSum(P_node(row_block_chi_ind,col_block_ind));
    [P_node(row_block_chi_ind,2),T_node(row_block_chi_ind,col_block_ind+1)] = LinearMaxSum(P_node(row_block_chi_ind,col_block_ind+1));
    T_node(row_block_chi_ind,1:2) = [1:K;(1:K)+K]';
end

%% linear time max sum
% calculate P(v) + \sum_{u\in chi(v)}M_{u->v}(v)
% time complexity is not O(K) now
% TODO better sorting algorithm with O(K)
function [res_sc,res_pt] = LinearMaxSum(data)
    %% parameters
    K = size(data,1); % top K
    max_nb_num = size(data,2); % maximum number of neighbor neighbors
    res_sc = zeros(K,1); % results that contain score
    res_pt = zeros(K,max_nb_num); % results that contain pointer value
    
    %% start from the first column
    cur_col = [(1:size(data,1))',data(:,1)]; % current column [index, score]
    cur_col = cur_col(cur_col(:,size(cur_col,2))~=0,:); % remove zero rows
    % if there is no score from tree below then return (meaning leave nodes)
    if numel(cur_col) == 0
        res_sc = zeros(size(data,1),1);
        res_pt = zeros(size(data,1),size(data,2));
        res_sc(1) = res_sc(1) + 1; 
        return
    end
    
    %% combine children score one at a time
    for i=2:max_nb_num
        next_col = [(1:size(data,1))',data(:,i)];
        next_col = next_col(next_col(:,size(next_col,2))~=0,:);
        % return if there is no leaves
        if numel(next_col)==0
            break
        end
        max_cur_row = min(size(cur_col,1),K);
        max_next_row = min(size(next_col,1),K);
        %res = [];
        res = zeros(max_cur_row*max_next_row,size(cur_col,2)+1);
        for p = 1:max_cur_row
            for q=1:max_next_row
                %res = [res;[cur_col(p,1:(size(cur_col,2)-1)),next_col(q,1),cur_col(p,size(cur_col,2))+next_col(q,2)]];
                res((p-1)*max_next_row+q,:) = [cur_col(p,1:(size(cur_col,2)-1)),next_col(q,1),cur_col(p,size(cur_col,2))+next_col(q,2)];
            end
        end
        res = sortrows(res,[-size(res,2)]);
        cur_col = res(1:min(K,size(res,1)),:);
    end
    
    %% collect results
    res_sc(1:size(cur_col,1),1) = cur_col(1:size(cur_col,1),size(cur_col,2)) + 1;
    res_pt(1:size(cur_col,1),1:(size(cur_col,2)-1)) = cur_col(1:size(cur_col,1),1:(size(cur_col,2)-1));
    return
end