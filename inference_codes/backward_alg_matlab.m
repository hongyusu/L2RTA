
function [Ymax_single, YmaxVal_single] = backward_alg_matlab(P_node, T_node, K, E, nlabel, node_degree)
    Ymax_single = zeros(1,K*nlabel);
    YmaxVal_single = zeros(1,K);
    % trace back
    node_degree(E(1,1)) = node_degree(E(1,1))+1;
    % root node
    for k=1:K
        if k==0
            break
        end
        % k'th best multilabel
        Y=zeros(nlabel,1);

        Q_node = P_node;
        zero_mask = zeros(K,2);

        % pick up the k'th best value from root
        p = E(1,1);
        row_block_par_ind = ((p-1)*K+1):p*K;
        [~,v] = sort(reshape(P_node(row_block_par_ind,1:2),2*K,1),'descend');
        col_pos = ceil((v(k)-1e-5)/K);
        row_pos = ceil(mod(v(k)-1e-5,K));
        YmaxVal_single(1,k) = P_node(row_pos,col_pos);
        zero_mask(row_pos,col_pos) = 1;
        Q_node(row_block_par_ind,1:2) = Q_node(row_block_par_ind,1:2) .* zero_mask;
        zero_mask(row_pos,col_pos) = 0;

        % now everthing is standardized, then we do loop
        p=0;

        for i=1:size(E,1)
            if p == E(i,1)
                continue
            end
            p = E(i,1);
            c = E(i,2);
            % get block of the score matrix
            row_block_par_ind = ((p-1)*K+1):p*K;
            % get current optimal score position
            Q_node(row_block_par_ind,1:2);
            index = find(Q_node(row_block_par_ind,1:2)~=0);
            col_pos = ceil((index-1e-5)/K);
            row_pos = ceil(mod(index-1e-5,K));
            % emit a label
            T_par_block = T_node(row_block_par_ind,1:2);
            %Y(p) = ceil((T_par_block(row_pos, col_pos)-1e-5)/K);
            Y(p) = col_pos;
            % number of children
            n_chi = node_degree(p)-1;
            %disp(sprintf('on %d, %d->%d, n_chi %d, index %d -->(%d,%d) emit %d',p,p,c,n_chi,index,row_pos,col_pos, Y(p)))
            % children in order
            cs = zeros(n_chi,1);
            j=0;
            while E(i+j,1)==p
                %[i,j,E(i+j,1),i+j,size(E,1),j+1]
                cs(j+1) = E(i+j,2);
                j=j+1;
                if i+j>size(E,1)
                    break;
                end
            end     
            % loop through children
            for j=size(cs,1):-1:1
                c = cs(j);
                c_pos = (n_chi-j+2);
                %disp(sprintf('  %d->%d, child-col %d',p,c,c_pos))
                col_block_c_ind = ((c_pos-1)*2+1):c_pos*2;
                block = T_node(row_block_par_ind,col_block_c_ind);
                index = block(row_pos,col_pos) + (col_pos-1)*K;
                c_col_pos = ceil((index-1e-5)/K);
                c_row_pos = ceil(mod(index-1e-5,K));
                row_block_c_ind = ((c-1)*K+1):c*K;
                T_chi_block = T_node(row_block_c_ind,1:2);
                c_index = T_chi_block(c_row_pos,c_col_pos);
                cc_col_pos = ceil((c_index-1e-5)/K);
                cc_row_pos = ceil(mod(c_index-1e-5,K));
                
                try
                zero_mask(cc_row_pos,cc_col_pos) = 1;
                catch err
                    E
                    [i,j,p,c,n_chi,c_pos,row_pos,col_pos,index,c_row_pos,c_col_pos,c_index,cc_row_pos, cc_col_pos]
                    asdfsd
                end
                %zero_mask(c_row_pos,c_col_pos) = 1;
                Q_node(row_block_c_ind,1:2) = Q_node(row_block_c_ind,1:2) .* zero_mask;
                %zero_mask(c_row_pos,c_col_pos) = 0;
                zero_mask(cc_row_pos,cc_col_pos) = 0;

                % leave node: emit directly
                if node_degree(c) == 1
                    Y(c) = cc_col_pos;
                end

                %disp(sprintf(' chi %d index %d:%d -->(%d,%d) %d -->(%d,%d) emit %d',...
                %    c, block(row_pos,col_pos),index,c_row_pos,c_col_pos,c_index,cc_row_pos,cc_col_pos,Y(c)))
            end
        end
        Ymax_single(1,(k-1)*nlabel+1:k*nlabel) = Y'*2-3;

    end
    node_degree(E(1,1)) = node_degree(E(1,1))-1;

    return
end

