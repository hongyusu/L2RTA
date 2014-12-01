%% get the worst margin violator from T*kappa violators
% Input: Y_kappa, Y_kappa_val
% Output:
function [Ymax, YmaxVal,break_flag] = find_worst_violator(Y_kappa,Y_kappa_val)
%     if nargin == 3
%         Y_ind_true = bin2dec(strrep(sprintf('%d ',(Y_true+1)/2),' ',''));
%     else
%         Y_ind_true = -1;
%     end
%tic
    % global variable
    l = size(Y_kappa,2)/size(Y_kappa_val,2);
    % local variable
    Y_kappa_ind = Y_kappa_val * 0;
    Y_kappa = (Y_kappa+1)/2;
    % assing decimal value for each multilabel
    for i=1:size(Y_kappa_val,1)
        for j = 1:size(Y_kappa_val,2)
            s = strrep(sprintf('%d ',Y_kappa(i,((j-1)*l+1):(j*l))),' ','');
            try
            Y_kappa_ind(i,j) = bin2dec(s);
            catch err
                Y_kappa(i,((j-1)*l+1):(j*l))
                s
            end
        end
    end
    if sum(Y_kappa_ind(:,1)==Y_kappa_ind(:,2))> 0
        Y_kappa_ind
        Y_kappa_val
        %reshape(Y_kappa,numel(Y_kappa)/10,10)
    end
    %
    break_flag=0;
    for i=1:size(Y_kappa_val,2)
        t_line = sum(Y_kappa_val(:,i));
        current_matrix_val = Y_kappa_val(:,1:i);
        current_matrix_ind = Y_kappa_ind(:,1:i);
%         unique_elements = unique(current_matrix_ind(current_matrix_ind~=Y_ind_true));
%         if size(unique_elements,1) == 0
%             continue
%         end
        unique_elements = unique(current_matrix_ind);
        if size(unique_elements) == 1
            break_flag=1;
            element_id = unique_elements(1);
            break
        end
        element_id=0;
        element_val=-1;
        element_id_mean=0;
        element_val_mean=-1;
        for j=1:size(unique_elements,1)
            current_val = sum(current_matrix_val(current_matrix_ind==unique_elements(j)));
            current_id = unique_elements(j);
            %[i,j,current_id, current_val]
            if current_val > element_val
                %element_val = sum(current_matrix_val(current_matrix_ind==unique_elements(j)));
                element_val = current_val;
                element_id = current_id;
            end
            current_val_mean = sum(current_matrix_val(current_matrix_ind==unique_elements(j)));
            current_id_mean = unique_elements(j);
            if current_val_mean > element_val_mean
                %element_val_mean = mean(current_matrix_val(current_matrix_ind==unique_elements(j)));
                element_val_mean = current_val_mean;
                element_id_mean = current_id_mean;                
            end
        end
        if element_val >= t_line
            break_flag=1;
            break
        end
    end
    if break_flag==0
        element_id = element_id_mean;
    end
    ind = find(Y_kappa_ind==element_id);
    ind = ind(1);
    i = ceil(mod(ind-1e-5, size(Y_kappa,1)));
    j = ceil((ind-1e-5) / size(Y_kappa,1));
    Ymax = Y_kappa(i,((j-1)*l+1):(j*l))*2-1;
    YmaxInd = Y_kappa_ind(i,j);
    YmaxVal = Y_kappa_val(i,j);
%toc
    return
end
