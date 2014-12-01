%% 
function [Ymax, YmaxVal] = majority_voting(Y_kappa,Y_kappa_val,nlabel,Y)

    use_n_label = 1;
    Ymax_tmp = sum(reshape( Y_kappa(:,1:(nlabel*use_n_label)),size(Y_kappa,1)*use_n_label,nlabel) .* repmat(reshape(Y_kappa_val(:,1:use_n_label),size(Y_kappa,1)*use_n_label,1),1,nlabel) );
    Ymax = (Ymax_tmp>0)*2-1;
    if sum(Y==Ymax) == nlabel
        Ymax(Ymax_tmp==min(Ymax_tmp)) = Ymax(Ymax_tmp==min(Ymax_tmp)) * (-1);
    end
    
    YmaxVal = mean(reshape(Y_kappa_val(:,1:(use_n_label)),size(Y_kappa,1)*use_n_label,1));
    
    return
end