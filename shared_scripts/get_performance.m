

% label should be +1/0

function [acc,vecacc,pre,rec,f1,auc1,auc2] = get_performance(Y,Ypred,YpredVal)

    acc = get_accuracy(Y,Ypred);
    vecacc = get_vecaccuracy(Y,Ypred);
    pre=get_precision(Y,Ypred);
    rec=get_recall(Y,Ypred);
    f1=get_f1(Y,Ypred);
    if nargin==3
        [auc1,auc2]=get_auc(Y,YpredVal);
    else
        auc1=0;
        auc2=0;
    end
end

function [auc1,auc2] = get_auc(Y,YpredVal)
    auc1 = 0;
    auc2 = 0;
    return
    AUC=zeros(1,size(Y,2));
    for i=1:size(Y,2)
        try
            [ax,ay,t,AUC(1,i)]=perfcurve(Y(:,i),YpredVal(:,i),1);
        catch
            [ax,ay,t,AUC(1,i)]=[0,0,0,0];
        end
    end
    auc1=mean(AUC);
    [ax,ay,t,auc2]=perfcurve(reshape(Y,numel(Y),1),reshape(YpredVal,numel(YpredVal),1),1);
end

function [acc] = get_accuracy(Y,Ypred)

    acc=1-sum(sum(abs(Y-Ypred)))/size(Y,1)/size(Y,2);

end

function [vecacc] = get_vecaccuracy(Y,Ypred)

    vecacc=sum(Y~=Ypred,2);
    vecacc=sum((vecacc==0))/numel(vecacc);

end

function [f1] = get_f1(Y,Ypred)

    f1=(2*get_precision(Y,Ypred)*get_recall(Y,Ypred))/(get_precision(Y,Ypred)+get_recall(Y,Ypred));

end

function [tp] = get_tp(Y,Ypred)

    tp = Y + Ypred;
    tp=(tp==2);
    tp=sum(sum(tp));
    
end

function [fp] = get_fp(Y,Ypred)

    fp=Y-Ypred;
    fp=(fp==-1);
    fp=sum(sum(fp));

end

function [tn] = get_tn(Y,Ypred)

    tn=Y+Ypred;
    tn=(tn==0);
    tn=sum(sum(tn));

end

function [fn] = get_fn(Y,Ypred)

    fn=Y-Ypred;
    fn=(fn==1);
    fn=sum(sum(fn));

end

function [pre] = get_precision(Y,Ypred)

    pre=(get_tp(Y,Ypred))/(get_tp(Y,Ypred)+get_fp(Y,Ypred));

end

function [rec] = get_recall(Y,Ypred)

    rec=(get_tp(Y,Ypred))/(get_tp(Y,Ypred)+get_fn(Y,Ypred));

end
