

%
% get stratified 'nfold'-folder cross validation index
%
function fInd = getCVIndex(Y,nfold)
    rand('twister', 0);
    fInd=[];
    Ysum=sum(Y,2);
    Yunique=unique(Ysum);
    for i=1:numel(Yunique)
        fInd=[fInd;[find(Ysum==Yunique(i)),randsample(nfold,length(find(Ysum==Yunique(i))),true)]];
    end
    [a,b]=sort(fInd(:,1));
    fInd=fInd(b,2);
end