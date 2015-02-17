function rtn=randTreeGenerator(N)
    usedNode=[];
    availableNode=1:N;
    edgelist=[];
    
    for i=1:N
        if i~=N
            sampleInd=randsample(find(~isnan(availableNode)),1);
        else
            sampleInd=find(~isnan(availableNode));
        end
        if length(usedNode) ~= 0
            if length(usedNode)==1
                edgelist=[edgelist;[usedNode, sampleInd]];
            else
                edgelist=[edgelist;[randsample(usedNode,1), sampleInd]];
            end
        end
        usedNode=[usedNode,sampleInd];
        availableNode(sampleInd)=NaN;
    end
    rtn=edgelist;
end