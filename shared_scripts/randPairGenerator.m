function rtn=randPairGenerator(N)
    usedNode=[];
    availableNode=1:N;
    edgelist=[];
    l=randsample(availableNode,length(availableNode));
    for i=1:floor(N/2)
            j=(i-1)*2;
            edgelist=[edgelist;[l(j+1),l(j+2)]];
    end
    if N>j+2
            edgelist=[edgelist;[l(j+2),l(j+3)]];
    end
    
    rtn=edgelist;
end
