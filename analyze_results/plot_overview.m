




function plot_overview()


    addpath('./savefigure/');
    
    dataset = {'toy10','toy50','emotions','yeast','scene'};

    for di = 1:length(dataset)
        
        
        dname = dataset{di};

        
        ts = [1, 5,10,15,20];
        ks = [1 2 4 6 8 10 12];
        cs = [1 10 100];
        gs = [1 5];


        
        for ig = 1:length(gs)
            
            
            figure1 = figure('visible','off');
            set(figure1, 'PaperUnits', 'centimeters');
            set(figure1, 'Position',[0 0 6000 3000]);


            g = gs(ig);

            for ic = 1:length(cs)

                for i = 1:length(ts)

                    t=ts(i);
                    c=cs(ic);

                    qd=char(34);
                    qs=char(39);
                    % training_1_err, training_ham_err, obj, duality_gap, gmax-g0, update
                    system(sprintf('for i in 1 2 4 6 8 10 12; do cat  ../outputs/backup_run/%s_tree_%d_f1_l2_k${i}_c%d_s%d_n1_RSTAs.log|tail -n3|head -n1|awk -F%s %s %s{print $6%s %s$9%s %s$11%s %s$13%s %s$14%s %s$32%s %s$33}%s|sed s/\\%%//g; done > tmp',...
                        dname,t,c,g,qs,qs,qs,qd,qd,qd,qd,qd,qd,qd,qd,qd,qd,qd,qd,qs));
                    x = dlmread('tmp');
                    % training_1_err, training_ham_err,test_1_err, test_ham_err, yi_tr, yi_ts
                    system(sprintf('for i in 1 2 4 6 8 10 12; do cat  ../outputs/backup_run/%s_tree_%d_f1_l2_k${i}_c%d_s%d_n1_RSTAs.log|tail -n2|head -n1|awk -F%s %s %s{print $4%s %s$7%s %s$10%s %s$13%s %s$18%s %s$24}%s|sed s/\\%%//g; done > tmp',...
                        dname,t,c,g,qs,qs,qs,qd,qd,qd,qd,qd,qd,qd,qd,qd,qd,qs));
                    y = dlmread('tmp');

                    x = [ks',x,y];

                    js = [2,3,4,5,6,10,11,12,13];
                    ind = 0;
                    for j = js
                        ind = ind+1;
                        subplot(length(cs),length(js),(ic-1)*length(js)+ind);
                        hold on
                        plot(x(:,1),x(:,j),'o-');


                        if i==1
                            xlabel('K')
                            title(sprintf('C%d', c));

                        end

                        if j==2
                            ylabel('Training error, multilabel');
                        end
                        if j==3
                            ylabel('Training error, microlabel');
                        end            
                        if j==4
                            ylabel('Objective');
                        end
                        if j==5
                            ylabel('Gap');
                        end  
                        if j==6
                            ylabel('Duality gap %');
                        end 
                        if j==10
                            ylabel('Test error, multilabel');
                        end
                        if j==11
                            ylabel('Test error, microlabel');
                        end
                        if j==12
                            ylabel('Yi Tr');
                        end
                        if j==13
                            ylabel('Yi Ts');
                        end
                        hold off
                    end

                end
                lgd = legend(strcat('T=',num2str(ts')));
                set(lgd,  'fontsize', 10, 'interpreter','latex','Position', [0,0.2,0.1,0.1]);

            end
            
            export_fig(sprintf('plot_%s_%d.jpg',dname,g),figure1)

        end

        
    end

end



