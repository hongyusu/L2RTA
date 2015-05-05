



function plot_learning_process(option)



    dataset = {'toy10','toy50','emotions','yeast','scene'};
    dataset = {'toy10'};
    ts = [5 10 15 20];
    ks = [1 2 4 6 8 10 12];
    cs = [1 10 100];
    gs = [1];  
    
    if option == 1
        plot_x_as_duality_gap(dataset,ts,ks,cs,gs)
    end
    if option == 2
        plot_x_as_iteration_count(dataset,ts,ks,cs,gs)
    end
    if option == 3
        plot_overall_statistics(dataset,ts,ks,cs,gs)
    end
    
    
end

function plot_overall_statistics(dataset,ts,ks,cs,gs)


    addpath('./savefigure/');
    qd=char(34);
    qs=char(39);
    
    for di = 1:length(dataset)
        
       
        dname = dataset{di};


        
        for gi = 1:length(gs)
            
            
            figure1 = figure('visible','off');
            set(figure1, 'PaperUnits', 'centimeters');
            set(figure1, 'Position',[0 0 7000 2000]);


            g = gs(gi);

            for ci = 1:length(cs)

                for ti = 1:length(ts)

                    t = ts(ti);
                    c = cs(ci);


                    % training_1_err, training_ham_err, obj, duality_gap, gmax-g0, update
                    system(sprintf('for i in 1 2 4 6 8 10 12; do cat  ../outputs/backup_run/%s_tree_%d_f1_l2_k${i}_c%d_s%d_n1_RSTAs.log|tail -n3|head -n1|awk -F%s %s %s{print $6%s %s$9%s %s$11%s %s$13%s %s$14%s %s$32%s %s$33}%s|sed s/\\%%//g; done > tmp',...
                        dname,t,c,g,qs,qs,qs,qd,qd,qd,qd,qd,qd,qd,qd,qd,qd,qd,qd,qs));
                    x = dlmread('tmp');
                    % training_1_err, training_ham_err,test_1_err, test_ham_err, yi_tr, yi_ts
                    system(sprintf('for i in 1 2 4 6 8 10 12; do cat  ../outputs/backup_run/%s_tree_%d_f1_l2_k${i}_c%d_s%d_n1_RSTAs.log|tail -n2|head -n1|awk -F%s %s %s{print $1%s %s$4%s %s$7%s %s$10%s %s$13%s %s$18%s %s$24}%s|sed s/://g|sed s/\\%%//g; done > tmp',...
                        dname,t,c,g,qs,qs,qs,qd,qd,qd,qd,qd,qd,qd,qd,qd,qd,qd,qd,qs));
                    y = dlmread('tmp');

                    x = [ks',x,y];

                    js = [2,3,4,5,6,7,8,9,12,13,14,15];
                    ind = 0;
                    for j = js
                        ind = ind+1;
                        subplot(length(cs),length(js),(ci-1)*length(js)+ind);
                        hold on
                        plot(x(:,1),x(:,j),'-');


                        if ti==1
                            xlabel('K')
                            title(sprintf('C%d', c));
                        end

                        if j==2
                            title(sprintf('Tr 0/1, C%d', c));
                        end
                        if j==3
                            title(sprintf('Tr hamming, C%d', c));
                        end            
                        if j==4
                            title(sprintf('Objective, C%d', c));
                        end
                        if j==5
                            title(sprintf('Gap, C%d', c));
                        end  
                        if j==6
                            title(sprintf('Duality gap %, C%d', c));
                        end 
                        if j==7
                            title(sprintf('Tree update %, C%d', c));
                        end 
                        if j==8
                            title(sprintf('Example update %, C%d', c));
                        end 
                        if j==9
                            title(sprintf('Running time %, C%d', c));
                        end 
                        if j==12
                            title(sprintf('Ts 0/1, C%d', c));
                        end
                        if j==13
                            title(sprintf('Ts hamming, C%d', c));
                        end
                        if j==14
                            title(sprintf('Yi tr, C%d', c));
                        end
                        if j==15
                            title(sprintf('Yi ts, C%d', c));
                        end
                        hold off
                    end

                end
                lgd = legend(strcat('T=',num2str(ts')));
                set(lgd,  'fontsize', 10, 'interpreter','latex','Position', [0,0.2,0.1,0.1]);

            end
            
            export_fig(sprintf('../plots/plot_overall_statistics_%s_%d_%d.jpg',dataset{di},gs(gi),cs(ci)),figure1)

        end

        
    end

end




function plot_x_as_iteration_count(dataset,ts,ks,cs,gs)

    addpath('./savefigure/');
    
    qd=char(34);
    qs=char(39);
   
    for di = 1:length(dataset)
        for gi = 1:length(gs)
            for ci = 1:length(cs)
                
                % initilize figures
                figure1 = figure('visible','off');
                set(figure1, 'PaperUnits', 'centimeters');
                set(figure1, 'Position',[0 0 6000 3000]);
                
                for ti = 1:length(ts)
                    for ki = 1:length(ks)

                        % training_1_err, training_ham_err, obj, duality_gap, gmax-g0, update
                        system(sprintf('cat  ../outputs/backup_run/%s_tree_%d_f1_l2_k%d_c%d_s%d_n1_RSTAs.log|head -n20|awk -F%s %s %s{print $1%s %s$6%s %s$9%s %s$11%s %s$13%s %s$14%s %s$32%s %s$33}%s|sed s/://g|sed s/\\%%//g > tmp',...
                            dataset{di},ts(ti),ks(ki),cs(ci),gs(gi),qs,qs,qs,qd,qd,qd,qd,qd,qd,qd,qd,qd,qd,qd,qd,qd,qd,qs));
                        
                        x = dlmread('tmp');
                        
                        js = [1 2 3 4 5 6 7 8];
                        for j =1:length(js)
                            
                            subplot(length(ks),length(js),j+(ki-1)*length(js))
                            hold on
                            plot(x(:,j),'-');
                            xlabel('Iteration count');
                            if j==1
                                title(sprintf('Running time, K=%d, C=%d',ks(ki),cs(ci)));
                            end
                            if j==2
                                title(sprintf('0/1 loss tr, K=%d, C=%d',ks(ki),cs(ci)));
                            end
                            if j==3
                                title(sprintf('Hamming loss tr, K=%d, C=%d',ks(ki),cs(ci)));
                            end
                            if j==4
                                title(sprintf('Objective, K=%d, C=%d',ks(ki),cs(ci)));
                            end
                            if j==5
                                title(sprintf('Gap value, K=%d, C=%d',ks(ki),cs(ci)));
                            end
                            if j==6
                                title(sprintf('Gap percentage, K=%d, C=%d',ks(ki),cs(ci)));
                            end
                            if j==7
                                title(sprintf('Tree update, K=%d, C=%d',ks(ki),cs(ci)));
                            end
                            if j==8
                                title(sprintf('Example update, K=%d, C=%d',ks(ki),cs(ci)));
                            end
                            hold off
                        end
                    end
                end
                lgd = legend(strcat('T=',num2str(ts')));
                set(lgd,  'fontsize', 10, 'interpreter','latex','Position', [0,0.2,0.1,0.1]);

                export_fig(sprintf('../plots/plot_learning_process_%s_%d_%d.jpg',dataset{di},gs(gi),cs(ci)),figure1)
            end
            

        end
    end
    
end

function plot_x_as_duality_gap(dataset,ts,ks,cs,gs)


    addpath('./savefigure/');
    
    qd=char(34);
    qs=char(39);
      
    
    for di = 1:length(dataset)
        
        for gi = 1:length(gs)

            
            for ci = 1:length(cs)


                figure1 = figure('visible','off');
                set(figure1, 'PaperUnits', 'centimeters');
                set(figure1, 'Position',[0 0 6000 3000]);
                for ti = 1:length(ts)

                    for ki = 1:length(ks)

                        % training_1_err, training_ham_err, obj, duality_gap, gmax-g0, update
                        system(sprintf('cat  ../outputs/backup_run/%s_tree_%d_f1_l2_k%d_c%d_s%d_n1_RSTAs.log|head -n20|awk -F%s %s %s{print $6%s %s$9%s %s$11%s %s$13%s %s$14%s %s$32%s %s$33}%s|sed s/\\%%//g > tmp',...
                            dataset{di},ts(ti),ks(ki),cs(ci),gs(gi),qs,qs,qs,qd,qd,qd,qd,qd,qd,qd,qd,qd,qd,qd,qd,qs));
                        
                        
                        x = dlmread('tmp');
                        
                        x(:,5) = log(x(:,5));
                        x(isinf(x)) = -5;
                        
                        js = [1 2 3 4 6 7];
                        for j =1:length(js)
                            subplot(length(ks),length(js),j+(ki-1)*length(js))
                            hold on
                            plot(x(:,5),x(:,js(j)),'-')
                            title(sprintf('T=%d, C=%d',ks(ki),cs(ci)));

                            xlabel('Duality gap percentage')
                            
                            if j==1
                                ylabel('1 error tr')
                            end
                            if j==2
                                ylabel('hamming error tr')
                            end
                            if j==3
                                ylabel('gap')
                            end
                            if j==4
                                ylabel('gap')
                            end
                            if j==5
                                ylabel('Update')
                            end
                            if j==6
                                ylabel('Update')
                            end
                            hold off
                        end
                    end
                end
                lgd = legend(strcat('K=',num2str(ts')));
                set(lgd,  'fontsize', 10, 'interpreter','latex','Position', [0,0.2,0.1,0.1]);

                export_fig(sprintf('plot_learning_process_%s_%d_%d.jpg',dataset{di},gs(gi),cs(ci)),figure1)
            end
            

        end
    end
    

end



