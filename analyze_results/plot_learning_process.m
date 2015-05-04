



function plot_learning_process(option)

    if option == 1
        plot_x_as_duality_gap()
    end
    if option == 2
        plot_x_as_iteration_count()
    end
end



function plot_x_as_iteration_count()

addpath('./savefigure/');
    
    qd=char(34);
    qs=char(39);
    
    
    dataset = {'toy10','toy50','emotions','yeast','scene'};
    %dataset = {'toy10'}
    
    
    ts = [1 5 10 15 20];
    ks = [1 2 4 6 8 10 12];
    cs = [1,10];
    gs = [1];    
    
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
                        
                        js = [1 2 3 4 5 6 7];
                        for j =1:length(js)
                            
                            subplot(length(ks),length(js),j+(ki-1)*length(js))
                            hold on
                            plot(x(:,j),'-');
                            xlabel('Iteration count');
                            if j==1
                                title(sprintf('Running time, T=%d, C=%d',ks(ki),cs(ci)));
                            end
                            if j==2
                                title(sprintf('0/1 loss tr, T=%d, C=%d',ks(ki),cs(ci)));
                            end
                            if j==3
                                title(sprintf('Hamming loss tr, T=%d, C=%d',ks(ki),cs(ci)));
                            end
                            if j==4
                                title(sprintf('Objective, T=%d, C=%d',ks(ki),cs(ci)));
                            end
                            if j==5
                                title(sprintf('Gap value, T=%d, C=%d',ks(ki),cs(ci)));
                            end
                            if j==6
                                title(sprintf('Gap percentage, T=%d, C=%d',ks(ki),cs(ci)));
                            end
                            if j==7
                                title(sprintf('Update, T=%d, C=%d',ks(ki),cs(ci)));
                            end
                            hold off
                        end
                    end
                end
                lgd = legend(strcat('K=',num2str(ts')));
                set(lgd,  'fontsize', 10, 'interpreter','latex','Position', [0,0.2,0.1,0.1]);

                export_fig(sprintf('../plots/plot_learning_process_%s_%d_%d.jpg',dataset{di},gs(gi),cs(ci)),figure1)
            end
            

        end
    end
    
end

function plot_x_as_duality_gap()


    addpath('./savefigure/');
    
    qd=char(34);
    qs=char(39);
    
    
    dataset = {'toy10','toy50','emotions','yeast','scene'};

    
    
    ts = [5 10 15 20];
    ks = [1 2 4 6 8 10 12];
    cs = [1 10 100];
    gs = [1];    
    
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



