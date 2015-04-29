




function plot_learning_process()


    addpath('./savefigure/');
    
    qd=char(34);
    qs=char(39);
    
    
    dataset = {'toy10','toy50','emotions'};
    ts = [1, 5,10,15,20];
    ks = [1 2 4 6 8 10 12];
    cs = [1 10 100];
    gs = [1 5];    
    
    for di = 1:length(dataset)
        
        for gi = 1:length(gs)

            figure1 = figure('visible','off');
            set(figure1, 'PaperUnits', 'centimeters');
            set(figure1, 'Position',[0 0 6000 3000]);
            
            for ci = 1:length(cs)

                for ki = 1:length(ks)

                    for ti = 1:length(ts)

                        % training_1_err, training_ham_err, obj, duality_gap, gmax-g0, update
                        system(sprintf('cat  ../outputs/backup_run/%s_tree_%d_f1_l2_k%d_c%d_s%d_n1_RSTAs.log|head -n20|awk -F%s %s %s{print $6%s %s$9%s %s$11%s %s$13%s %s$14%s %s$32%s %s$33}%s|sed s/\\%%//g > tmp',...
                            dataset{di},ts(ti),ks(ki),cs(ci),gs(gi),qs,qs,qs,qd,qd,qd,qd,qd,qd,qd,qd,qd,qd,qd,qd,qs));
                        x = dlmread('tmp');
                        js = [1 2 3 4 6 7];
                        for j =1:length(js)
                            subplot(length(ks),length(js),j+(ki-1)*length(js))
                            plot(log(x(:,5)),x(:,js(j)),'-')
                            hold on
                            if 1==ti
                                title(sprintf('K=%d, C=%d',ks(ki),cs(ci)));
                            end
                            xlabel('Duality gap')
                            
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
                                ylabel('1 error ts')
                            end
                            if j==6
                                ylabel('hamming error ts')
                            end
                        end
                    end
                end

                lgd = legend(strcat('T=',num2str(ts')));
                set(lgd,  'fontsize', 10, 'interpreter','latex','Position', [0,0.2,0.1,0.1]);
                
                export_fig(sprintf('plot_learning_process_%s_%d_%d.jpg',dataset{di},gs(gi),cs(ci)),figure1)

            end
        end
    end
    

end



