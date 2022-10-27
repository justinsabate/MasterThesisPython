% Shows the results of the perceptual mixing time prediction from model
% and data based predictors. If data based prediction was selected it plots
% the curve of the echo density of each channel and marks the corresponding
% perceptual mixing times.
%
%
% tmp50_model_based       - average perceptual mixing time, model based prediction
% tmp95_model_based       - 95%-point perceptual mixing time, model based prediction
% tmp50_data_based        - average  mixing time, data based prediction
% tmp95_data_based        - 95%-point perceptual mixing time, data based prediction
% tmp50_interchannel_mean_data_based   - interchannel average of average perceptual mixing time, data based prediction, average from all channels
% tmp95_interchannel_mean_data_based   - interchannel average of 95%-point perceptual mixing time, data based prediction, average from all channels
% echo_dens               - echo density (cf. Abel & Huang (2006))
% fs                      - sampling frequency
% do_print                - save plot, [1/0]
%
% A. Lindau, L. Kosanke, 2011
% alexander.lindau@tu-berlin.de
% audio communication group
% Technical University of Berlin
%-------------------------------------------------------------------------%
function plot_tmp(tmp50_model_based, tmp95_model_based, tmp50_data_based, tmp95_data_based, tmp50_interchannel_mean_data_based, tmp95_interchannel_mean_data_based, echo_dens, fs, do_print)

% figure properties
hFigureHandle = figure;
set(hFigureHandle,'PaperUnits', 'centimeters', 'Units', 'centimeters')
set(hFigureHandle,'PaperPosition', [0 0 17 12], 'Position', [0 0 17 12])

if isempty(tmp50_data_based)
    plot([4 4],[0.5 5.5],'k',[0 10],[4.6 4.6],'k')
    text(0.1,5,'MODEL-BASED','FontSize',12)
    text(0.1,4,['t_{mp50}= ',num2str(round(tmp50_model_based)),' ms'],'FontSize',10)
    text(0.1,3,['t_{mp95}= ',num2str(round(tmp95_model_based)),' ms'],'FontSize',10)
    text(4.5,5,'DATA-BASED','FontSize',12)
    text(4.5,3,'No results available.','FontSize',10)
    axis([0 12 0 6]),axis off
    set(hFigureHandle,'PaperPosition', [0 0 13 6], 'Position', [0 0 13 6])
    
else
    % time vector for echo density plot
    t = linspace(0,length(echo_dens)/fs*1000,length(echo_dens));
    
    subplot(2,1,1);hold on
    
    % plot estimated tmp-values
    for n = 1:length(tmp50_data_based)
        
        if round(tmp50_data_based(n)/1000*fs) > length(echo_dens)
            plot(tmp50_data_based(n),1,'or','LineWidth',4)
        else
            plot(t(round(tmp50_data_based(n)/1000*fs)),echo_dens(round(tmp50_data_based(n)/1000*fs),n),'or','LineWidth',4)
        end
        
        if round(tmp95_data_based(n)/1000*fs) > length(echo_dens)
            plot(tmp95_data_based(n),1,'og','LineWidth',4)
        else
            plot(t(round(tmp95_data_based(n)/1000*fs)),echo_dens(round(tmp95_data_based(n)/1000*fs),n),'og','LineWidth',4)
        end
        
    end
    legend('t_{mp50}','t_{mp95}','Location','SouthEast')
    
    % plot echo density
    plot(t,echo_dens,'k')
    xlabel('t in [ms]','FontSize',8), ylabel('echo density','FontSize',8), title('Data-based prediction using echo-densitiy-approach from Abel & Huang (2006, criterion I)','FontSize',8)
    
    hold off
    set(gca,'XScale','log')
    axis([1 max(max(t),max(max(tmp50_data_based),max(tmp95_data_based))) 0 1.2]),grid on
    
    subplot(2,1,2)
    plot([4 4],[0.5 5.5],'k',[0 13.5],[4.6 4.6],'k')
    % write results of model based predicition into figure
    if isempty(tmp50_model_based)
        text(0.1,5,'MODEL-BASED','FontSize',12)
        text(0.1,3,'No results available.','FontSize',10)
    else
        text(0.1,5,'MODEL-BASED','FontSize',12)
        text(0.1,4,['t_{mp50}= ',num2str(round(tmp50_model_based)),' ms'],'FontSize',10)
        text(0.1,3,['t_{mp95}= ',num2str(round(tmp95_model_based)),' ms'],'FontSize',10)
    end
    
    % write results of data based prediction into figure
    
    % results of prediction for more than one channel
    if length(tmp50_data_based)>1
        for n = 1:length(tmp50_data_based)
            text(4.5,5,'DATA-BASED','FontSize',12)
            text(4.5+(n-1)*4,4,['t_{mp50} (ch. ',num2str(n)',') = ',num2str(round(tmp50_data_based(n))),' ms'],'FontSize',10)
            text(4.5+(n-1)*4,3,['t_{mp95} (ch. ',num2str(n)',') = ',num2str(round(tmp95_data_based(n))),' ms'],'FontSize',10)
            text(4.5,2,['t_{mp50} (interchannel mean) = ',num2str(round(tmp50_interchannel_mean_data_based)),' ms'],'FontSize',10)
            text(4.5,1,['t_{mp95} (interchannel mean) = ',num2str(round(tmp95_interchannel_mean_data_based)),' ms'],'FontSize',10)
        end
    else
        % results of prediction for one channel
        text(4.5,5,'DATA-BASED','FontSize',12)
        text(4.5,4,['t_{mp50} = ',num2str(round(tmp50_data_based)),' ms'],'FontSize',10)
        text(4.5,3,['t_{mp95} = ',num2str(round(tmp95_data_based)),' ms'],'FontSize',10)
    end
end

axis([0 12 0 6]),axis off

% print
if do_print == 1
    print('-dtiff','-r300','tmp_prediction_results')
end