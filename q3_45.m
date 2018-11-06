load("trainAnno.mat");
[trD, trLb, valD, valLb, trRegs, valRegs] = HW4_Utils.getPosAndRandomNeg();
[W, b, alpha, fval] = qpSVM(trD, trLb);
[d, n] = size(trD);
PosD = trD(:, trLb>0);
NegD = trD(:, trLb<0 & alpha<0.1); 
fvals = [];
ap_array = [];

for iter = 1:10
    disp("iteration : ");
    disp(iter);
    PosD = trD(:, trLb>0);
    NegD = trD(:, trLb<0 & alpha<0.1);

    HW4_Utils.genRsltFile(W, b, "train", "q3_rects");
    load("q3_rects.mat");
    hard_neg = [];
    for i = 1:length(rects)
        im = imread(sprintf('%s/trainIms/%04d.jpg', HW4_Utils.dataDir, i));
        [imH, imW,~] = size(im);
        current_rect = rects{i};
        badIdxs = or(current_rect(3,:) > imW, current_rect(4,:) > imH);
        current_rect = current_rect(:,~badIdxs);
        ubs = ubAnno{i};
        overlaps = [];
        for j = 1:size(ubs, 2)
            overlap = HW4_Utils.rectOverlap(current_rect, ubs(:, j));
            overlaps = [overlaps, overlap];
        end        
        
        for j = 1:length(current_rect)
            if current_rect(5, j) > 0
               continue 
            end
            break_flag = 0;
            for k = 1:size(ubs, 2)
                if overlaps(j, k) > 0.3
                    break_flag = 1;
                    break;
                end
            end
            if break_flag == 0
              
                imReg = im(int16(current_rect(2, j)):int16(current_rect(4, j)), int16(current_rect(1, j)):int16(current_rect(3, j)), :);
                imReg = imresize(imReg, HW4_Utils.normImSz);
                
                feat = HW4_Utils.cmpFeat(rgb2gray(imReg));
                feat = feat / norm(feat);
                hard_neg = [hard_neg, feat];
                
                if size(hard_neg, 2) > 1000
                    break;
                end
            end
            if size(hard_neg, 2) > 1000
                break;
            end
        end
        if size(hard_neg, 2) > 1000
            break;
        end
    end
    NegD = [NegD, hard_neg];
    NegLb = -ones(size(NegD, 2), 1);
    trD = [];
    trD = [trD, PosD];
    trD = [trD, NegD];
    PosLb = ones(size(PosD, 2), 1);
    trLb = [PosLb; NegLb];

    [W, b, alpha, fval] = qpSVM(trD, trLb);

    fvals = [fvals, fval];
    
    HW4_Utils.genRsltFile(W, b, "val", "q3_temp");

    [ap, prec, rec] = HW4_Utils.cmpAP("q3_temp", "val");
    ap_array = [ap_array, ap];
    
end
numbers = linspace(1, 10, 10);
subplot(2,1,1);
plot(numbers, fvals);
title('Objective values')

subplot(2,1, 2);
plot(numbers, ap_array);
title('APs')

HW4_Utils.genRsltFile(W, b, "test", "109369879");

function [W, b, alpha, fval] = qpSVM(trD, trLb)
    C = 10; % C = 0.1;
    [~,n] = size(trD);
    k = trD'*trD;
    H = diag(trLb)*k*diag(trLb);
    f = -ones(1,n);
    A = zeros(1,n);
    b = 0;
    Aeq = trLb';
    beq = 0;
    lb = zeros(n,1);
    ub = C*ones(n,1);
    [alpha, fval] = quadprog(double(H),f,A,b,Aeq,beq,lb,ub);
    fval = -fval;
    alpha_n = diag(alpha);
    W = (trLb'*alpha_n*trD')';
  
    %b = trLb - trD'*W;
    %b = mean(b);
    
    [~, index] = min(abs(alpha-0.05));
    b = trLb(index) - (W' * trD(:, index));
end