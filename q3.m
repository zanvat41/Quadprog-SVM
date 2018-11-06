[trD, trLb, valD, valLb, trRegs, valRegs] = HW4_Utils.getPosAndRandomNeg();
[W, b] = qpSVM(trD, trLb);
HW4_Utils.genRsltFile(W, b, "val", "q31");
[ap, prec, rec] = HW4_Utils.cmpAP("q31", "val");

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
    [alpha, fval] = quadprog(H,f,A,b,Aeq,beq,lb,ub);
    fval = -fval;
    alpha_n = diag(alpha);
    W = (trLb'*alpha_n*trD')';
  
    %b = trLb - trD'*W;
    %b = mean(b);
    
    [~, index] = min(abs(alpha-0.05));
    b = trLb(index) - (W' * trD(:, index));
end