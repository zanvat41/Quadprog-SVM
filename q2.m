[W, b, alpha, fval] = qpSVM(trD, trLb);
res = sign(valD'*W + b);
[size1, size2] = size(valLb);
accuracy = nnz(valLb==res)/size(valLb, 1);
confusion_matrix = confusionmat(valLb, res);

res_temp = valD'*W + b;
support_vectors = size(res_temp(res_temp >= -1 & res_temp<=1));
%support_vectors = find(alpha<0.1);
%num_sv = size(support_vectors, 1);

function [W, b, alpha, fval] = qpSVM(trD, trLb)
    C = 0.1; % C = 0.1;
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