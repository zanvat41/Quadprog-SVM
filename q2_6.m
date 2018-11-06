[d,n] = size(trD);
trLb0 = ones(n, 1);
trLb1 = ones(n, 1);
trLb2 = ones(n, 1);
trLb3 = ones(n, 1);
trLb4 = ones(n, 1);
trLb5 = ones(n, 1);
trLb6 = ones(n, 1);
trLb7 = ones(n, 1);
trLb8 = ones(n, 1);
trLb9 = ones(n, 1);

for i=1:n
    if trLb(i) ~= 1
        trLb0(i) = -1;
    end
    
    if trLb(i) ~= 2
        trLb1(i) = -1;
    end
    
    if trLb(i) ~= 3
        trLb2(i) = -1;
    end
    
    if trLb(i) ~= 4
        trLb3(i) = -1;
    end
        
    if trLb(i) ~= 5
        trLb4(i) = -1;
    end
        
    if trLb(i) ~= 6
        trLb5(i) = -1;
    end
        
    if trLb(i) ~= 7
        trLb6(i) = -1;
    end
        
    if trLb(i) ~= 8
        trLb7(i) = -1;
    end
        
    if trLb(i) ~= 9
        trLb8(i) = -1;
    end
        
    if trLb(i) ~= 10
        trLb9(i) = -1;
    end
end

[W0, b0, alpha0, f_0] = qpSVM(trD, trLb0);
[W1, b1, alpha1, f_1] = qpSVM(trD, trLb1);
[W2, b2, alpha2, f_2] = qpSVM(trD, trLb2);
[W3, b3, alpha3, f_3] = qpSVM(trD, trLb3);
[W4, b4, alpha4, f_4] = qpSVM(trD, trLb4);
[W5, b5, alpha5, f_5] = qpSVM(trD, trLb5);
[W6, b6, alpha6, f_6] = qpSVM(trD, trLb6);
[W7, b7, alpha7, f_7] = qpSVM(trD, trLb7);
[W8, b8, alpha8, f_8] = qpSVM(trD, trLb8);
[W9, b9, alpha9, f_9] = qpSVM(trD, trLb9);

[size1, size2] = size(valLb);

res0 = valD'*W0 + b0;
res1 = valD'*W1 + b1;
res2 = valD'*W2 + b2;
res3 = valD'*W3 + b3;
res4 = valD'*W4 + b4;
res5 = valD'*W5 + b5;
res6 = valD'*W6 + b6;
res7 = valD'*W7 + b7;
res8 = valD'*W8 + b8;
res9 = valD'*W9 + b9;

res = ones(size1, 1);

count = 0;
for i = 1:size1
    if res1(i) > res0(i)
        res0(i) = res1(i);
        res(i) = 2;
    end

    if res2(i) > res0(i)
        res0(i) = res2(i);
        res(i) = 3;
    end

    if res3(i) > res0(i)
        res0(i) = res3(i);
        res(i) = 4;
    end

    if res4(i) > res0(i)
        res0(i) = res4(i);
        res(i) = 5;
    end

    if res5(i) > res0(i)
        res0(i) = res5(i);
        res(i) = 6;
    end

    if res6(i) > res0(i)
        res0(i) = res6(i);
        res(i) = 7;
    end

    if res7(i) > res0(i)
        res0(i) = res7(i);
        res(i) = 8;
    end

    if res8(i) > res0(i)
        res0(i) = res8(i);
        res(i) = 9;
    end
 
    if res9(i) > res0(i)
        res0(i) = res9(i);
        res(i) = 10;
    end
    
    if res(i) == valLb(i)
        count = count + 1;
    end
    
end

[size1, size2] = size(tstD);

res0 = tstD'*W0 + b0;
res1 = tstD'*W1 + b1;
res2 = tstD'*W2 + b2;
res3 = tstD'*W3 + b3;
res4 = tstD'*W4 + b4;
res5 = tstD'*W5 + b5;
res6 = tstD'*W6 + b6;
res7 = tstD'*W7 + b7;
res8 = tstD'*W8 + b8;
res9 = tstD'*W9 + b9;

res = ones(size2, 1);

for i = 1:size2
    if res1(i) >= res0(i)
        res0(i) = res1(i);
        res(i) = 2;
    end

    if res2(i) >= res0(i)
        res0(i) = res2(i);
        res(i) = 3;
    end

    if res3(i) >= res0(i)
        res0(i) = res3(i);
        res(i) = 4;
    end

    if res4(i) >= res0(i)
        res0(i) = res4(i);
        res(i) = 5;
    end

    if res5(i) >= res0(i)
        res0(i) = res5(i);
        res(i) = 6;
    end

    if res6(i) >= res0(i)
        res0(i) = res6(i);
        res(i) = 7;
    end

    if res7(i) >= res0(i)
        res0(i) = res7(i);
        res(i) = 8;
    end

    if res8(i) >= res0(i)
        res0(i) = res8(i);
        res(i) = 9;
    end
 
    if res9(i) >= res0(i)
        res0(i) = res9(i);
        res(i) = 10;
    end    
end


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