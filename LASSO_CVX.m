%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% LASSO regression using CVX
% arg min_{B} 0.5*||X - A*B||_{2}^{2} + gamma*||B||_{1}
%
% CVX is a Matlab-based modeling system for convex optimization. CVX turns Matlab
% into a modeling language, allowing constraints and objectives to be specified
% using standard Matlab expression syntax
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function B = lasso_cvx(X, A, gamma)

% vectorize
I = speye(c);
J = kron(I,A);
x = X(:);

% Get the size of B
c = size(X,2);
r = size(A,2);

cvx_begin
    cvx_quiet(true);
    variable b(r*c,1);
    minimize(0.5*sum_square(x - J*b) + gamma*norm(b,1));
cvx_end
B = reshape(b,[r,c]);
