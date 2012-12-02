clear all
close all
clc

m = 100;

% Construct a matrix
A = 2.5 * eye(m, m);

for i=1:m
    for j=1:m
        if (i == j-5):
            A[i][j] = 5.0;
    end
end

spy(A)

% Check with MATLAB's eigs function


% Check with power iteration and backslash notation