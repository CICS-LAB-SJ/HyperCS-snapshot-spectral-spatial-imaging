% fwhmg  Generate Gaussian kernels for the spectral sparsifying dictionary.
%
%   output = fwhmg(FWHM, space, min_x, max_x)
%     FWHM  : full width at half maximum of each kernel [nm]
%     space : center-to-center spacing between kernels [nm]
%     min_x, max_x : wavelength range [nm]
%   Returns a (max_x-min_x+1) x floor(range/space) matrix whose columns are
%   Gaussian kernels centered at min_x, min_x+space, ...

function [ output ] = fwhmg(FWHM, space, min_x, max_x)

x = min_x : 1 : max_x; 
x = x * 1e-9; 
min_x = min_x * 1e-9;

% number of kernels
iter = length(x) / space;

% Gaussian standard deviation from FWHM
sigma = FWHM * 1e-9 / (2 * sqrt(2 * log(2)));

output = zeros(length(x), iter);

for i = 1 : iter
    output(:, i) =  exp(-0.5 * (x - (min_x + space * (i-1) * 1e-9)).^2 / sigma^2);
end

end

