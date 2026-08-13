%% Spectral_Recovery.m
%  Pixel-wise spectral reconstruction of the aligned 9-band sub-image stacks.
%
%  For each spatial pixel, the 9 filtered intensities are decoded into a
%  250-band spectrum (500-749 nm, 1 nm spacing) by solving a non-negative
%  l1-regularized least-squares problem (l1_ls_nonneg) on a Gaussian
%  sparsifying dictionary. The sensing matrix combines the measured
%  transmittance spectra of the nine selected filters (F9 configuration)
%  with the quantum efficiency of the sensor.
%
%  Paths are anchored to this file's location, so the script can be run
%  from any current folder (after Spatial_Recovery.py and Align.m):
%      >> addpath(fullfile('HyperCS','Code')); Spectral_Recovery
%
%  Input : Results/<sample>_Imgs.mat  (variable Imgs, H x W x 9)
%          Data/Trr.mat, Data/QE.mat
%  Output: Results/<sample>_recover.mat
%          recover     : H x W x 250  (500-749 nm)
%          recover_230 : H x W x 230  (505-734 nm, reported band range)
%
%  The l1_ls solver was developed by Koh, Kim, and Boyd
%  (https://web.stanford.edu/~boyd/l1_ls/).

close all; clc; clear;

% Repository root (parent of Code/), independent of the current folder
repoRoot   = fileparts(fileparts(mfilename('fullpath')));
dataDir    = fullfile(repoRoot, 'Data');
resultsDir = fullfile(repoRoot, 'Results');

%% ----------------------------------------------------------- parameters --
samples = {'letter_CS_CR4', 'letter_CI_CR4'};

lambda  = 0.2;    % l1 regularization parameter
rel_tol = 1e-2;   % relative target duality gap

% Indices of the nine selected filters (F9 configuration) in Trr
idxTrr = [14, 15, 16, ...
          20, 21, 22, ...
          26, 27, 28];

% The interior-point solver occasionally tightens the PCG tolerance beyond
% machine precision; the resulting warning is harmless and suppressed here.
warning('off', 'MATLAB:pcg:tooSmallTolerance');

%% ----------------------------------------------- sensing matrix and dict --
load(fullfile(dataDir, 'Trr.mat'));
load(fullfile(dataDir, 'QE.mat'));

A = Trr(idxTrr, 1:250);     % filter transmittance (9 x 250)
B = A .* QE(1:250);         % include sensor quantum efficiency

% Gaussian sparsifying dictionary (multi-FWHM blocks over 500-749 nm)
G1 = fwhmg(30, 5,  500, 749);
G2 = fwhmg(40, 5,  500, 749);
G3 = fwhmg(50, 10, 500, 749);
G4 = fwhmg(55, 10, 500, 749);
G5 = fwhmg(60, 10, 500, 749);
G6 = fwhmg(65, 10, 500, 749);
G7 = fwhmg(70, 10, 500, 749);
G8 = fwhmg(75, 10, 500, 749);
GG = [G1, G2, G3, G4, G5, G6, G7, G8];

C = B * GG;                 % sensing matrix in the sparse domain

%% -------------------------------------------- pixel-wise reconstruction --
for si = 1:numel(samples)
    name = samples{si};
    inputFile  = fullfile(resultsDir, [name '_Imgs.mat']);
    outputFile = fullfile(resultsDir, [name '_recover.mat']);
    fprintf('\n=== %s ===\n', name);

    load(inputFile);            % Imgs (H x W x 9)
    Imgs = uint8(Imgs);

    [h, w, ~] = size(Imgs);
    recover = zeros(h, w, 250);
    total   = h * w;

    fprintf('Spectral reconstruction: %d pixels\n', total);
    tic;
    for idx = 1:total
        [x, y] = ind2sub([h, w], idx);
        y9 = im2double(squeeze(Imgs(x, y, :)));
        [s, ~] = l1_ls_nonneg(C, y9, lambda, rel_tol, true);
        recover(x, y, :) = GG * s;

        if mod(idx, max(1, floor(total/100))) == 0 || idx == total
            fprintf('\rProgress: %6.2f%%', 100 * idx / total);
        end
    end
    fprintf('\n');
    toc;

    % ------------------------------------------------------------- save --
    recover_230 = recover(:, :, 6:235);   % 505-734 nm (reported band range)

    if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
    save(outputFile, 'recover', 'recover_230');
    fprintf('Saved: %s\n', outputFile);
end

warning('on', 'MATLAB:pcg:tooSmallTolerance');
