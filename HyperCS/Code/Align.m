%% Align.m
%  Sub-image cropping and registration of the spatially reconstructed frames.
%
%  Nine sub-images are located within each reconstructed full frame by
%  normalized cross-correlation (normxcorr2) against a manually selected
%  template ROI. Detected peaks are fitted to a regular 3x3 grid, refined
%  to sub-pixel accuracy by quadratic interpolation of the correlation map,
%  cropped at floating-point positions, and further co-registered to the
%  first sub-image by translation-only registration. The cropped patches
%  are re-indexed to the calibrated filter order and stacked as H x W x 9.
%
%  Paths are anchored to this file's location, so the script can be run
%  from any current folder (after Spatial_Recovery.py):
%      >> addpath(fullfile('HyperCS','Code')); Align
%
%  Input : Results/<sample>_recon.mat  (variable CR4, spatial recon output)
%  Output: Results/<sample>_Imgs.mat   (variable Imgs, H x W x 9)

clear; close all; clc

% Repository root (parent of Code/), independent of the current folder
repoRoot   = fileparts(fileparts(mfilename('fullpath')));
resultsDir = fullfile(repoRoot, 'Results');

%% ----------------------------------------------------------- parameters --
% Samples to process and their template ROI corners [x y]
% (manually selected on each reconstructed frame)
samples(1).name   = 'letter_CS_CR4';
samples(1).Points = [116,   1;  147,   1;
                     116,  32;  147,  32];

samples(2).name   = 'letter_CI_CR4';
samples(2).Points = [116,   1;  147,   1;
                     116,  32;  147,  32];

inputVar = 'CR4';

% Save order: re-index the detected 3x3 grid to the calibrated filter order
saveOrder = [7 8 9 4 5 6 1 2 3];

useGradient = false;   % true: edge-based matching (for strong band intensity differences)
refineToRef = true;    % iterative translation refinement to sub-image 1
nIter       = 3;       % refinement iterations

%% -------------------------------------------------------- process loop ---
for si = 1:numel(samples)
    name   = samples(si).name;
    Points = samples(si).Points;

    inputFile  = fullfile(resultsDir, [name '_recon.mat']);
    outputFile = fullfile(resultsDir, [name '_Imgs.mat']);
    fprintf('\n=== %s ===\n', name);

    S   = load(inputFile);
    Img = double(S.(inputVar));

    xs = Points(:,1);  ys = Points(:,2);
    x0 = min(xs);      y0 = min(ys);
    w  = max(xs) - x0 + 1;
    h  = max(ys) - y0 + 1;

    template = Img(y0:(y0+h-1), x0:(x0+w-1));
    [tH, tW] = size(template);

    % -------------- 9-ROI detection: NMS + 3x3 grid fit + sub-pixel -------
    if useGradient
        Imgm = imgradient(Img);  tmpl = imgradient(template);
    else
        Imgm = Img;              tmpl = template;
    end

    C       = normxcorr2(tmpl, Imgm);
    C_valid = C(tH:end-tH+1, tW:end-tW+1);   % (r,c) == ROI top-left (y,x)
    C_valid(isnan(C_valid)) = -inf;

    % 1) Non-maximum suppression: extract separated candidate peaks
    supR = round(min(tH,tW)*0.8);
    Cw   = C_valid;  cand = [];
    for i = 1:10
        [mv, mi] = max(Cw(:));
        if mv < 0.3, break; end
        [py, px] = ind2sub(size(Cw), mi);
        cand = [cand; px, py, mv];           %#ok<AGROW>
        y1 = max(1,py-supR); y2 = min(size(Cw,1),py+supR);
        x1 = max(1,px-supR); x2 = min(size(Cw,2),px+supR);
        Cw(y1:y2, x1:x2) = -inf;
    end

    % 2) Fit candidates to a regular 3x3 grid
    colX = regularAxis(cluster1d(cand(:,1), 3));
    rowY = regularAxis(cluster1d(cand(:,2), 3));

    % 3) Per-cell peak search + sub-pixel refinement
    R = round(min(tH,tW)*0.4);
    gridXY = zeros(9,2);  gi = 1;
    for r = 1:3
      for c = 1:3
        ex = colX(c);  ey = rowY(r);
        x1 = max(1,round(ex-R)); x2 = min(size(C_valid,2),round(ex+R));
        y1 = max(1,round(ey-R)); y2 = min(size(C_valid,1),round(ey+R));
        sub = C_valid(y1:y2, x1:x2);
        [mv, mi] = max(sub(:));
        if mv > 0.5
            [ly, lx] = ind2sub(size(sub), mi);
            gx = x1+lx-1;  gy = y1+ly-1;
            [ddx, ddy] = subpix(C_valid, gx, gy);
            gridXY(gi,:) = [gx+ddx, gy+ddy];   % floating-point top-left
        else
            gridXY(gi,:) = [ex, ey];
        end
        gi = gi + 1;
      end
    end

    % ---------- sub-pixel crop + translation refinement to sub-image 1 ----
    tlFloat = gridXY(saveOrder, :);

    [Xg, Yg] = meshgrid(0:w-1, 0:h-1);
    maxX = size(Img,2) - w + 1;
    maxY = size(Img,1) - h + 1;
    cropAt = @(tl) interp2(Img, ...
        Xg + min(max(tl(1),1),maxX), ...
        Yg + min(max(tl(2),1),maxY), 'cubic', 0);

    Imgs = zeros(h, w, 9);
    for k = 1:9
        Imgs(:,:,k) = cropAt(tlFloat(k,:));
    end

    if refineToRef
        ref = Imgs(:,:,1);
        for it = 1:nIter
            for k = 2:9
                tf = imregcorr(mat2gray(Imgs(:,:,k)), mat2gray(ref), 'translation');
                tlFloat(k,:) = tlFloat(k,:) - tf.Translation;
                Imgs(:,:,k)  = cropAt(tlFloat(k,:));   % single interpolation from the source
            end
        end
    end

    % ---------------------------------------------------- visualization ---
    figure('Name', name); imshow(Img, []); hold on;
    for k = 1:9
        rectangle('Position', [round(gridXY(k,1)), round(gridXY(k,2)), w, h], ...
                  'EdgeColor', 'r', 'LineWidth', 1);
    end
    title(sprintf('%s: detected 9 sub-image ROIs', name), 'Interpreter', 'none');
    hold off;

    figure('Name', [name ' sum']); imshow(mat2gray(sum(Imgs, 3)));
    title(sprintf('%s: sum of the 9 aligned sub-images', name), 'Interpreter', 'none');

    % -------------------------------------------------------------- save --
    if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
    save(outputFile, 'Imgs');
    fprintf('Saved: %s  (size = [%d x %d x %d])\n', outputFile, size(Imgs));
end

%% -------------------------------------------------------------- helpers --
function ctr = cluster1d(v, k)
    v = sort(v(:));
    if numel(v) <= k
        ctr = v; ctr(end+1:k) = v(end); ctr = sort(ctr(1:k)); return;
    end
    d = diff(v);  [~, g] = maxk(d, k-1);  g = sort(g);
    e = [0; g; numel(v)];  ctr = zeros(k,1);
    for i = 1:k, ctr(i) = median(v(e(i)+1:e(i+1))); end
end

function a = regularAxis(c)
    c = sort(c(:));
    if numel(c) >= 3
        p = (c(3)-c(1))/2;  a = [c(1); c(1)+p; c(1)+2*p];
    else
        a = c;
    end
end

function [dx,dy] = subpix(C, px, py)
    dx = 0; dy = 0; [H,W] = size(C);
    if px>1 && px<W
        l = C(py,px-1); c = C(py,px); r = C(py,px+1); d = l-2*c+r;
        if d~=0, dx = 0.5*(l-r)/d; end
    end
    if py>1 && py<H
        u = C(py-1,px); c = C(py,px); dn = C(py+1,px); d = u-2*c+dn;
        if d~=0, dy = 0.5*(u-dn)/d; end
    end
    dx = max(min(dx,0.5),-0.5);  dy = max(min(dy,0.5),-0.5);
end
