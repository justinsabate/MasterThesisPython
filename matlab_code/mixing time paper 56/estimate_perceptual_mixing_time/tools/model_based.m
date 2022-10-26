% Computes the perceptual mixing time from model based predictors as
% described in:
%
% Lindau, A.; Kosanke, L.; Weinzierl, S.: 
% "Perceptual evaluation of model- and signal-based predictors of the 
% mixing time in binaural room impulse responses", In:  J. A. Eng. Soc.
% 
% 
% tmp50    - average perceptual mixing time
% tmp95    - 95%-point perceptual mixing time
%
% h         - height
% b         - width
% l         - length
%
% call:
% [tmp50, tmp95] = model_based(h,b,l)
%
% A. Lindau, L. Kosanke, 2011
% alexander.lindau@tu-berlin.de
% audio communication group
% Technical University of Berlin
%-------------------------------------------------------------------------%
function [tmp50, tmp95] = model_based(h,b,l)

% calculate room properties
V       = h * l * b;                % volume
S       = 2*l*b + 2*b*h + 2*l*h;    % surface area
    
fprintf('\nVolume: %5.2f m³.',V)
fprintf('\nSurface area: %5.2f m².\n',S)

% predict tmp from linear models 
tmp50  = 20.08 * V/S + 12;             
tmp95  = 0.0117 * V + 50.1;               




