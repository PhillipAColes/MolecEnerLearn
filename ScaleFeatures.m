function [ X_scaled ] = ScaleFeatures( X )
%SCALEFEATURES Scales features vectors using mean normalisation
    X_av = mean(X);
    X_av_mat = repmat(X_av,size(X,1),1);
    X_range = range(X);
    X_range_mat = repmat(X_range,size(X,1),1);
    X_scaled = (X - X_av_mat) ./ X_range_mat;
end

