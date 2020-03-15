function [ cost grad ] = NNBackPropagation(nn_weights, X, y, num_layers, ...
                                           num_data_samples, num_units, ...
                                           activation_function_type, lambda)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

weights_array = Vec2CellArray(nn_weights,num_layers,num_units);

activation = ForwardPropagation(weights_array, num_layers,...
                                   num_data_samples, num_units, ...
                                   activation_function_type, X);
                               
hypothesis = activation{num_layers}
output_error = hypothesis' - y'

cost = ComputeCost(output_error, weights_array, ...
                                num_data_samples, num_layers, lambda);


activation_error(num_layers-1) = {output_error}

grad = cellfun(@(x) x*0,weights_array,'un',0)

for t = 1:num_data_samples
    
    activation_error = {output_error(t)}
    grad{num_layers-1} = grad{num_layers-1} + ...
                         activation_error{num_layers-1}(:)'*activation{num_layers-1}(:)'
    
    activation_error{num_layers-2} = activation_error{num_layers-1}(:,t)*weights_array{num_layers-1}'
    %for layer = num_layers-2,-1,1
        
       % if strcmp(activation_function_type{layer},'sigmoid')
   %     activation_error{layer} = activation_error{layer+1}*weights_array{layer}...
      %                          .*activation{layer-1}.*(1-activation{layer-1});
       % delta2 = delta3'*Theta2.*a2(:,i)'.*(1-a2(:,i))';
        
   % end
    
end

end

