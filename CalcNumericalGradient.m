function [ numerical_grad ] = CalcNumericalGradient(weights_array, num_layers,...
                                   num_data_samples, num_units, ...
                                   activation_function_type, X, y, lambda)
%CALCNUMERICALGRADIENT calculates gradient numerically...

% unrolled_weights = []
% for layer = 1:num-layers-1
%     unrolled_weights = [unrolled_weights weights_array]
% end
% 
% % Number of elements in cell array
% sum(cellfun(@numel,weights_array));

delta = 1.0e-4

for layer = 1:num_layers-1
    
    for i = 1:size(weights_array{layer},1)
        
        for j = 1:size(weights_array{layer},2)

            weights_array_pl = weights_array;
            weights_array_pl{layer}(i,j) = weights_array_pl{layer}(i,j) + delta

            [activation] = ForwardPropagation(weights_array_pl, num_layers, num_data_samples, num_units, ...
                                       activation_function_type, X)                             
            error = activation{num_layers}' - y';

            J_pl = ComputeCost(error, weights_array_pl, num_data_samples, num_layers, lambda)

            weights_array_mi = weights_array;
            weights_array_mi{layer}(i,j) = weights_array{layer}(i,j) - delta;

            [activation] = ForwardPropagation(weights_array_mi, num_layers, num_data_samples, num_units, ...
                                       activation_function_type, X)                             
            error = activation{num_layers}' - y';

            J_mi = ComputeCost(error, weights_array_mi, num_data_samples, num_layers, lambda)
            
            numerical_grad{layer}(i,j) = (J_pl - J_mi) / (2*delta);
        
        end
        
    end
                                
end

end                              