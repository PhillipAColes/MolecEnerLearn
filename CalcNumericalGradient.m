function [ numerical_grad ] = CalcNumericalGradient(weights_array, num_layers,...
                                   num_data_samples, num_units, ...
                                   activation_function_type, X, y, lambda)
%CALCNUMERICALGRADIENT calculates gradient numerically...

% Small perturbation to be added to each weight. 0.0001 is sufficiently
% small for an accurate approximation of the gradient, too small and we get
% numerical problems
delta = 1.0e-4;

% Approximate the derivative of each weight
for layer = 1:num_layers-1
    
    for i = 1:size(weights_array{layer},1)
        
        for j = 1:size(weights_array{layer},2)
            
            % Add a small perturbation to the weight
            weights_array_pl = weights_array;
            weights_array_pl{layer}(i,j) = weights_array_pl{layer}(i,j) + delta;
            
            % Perform forward propagation to get the cost
            activation = ForwardPropagation(weights_array_pl, num_layers, num_data_samples, num_units, ...
                                       activation_function_type, X);
            % Error on our hypothesis                       
            error = activation{num_layers}' - y';
            
            % The bias terms are not regularised
            if( i == 1 )
            J_pl = ComputeCost(error, weights_array_pl, num_data_samples, num_layers, 0);
            else
            J_pl = ComputeCost(error, weights_array_pl, num_data_samples, num_layers, lambda);
            end

            weights_array_mi = weights_array;
            weights_array_mi{layer}(i,j) = weights_array_mi{layer}(i,j) - delta;

            [activation] = ForwardPropagation(weights_array_mi, num_layers, num_data_samples, num_units, ...
                                       activation_function_type, X);                          
            error = activation{num_layers}' - y';
            
            if( i == 1 )
            J_mi = ComputeCost(error, weights_array_mi, num_data_samples, num_layers, 0);
            else
            J_mi = ComputeCost(error, weights_array_mi, num_data_samples, num_layers, lambda);
            end
            
            numerical_grad{layer}(i,j) = (J_pl - J_mi) / (2*delta);
        
        end
        
    end
                                
end

end                              