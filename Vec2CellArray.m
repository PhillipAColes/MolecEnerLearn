function [ reshaped_weights ] = Vec2CellArray(nn_weights, num_layers, num_units)
%VEC2CELLARRAY reshapes the unrolled NN weights (nn_weights) back into 
%              their original matrix form as given by weights_array in the
%              main program

for i = 1:num_layers-1
    
    num_weights = (num_units(1:end-1)+1).*(num_units(2:end));
    offset = sum(num_weights(1:i)) - num_weights(i);
    reshaped_weights(i) = { reshape( nn_weights(offset+1:offset+num_weights(i)) ...
                              , [num_units(i)+1 , num_units(i+1)] )};
    
end

end

