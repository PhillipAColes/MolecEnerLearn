clear all;  clc
cd C:\Users\Phillip\Workspace\ML\RovibEner

fprintf('Loading data ...\n');
data = load('31P-1H3__SAlTY_100000n_nu0.states');
%data = load('31P-1H3__SAlTY_100000n.states');
fprintf('... done\n');

%SAlTY term values
y = data(:,2);

%rigid rotor terms
J = data(:,4);
K = data(:,7);
J2 = J.^2;
K2 = K.^2;
K4 = K.^4;
JJ1 = J.*(J+1);
JJ2 = J2.*(J+1).^2;
JK = J.*(J+1).*K2;

X = [ JJ1 JJ2 K2 K4 JK ];%rigid rotor for only J and K

%SAlTY quantum numbers [v1 v2 v3 v4 l3 l4 l J K]
%X = [data(:,10:15) data(:,9) data(:,4) data(:,7) JJ1 JJ2 JK K4 ];

%%
clear all;  clc
cd C:\Users\Phillip\Workspace\ML\RovibEner
data = load('backproptest-1');
y = data(:,1);
X = [data(:,2) data(:,3) data(:,4)];

%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%%
%%%% user should modify the below %%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%%%%
%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%%

% number of hidden layers
num_hidden_layers = 3;

% number of units in each hidden layer
num_hidden_units = [3 5 3]

% activation function types
activation_function_type = {'linear', 'linear', 'linear', 'linear'};

feature_scaling_tf = false;

% regularisation parameter
lambda = 1

%%%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%%%%

%%% simple test case performed with pen and paper ~%%
% y = 10;
% X = [1 1]
% num_hidden_layers = 1
% num_hidden_units = [ 2 ];
% activation_function_type = {'linear', 'linear'};
% feature_scaling_tf = false;
% lambda = 1
% weights array set to arrays of ones
%%%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%%


% number of data samples
num_data_samples = length(y);

% number of units in each layer
num_units = [ size(X,2) num_hidden_units 1 ];

% total number of layers
num_layers = num_hidden_layers + 2

% preliminary checks for user input parameters
if size(num_hidden_units,2) ~= num_hidden_layers
    error(['Error: array containing number of hidden units in each layer'...
    ' needs to reflect number of hidden layers']);
elseif size(activation_function_type,2) ~= num_hidden_layers+1;
     error(['Error: need to define exactly ',num2str(num_hidden_layers+1),...
    ' activation functions']);       
end

% constant used for initialising weights
epsilon_init = 0.12;

% Initialise weights array
for layer = 1:num_hidden_layers+1
    
    % num_units(layer)+1 <=== +1 comes from bias term 
    weights_array(layer) = ...
    {rand(num_units(layer)+1,num_units(layer+1)) * 2 * epsilon_init - epsilon_init};
    
    % for testing:
    %weights_array(layer) = {ones(num_units(layer)+1,num_units(layer+1))}
end

if feature_scaling_tf == true
    % scale features
    X_av = mean(X);
    X_av_mat = repmat(X_av,size(X,1),1);
    X_range = range(X);
    X_range_mat = repmat(X_range,size(X,1),1);
    X_scaled = (X - X_av_mat) ./ X_range_mat;

    % activation of first layer is just the input data (features)
    activation(1) = {[ones(num_data_samples,1) X_scaled]};
else
    
    activation(1) = {[ones(num_data_samples,1) X]};
end

%%~~~~~~~~~~~~~~~~~~~~~~%%
%%% Forward propagation %%
%%~~~~~~~~~~~~~~~~~~~~~~%%

for layer = 1:num_layers-1
layer
    if strcmp(activation_function_type{layer},'sigmoid')
        fprintf('layer %d uses sigmoid activation function \n',layer)
        z = activation{layer}*weights_array{layer};
        activation_next_layer = 1 ./ (1+exp(-z));
    elseif strcmp(activation_function_type{layer},'linear')
        fprintf('layer %d uses linear activation function \n',layer)
        activation_next_layer = activation{layer}*weights_array{layer};
    end
    
    % Fill cell of node activation arrays using forward propogation
    if layer+1 == num_layers
        % No bias term added for output layer
        activation(layer+1) = {activation_next_layer};
    else
       % Array of ones for bias term
       activation(layer+1) = {[ones(num_data_samples,1) activation_next_layer]};
    end

    % initialise cell array containing node activation errors for each layer
    % except input layer
    if layer == num_layers-1
        % No bias term in final (output) layer        
        activation_error(layer) = {zeros(num_units(layer+1),num_data_samples)};
    else
        
        activation_error(layer) = {zeros(num_units(layer+1)+1,num_data_samples)};
    end
    
end

%fprintf('predictions for %d data points are: \n',num_data_samples);
hypothesis = activation{num_layers}

%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%%
%%%% Now to determine the activation error of each node %%
%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%%
   
% Activation error of final (output) layer
activation_error{num_layers-1} = activation{num_layers}' - y';

cost = 1/(2*num_data_samples)*activation_error{num_layers-1}*activation_error{num_layers-1}';

% Activation error of the final hidden layer
if strcmp(activation_function_type{num_layers-1},'sigmoid')
    activation_error{num_layers-2} = weights_array{num_layers-1}*activation_error{num_layers-1}...
                                     .*activation{num_layers-1}'.*(1-activation{num_layers-1})';                            
elseif strcmp(activation_function_type{num_layers-1},'linear')    
    activation_error{num_layers-2} = weights_array{num_layers-1}*activation_error{num_layers-1};
end

% Activation error for all other hidden layers
for layer = num_layers-2:-1:2
    
    layer;
    if strcmp(activation_function_type{layer},'sigmoid')
        activation_error{layer-1} = weights_array{layer}*activation_error{layer}(2:end,:)...
                                    .*activation{layer}'.*(1-activation{layer})';
    elseif strcmp(activation_function_type{layer},'linear')
        activation_error{layer-1} = weights_array{layer}*activation_error{layer}(2:end,:);
    end
    
end

%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%%
%%%% Now to calculate regularized gradient %%
%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%%

unroll_grad = [];

% Now we accumulate the gradient of the cost w.r.t the weights by looping 
% over each training example and summing their respective contributions
% Currently using for-loops, vectorise inner loop later.
for layer = 1:num_layers-1 
    
   % initialise the error accumulator to zero 
   sum_grad(layer) =  {zeros(num_units(layer+1),num_units(layer)+1)};
   
   % For each training example
   for t = 1:num_data_samples
       
       if layer == num_layers-1
           % Final layer (output) contains no bias node
           sum_grad{layer} = sum_grad{layer} + activation_error{layer}(:,t)*activation{layer}(t,:);
       else
           % Bias node in layer 'L' not connected to nodes in layer (L-1)
           sum_grad{layer} = sum_grad{layer} + activation_error{layer}(2:end,t)*activation{layer}(t,:);
       end
       
   end
   
   % Now to calculate regularized gradient. Weights connected to bias 
   % nodes are not regularized.
   grad_with_reg{layer} = (1/num_data_samples) .* (sum_grad{layer} + ...
       [ zeros(1,size(weights_array{layer},2)) ; lambda.*weights_array{layer}(2:end,:) ]');
   
   
   unroll_grad = [unroll_grad ; grad_with_reg{layer}(:)];
   
end
