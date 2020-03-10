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
y = data(:,1)
X = [data(:,2) data(:,3) data(:,4)]

% user should modify the below ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%%%%

% number of hidden layers
num_hidden_layers = 1;

% number of units in each hidden layer
%num_hidden_units = [ 20 10 ];
num_hidden_units = [3]

% activation function types
activation_function_type = {'linear', 'linear'};

feature_scaling_tf = false;

%%%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%%%%

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
epsilon_init = 0.12

% create cell array of weights, accessed by curly brackets 'weights_array{layer}', 
% individual elements accessed by 'weights_array{layer}(i,j)'
for layer = 1:num_hidden_layers+1
    
    % num_units(layer)+1 <=== +1 comes from bias term 
    weights_array(layer) = ...
    {rand(num_units(layer)+1,num_units(layer+1)) * 2 * epsilon_init - epsilon_init}
    
    % for testing:
    %weights_array(layer) = {ones(num_units(layer),num_units(layer+1))}
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
    
    activation(1) = {[ones(num_data_samples,1) X]}
end

% forward propagation
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
    
    if layer+1 == num_layers
        activation(layer+1) = {activation_next_layer}
    else
       % Fill cell of node activation arrays using forward propogation
       % Array of ones for bias term
       activation(layer+1) = {[ones(num_data_samples,1) activation_next_layer]}
    end

    % initialise cell array containing node activation errors for each layer
    % except input features. No bias term included for activation error
    activation_error(layer) = {zeros(num_data_samples,num_units(layer+1))}
    
end

fprintf('predictions for %d data points are: \n',num_data_samples)
hypothesis = activation{num_layers}

% don't consider input data to have any activation error
activation_error{num_layers-1} = activation{num_layers} - y

%%
for layer = num_layers-1:-1:2
    
    layer
    if strcmp(activation_function_type{layer},'sigmoid')
    exit
    elseif strcmp(activation_function_type{layer},'linear')
    activation_error()
    end
    
end
