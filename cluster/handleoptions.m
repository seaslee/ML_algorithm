function s = handleoptions(s, options)
% handle the list of option parameters in the function
% Parameters:
%   -s: structure to contain the option parameters
%   -options: option parameter list of the function
% Return:
%   -s: the updated structure of the option parameters
% format: "name",value,"name",value

names = options(1:2:end);
values = options(2:2:end);

n_name = length(names);
n_values = length(values);

if n_name ~= n_values,
    error('invalid option parameters');
end

for n=1:n_name,
    name = names{n};
    val = values{n};

    if isfield(s, name),
        s.(name) = val;
    else
        error('invalid option parameters', 'invalid parameter name', name);
    end
end

end

