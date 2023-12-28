#include "Neural_Network.h"
#include <godot_cpp/classes/file_access.hpp>

void NeuralNetwork::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("set_fitness", "_fitness"), &NeuralNetwork::set_fitness);
    ClassDB::bind_method(D_METHOD("get_fitness"), &NeuralNetwork::get_fitness);
    ClassDB::bind_method(D_METHOD("set_raycasts", "_raycasts"), &NeuralNetwork::set_raycasts);
    ClassDB::bind_method(D_METHOD("set_color", "_color"), &NeuralNetwork::set_color);
    ClassDB::bind_method(D_METHOD("get_color"), &NeuralNetwork::get_color);
    ClassDB::bind_method(D_METHOD("init", "_input_nodes", "_hidden_nodes", "_output_nodes"), &NeuralNetwork::init);
    ClassDB::bind_method(D_METHOD("set_nn_color"), &NeuralNetwork::set_nn_color);
    ClassDB::bind_method(D_METHOD("set_activation_function", "hidden_func", "output_func"), &NeuralNetwork::set_activation_function);
    ClassDB::bind_method(D_METHOD("predict", "input_array"), &NeuralNetwork::predict);
    ClassDB::bind_method(D_METHOD("train", "input_array", "target_array"), &NeuralNetwork::train);
    ClassDB::bind_method(D_METHOD("get_inputs_from_raycasts"), &NeuralNetwork::get_inputs_from_raycasts);
    ClassDB::bind_method(D_METHOD("get_prediction_from_raycasts", "optional_val"), &NeuralNetwork::get_prediction_from_raycasts);
    ClassDB::bind_method(D_METHOD("get_distance", "_raycast"), &NeuralNetwork::get_distance);
    ClassDB::bind_static_method("NeuralNetwork", D_METHOD("reproduce", "a", "b"), &NeuralNetwork::reproduce);
    ClassDB::bind_static_method("NeuralNetwork", D_METHOD("mutate", "nn", "callback"), &NeuralNetwork::mutate);
    ClassDB::bind_method(D_METHOD("mutate_callable_reproduced", "value", "row", "col"), &NeuralNetwork::mutate_callable_reproduced);
    ClassDB::bind_static_method("NeuralNetwork", D_METHOD("copy", "nn"), &NeuralNetwork::copy);
    ClassDB::bind_static_method("NeuralNetwork", D_METHOD("mutate_callable", "value", "row", "col"), &NeuralNetwork::mutate_callable);
    ClassDB::bind_method(D_METHOD("save", "path"), &NeuralNetwork::save);
    ClassDB::bind_static_method("NeuralNetwork", D_METHOD("load", "path"), &NeuralNetwork::load);
}

void NeuralNetwork::set_fitness(double _fitness)
{
    fitness = _fitness;
}

double NeuralNetwork::get_fitness()
{
    return fitness;
}

void NeuralNetwork::set_raycasts(const Array &_raycasts)
{
    raycasts = _raycasts.duplicate();
}

void NeuralNetwork::set_color(const Color &_color)
{
    color = _color;
}

Color NeuralNetwork::get_color()
{
    return color;
}

void NeuralNetwork::init(int _input_nodes, int _hidden_nodes, int _output_nodes)
{
    input_nodes = _input_nodes;
    hidden_nodes = _hidden_nodes;
    output_nodes = _output_nodes;

    weights_input_hidden->init(hidden_nodes, input_nodes);
    weights_hidden_output->init(output_nodes, hidden_nodes);
    weights_input_hidden->rand();
    weights_hidden_output->rand();

    bias_hidden->init(hidden_nodes, 1);
    bias_output->init(output_nodes, 1);
    bias_hidden->rand();
    bias_output->rand();

    set_activation_function(activation_functions["SIGMOID"], activation_functions["SIGMOID"]);
    set_nn_color();
}

void NeuralNetwork::set_nn_color()
{
    color = Color(Matrix::average(weights_input_hidden), Matrix::average(weights_hidden_output), Matrix::average(bias_hidden));
}

void NeuralNetwork::set_activation_function(Dictionary hidden_func, Dictionary output_func)
{
    hidden_activation = hidden_func;
    output_activation = output_func;
}

Array NeuralNetwork::predict(const Array &input_array)
{
    Ref<Matrix> inputs = Matrix::from_array(input_array);
    Ref<Matrix> hidden = Matrix::dot_product(weights_input_hidden, inputs);

    hidden = Matrix::add(hidden, bias_hidden);
    hidden = Matrix::map(hidden, hidden_activation["function"]);

    Ref<Matrix> output = Matrix::dot_product(weights_hidden_output, hidden);

    output = Matrix::add(output, bias_output);
    output = Matrix::map(output, output_activation["function"]);

    Array result = Matrix::to_array(output);

    return result;
}

void NeuralNetwork::train(const Array &input_array, const Array &target_array)
{
    Ref<Matrix> inputs = Matrix::from_array(input_array);
    Ref<Matrix> targets = Matrix::from_array(target_array);

    Ref<Matrix> hidden = Matrix::dot_product(weights_input_hidden, inputs);
    hidden = Matrix::add(hidden, bias_hidden);
    hidden = Matrix::map(hidden, hidden_activation["function"]);

    Ref<Matrix> outputs = Matrix::dot_product(weights_hidden_output, hidden);
    outputs = Matrix::add(outputs, bias_output);
    outputs = Matrix::map(outputs, output_activation["function"]);

    Array outputs_arr = Matrix::to_array(outputs);
    UtilityFunctions::print(outputs_arr);

    Ref<Matrix> output_errors = Matrix::subtract(targets, outputs);

    Ref<Matrix> gradients = Matrix::map(outputs, output_activation["derivative"]);
    gradients = Matrix::multiply(gradients, output_errors);
    gradients = Matrix::scalar(gradients, learning_rate);

    Ref<Matrix> hidden_T = Matrix::transpose(hidden);
    Ref<Matrix> weight_ho_deltas = Matrix::dot_product(gradients, hidden_T);

    weights_hidden_output = Matrix::add(weights_hidden_output, weight_ho_deltas);
    bias_output = Matrix::add(bias_output, gradients);

    Ref<Matrix> who_t = Matrix::transpose(weights_hidden_output);
    Ref<Matrix> hidden_errors = Matrix::multiply(who_t, output_errors);

    Ref<Matrix> hidden_gradient = Matrix::map(hidden, hidden_activation["derivative"]);

    hidden_gradient = Matrix::multiply(hidden_gradient, hidden_errors);
    hidden_gradient = Matrix::scalar(hidden_gradient, learning_rate);

    Ref<Matrix> inputs_T = Matrix::transpose(inputs);
    Ref<Matrix> weight_ih_deltas = Matrix::dot_product(hidden_gradient, inputs_T);

    weights_input_hidden = Matrix::add(weights_input_hidden, weight_ih_deltas);
    bias_hidden = Matrix::add(bias_hidden, hidden_gradient);
}

Array NeuralNetwork::get_inputs_from_raycasts()
{
    Array inputs;
    for (int i = 0; i < raycasts.size(); ++i)
    {
        RayCast2D *raycast = Object::cast_to<RayCast2D>(raycasts[i]);
        if (raycast)
        {
            inputs.append(get_distance(raycast));
        }
    }

    return inputs;
}

Array NeuralNetwork::get_prediction_from_raycasts(const Array &optional_val = Array())
{
    Array inputs = get_inputs_from_raycasts();
    if (optional_val.size() > 0)
    {
        inputs.append(optional_val);
    }
    return predict(inputs);
}

double NeuralNetwork::get_distance(RayCast2D *_raycast)
{
    if (_raycast->is_colliding())
    {
        return _raycast->get_collision_point().distance_to(_raycast->get_global_position());
    }
    else
    {
        return 1.0;
    }
}

Ref<NeuralNetwork> NeuralNetwork::reproduce(Ref<NeuralNetwork> a, Ref<NeuralNetwork> b)
{
    Ref<NeuralNetwork> child = memnew(NeuralNetwork);
    child->init(a->input_nodes, a->hidden_nodes, a->output_nodes);
    child->weights_input_hidden = Matrix::copy(a->weights_input_hidden);
    child->weights_hidden_output = Matrix::copy(a->weights_hidden_output);
    child->bias_hidden = Matrix::copy(a->bias_hidden);
    child->bias_output = Matrix::copy(a->bias_output);

    Ref<RandomNumberGenerator> rng = memnew(RandomNumberGenerator);
    rng->randomize();

    for (int i = 0; i < child->weights_input_hidden->get_rows(); ++i)
    {
        for (int j = 0; j < child->weights_input_hidden->get_cols(); ++j)
        {
            if (rng->randf() < 0.5)
            {
                child->weights_input_hidden->set_at(i, j, b->weights_input_hidden->get_at(i, j));
            }
        }
    }

    for (int i = 0; i < child->weights_hidden_output->get_rows(); ++i)
    {
        for (int j = 0; j < child->weights_hidden_output->get_cols(); ++j)
        {
            if (rng->randf() < 0.5)
            {
                child->weights_hidden_output->set_at(i, j, b->weights_hidden_output->get_at(i, j));
            }
        }
    }

    for (int i = 0; i < child->bias_hidden->get_rows(); ++i)
    {
        for (int j = 0; j < child->bias_hidden->get_cols(); ++j)
        {
            if (rng->randf() < 0.5)
            {
                child->bias_hidden->set_at(i, j, b->bias_hidden->get_at(i, j));
            }
        }
    }

    for (int i = 0; i < child->bias_output->get_rows(); ++i)
    {
        for (int j = 0; j < child->bias_output->get_cols(); ++j)
        {
            if (rng->randf() < 0.5)
            {
                child->bias_output->set_at(i, j, b->bias_output->get_at(i, j));
            }
        }
    }

    return child;
}

Ref<NeuralNetwork> NeuralNetwork::mutate(Ref<NeuralNetwork> nn, Callable callback)
{
    Ref<NeuralNetwork> child = memnew(NeuralNetwork);
    child->init(nn->input_nodes, nn->hidden_nodes, nn->output_nodes);
    child->weights_input_hidden = Matrix::copy(nn->weights_input_hidden);
    child->weights_hidden_output = Matrix::copy(nn->weights_hidden_output);
    child->bias_hidden = Matrix::copy(nn->bias_hidden);
    child->bias_output = Matrix::copy(nn->bias_output);

    child->weights_input_hidden = Matrix::map(child->weights_input_hidden, callback);
    child->weights_hidden_output = Matrix::map(child->weights_hidden_output, callback);
    child->bias_hidden = Matrix::map(child->bias_hidden, callback);
    child->bias_output = Matrix::map(child->bias_output, callback);

    return child;
}

double NeuralNetwork::mutate_callable_reproduced(double value, int row, int col)
{
    Ref<RandomNumberGenerator> rng = memnew(RandomNumberGenerator);
    rng->randomize();
    if (rng->randf() < 0.1)
    {
        return value + rng->randf_range(-0.1, 0.1);
    }
    else
    {
        return value;
    }
}

Ref<NeuralNetwork> NeuralNetwork::copy(Ref<NeuralNetwork> nn)
{
    Ref<NeuralNetwork> child = memnew(NeuralNetwork);
    child->init(nn->input_nodes, nn->hidden_nodes, nn->output_nodes);
    child->weights_input_hidden = Matrix::copy(nn->weights_input_hidden);
    child->weights_hidden_output = Matrix::copy(nn->weights_hidden_output);
    child->bias_hidden = Matrix::copy(nn->bias_hidden);
    child->bias_output = Matrix::copy(nn->bias_output);

    return child;
}

double NeuralNetwork::mutate_callable(double value, int row, int col)
{
    Ref<RandomNumberGenerator> rng = memnew(RandomNumberGenerator);
    rng->randomize();
    if (rng->randf() < 0.1)
    {
        return value + rng->randf_range(-0.1, 0.1);
    }
    else
    {
        return value;
    }
}

void NeuralNetwork::save(const String &path)
{
    Dictionary nn_data;
    nn_data["input_nodes"] = input_nodes;
    nn_data["hidden_nodes"] = hidden_nodes;
    nn_data["output_nodes"] = output_nodes;
    nn_data["weights_input_hidden"] = Matrix::to_array(weights_input_hidden);
    nn_data["weights_hidden_output"] = Matrix::to_array(weights_hidden_output);
    nn_data["bias_hidden"] = Matrix::to_array(bias_hidden);
    nn_data["bias_output"] = Matrix::to_array(bias_output);
    nn_data["hidden_activation"] = hidden_activation;
    nn_data["output_activation"] = output_activation;
    nn_data["learning_rate"] = learning_rate;
    nn_data["fitness"] = fitness;
    nn_data["color"] = color.to_html(false);

    Ref<FileAccess> file = FileAccess::open(path, FileAccess::WRITE);
    file->store_var(nn_data);
    file->close();
}

Ref<NeuralNetwork> NeuralNetwork::load(const String &path)
{
    Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
    Dictionary nn_data_dict = file->get_var();
    file->close();

    Ref<NeuralNetwork> nn = memnew(NeuralNetwork);
    nn->init(nn_data_dict["input_nodes"], nn_data_dict["hidden_nodes"], nn_data_dict["output_nodes"]);
    nn->weights_input_hidden = Matrix::from_array(nn_data_dict["weights_input_hidden"]);
    nn->weights_hidden_output = Matrix::from_array(nn_data_dict["weights_hidden_output"]);
    nn->bias_hidden = Matrix::from_array(nn_data_dict["bias_hidden"]);
    nn->bias_output = Matrix::from_array(nn_data_dict["bias_output"]);
    nn->hidden_activation = nn_data_dict["hidden_activation"];
    nn->output_activation = nn_data_dict["output_activation"];
    nn->learning_rate = nn_data_dict["learning_rate"];
    nn->fitness = nn_data_dict["fitness"];
    nn->color = Color(nn_data_dict["color"]);

    return nn;
}

NeuralNetwork::NeuralNetwork()
{

    activation = Ref<Activation>(memnew(Activation));
    activation->init_functions();
    weights_input_hidden = Ref<Matrix>(memnew(Matrix));
    weights_hidden_output = Ref<Matrix>(memnew(Matrix));
    bias_hidden = Ref<Matrix>(memnew(Matrix));
    bias_output = Ref<Matrix>(memnew(Matrix));

    activation_functions = activation->get_functions();

    input_nodes = 0;
    hidden_nodes = 0;
    output_nodes = 0;
    learning_rate = 0.1;
    fitness = 0;
    color = Color(1, 1, 1);
}

NeuralNetwork::~NeuralNetwork()
{
    activation_functions.clear();
    raycasts.clear();
}