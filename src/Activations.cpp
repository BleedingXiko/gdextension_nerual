#include "Activations.h"
#include <godot_cpp/core/math.hpp>

using namespace godot;

void Activation::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("get_functions"), &Activation::get_functions);
    ClassDB::bind_method(D_METHOD("init_functions"), &Activation::init_functions);
    ClassDB::bind_method(D_METHOD("sigmoid", "value", "_row", "_col"), &Activation::sigmoid);
    ClassDB::bind_method(D_METHOD("dsigmoid", "value", "_row", "_col"), &Activation::dsigmoid);
    ClassDB::bind_method(D_METHOD("relu", "value", "_row", "_col"), &Activation::relu);
    ClassDB::bind_method(D_METHOD("drelu", "value", "_row", "_col"), &Activation::drelu);
    ClassDB::bind_method(D_METHOD("tanh_", "value", "_row", "_col"), &Activation::tanh_);
    ClassDB::bind_method(D_METHOD("dtanh", "value", "_row", "_col"), &Activation::dtanh);
    ClassDB::bind_method(D_METHOD("arcTan", "value", "_row", "_col"), &Activation::arcTan);
    ClassDB::bind_method(D_METHOD("darcTan", "value", "_row", "_col"), &Activation::darcTan);
    ClassDB::bind_method(D_METHOD("prelu", "value", "_row", "_col"), &Activation::prelu);
    ClassDB::bind_method(D_METHOD("dprelu", "value", "_row", "_col"), &Activation::dprelu);
    ClassDB::bind_method(D_METHOD("elu", "value", "_row", "_col"), &Activation::elu);
    ClassDB::bind_method(D_METHOD("delu", "value", "_row", "_col"), &Activation::delu);
    ClassDB::bind_method(D_METHOD("softplus", "value", "_row", "_col"), &Activation::softplus);
    ClassDB::bind_method(D_METHOD("dsoftplus", "value", "_row", "_col"), &Activation::dsoftplus);
    ClassDB::bind_method(D_METHOD("swish", "value", "_row", "_col", "beta"), &Activation::swish);
    ClassDB::bind_method(D_METHOD("dswish", "value", "_row", "_col", "beta"), &Activation::dswish);
    ClassDB::bind_method(D_METHOD("mish", "value", "_row", "_col"), &Activation::mish);
    ClassDB::bind_method(D_METHOD("dmish", "value", "_row", "_col"), &Activation::dmish);
    ClassDB::bind_method(D_METHOD("linear", "value", "_row", "_col"), &Activation::linear);
    ClassDB::bind_method(D_METHOD("dlinear", "value", "_row", "_col"), &Activation::dlinear);
}

Activation::Activation()
{
    init_functions();
}

Activation::~Activation()
{
    functions.clear();
}

Dictionary Activation::get_functions(){
    return functions;
}

void Activation::init_functions()
{
    functions = Dictionary();

    // SIGMOID
    Dictionary sigmoid_dict;
    sigmoid_dict["function"] = Callable(this, "sigmoid");
    sigmoid_dict["derivative"] = Callable(this, "dsigmoid");
    sigmoid_dict["name"] = "SIGMOID";
    functions["SIGMOID"] = sigmoid_dict;

    // RELU
    Dictionary relu_dict;
    relu_dict["function"] = Callable(this, "relu");
    relu_dict["derivative"] = Callable(this, "drelu");
    relu_dict["name"] = "RELU";
    functions["RELU"] = relu_dict;

    // TANH
    Dictionary tanh_dict;
    tanh_dict["function"] = Callable(this, "tanh_");
    tanh_dict["derivative"] = Callable(this, "dtanh");
    tanh_dict["name"] = "TANH";
    functions["TANH"] = tanh_dict;

    // ARCTAN
    Dictionary arctan_dict;
    arctan_dict["function"] = Callable(this, "arcTan");
    arctan_dict["derivative"] = Callable(this, "darcTan");
    arctan_dict["name"] = "ARCTAN";
    functions["ARCTAN"] = arctan_dict;

    // PRELU
    Dictionary prelu_dict;
    prelu_dict["function"] = Callable(this, "prelu");
    prelu_dict["derivative"] = Callable(this, "dprelu");
    prelu_dict["name"] = "PRELU";
    functions["PRELU"] = prelu_dict;

    // ELU
    Dictionary elu_dict;
    elu_dict["function"] = Callable(this, "elu");
    elu_dict["derivative"] = Callable(this, "delu");
    elu_dict["name"] = "ELU";
    functions["ELU"] = elu_dict;

    // SOFTPLUS
    Dictionary softplus_dict;
    softplus_dict["function"] = Callable(this, "softplus");
    softplus_dict["derivative"] = Callable(this, "dsoftplus");
    softplus_dict["name"] = "SOFTPLUS";
    functions["SOFTPLUS"] = softplus_dict;

    // SWISH
    Dictionary swish_dict;
    swish_dict["function"] = Callable(this, "swish");
    swish_dict["derivative"] = Callable(this, "dswish");
    swish_dict["name"] = "SWISH";
    functions["SWISH"] = swish_dict;

    // MISH
    Dictionary mish_dict;
    mish_dict["function"] = Callable(this, "mish");
    mish_dict["derivative"] = Callable(this, "dmish");
    mish_dict["name"] = "MISH";
    functions["MISH"] = mish_dict;

    // LINEAR
    Dictionary linear_dict;
    linear_dict["function"] = Callable(this, "linear");
    linear_dict["derivative"] = Callable(this, "dlinear");
    linear_dict["name"] = "LINEAR";
    functions["LINEAR"] = linear_dict;
}


// Implementation of activation functions
double Activation::sigmoid(double value, int _row, int _col)
{
    return 1.0 / (1.0 + Math::exp(-value));
}

double Activation::dsigmoid(double value, int _row, int _col)
{
    return value * (1.0 - value);
}

double Activation::relu(double value, int _row, int _col)
{
    return Math::max(0.0, value);
}

double Activation::drelu(double value, int _row, int _col)
{
    return (value < 0) ? 0.0 : 1.0;
}

double Activation::tanh_(double value, int _row, int _col)
{
    return Math::tanh(value);
}

double Activation::dtanh(double value, int _row, int _col)
{
    return 1.0 - Math::pow(Math::tanh(value), 2);
}

double Activation::arcTan(double value, int _row, int _col)
{
    return Math::atan(value);
}

double Activation::darcTan(double value, int _row, int _col)
{
    return 1.0 / (1.0 + Math::pow(value, 2));
}

double Activation::prelu(double value, int _row, int _col)
{
    double alpha = 0.1;
    return (value < 0) ? alpha * value : value;
}

double Activation::dprelu(double value, int _row, int _col)
{
    double alpha = 0.1;
    return (value < 0) ? alpha : 1.0;
}

double Activation::elu(double value, int _row, int _col)
{
    double alpha = 0.1;
    return (value < 0) ? alpha * (Math::exp(value) - 1) : value;
}

double Activation::delu(double value, int _row, int _col)
{
    double alpha = 0.1;
    return (value < 0) ? alpha * Math::exp(value) : 1.0;
}

double Activation::softplus(double value, int _row, int _col)
{
    return Math::log(1.0 + Math::exp(value));
}

double Activation::dsoftplus(double value, int _row, int _col)
{
    return 1.0 / (1.0 + Math::exp(-value));
}

double Activation::swish(double value, int _row, int _col, double beta)
{
    return value * sigmoid(beta * value, _row, _col);
}

double Activation::dswish(double value, int _row, int _col, double beta)
{
    return beta * value * dsigmoid(beta * value, _row, _col) + sigmoid(beta * value, _row, _col);
}

double Activation::mish(double value, int _row, int _col)
{
    return value * tanh_(softplus(value, _row, _col), _row, _col);
}

double Activation::dmish(double value, int _row, int _col)
{
    double sp = softplus(value, _row, _col);
    double tsp = tanh_(sp, _row, _col);
    return tsp + value * dtanh(sp, _row, _col) * dsoftplus(value, _row, _col);
}

double Activation::linear(double value, int _row, int _col)
{
    return value;
}

double Activation::dlinear(double value, int _row, int _col)
{
    return 1.0;
}
