#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H


#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/color.hpp>
#include <godot_cpp/classes/ray_cast2d.hpp>
#include <godot_cpp/variant/callable.hpp>

#include "Matrix.h"
#include "Activations.h"

using namespace godot;

class NeuralNetwork : public RefCounted {
    GDCLASS(NeuralNetwork, RefCounted)

private:
    Ref<Activation> activation;

    int input_nodes;
    int hidden_nodes;
    int output_nodes;

    Ref<Matrix> weights_input_hidden;
    Ref<Matrix> weights_hidden_output;

    Ref<Matrix> bias_hidden;
    Ref<Matrix> bias_output;

    Dictionary hidden_activation;
    Dictionary output_activation;

    double learning_rate;
    double fitness;
    Array raycasts;
    Color color;
    Dictionary activation_functions;
protected:
    static void _bind_methods();


public:
    void set_fitness(double _fitness);
    double get_fitness();
    void set_raycasts(const Array& _raycasts);
    void set_color(const Color& _color);
    Color get_color();

    NeuralNetwork();
    ~NeuralNetwork();

    void init(int _input_nodes, int _hidden_nodes, int _output_nodes);

    void set_nn_color();
    void set_activation_function(Dictionary hidden_func, Dictionary output_func);
    Array predict(const Array& input_array);
    void train(const Array& input_array, const Array& target_array);
    Array get_inputs_from_raycasts();
    Array get_prediction_from_raycasts(const Array& optional_val);
    double get_distance(RayCast2D* _raycast);
    static Ref<NeuralNetwork> NeuralNetwork::reproduce(Ref<NeuralNetwork> a, Ref<NeuralNetwork> b);
    static Ref<NeuralNetwork> mutate(Ref<NeuralNetwork> nn, Callable callback);
    double NeuralNetwork::mutate_callable_reproduced(double value, int row, int col);
    static Ref<NeuralNetwork> copy(Ref<NeuralNetwork> nn);
    static double mutate_callable(double value, int row, int col);
    void save(const String& path);
    static Ref<NeuralNetwork> load(const String& path);
};

#endif // NEURAL_NETWORK_H
