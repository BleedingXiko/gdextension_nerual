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

#include <../eigen/Eigen/Dense>

using namespace godot;

class NeuralNetwork : public Reference {
    GDCLASS(NeuralNetwork)

private:
    int input_nodes;
    int hidden_nodes;
    int output_nodes;

    Eigen::MatrixXd weights_input_hidden;
    Eigen::MatrixXd weights_hidden_output;

    Eigen::MatrixXd bias_hidden;
    Eigen::MatrixXd bias_output;

    double learning_rate;
    Dictionary hidden_activation;
    Dictionary output_activation;
    double fitness;
    Color color;
    Array godot_raycasts;

protected:
    static void _bind_methods();


public:
    NeuralNetwork();
    ~NeuralNetwork();

    void set_nn_color();
    void set_activation_function(Dictionary hidden_func, Dictionary output_func);
    Array predict(Array input_array);
    void train(Array input_array, Array target_array);
    Array get_inputs_from_raycasts();
    Array get_prediction_from_raycasts(Array optional_val);
    double get_distance(Ref<RayCast2D> _raycast);
    static Ref<NeuralNetwork> reproduce(Ref<NeuralNetwork> a, Ref<NeuralNetwork> b);
    static Ref<NeuralNetwork> mutate(Ref<NeuralNetwork> nn, Callable callback);
    static double mutate_callable_reproduced(double value, int row, int col);
    static Ref<NeuralNetwork> copy(Ref<NeuralNetwork> nn);
    static double mutate_callable(double value, int row, int col);
    void save(String path);
    static Ref<NeuralNetwork> load(String path);
};

#endif // NEURAL_NETWORK_H
