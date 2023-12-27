#ifndef QTABLE_H
#define QTABLE_H

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/callable.hpp>

#include <../eigen/Eigen/Dense>


using namespace godot;

class QTable : public RefCounted {
    GDCLASS(QTable, RefCounted)

private:
    int observation_space;
    int action_spaces;
    Eigen::MatrixXd Table;


    double exploration_probability;
    double exploration_decreasing_decay;
    double min_exploration_probability;
    double discounted_factor;
    double learning_rate;
    int decay_per_steps;
    int steps_completed;
    bool random_weights;

    int previous_state;
    int previous_action;

    int max_state_value;
    bool is_learning;
    bool print_debug_info;

    void load_table(const Array& arr);
    Array save_table();
    
protected:
    static void _bind_methods();


public:
    void init(int n_observations, int n_action_spaces, const Dictionary& config);

    double get_exploration_probability();

    int predict(const Array& current_states, double reward_of_previous_state);
    int create_composite_state(const Array& current_state);

    void save(const String& path);
    void load(const String& path);


    //Eigen Interface
    Array get_data();
    void set_at(int _row, int _col, double value);
    double get_at(int _row, int _col);
    int index_of_max_from_row(int _row);
    double max_from_row(int _row);


    QTable();
    ~QTable();
};

#endif // QTABLE_H