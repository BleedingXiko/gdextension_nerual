#ifndef QTABLE_H
#define QTABLE_H

#include "Matrix.h"

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/classes/random_number_generator.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/classes/file_access.hpp>

using namespace godot;

class QTable : public RefCounted
{
    GDCLASS(QTable, RefCounted)

private:
    Ref<RandomNumberGenerator> rng;

    int observation_space;
    int action_spaces;

    Ref<Matrix> Table;
    Ref<Matrix> VisitCounts;
    Ref<Matrix> TotalVisitCounts;

    String exploration_strategy;
    double exploration_parameter; // temperature for softmax, exploration_paramter for ucb, etc.

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

protected:
    static void _bind_methods();

public:
    void init(int n_observations, int n_action_spaces, const Dictionary &config);
    void set_variables(const Dictionary &config);

    double get_exploration_probability();

    int predict(const Array &current_states, double reward_of_previous_state);
    //maybe add in multiple? this can be done in gdscript by the user
    int create_composite_state(const Array &current_state);

    void save(const String &path);
    void load(const String &path, const Dictionary &config);

    int selectStrategy(const String &exploration_strategy, int action_spaces, int chosen_state);
    int epsilonGreedyStrategy(int action_spaces, int chosen_state);
    Array normalizeArray(const Array &input);
    int softmaxExploration(int action_spaces, int chosen_state, double temperature);
    int thompsonSampling(int action_spaces, int chosen_state);
    int ucbExploration(int action_spaces, int chosen_state, int total_steps, double exploration_parameter);

    int get_visits(int state, int action);
    int get_total_visits(int state);
    void increment_visits(int state, int action);
    void increment_total_visits(int state);

    int sum(const Array &array);

    QTable();
    ~QTable();
};

#endif // QTABLE_H