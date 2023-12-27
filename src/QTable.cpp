#include "QTable.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/random_number_generator.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/classes/random_number_generator.hpp>


using namespace godot;

void QTable::_bind_methods() {
    ClassDB::bind_method(D_METHOD("init", "n_observations", "n_action_spaces", "config"), &QTable::init);
    ClassDB::bind_method(D_METHOD("get_data"), &QTable::get_data);
    ClassDB::bind_method(D_METHOD("set_at", "_row", "_col", "value"), &QTable::set_at);
    ClassDB::bind_method(D_METHOD("get_at", "_row", "_col"), &QTable::get_at);
    ClassDB::bind_method(D_METHOD("index_of_max_from_row", "_row"), &QTable::index_of_max_from_row);
    ClassDB::bind_method(D_METHOD("max_from_row", "_row"), &QTable::max_from_row);
    ClassDB::bind_method(D_METHOD("get_exploration_probability"), &QTable::get_exploration_probability);
    ClassDB::bind_method(D_METHOD("predict", "current_states", "reward_of_previous_state"), &QTable::predict);
    ClassDB::bind_method(D_METHOD("create_composite_state", "current_state"), &QTable::create_composite_state);
    ClassDB::bind_method(D_METHOD("load", "path"), &QTable::load);
    ClassDB::bind_method(D_METHOD("save", "path"), &QTable::save);
    ClassDB::bind_method(D_METHOD("load_table", "arr"), &QTable::load_table);
    ClassDB::bind_method(D_METHOD("save_table"), &QTable::save_table);
}

void QTable::init(int n_observations, int n_action_spaces, const Dictionary& config) {

    observation_space = n_observations;
    action_spaces = n_action_spaces;


    is_learning = config.get("is_learning", is_learning);
	max_state_value = config.get("max_state_value", max_state_value);
	exploration_decreasing_decay = config.get("exploration_decreasing_decay", exploration_decreasing_decay);
	min_exploration_probability = config.get("min_exploration_probability", min_exploration_probability);
	discounted_factor = config.get("discounted_factor", discounted_factor);
	learning_rate = config.get("learning_rate", learning_rate);
	decay_per_steps = config.get("decay_per_steps", decay_per_steps);
	print_debug_info = config.get("print_debug_info", print_debug_info);
	random_weights = config.get("random_weights", random_weights);


    Table = Eigen::MatrixXd::Constant(observation_space, action_spaces, 0.0);

    if(random_weights){
        Table.setRandom();
    };
    

}


int QTable::predict(const Array& current_states, double reward_of_previous_state) 
{
    int chosen_state = create_composite_state(current_states);

    Ref<RandomNumberGenerator> rng = memnew(RandomNumberGenerator);
    rng->randomize();

    if (is_learning && previous_state != -100) {
        double old_value = get_at(previous_state, previous_action);
        double max_future_q = max_from_row(chosen_state);
        double new_value = (1 - learning_rate) * old_value + learning_rate * (reward_of_previous_state + discounted_factor * max_future_q);
        set_at(previous_state, previous_action, new_value);
    }

    int action_to_take;
    if (rng->randf() < exploration_probability) {
        action_to_take = rng->randi() % action_spaces;
    } else {
        action_to_take = index_of_max_from_row(chosen_state);
    }

    if (is_learning) {
        previous_state = chosen_state;
        previous_action = action_to_take;
        if (steps_completed % decay_per_steps == 0) {
            exploration_probability = UtilityFunctions::max(min_exploration_probability, exploration_probability - exploration_decreasing_decay);
            if (print_debug_info) {
                UtilityFunctions::print("Total steps completed:", steps_completed);
                UtilityFunctions::print("Current exploration probability:", exploration_probability);
                UtilityFunctions::print("Q-Table data:", get_data());
                UtilityFunctions::print("-----------------------------------------------------------------------------------------");
            }
        }
    }

    steps_completed += 1;
    return action_to_take;
}



int QTable::create_composite_state(const Array& current_states) {
    int composite_state = 0;
    int multiplier = 1;
    for (int i = 0; i < current_states.size(); ++i) {
        int state = current_states[i];
        composite_state += state * multiplier;
        multiplier *= max_state_value;
    }
    return composite_state;
}


void QTable::set_at(int _row, int _col, double value){
    Table(_row, _col) = value;
}
double QTable::get_at(int _row, int _col){
    return Table(_row, _col);
}

double QTable::get_exploration_probability() {
    return exploration_probability;
}

Array QTable::get_data() {
    Array arr;
    for (int i = 0; i < Table.rows(); ++i){
        Array row;
        for (int j = 0; j < Table.cols(); ++j){
            row.append(Table(i,j));
        }

        arr.append(row);
    }
    return arr;
}
void QTable::save(const String& path)
{
    Ref<FileAccess> file = FileAccess::open(path, FileAccess::WRITE);

    Array qTableData = get_data();
    file->store_var(qTableData);

    file->close();
    
}

void QTable::load(const String& path)
{
    Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
    // Load Q-table data
    Array qTableData = file->get_var();

    file->close();

    // Update Q-table with loaded data
    load_table(qTableData);

    // Additional initialization
    is_learning = false;
    exploration_probability = min_exploration_probability;
}


int QTable::index_of_max_from_row(int _row) {
    if (_row < 0 || _row >= Table.rows()) {
        ERR_PRINT("Row index out of bounds.");
        return 0.0;
    }

    int col_index;
    Table.row(_row).maxCoeff(&col_index);
    return col_index;


}

double QTable::max_from_row(int _row){
    if (_row < 0 || _row >= Table.rows()) {
        ERR_PRINT("Row index out of bounds.");
        return 0.0;
    }

    float max_value = Table.row(_row).maxCoeff();

    return max_value;
}





Array QTable::save_table(){
    Array arr = get_data();
    return arr;
}

void QTable::load_table(const Array& arr) {
    int rows = arr.size();
    Array temp = arr[0];
    int cols = temp.size();
    Dictionary dict = {};
    init(rows, cols, dict);
    for (int i = 0; i < rows; ++i) {
        Array row = arr[i];  
        for (int j = 0; j < cols; ++j) {
            double value = static_cast<double>(row[j]); 
            Table(i, j) = value;  
        }
    }
}


QTable::QTable() {

    exploration_probability = 1.0;
    exploration_decreasing_decay = 0.01;
    min_exploration_probability = 0.05;
    discounted_factor = 0.9;
    learning_rate = 0.2;
    decay_per_steps = 100;
    steps_completed = 0;
    random_weights = false;

    previous_state = -100;
    previous_action = 0;

    max_state_value = 2;
    is_learning = true;
    print_debug_info = false;


}
QTable::~QTable() {

}