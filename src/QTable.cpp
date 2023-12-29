#include "QTable.h"

using namespace godot;

void QTable::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("init", "n_observations", "n_action_spaces", "config"), &QTable::init);
    ClassDB::bind_method(D_METHOD("get_exploration_probability"), &QTable::get_exploration_probability);
    ClassDB::bind_method(D_METHOD("predict", "current_states", "reward_of_previous_state"), &QTable::predict);
    ClassDB::bind_method(D_METHOD("create_composite_state", "current_state"), &QTable::create_composite_state);
    ClassDB::bind_method(D_METHOD("load", "path"), &QTable::load);
    ClassDB::bind_method(D_METHOD("save", "path"), &QTable::save);
}

void QTable::init(int n_observations, int n_action_spaces, const Dictionary &config)
{

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

    exploration_strategy = config.get("exploration_strategy", exploration_strategy);
    exploration_parameter = config.get("exploration_parameter", exploration_parameter);

    Table->init(observation_space, action_spaces);
    VisitCounts->init(observation_space, action_spaces);
    TotalVisitCounts->init(observation_space, action_spaces);
    if (random_weights)
    {
        Table->rand();
    };
}

int QTable::predict(const Array &current_states, double reward_of_previous_state)
{
    int chosen_state = create_composite_state(current_states);

    if (is_learning && previous_state != -100)
    {
        double old_value = Table->get_at(previous_state, previous_action);
        double max_future_q = Table->max_from_row(chosen_state);
        double new_value = (1 - learning_rate) * old_value + learning_rate * (reward_of_previous_state + discounted_factor * max_future_q);
        Table->set_at(previous_state, previous_action, new_value);
        increment_visits(previous_state, previous_action);
        increment_total_visits(previous_state);
    }

    int action_to_take = selectStrategy(exploration_strategy, action_spaces, chosen_state);

    if (is_learning)
    {
        previous_state = chosen_state;
        previous_action = action_to_take;
    }

    if (print_debug_info && steps_completed % decay_per_steps == 0)
    {
        UtilityFunctions::print("Total steps completed:", steps_completed);
        UtilityFunctions::print("Current exploration probability:", exploration_probability);
        UtilityFunctions::print("Q-Table data:", Table->get_data());
        UtilityFunctions::print("-----------------------------------------------------------------------------------------");
    }

    steps_completed += 1;
    return action_to_take;
}

int QTable::selectStrategy(const String &exploration_strategy, int action_spaces, int chosen_state)
{
    if (exploration_strategy == "epsilon_greedy")
    {
        return epsilonGreedyStrategy(action_spaces, chosen_state);
    }
    else if (exploration_strategy == "softmax")
    {
        return softmaxExploration(action_spaces, chosen_state, exploration_parameter); // 0 - 1 1 being more exploration
    }
    else if (exploration_strategy == "thompson_sampling")
    {
        return thompsonSampling(action_spaces, chosen_state);
    }
    else if (exploration_strategy == "ucb")
    {
        return ucbExploration(action_spaces, chosen_state, steps_completed, exploration_parameter); // higher = more exploration 1 - 50
    }
    else
    {
        ERR_PRINT("Unknown exploration strategy: ");
        return epsilonGreedyStrategy(action_spaces, chosen_state);
    }
}

// Define different exploration strategies as functions
int QTable::epsilonGreedyStrategy(int action_spaces, int chosen_state)
{
    if (steps_completed % decay_per_steps == 0)
    {
        exploration_probability = UtilityFunctions::max(min_exploration_probability, exploration_probability - exploration_decreasing_decay);
    }

    if (rng->randf() < exploration_probability)
    {
        return rng->randi() % action_spaces;
    }
    else
    {
        return Table->index_of_max_from_row(chosen_state);
    }
}

Array QTable::normalizeArray(const Array &input)
{
    Array normalizedArray;

    double sum = 0.0;
    for (int i = 0; i < input.size(); ++i)
    {
        sum += static_cast<double>(input[i]);
    }

    for (int i = 0; i < input.size(); ++i)
    {
        normalizedArray.append(static_cast<double>(input[i]) / sum);
    }

    return normalizedArray;
}

int QTable::softmaxExploration(int action_spaces, int chosen_state, double temperature)
{
    Array action_probabilities;

    // Calculate unnormalized action probabilities using Q-values
    for (int action = 0; action < action_spaces; ++action)
    {
        double q_value = Table->get_at(chosen_state, action);

        action_probabilities.append(Math::exp(q_value / temperature));
    }

    // Normalize the probabilities
    action_probabilities = normalizeArray(action_probabilities);

    // Choose an action based on the probabilities
    double cumulative_probability = 0.0;
    for (int action = 0; action < action_spaces; ++action)
    {
        cumulative_probability += static_cast<double>(action_probabilities[action]);
        if (rng->randf() < cumulative_probability)
        {
            return action;
        }
    }

    // Fallback to a random action (should not reach here)
    return rng->randi() % action_spaces;
}

int QTable::thompsonSampling(int action_spaces, int chosen_state)
{
    Array sampled_values;

    // Assume binary reward, use Beta distribution for sampling
    for (int action = 0; action < action_spaces; ++action)
    {
        double alpha = get_visits(chosen_state, action) + 1;
        double beta = get_total_visits(chosen_state) - alpha + 1;

        // Sample from Beta distribution
        double sampled_value = pow(rng->randf(), 1.0 / alpha) / (pow(rng->randf(), 1.0 / alpha) + pow(rng->randf(), 1.0 / beta));

        sampled_values.append(sampled_value);
    }

    // Choose the action with the highest sampled value
    int action_to_take = sampled_values.find(sampled_values.max());
    
    return action_to_take;
}

int QTable::ucbExploration(int action_spaces, int chosen_state, int total_steps, double exploration_parameter)
{
    Array ucb_values;

    // Calculate UCB values for each action
    for (int action = 0; action < action_spaces; ++action)
    {
        double q_value = Table->get_at(chosen_state, action);
        double exploration_bonus = exploration_parameter * Math::sqrt(Math::log(static_cast<double>(total_steps + 1)) / (get_visits(chosen_state, action) + 1));
        ucb_values.append(q_value + exploration_bonus);
    }

    // Choose the action with the highest UCB value
    return ucb_values.find(ucb_values.max());
}

int QTable::get_visits(int state, int action)
{
    // Get the number of visits to a specific state-action pair
    return VisitCounts->get_at(state, action);
}

// Custom sum function
int QTable::sum(const Array &array)
{
    int result = 0;
    for (int i = 0; i < array.size(); ++i)
    {
        Variant element = array[i];
        if (element.get_type() == Variant::INT)
        {
            result += element.operator int();
        }
        if (element.get_type() == Variant::FLOAT)
        {
            result += element.operator float();
        }
        // You might need additional checks for other types like Variant::REAL
    }
    return result;
}

int QTable::get_total_visits(int state)
{
    // Get the total number of visits to a state
    Array count = TotalVisitCounts->get_data();

    return sum(count[state]);
}

void QTable::increment_visits(int state, int action)
{
    // Increment the number of visits to a specific state-action pair
    double current_value = VisitCounts->get_at(state, action);
    double new_value = current_value + 1;
    VisitCounts->set_at(state, action, new_value);
}

void QTable::increment_total_visits(int state)
{
    // Increment the total number of visits to a state
    double value = TotalVisitCounts->get_at(state, 0) + 1;
    TotalVisitCounts->set_at(state, 0, value);
}

int QTable::create_composite_state(const Array &current_states)
{
    int composite_state = 0;
    int multiplier = 1;
    for (int i = 0; i < current_states.size(); ++i)
    {
        int state = current_states[i];
        composite_state += state * multiplier;
        multiplier *= max_state_value;
    }
    return composite_state;
}

double QTable::get_exploration_probability()
{
    return exploration_probability;
}

void QTable::save(const String &path)
{
    Ref<FileAccess> file = FileAccess::open(path, FileAccess::WRITE);

    Array qTableData = Table->get_data();
    file->store_var(qTableData);

    file->close();
}

void QTable::load(const String &path)
{
    Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
    // Load Q-table data
    Array qTableData = file->get_var();

    file->close();

    // Update Q-table with loaded data
    Table = Table->load(qTableData);

    // Additional initialization
    is_learning = false;
    exploration_probability = min_exploration_probability;
}

QTable::QTable()
{
    Table = Ref<Matrix>(memnew(Matrix));
    VisitCounts = Ref<Matrix>(memnew(Matrix));
    TotalVisitCounts = Ref<Matrix>(memnew(Matrix));
    rng = Ref<RandomNumberGenerator>(memnew(RandomNumberGenerator));
    rng->randomize();

    exploration_strategy = "epsilon_greedy";

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
QTable::~QTable()
{
}