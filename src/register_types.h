#ifndef NEURAL_REGISTER_TYPES_H
#define NEURAL_REGISTER_TYPES_H

#include <godot_cpp/core/class_db.hpp>

using namespace godot;

void initialize_neural_module(ModuleInitializationLevel p_level);
void uninitialize_neural_module(ModuleInitializationLevel p_level);

#endif // NEURAL_REGISTER_TYPES_H