#ifndef ACTIVATION_H
#define ACTIVATION_H


#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/callable.hpp>

using namespace godot;
    class Activation : public RefCounted {
        GDCLASS(Activation, RefCounted)
    

    private:


    protected:
        static void _bind_methods();


    public:
        Activation();
        ~Activation();
        Dictionary functions;

        void init_functions();

        Dictionary get_functions();

        // Activation functions
         double sigmoid(double value, int _row, int _col);
         double dsigmoid(double value, int _row, int _col);
         double relu(double value, int _row, int _col);
         double drelu(double value, int _row, int _col);
         double tanh_(double value, int _row, int _col);
         double dtanh(double value, int _row, int _col);
         double arcTan(double value, int _row, int _col);
         double darcTan(double value, int _row, int _col);
         double prelu(double value, int _row, int _col);
         double dprelu(double value, int _row, int _col);
         double elu(double value, int _row, int _col);
         double delu(double value, int _row, int _col);
         double softplus(double value, int _row, int _col);
         double dsoftplus(double value, int _row, int _col);
         double swish(double value, int _row, int _col, double beta);
         double dswish(double value, int _row, int _col, double beta);
         double mish(double value, int _row, int _col);
         double dmish(double value, int _row, int _col);
         double linear(double value, int _row, int _col);
         double dlinear(double value, int _row, int _col);
    };

#endif // ACTIVATION_H
