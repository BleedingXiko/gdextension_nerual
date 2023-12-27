#ifndef MATRIX_H
#define MATRIX_H

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/random_number_generator.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/core/class_db.hpp>

#include <../eigen/Eigen/Dense>

using namespace godot;

class Matrix : public RefCounted
{
    GDCLASS(Matrix, RefCounted)

private:
    Eigen::MatrixXd data;

protected:
    static void _bind_methods();

public:
    Matrix();
    ~Matrix();
    void init(int _rows, int _cols);

    Array get_data();
    void set_at(int _row, int _col, double _value);
    double get_at(int _row, int _col);
    void rand();

    static Ref<Matrix> from_array(const Array &arr);
    static Array to_array(const Ref<Matrix> matrix);
    static Ref<Matrix> add(const Ref<Matrix> a, const Ref<Matrix> b);
    static Ref<Matrix> subtract(const Ref<Matrix> a, const Ref<Matrix> b);
    static Ref<Matrix> scalar(const Ref<Matrix> matrix, float value);
    static Ref<Matrix> dot_product(const Ref<Matrix> a, const Ref<Matrix> b);
    // static Matrix* dot_divide(const Matrix* a, const Matrix* b);
    static Ref<Matrix> multiply(const Ref<Matrix> a, const Ref<Matrix> b);
    // static Matrix* divide(const Matrix* a, const Matrix* b);
    static Ref<Matrix> transpose(const Ref<Matrix> matrix);
    static Ref<Matrix> map(const Ref<Matrix> matrix, Callable callback);
    static Ref<Matrix> random(const Ref<Matrix> a, const Ref<Matrix> b);
    static Ref<Matrix> copy(const Ref<Matrix> matrix);
    static float average(const Ref<Matrix> matrix);

    int index_of_max_from_row(int _row);
    double max_from_row(int _row);
    static Ref<Matrix> load(const Array &arr);
    Array save();
};

#endif // MATRIX_H