#include "Matrix.h"

using namespace godot;

void Matrix::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("init", "_rows", "_cols"), &Matrix::init);
    ClassDB::bind_method(D_METHOD("get_rows"), &Matrix::get_rows);
    ClassDB::bind_method(D_METHOD("get_cols"), &Matrix::get_cols);
    ClassDB::bind_method(D_METHOD("get_data"), &Matrix::get_data);
    ClassDB::bind_method(D_METHOD("set_at", "_rows", "_cols", "_value"), &Matrix::set_at);
    ClassDB::bind_method(D_METHOD("get_at", "_rows", "_cols"), &Matrix::get_at);
    ClassDB::bind_method(D_METHOD("save"), &Matrix::save);
    ClassDB::bind_method(D_METHOD("index_of_max_from_row", "_row"), &Matrix::index_of_max_from_row);
    ClassDB::bind_method(D_METHOD("indices_of_max_from_row", "_row", "threshold"), &Matrix::indices_of_max_from_row);
    ClassDB::bind_method(D_METHOD("max_from_row", "_row"), &Matrix::max_from_row);
    ClassDB::bind_method(D_METHOD("rand"), &Matrix::rand);

    // Static Methods
    ClassDB::bind_static_method("Matrix", D_METHOD("to_array", "matrix"), &Matrix::to_array);
    ClassDB::bind_static_method("Matrix", D_METHOD("from_array", "arr"), &Matrix::from_array);
    ClassDB::bind_static_method("Matrix", D_METHOD("add", "a", "b"), &Matrix::add);
    ClassDB::bind_static_method("Matrix", D_METHOD("subtract", "a", "b"), &Matrix::subtract);
    ClassDB::bind_static_method("Matrix", D_METHOD("scalar", "matrix", "value"), &Matrix::scalar);
    ClassDB::bind_static_method("Matrix", D_METHOD("dot_product", "a", "b"), &Matrix::dot_product);
    ClassDB::bind_static_method("Matrix", D_METHOD("multiply", "a", "b"), &Matrix::multiply);
    ClassDB::bind_static_method("Matrix", D_METHOD("transpose", "matrix"), &Matrix::transpose);
    ClassDB::bind_static_method("Matrix", D_METHOD("map", "matrix", "callback"), &Matrix::map);
    ClassDB::bind_static_method("Matrix", D_METHOD("average", "matrix"), &Matrix::average);
    ClassDB::bind_static_method("Matrix", D_METHOD("random", "a", "b"), &Matrix::random);
    ClassDB::bind_static_method("Matrix", D_METHOD("copy", "matrix"), &Matrix::copy);
    ClassDB::bind_static_method("Matrix", D_METHOD("load", "arr"), &Matrix::load);
}

void Matrix::init(int _rows, int _cols)
{
    data = Eigen::MatrixXd::Constant(_rows, _cols, 0.0);
}

int Matrix::get_rows()
{
    return data.rows();
}

int Matrix::get_cols()
{
    return data.cols();
}

void Matrix::set_at(int _row, int _col, double _value)
{
    data(_row, _col) = _value;
}
double Matrix::get_at(int _row, int _col)
{
    return data(_row, _col);
}

Ref<Matrix> Matrix::from_array(const Array &arr)
{
    Ref<Matrix> result = memnew(Matrix);
    result->init(arr.size(), 1);
    for (int i = 0; i < result->data.rows(); ++i)
    {
        double value = static_cast<double>(arr[i]); // Cast to double
        result->data(i, 0) = value;
    }

    return result;
}

Array Matrix::to_array(const Ref<Matrix> matrix)
{
    Array arr;
    for (int i = 0; i < matrix->data.rows(); ++i)
    {
        // Array row;
        for (int j = 0; j < matrix->data.cols(); ++j)
        {
            // row.append(matrix->data(i,j));
            arr.append(matrix->data(i, j));
        }

        // arr.append(row);
    }
    return arr;
}

void Matrix::rand()
{
    data.setRandom();
}

Ref<Matrix> Matrix::add(const Ref<Matrix> a, const Ref<Matrix> b)
{
    if (a->data.rows() != b->data.rows() || a->data.cols() != b->data.cols())
    {
        // Print error message and return null pointer
        ERR_PRINT("Matrices dimensions do not match.\n");
        return nullptr;
    }
    Ref<Matrix> result = memnew(Matrix);
    result->init(a->data.rows(), a->data.cols());

    result->data = a->data + b->data;

    return result;
}

Ref<Matrix> Matrix::subtract(const Ref<Matrix> a, const Ref<Matrix> b)
{
    if (a->data.rows() != b->data.rows() || a->data.cols() != b->data.cols())
    {
        // Print error message and return null pointer
        ERR_PRINT("Matrices dimensions do not match.\n");
        return nullptr;
    }
    Ref<Matrix> result = memnew(Matrix);
    result->init(a->data.rows(), a->data.cols());

    result->data = a->data - b->data;

    return result;
}

Ref<Matrix> Matrix::scalar(const Ref<Matrix> matrix, float value)
{
    Ref<Matrix> result = memnew(Matrix);
    result->init(matrix->data.rows(), matrix->data.cols());

    result->data = matrix->data * value;

    return result;
}

Array Matrix::get_data()
{
    Array arr;
    for (int i = 0; i < data.rows(); ++i)
    {
        Array row;
        for (int j = 0; j < data.cols(); ++j)
        {
            row.append(data(i, j));
        }

        arr.append(row);
    }
    return arr;
}

Ref<Matrix> Matrix::dot_product(const Ref<Matrix> a, const Ref<Matrix> b)
{
    Ref<Matrix> result = memnew(Matrix);
    result->init(a->data.rows(), b->data.cols());

    result->data = a->data * b->data;

    return result;
}

// dot divide here

Ref<Matrix> Matrix::multiply(const Ref<Matrix> a, const Ref<Matrix> b)
{
    if (a->data.rows() != b->data.rows() || a->data.cols() != b->data.cols())
    {
        // Print error message and return null pointer
        ERR_PRINT("Matrices dimensions do not match.\n");
        return nullptr;
    }

    Ref<Matrix> result = memnew(Matrix);
    result->init(a->data.rows(), a->data.cols());

    result->data = a->data * b->data;

    return result;
}

// divide

Ref<Matrix> Matrix::transpose(const Ref<Matrix> matrix)
{
    Ref<Matrix> result = memnew(Matrix);
    result->init(matrix->data.cols(), matrix->data.rows());

    result->data = matrix->data.transpose();

    return result;
}

Ref<Matrix> Matrix::map(const Ref<Matrix> matrix, Callable callback)
{
    Ref<Matrix> result = memnew(Matrix);
    result->init(matrix->data.rows(), matrix->data.cols());

    for (int i = 0; i < matrix->data.rows(); ++i)
    {
        for (int j = 0; j < matrix->data.cols(); ++j)
        {
            Variant value = matrix->data(i, j);
            Array args;
            args.append(value);
            args.append(i);
            args.append(j);

            Variant result_value = callback.callv(args);
            result->data(i, j) = result_value;
        }
    }

    return result;
}

Ref<Matrix> Matrix::random(const Ref<Matrix> a, const Ref<Matrix> b)
{
    Ref<Matrix> result = memnew(Matrix);
    Ref<RandomNumberGenerator> rng = memnew(RandomNumberGenerator);
    result->init(a->data.rows(), a->data.cols());

    for (int i = 0; i < result->data.rows(); ++i)
    {
        for (int j = 0; j < result->data.cols(); ++j)
        {

            if (rng->randf() < 0.5)
            {
                result->data(i, j) = a->data(i, j);
            }
            else
            {
                result->data(i, j) = b->data(i, j);
            }
        }
    }
    return result;
}

Ref<Matrix> Matrix::copy(const Ref<Matrix> matrix)
{
    Ref<Matrix> result = memnew(Matrix);
    result->init(matrix->data.rows(), matrix->data.cols());

    for (int i = 0; i < result->data.rows(); ++i)
    {
        for (int j = 0; j < result->data.cols(); ++j)
        {

            result->data(i, j) = matrix->data(i, j);
        }
    }

    return result;
}

int Matrix::index_of_max_from_row(int _row)
{
    if (_row < 0 || _row >= data.rows())
    {
        ERR_PRINT("Row index out of bounds.");
        return 0.0;
    }

    int col_index;
    data.row(_row).maxCoeff(&col_index);
    return col_index;
}

Array Matrix::indices_of_max_from_row(int _row, double threshold)
{
    if (_row < 0 || _row >= data.rows())
    {
        ERR_PRINT("Row index out of bounds.");
        Array err;
        err.append(0.0);
        return err;
    }

    Array arr;
    double max_value = max_from_row(_row);
    for (int i = 0; i < data.cols(); ++i)
    {
        double value = data(_row, i);
        double diff = UtilityFunctions::abs(value - max_value);
        if (diff < threshold)
        {
            arr.append(i);
        }
    }

    arr.append(index_of_max_from_row(_row));
    return arr;
}

double Matrix::max_from_row(int _row)
{
    if (_row < 0 || _row >= data.rows())
    {
        ERR_PRINT("Row index out of bounds.");
        return 0.0;
    }

    float max_value = data.row(_row).maxCoeff();

    return max_value;
}

float Matrix::average(const Ref<Matrix> matrix)
{
    int total_elements = matrix->data.rows() * matrix->data.cols();

    if (total_elements == 0)
    {
        ERR_PRINT("Matrix is empty");
        return 0.0;
    }

    float sum = matrix->data.sum();

    float avg = sum / total_elements;

    return avg;
}

Array Matrix::save()
{
    Array arr = get_data();
    return arr;
}

Ref<Matrix> Matrix::load(const Array &arr)
{
    Ref<Matrix> result = memnew(Matrix);
    int rows = arr.size();
    Array temp = arr[0];
    int cols = temp.size();
    result->init(rows, cols);
    for (int i = 0; i < rows; ++i)
    {
        Array row = arr[i]; // Assuming each element of arr is also an Array
        for (int j = 0; j < cols; ++j)
        {
            double value = static_cast<double>(row[j]); // Cast to double
            result->data(i, j) = value;
        }
    }

    return result;
}

Matrix::Matrix()
{
    // Initialize any variables here.
}
Matrix::~Matrix()
{
}