
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

double compute_sum(py::array_t<double> arr) {
    auto buf = arr.request();
    double *ptr = static_cast<double*>(buf.ptr);
    double sum = 0.0;

    for (size_t i = 0; i < buf.size; ++i) {
        sum += ptr[i];
    }

    return sum;
}

PYBIND11_MODULE(fast_math, m) {
    m.def("compute_sum", &compute_sum, "Fast sum computation");
}
