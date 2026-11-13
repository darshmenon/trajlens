// Optional C++ core for trajlens: numpy.gradient for a (n, m) array along axis 0 with non-uniform sample times,
// edge_order 1 or 2. Same formulas as numpy (second-order accurate interior and edges), so results match to rounding.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

namespace py = pybind11;
using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

static Arr gradient(Arr t, Arr f, int edge_order) {
  if (t.ndim() != 1 || f.ndim() != 2 || f.shape(0) != t.shape(0)) throw std::invalid_argument("gradient: t is (n,), f is (n, m)");
  if (edge_order != 1 && edge_order != 2) throw std::invalid_argument("gradient: edge_order must be 1 or 2");
  const py::ssize_t n = t.shape(0), m = f.shape(1);
  if (n < edge_order + 1) throw std::invalid_argument("gradient: not enough samples for this edge_order");
  Arr out({n, m});
  const double * x = t.data();
  const double * y = f.data();
  double * g = out.mutable_data();
  {
    py::gil_scoped_release release;
    for (py::ssize_t i = 1; i + 1 < n; ++i) {
      const double hs = x[i] - x[i - 1], hd = x[i + 1] - x[i];
      const double a = -hd / (hs * (hd + hs)), b = (hd - hs) / (hd * hs), c = hs / (hd * (hd + hs));
      for (py::ssize_t j = 0; j < m; ++j) g[i * m + j] = a * y[(i - 1) * m + j] + b * y[i * m + j] + c * y[(i + 1) * m + j];
    }
    if (edge_order == 1) {
      const double d0 = x[1] - x[0], d1 = x[n - 1] - x[n - 2];
      for (py::ssize_t j = 0; j < m; ++j) {
        g[j] = (y[m + j] - y[j]) / d0;
        g[(n - 1) * m + j] = (y[(n - 1) * m + j] - y[(n - 2) * m + j]) / d1;
      }
    } else {
      const double d1 = x[1] - x[0], d2 = x[2] - x[1];
      const double a0 = -(2 * d1 + d2) / (d1 * (d1 + d2)), b0 = (d1 + d2) / (d1 * d2), c0 = -d1 / (d2 * (d1 + d2));
      const double e1 = x[n - 2] - x[n - 3], e2 = x[n - 1] - x[n - 2];
      const double a1 = e2 / (e1 * (e1 + e2)), b1 = -(e2 + e1) / (e1 * e2), c1 = (2 * e2 + e1) / (e2 * (e1 + e2));
      for (py::ssize_t j = 0; j < m; ++j) {
        g[j] = a0 * y[j] + b0 * y[m + j] + c0 * y[2 * m + j];
        g[(n - 1) * m + j] = a1 * y[(n - 3) * m + j] + b1 * y[(n - 2) * m + j] + c1 * y[(n - 1) * m + j];
      }
    }
  }
  return out;
}

PYBIND11_MODULE(_fast, mod) {
  mod.doc() = "C++ non-uniform gradient for trajlens";
  mod.def("gradient", &gradient, py::arg("t"), py::arg("f"), py::arg("edge_order") = 2);
}
