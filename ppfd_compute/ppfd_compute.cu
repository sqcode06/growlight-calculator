#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {

__global__ void compute_phi_kernel(
    const double* __restrict__ xs,          // [L]
    const double* __restrict__ ys,          // [L]
    const double* __restrict__ ms,          // [L]
    const double* __restrict__ Ie0s,        // [L]
    const double* __restrict__ spd,         // [nLambda * L]
    const double* __restrict__ photon_conv, // [nLambda]
    const double* __restrict__ x_coords,    // [nx]
    double z,
    double y_s,
    double* __restrict__ Z,                 // [nLambda * nx]
    int L,
    int nLambda,
    int nx
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // x-index
    int k = blockIdx.y * blockDim.y + threadIdx.y;  // lam_-index

    if (j < nx && k < nLambda) {
        double acc = 0.0;
        double lam_conv = photon_conv[k];
        // sum over all LEDs
        for (int i = 0; i < L; ++i) {
            double dx = x_coords[j] - xs[i];
            double dy = y_s       - ys[i];
            double r  = sqrt(dx*dx + dy*dy + z*z);
            if (r <= 0.0) continue;
            double cos_t = z / r;
            if (cos_t <= 0.0) continue;
            double coef = Ie0s[i] * pow(cos_t, ms[i]) / (r * r);
            acc += spd[k * L + i] * lam_conv * coef;
            acc += spd[k * L + i] * lam_conv * coef;
        }
        Z[k * nx + j] = acc;
    }
}

torch::Tensor compute_ppfd_CUDA(
    torch::Tensor _xs,
    torch::Tensor _ys,
    torch::Tensor _y_s,
    torch::Tensor _Ie0s,
    torch::Tensor _ms,
    torch::Tensor lam_m,            // not used in kernel, assume pre-multiplied into photon_conv
    torch::Tensor _photon_conv,
    torch::Tensor _spd,
    torch::Tensor _x_coords,
    torch::Tensor _z
) {
    // Ensure inputs are contiguous and on CUDA
    _xs          = _xs.contiguous();
    _ys          = _ys.contiguous();
    _Ie0s        = _Ie0s.contiguous();
    _ms          = _ms.contiguous();
    _photon_conv = _photon_conv.contiguous();
    _spd         = _spd.contiguous();
    _x_coords    = _x_coords.contiguous();

    // Dimensions
    int L        = _xs.size(0);
    int nLambda  = _photon_conv.size(0);
    int nx       = _x_coords.size(0);

    // Extract scalar values
    double z     = _z.item<double>();
    double y_s   = _y_s.item<double>();

    // Prepare output tensor [nLambda, nx]
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat64)
                       .device(_xs.device());
    torch::Tensor Z = torch::zeros({nLambda, nx}, options);

    // Launch parameters
    const int TPB_X = 16;
    const int TPB_Y = 16;
    dim3 threads(TPB_X, TPB_Y);
    dim3 blocks(
        (nx      + TPB_X - 1) / TPB_X,
        (nLambda + TPB_Y - 1) / TPB_Y
    );

    // Raw pointers
    const double* xs_ptr          = _xs.data_ptr<double>();
    const double* ys_ptr          = _ys.data_ptr<double>();
    const double* ms_ptr          = _ms.data_ptr<double>();
    const double* Ie0s_ptr        = _Ie0s.data_ptr<double>();
    const double* spd_ptr         = _spd.data_ptr<double>();
    const double* photon_conv_ptr = _photon_conv.data_ptr<double>();
    const double* x_coords_ptr    = _x_coords.data_ptr<double>();
    double*       Z_ptr           = Z.data_ptr<double>();

    // Launch the kernel
    compute_phi_kernel<<<blocks, threads>>>(
        xs_ptr, ys_ptr, ms_ptr, Ie0s_ptr,
        spd_ptr, photon_conv_ptr, x_coords_ptr,
        z, y_s, Z_ptr,
        L, nLambda, nx
    );
    cudaDeviceSynchronize();

    return Z;
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_ppfd", &compute_ppfd_CUDA, "Compute spectral PPFD on CUDA");
}
