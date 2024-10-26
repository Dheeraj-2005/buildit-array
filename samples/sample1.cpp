#include "barray.h"


// BArray Program to be compiled
void test_program(builder::dyn_var<int*> arr1, int dim1, int dim2, int dim3) {
	barray::current_device = barray::DEVICE_HOST;
	// Declare 3 arrays, with 3 dimensions
	// The first arrays used a buffer passed in
	barray::barray<int> y(arr1, {dim1, dim2, dim3});
	barray::barray<int> z({dim1, dim2, dim3});
	barray::barray<int> w({dim1, dim2, dim3});

	// Initialize Z to  all zeroes
	z = 0;
	// Initialize W to all ones
	w = 1;
	z.finalize();
	w.to_device();
	z.to_device();
	y.allocate_device();
	// Compute Z + W and store it in Y
	barray::run_on_gpu([&] () {
		y=2 * z + 3 * w + 3;
	});

}


int main(int argc, char* argv[]) {
	barray::generate_barray_program(test_program, "test", std::cout, 32, 16, 8);
	return 0;
}
