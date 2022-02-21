#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct LBMpara
{
	int Nx;
	int Ny;
	int Nz;
	double rho0;
	double ux0;
	double uy0;
	double uz0;
	double gravity;
};

class MRTGPU
{
public:
	void init(LBMpara params);

	void feq();
	void rate_strain();
	void mf_meq();
	void Invf_feq();
	void collision();
	void boundary();
	void macroscopic();

	void output(int t);
	void freemem();

protected:
	dim3 block, grid;

	int* d_geo, * h_geo;

	double* d_ux, * h_ux;
	double* d_uy, * h_uy;
	double* d_uz, * h_uz;
	double* d_rho, * h_rho;
	
	double* d_f, * h_f;
	double* d_f_post, * h_f_post;
	double* d_feq;
	double* m_f, * m_eq;
	double* inv_f, * inv_feq;

	double* s_xx, * s_xy, * s_xz;
	double* s_yx, * s_yy, * s_yz;
	double* s_zx, * s_zy, * s_zz;
	double* sigma;

	double* output_rho, * output_ux, * output_uy, * output_uz;

	int Nx;
	int Ny;
	int Nz;
	double rho0;
	double ux0;
	double uy0;
	double uz0;
	double gravity;
};