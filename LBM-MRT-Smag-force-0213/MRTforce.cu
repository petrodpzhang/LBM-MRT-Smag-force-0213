#include "MRTforce.cuh"
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

const int Q = 19;
__constant__ int cx[19];
__constant__ int cy[19];
__constant__ int cz[19];
__constant__ double w[19];
__constant__ int N[3];

//1.initialize
void MRTGPU::init(LBMpara params)
{
	Nx = params.Nx;
	Ny = params.Ny;
	Nz = params.Nz;
	rho0 = params.rho0;
	ux0 = params.ux0;
	uy0 = params.uy0;
	uz0 = params.uz0;
	gravity = params.gravity;
	int Nlattice = Nx * Ny * Nz;
	int QNlattice = Nx * Ny * Nz * Q;

	int _cx[19] = { 0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0 };
	int _cy[19] = { 0,0,0,1,-1,0,0,1,1,-1,-1,0,0,0,0,1,-1,1,-1 };
	int _cz[19] = { 0,0,0,0,0,1,-1,0,0,0,0,1,1,-1,-1,1,1,-1,-1 };//Yu's setting
	double _w[19] = { 1.0 / 3.0,1.0 / 18.0,1.0 / 18.0,1.0 / 18.0,1.0 / 18.0,1.0 / 18.0,1.0 / 18.0,1.0 / 36.0,1.0 / 36.0,1.0 / 36.0,
		1.0 / 36.0,1.0 / 36.0,1.0 / 36.0,1.0 / 36.0,1.0 / 36.0,1.0 / 36.0,1.0 / 36.0,1.0 / 36.0,1.0 / 36.0 };
	int _N[3] = { Nx,Ny,Nz };

	cudaMemcpyToSymbol(cx, _cx, sizeof(int) * Q);
	cudaMemcpyToSymbol(cy, _cy, sizeof(int) * Q);
	cudaMemcpyToSymbol(cz, _cz, sizeof(int) * Q);
	cudaMemcpyToSymbol(w, _w, sizeof(double) * Q);
	cudaMemcpyToSymbol(N, _N, sizeof(int) * 3);

	int threadsAlongX = 8;
	int threadsAlongY = 8;
	int threadsAlongZ = 8;

	block = dim3(threadsAlongX, threadsAlongY, threadsAlongZ);
	grid = dim3(1 + (Nx - 1) / threadsAlongX, 1 + (Ny - 1) / threadsAlongY, 1 + (Nz - 1) / threadsAlongZ);

	// allocate memory on CPU and GPU 
	h_geo = (int*)malloc(sizeof(int) * Nlattice);
	h_ux = (double*)malloc(sizeof(double) * Nlattice);
	h_uy = (double*)malloc(sizeof(double) * Nlattice);
	h_uz = (double*)malloc(sizeof(double) * Nlattice);
	h_rho = (double*)malloc(sizeof(double) * Nlattice);	
	h_f = (double*)malloc(sizeof(double) * Nlattice * Q);
	h_f_post = (double*)malloc(sizeof(double) * Nlattice * Q);

	output_rho = (double*)malloc(sizeof(double) * Nlattice);
	output_ux = (double*)malloc(sizeof(double) * Nlattice);
	output_uy = (double*)malloc(sizeof(double) * Nlattice);
	output_uz = (double*)malloc(sizeof(double) * Nlattice);

	cudaMalloc((void**)&d_geo, Nlattice * sizeof(int));
	cudaMalloc((void**)&d_f, sizeof(double) * Nlattice * Q);
	cudaMalloc((void**)&d_f_post, sizeof(double) * Nlattice * Q);
	cudaMalloc((void**)&d_feq, sizeof(double) * Nlattice * Q);
	cudaMalloc((void**)&m_f, sizeof(double) * Nlattice * Q);
	cudaMalloc((void**)&m_eq, sizeof(double) * Nlattice * Q);
	cudaMalloc((void**)&inv_f, sizeof(double) * Nlattice * Q);
	cudaMalloc((void**)&inv_feq, sizeof(double) * Nlattice * Q);

	cudaMalloc((void**)&d_rho, Nlattice * sizeof(double));
	cudaMalloc((void**)&d_ux, Nlattice * sizeof(double));
	cudaMalloc((void**)&d_uy, Nlattice * sizeof(double));
	cudaMalloc((void**)&d_uz, Nlattice * sizeof(double));
	
	cudaMalloc((void**)&s_xx, Nlattice * sizeof(double));
	cudaMalloc((void**)&s_xy, Nlattice * sizeof(double));
	cudaMalloc((void**)&s_xz, Nlattice * sizeof(double));
	cudaMalloc((void**)&s_yx, Nlattice * sizeof(double));
	cudaMalloc((void**)&s_yy, Nlattice * sizeof(double));
	cudaMalloc((void**)&s_yz, Nlattice * sizeof(double));
	cudaMalloc((void**)&s_zx, Nlattice * sizeof(double));
	cudaMalloc((void**)&s_zy, Nlattice * sizeof(double));
	cudaMalloc((void**)&s_zz, Nlattice * sizeof(double));
	cudaMalloc((void**)&sigma, Nlattice * sizeof(double));

	ifstream inf("D:\\Structrue-96.dat"); //read the geo file

	int i = 0;

	while (inf >> h_geo[i])
		++i;
	inf.close();

	for (int z = 0; z < Nz; z++)
	{
		for (int y = 0; y < Ny; y++)
		{
			for (int x = 0; x < Nx; x++)
			{
				int index = z * Nx * Ny + y * Nx + x;

				if (h_geo[index] == 0 || h_geo[index] == 3 || h_geo[index] == 4)
				{
					h_ux[index] = ux0;
					h_uy[index] = uy0;
					h_uz[index] = uz0 - gravity / 2;					
					h_rho[index] = rho0;
				}
				else if (h_geo[index] == 1)
				{
					h_ux[index] = 0.0;
					h_uy[index] = 0.0;
					h_uz[index] = 0.0;					
					h_rho[index] = 0.0;
				}
				else if (h_geo[index] == 2 || h_geo[index] == 5 || h_geo[index] == 6)
				{
					h_ux[index] = 0.0;
					h_uy[index] = 0.0;
					h_uz[index] = 0.0;					
					h_rho[index] = rho0;
				}
			}
		}
	}
	for (int z = 0; z < Nz; z++)
	{
		for (int y = 0; y < Ny; y++)
		{
			for (int x = 0; x < Nx; x++)
			{
				int index = z * Nx * Ny + y * Nx + x;

				for (int q = 0; q < Q; q++)
				{
					if (h_geo[index] == 0 || h_geo[index] == 2 || h_geo[index] == 3 || h_geo[index] == 4 || h_geo[index] == 5 || h_geo[index] == 6)
					{
						h_f[Nlattice * q + index] = _w[q] * h_rho[index] * (1 + 3.0 * (_cx[q] * h_ux[index] + _cy[q] * h_uy[index] + _cz[q] * h_uz[index])
							+ 4.5 * (_cx[q] * h_ux[index] + _cy[q] * h_uy[index] + _cz[q] * h_uz[index]) * (_cx[q] * h_ux[index] + _cy[q] * h_uy[index] + _cz[q] * h_uz[index])
							- 1.5 * (h_ux[index] * h_ux[index] + h_uy[index] * h_uy[index] + h_uz[index] * h_uz[index]));
						h_f_post[Nlattice * q + index] = 0.0;
					}
					else if (h_geo[index] == 1)
					{
						h_f[Nlattice * q + index] = 0.0;
						h_f_post[Nlattice * q + index] = 0.0;
					}
				}
			}
		}
	}
	cudaMemcpy(d_geo, h_geo, Nlattice * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_f, h_f, QNlattice * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_f_post, h_f_post, QNlattice * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ux, h_ux, Nlattice * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_uy, h_uy, Nlattice * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_uz, h_uz, Nlattice * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rho, h_rho, Nlattice * sizeof(double), cudaMemcpyHostToDevice);		
}

//2.gpu side functions
__global__ void kernelfeq(int* __restrict__ d_geo, double* __restrict__ d_feq,
	double* __restrict__ d_rho, double* __restrict__ d_ux, double* __restrict__ d_uy, double* __restrict__ d_uz);

__global__ void kernelrate_strain(int* __restrict__ d_geo, double* __restrict__ d_feq, double* __restrict__ d_f,
	double* __restrict__ s_xx, double* __restrict__ s_xy, double* __restrict__ s_xz, 
	double* __restrict__ s_yx, double* __restrict__ s_yy, double* __restrict__ s_yz,
	double* __restrict__ s_zx, double* __restrict__ s_zy, double* __restrict__ s_zz, double* __restrict__ sigma);

__global__ void kernelmf_meq(int* __restrict__ d_geo, double* __restrict__ d_f, double* __restrict__ m_f, double* __restrict__ m_eq,
	double* __restrict__ sigma, double* __restrict__ d_rho, double* __restrict__ d_ux, double* __restrict__ d_uy, double* __restrict__ d_uz);

__global__ void kernelInvf_feq(int* __restrict__ d_geo, double* __restrict__ m_f, double* __restrict__ m_eq, 
	double* __restrict__ inv_f, double* __restrict__ inv_feq);

__global__ void kernelcollision(int* __restrict__ d_geo, double* __restrict__ inv_f, double* __restrict__ inv_feq, 
	double* __restrict__ d_f, double* __restrict__ d_f_post);

__global__ void kernelboundary(int* __restrict__ d_geo, double* __restrict__ d_f, double* __restrict__ d_rho);

__global__ void kernelmacroscopic(int* __restrict__ d_geo, double* __restrict__ d_f, double* __restrict__ d_rho,
	double* __restrict__ d_ux, double* __restrict__ d_uy, double* __restrict__ d_uz);

//3.class functions
void MRTGPU::feq()
{
	kernelfeq << <grid, block >> > (d_geo, d_feq, d_rho, d_ux, d_uy, d_uz);
	cudaDeviceSynchronize();
}
void MRTGPU::rate_strain()
{
	kernelrate_strain << <grid, block >> > (d_geo, d_feq, d_f, s_xx, s_xy, s_xz, s_yx, s_yy, s_yz, s_zx, s_zy, s_zz, sigma);
	cudaDeviceSynchronize();
}
void MRTGPU::mf_meq()
{
	kernelmf_meq << <grid, block >> > (d_geo, d_f, m_f, m_eq, sigma, d_rho, d_ux, d_uy, d_uz);
	cudaDeviceSynchronize();
}
void MRTGPU::Invf_feq()
{
	kernelInvf_feq << <grid, block >> > (d_geo, m_f, m_eq, inv_f, inv_feq);
	cudaDeviceSynchronize();
}
void MRTGPU::collision()
{
	kernelcollision << <grid, block >> > (d_geo, inv_f, inv_feq, d_f, d_f_post);
	cudaDeviceSynchronize();
}
void MRTGPU::boundary()
{
	kernelboundary << <grid, block >> > (d_geo, d_f, d_rho);
	cudaDeviceSynchronize();
}
void MRTGPU::macroscopic()
{
	kernelmacroscopic << <grid, block >> > (d_geo, d_f, d_rho, d_ux, d_uy, d_uz);
	cudaDeviceSynchronize();
}

//4.gpu functions definition
__global__ void kernelfeq(int* __restrict__ d_geo, double* __restrict__ d_feq,
	double* __restrict__ d_rho, double* __restrict__ d_ux, double* __restrict__ d_uy, double* __restrict__ d_uz)
{
	const double rho0 = 1.0;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	int index = k * N[0] * N[1] + j * N[0] + i;
	int Nlattice = N[0] * N[1] * N[2];

	for (int q = 0; q < 19; q++)
	{
		if (d_geo[index] == 0 || d_geo[index] == 2 || d_geo[index] == 3 || d_geo[index] == 4 || d_geo[index] == 5 || d_geo[index] == 6)
			d_feq[index + Nlattice * q] = w[q] * (d_rho[index] + rho0 * (3.0 * (cx[q] * d_ux[index] + cy[q] * d_uy[index] + cz[q] * d_uz[index])
				+ 4.5 * (cx[q] * d_ux[index] + cy[q] * d_uy[index] + cz[q] * d_uz[index]) * (cx[q] * d_ux[index] + cy[q] * d_uy[index] + cz[q] * d_uz[index])
				- 1.5 * (d_ux[index] * d_ux[index] + d_uy[index] * d_uy[index] + d_uz[index] * d_uz[index])));
		else if (d_geo[index] == 1)
			d_feq[index + Nlattice * q] = 0.0;
	}
}

__global__ void kernelrate_strain(int* __restrict__ d_geo, double* __restrict__ d_feq, double* __restrict__ d_f,
	double* __restrict__ s_xx, double* __restrict__ s_xy, double* __restrict__ s_xz,
	double* __restrict__ s_yx, double* __restrict__ s_yy, double* __restrict__ s_yz,
	double* __restrict__ s_zx, double* __restrict__ s_zy, double* __restrict__ s_zz, double* __restrict__ sigma)
{
	const int Q = 19;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	int index = k * N[0] * N[1] + j * N[0] + i;
	int Nlattice = N[0] * N[1] * N[2];

	s_xx[index] = 0.0; s_xy[index] = 0.0; s_xz[index] = 0.0;
	s_yx[index] = 0.0; s_yy[index] = 0.0; s_yz[index] = 0.0;
	s_zx[index] = 0.0; s_zy[index] = 0.0; s_zz[index] = 0.0;

	for (int q = 0; q < Q; q++)
	{
		s_xx[index] += (d_f[index + Nlattice * q] - d_feq[index + Nlattice * q]) * cx[q] * cx[q];
		s_xy[index] += (d_f[index + Nlattice * q] - d_feq[index + Nlattice * q]) * cx[q] * cy[q];
		s_xz[index] += (d_f[index + Nlattice * q] - d_feq[index + Nlattice * q]) * cx[q] * cz[q];
		s_yx[index] += (d_f[index + Nlattice * q] - d_feq[index + Nlattice * q]) * cx[q] * cy[q];
		s_yy[index] += (d_f[index + Nlattice * q] - d_feq[index + Nlattice * q]) * cy[q] * cy[q];
		s_yz[index] += (d_f[index + Nlattice * q] - d_feq[index + Nlattice * q]) * cy[q] * cz[q];
		s_zx[index] += (d_f[index + Nlattice * q] - d_feq[index + Nlattice * q]) * cx[q] * cz[q];
		s_zy[index] += (d_f[index + Nlattice * q] - d_feq[index + Nlattice * q]) * cy[q] * cz[q];
		s_zz[index] += (d_f[index + Nlattice * q] - d_feq[index + Nlattice * q]) * cz[q] * cz[q];
	}
	if (d_geo[index] == 0 || d_geo[index] == 2 || d_geo[index] == 3 || d_geo[index] == 4 || d_geo[index] == 5 || d_geo[index] == 6)
		sigma[index] = sqrt(2 * (s_xx[index] * s_xx[index] + s_xy[index] * s_xy[index] + s_xz[index] * s_xz[index]
			+ s_yx[index] * s_yx[index] + s_yy[index] * s_yy[index] + s_yz[index] * s_yz[index]
			+ s_zx[index] * s_zx[index] + s_zy[index] * s_zy[index] + s_zz[index] * s_zz[index]));
	else if (d_geo[index] == 1)
		sigma[index] = 0.0;
}

__global__ void kernelmf_meq(int* __restrict__ d_geo, double* __restrict__ d_f, double* __restrict__ m_f, double* __restrict__ m_eq, 
	double* __restrict__ sigma,	double* __restrict__ d_rho, double* __restrict__ d_ux, double* __restrict__ d_uy, double* __restrict__ d_uz)
{
	const double rho0 = 1.0;
	const double tau = 0.55;/////////////calculaton
	const double C_Smag = 0.16;

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	int index = k * N[0] * N[1] + j * N[0] + i;
	int Nlattice = N[0] * N[1] * N[2];

	const double S_diag[19] = { 0.0, 1.19,1.4,0.0,1.2,0.0,1.2,0.0,1.2,1.0,1.4,1.0,1.4,1.0,1.0,1.0,1.98,1.98,1.98 };
	const double F_diag[19] = { 19.0,2394.0,252.0,10.0,40.0,10.0,40.0,10.0,40.0,36.0,72.0,12.0,24.0,4.0,4.0,4.0,8.0,8.0,8.0 };

	const double mm[19][19] = {
	{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0},
	{-30.0,-11.0,-11.0,-11.0,-11.0,-11.0,-11.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0},
	{12.0,-4.0,-4.0,-4.0,-4.0,-4.0,-4.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0},
	{0.0,1.0,-1.0,0.0,0.0,0.0,0.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,0.0,0.0,0.0,0.0},
	{0.0,-4.0,4.0,0.0,0.0,0.0,0.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,0.0,0.0,0.0,0.0},
	{0.0,0.0,0.0,1.0,-1.0,0.0,0.0,1.0,1.0,-1.0,-1.0,0.0,0.0,0.0,0.0,1.0,-1.0,1.0,-1.0},
	{0.0,0.0,0.0,-4.0,4.0,0.0,0.0,1.0,1.0,-1.0,-1.0,0.0,0.0,0.0,0.0,1.0,-1.0,1.0,-1.0},
	{0.0,0.0,0.0,0.0,0.0,1.0,-1.0,0.0,0.0,0.0,0.0,1.0,1.0,-1.0,-1.0,1.0,1.0,-1.0,-1.0},
	{0.0,0.0,0.0,0.0,0.0,-4.0,4.0,0.0,0.0,0.0,0.0,1.0,1.0,-1.0,-1.0,1.0,1.0,-1.0,-1.0},
	{0.0,2.0,2.0,-1.0,-1.0,-1.0,-1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,-2.0,-2.0,-2.0,-2.0},
	{0.0,-4.0,-4.0,2.0,2.0,2.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,-2.0,-2.0,-2.0,-2.0},
	{0.0,0.0,0.0,1.0,1.0,-1.0,-1.0,1.0,1.0,1.0,1.0,-1.0,-1.0,-1.0,-1.0,0.0,0.0,0.0,0.0},
	{0.0,0.0,0.0,-2.0,-2.0,2.0,2.0,1.0,1.0,1.0,1.0,-1.0,-1.0,-1.0,-1.0,0.0,0.0,0.0,0.0},
	{0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,-1.0,-1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
	{0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,-1.0,-1.0,1.0},
	{0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,-1.0,-1.0,1.0,0.0,0.0,0.0,0.0},
	{0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,-1.0,1.0,-1.0,-1.0,1.0,-1.0,1.0,0.0,0.0,0.0,0.0},
	{0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0,-1.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,-1.0,1.0,-1.0},
	{0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,-1.0,-1.0,-1.0,-1.0,1.0,1.0} };

	for (int q = 0; q < 19; q++)
	{
		m_f[index + Nlattice * q] = 0.0;
		m_eq[index + Nlattice * q] = 0.0;
	}
	if (d_geo[index] == 0 || d_geo[index] == 2 || d_geo[index] == 3 || d_geo[index] == 4 || d_geo[index] == 5 || d_geo[index] == 6)
	{
		for (int q = 0; q < Q; q++)
		{
			for (int qq = 0; qq < Q; qq++)
			{
				if (q == 0 || q == 1 || q == 2 || q == 3 || q == 4 || q == 5 || q == 6 || q == 7 || q == 8 || q == 10 || q == 12 || q == 16 || q == 17 || q == 18)
					m_f[index + Nlattice * q] += S_diag[q] / F_diag[q] * mm[q][qq] * d_f[index + Nlattice * qq];
				else if (q == 9 || q == 11 || q == 13 || q == 14 || q == 15)
					m_f[index + Nlattice * q] += mm[q][qq] * d_f[index + Nlattice * qq] / (0.5 * (sqrt(tau * tau + 18.0 * C_Smag * C_Smag * sigma[index]) + tau)) / F_diag[q];
			}//tau_t = 0.5*(pow(pow(tau,2) + 18.0*pow(C_Smagorinsky,2)*sigma[index],0.5) - tau);
		}
		m_eq[index + Nlattice * 0] = S_diag[0] / F_diag[0] * d_rho[index];
		m_eq[index + Nlattice * 1] = S_diag[1] / F_diag[1] * ((-11.0) * d_rho[index] + 19.0 * (d_ux[index] * d_ux[index] + d_uy[index] * d_uy[index] + d_uz[index] * d_uz[index]));
		m_eq[index + Nlattice * 2] = S_diag[2] / F_diag[2] * (-475.0 / 63.0) * (d_ux[index] * d_ux[index] + d_uy[index] * d_uy[index] + d_uz[index] * d_uz[index]);
		m_eq[index + Nlattice * 3] = S_diag[3] / F_diag[3] * d_ux[index];
		m_eq[index + Nlattice * 4] = S_diag[4] / F_diag[4] * (-2.0 / 3.0) * d_ux[index];
		m_eq[index + Nlattice * 5] = S_diag[5] / F_diag[5] * d_uy[index];
		m_eq[index + Nlattice * 6] = S_diag[6] / F_diag[6] * (-2.0 / 3.0) * d_uy[index];
		m_eq[index + Nlattice * 7] = S_diag[7] / F_diag[7] * d_uz[index];
		m_eq[index + Nlattice * 8] = S_diag[8] / F_diag[8] * (-2.0 / 3.0) * d_uz[index];
		m_eq[index + Nlattice * 9] = 2.0 * d_ux[index] * d_ux[index] - (d_uy[index] * d_uy[index] + d_uz[index] * d_uz[index])
			/ (0.5 * (sqrt(tau * tau + 18.0 * C_Smag * C_Smag * sigma[index]) + tau)) / F_diag[9];
		m_eq[index + Nlattice * 10] = 0.0;
		m_eq[index + Nlattice * 11] = (d_uy[index] * d_uy[index] - d_uz[index] * d_uz[index])
			/ (0.5 * (sqrt(tau * tau + 18.0 * C_Smag * C_Smag * sigma[index]) + tau)) / F_diag[11];
		m_eq[index + Nlattice * 12] = 0.0;
		m_eq[index + Nlattice * 13] = d_ux[index] * d_uy[index]
			/ (0.5 * (sqrt(tau * tau + 18.0 * C_Smag * C_Smag * sigma[index]) + tau)) / F_diag[13];
		m_eq[index + Nlattice * 14] = d_uy[index] * d_uz[index]
			/ (0.5 * (sqrt(tau * tau + 18.0 * C_Smag * C_Smag * sigma[index]) + tau)) / F_diag[14];
		m_eq[index + Nlattice * 15] = d_ux[index] * d_uz[index]
			/ (0.5 * (sqrt(tau * tau + 18.0 * C_Smag * C_Smag * sigma[index]) + tau)) / F_diag[15];
		m_eq[index + Nlattice * 16] = 0.0;
		m_eq[index + Nlattice * 17] = 0.0;
		m_eq[index + Nlattice * 18] = 0.0;
	}
	if (d_geo[index] == 1)
	{
		for (int q = 0; q < 19; q++)
		{
			m_f[index + Nlattice * q] = 0.0;
			m_eq[index + Nlattice * q] = 0.0;
		}
	}
}

__global__ void kernelInvf_feq(int* __restrict__ d_geo, double* __restrict__ m_f, double* __restrict__ m_eq,
	double* __restrict__ inv_f, double* __restrict__ inv_feq)
{
	const int Q = 19;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	int index = k * N[0] * N[1] + j * N[0] + i;
	int Nlattice = N[0] * N[1] * N[2];

	const double mmt[19][19] = {
	{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0},
	{-30.0,-11.0,-11.0,-11.0,-11.0,-11.0,-11.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0,8.0},
	{12.0,-4.0,-4.0,-4.0,-4.0,-4.0,-4.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0},
	{0.0,1.0,-1.0,0.0,0.0,0.0,0.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,0.0,0.0,0.0,0.0},
	{0.0,-4.0,4.0,0.0,0.0,0.0,0.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0,0.0,0.0,0.0,0.0},
	{0.0,0.0,0.0,1.0,-1.0,0.0,0.0,1.0,1.0,-1.0,-1.0,0.0,0.0,0.0,0.0,1.0,-1.0,1.0,-1.0},
	{0.0,0.0,0.0,-4.0,4.0,0.0,0.0,1.0,1.0,-1.0,-1.0,0.0,0.0,0.0,0.0,1.0,-1.0,1.0,-1.0},
	{0.0,0.0,0.0,0.0,0.0,1.0,-1.0,0.0,0.0,0.0,0.0,1.0,1.0,-1.0,-1.0,1.0,1.0,-1.0,-1.0},
	{0.0,0.0,0.0,0.0,0.0,-4.0,4.0,0.0,0.0,0.0,0.0,1.0,1.0,-1.0,-1.0,1.0,1.0,-1.0,-1.0},
	{0.0,2.0,2.0,-1.0,-1.0,-1.0,-1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,-2.0,-2.0,-2.0,-2.0},
	{0.0,-4.0,-4.0,2.0,2.0,2.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,-2.0,-2.0,-2.0,-2.0},
	{0.0,0.0,0.0,1.0,1.0,-1.0,-1.0,1.0,1.0,1.0,1.0,-1.0,-1.0,-1.0,-1.0,0.0,0.0,0.0,0.0},
	{0.0,0.0,0.0,-2.0,-2.0,2.0,2.0,1.0,1.0,1.0,1.0,-1.0,-1.0,-1.0,-1.0,0.0,0.0,0.0,0.0},
	{0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,-1.0,-1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0},
	{0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,-1.0,-1.0,1.0},
	{0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,-1.0,-1.0,1.0,0.0,0.0,0.0,0.0},
	{0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,-1.0,1.0,-1.0,-1.0,1.0,-1.0,1.0,0.0,0.0,0.0,0.0},
	{0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0,-1.0,1.0,1.0,0.0,0.0,0.0,0.0,1.0,-1.0,1.0,-1.0},
	{0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,-1.0,-1.0,-1.0,-1.0,1.0,1.0} };

	for (int q = 0; q < Q; q++)
	{
		inv_f[index + Nlattice * q] = 0.0;
		inv_feq[index + Nlattice * q] = 0.0;
	}
	if (d_geo[index] == 0 || d_geo[index] == 2 || d_geo[index] == 3 || d_geo[index] == 4 || d_geo[index] == 5 || d_geo[index] == 6)
	{
		for (int q = 0; q < Q; q++)
		{
			for (int qq = 0; qq < Q; qq++)
			{
				inv_f[index + Nlattice * q] += mmt[qq][q] * m_f[index + Nlattice * qq];
				inv_feq[index + Nlattice * q] += mmt[qq][q] * m_eq[index + Nlattice * qq];
			}
		}
	}
}

__global__ void kernelcollision(int* __restrict__ d_geo, double* __restrict__ inv_f, double* __restrict__ inv_feq,
	double* __restrict__ d_f, double* __restrict__ d_f_post)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	int index = k * N[0] * N[1] + j * N[0] + i;
	int Nlattice = N[0] * N[1] * N[2];
	int indexf[19];
	const double gravity = 0.00098;

	for (int q = 0; q < 19; q++)
	{
		int i_1 = (i + cx[q] + N[0]) % N[0];
		int j_1 = (j + cy[q] + N[1]) % N[1];
		int k_1 = (k + cz[q] + N[2]) % N[2];
		indexf[q] = k_1 * N[0] * N[1] + j_1 * N[0] + i_1;

		if (d_geo[index] == 0 || d_geo[index] == 2 || d_geo[index] == 3 || d_geo[index] == 4 || d_geo[index] == 5 || d_geo[index] == 6)
			d_f_post[indexf[q] + Nlattice * q] = d_f[index + Nlattice * q] - (inv_f[index + Nlattice * q] - inv_feq[index + Nlattice * q])
			- 3.0 * w[q] * (cz[q] * gravity);

		else if (d_geo[index] == 1)
			d_f_post[indexf[q] + Nlattice * q] = 0.0;
	}

	for (int q = 0; q < 19; q++) //swap
	{
		d_f[index + Nlattice * q] = d_f_post[index + Nlattice * q];
		d_f[index + Nlattice * q] = d_f_post[index + Nlattice * q];
	}
}

__global__ void kernelboundary(int* __restrict__ d_geo, double* __restrict__ d_f, double* __restrict__ d_rho)
{
	const double u0 = 0.02 - 0.00098 / 2;///////////////////////////////////calculation
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	int index = k * N[0] * N[1] + j * N[0] + i;
	int index_out = (k - 1) * N[0] * N[1] + j * N[0] + i;
	int Nlattice = N[0] * N[1] * N[2];
	int indexf[19];

	for (int q = 0; q < 19; q++)
	{
		int i_1 = (i + cx[q] + N[0]) % N[0];
		int j_1 = (j + cy[q] + N[1]) % N[1];
		int k_1 = (k + cz[q] + N[2]) % N[2];
		indexf[q] = k_1 * N[0] * N[1] + j_1 * N[0] + i_1;

		if (d_geo[index] == 2 || d_geo[index] == 5 || d_geo[index] == 6)
		{
			if (d_geo[indexf[q]] == 1)
			{
				if (q == 1) d_f[index + Nlattice * 2] = d_f[index + Nlattice * q];
				else if (q == 2) d_f[index + Nlattice * 1] = d_f[index + Nlattice * q];
				else if (q == 3) d_f[index + Nlattice * 4] = d_f[index + Nlattice * q];
				else if (q == 4) d_f[index + Nlattice * 3] = d_f[index + Nlattice * q];
				else if (q == 5) d_f[index + Nlattice * 6] = d_f[index + Nlattice * q];
				else if (q == 6) d_f[index + Nlattice * 5] = d_f[index + Nlattice * q];
				else if (q == 7) d_f[index + Nlattice * 10] = d_f[index + Nlattice * q];
				else if (q == 8) d_f[index + Nlattice * 9] = d_f[index + Nlattice * q];
				else if (q == 9) d_f[index + Nlattice * 8] = d_f[index + Nlattice * q];
				else if (q == 10) d_f[index + Nlattice * 7] = d_f[index + Nlattice * q];
				else if (q == 11) d_f[index + Nlattice * 14] = d_f[index + Nlattice * q];
				else if (q == 12) d_f[index + Nlattice * 13] = d_f[index + Nlattice * q];
				else if (q == 13) d_f[index + Nlattice * 12] = d_f[index + Nlattice * q];
				else if (q == 14) d_f[index + Nlattice * 11] = d_f[index + Nlattice * q];
				else if (q == 15) d_f[index + Nlattice * 18] = d_f[index + Nlattice * q];
				else if (q == 16) d_f[index + Nlattice * 17] = d_f[index + Nlattice * q];
				else if (q == 17) d_f[index + Nlattice * 16] = d_f[index + Nlattice * q];
				else if (q == 18) d_f[index + Nlattice * 15] = d_f[index + Nlattice * q];
			}
		}
		else if (d_geo[index] == 3)
		{
			if (q == 6)
				d_f[index + Nlattice * 5] = d_f[index + Nlattice * q] + u0 * d_rho[index] / 3.0;
			else if (q == 13)
				d_f[index + Nlattice * 12] = d_f[index + Nlattice * q] + u0 * d_rho[index] / 6.0
				- 0.5 * (cx[12] * (d_f[index + Nlattice * 1] - d_f[index + Nlattice * 2])
					+ cy[12] * (d_f[index + Nlattice * 3] - d_f[index + Nlattice * 4]));
			else if (q == 14)
				d_f[index + Nlattice * 11] = d_f[index + Nlattice * q] + u0 * d_rho[index] / 6.0
				- 0.5 * (cx[11] * (d_f[index + Nlattice * 1] - d_f[index + Nlattice * 2])
					+ cy[11] * (d_f[index + Nlattice * 3] - d_f[index + Nlattice * 4]));
			else if (q == 17)
				d_f[index + Nlattice * 16] = d_f[index + Nlattice * q] + u0 * d_rho[index] / 6.0
				- 0.5 * (cx[16] * (d_f[index + Nlattice * 1] - d_f[index + Nlattice * 2])
					+ cy[16] * (d_f[index + Nlattice * 3] - d_f[index + Nlattice * 4]));
			else if (q == 18)
				d_f[index + Nlattice * 15] = d_f[index + Nlattice * q] + u0 * d_rho[index] / 6.0
				- 0.5 * (cx[15] * (d_f[index + Nlattice * 1] - d_f[index + Nlattice * 2])
					+ cy[15] * (d_f[index + Nlattice * 3] - d_f[index + Nlattice * 4]));
		}
		else if (d_geo[index] == 4)
		{
			if (q == 5)
				d_f[index + Nlattice * 6] = d_f[index_out + Nlattice * 6];
			else if (q == 11)
				d_f[index + Nlattice * 14] = d_f[index_out + Nlattice * 14];
			else if (q == 12)
				d_f[index + Nlattice * 13] = d_f[index_out + Nlattice * 13];
			else if (q == 15)
				d_f[index + Nlattice * 18] = d_f[index_out + Nlattice * 18];
			else if (q == 16)
				d_f[index + Nlattice * 17] = d_f[index_out + Nlattice * 17];
		}
	}
}

__global__ void kernelmacroscopic(int* __restrict__ d_geo, double* __restrict__ d_f, double* __restrict__ d_rho,
	double* __restrict__ d_ux, double* __restrict__ d_uy, double* __restrict__ d_uz)
{
	const double rho0 = 1.0;
	const double gravity = 0.00098;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	int index = k * N[0] * N[1] + j * N[0] + i;
	int Nlattice = N[0] * N[1] * N[2];
	
	d_rho[index] = 0.0;
	d_ux[index] = 0.0;
	d_uy[index] = 0.0;
	d_uz[index] = 0.0;

	if (d_geo[index] == 0 || d_geo[index] == 3|| d_geo[index] == 4)
	{
		for (int q = 0; q < 19; q++)
		{
			d_rho[index] += d_f[index + Nlattice * q];
			d_ux[index] += cx[q] * d_f[index + Nlattice * q];
			d_uy[index] += cy[q] * d_f[index + Nlattice * q];
			d_uz[index] += cz[q] * d_f[index + Nlattice * q];
		}
		d_uz[index] += d_uz[index] - gravity / 2;
	}
	else if (d_geo[index] == 2 || d_geo[index] == 5 || d_geo[index] == 6)
	{
		for (int q = 0; q < 19; q++)
		{
			d_rho[index] += d_f[index + Nlattice * q];
			d_ux[index] = 0.0;
			d_uy[index] = 0.0;
			d_uz[index] = 0.0;//no slip wall
		}
	}
	else if (d_geo[index] == 1)
	{
		d_rho[index] = 0.0;
		d_ux[index] = 0.0;
		d_uy[index] = 0.0;
		d_uz[index] = 0.0;
	}	
}

void MRTGPU::output(int t)
{
	ofstream outputfile;
	stringstream sfile;
	sfile << "D:\\MRTforce-" << t << ".dat";
	string datafilename = sfile.str();

	outputfile.open(datafilename.c_str());
	outputfile << "TITLE = \"Case Data\"" << endl;
	outputfile << "VARIABLES = \"X\", \"Y\", \"Z\", \"Density\", \"Uz\"" << endl;
	outputfile << "ZONE I = 96, J = 104, K = 104, DATAPACKING = POINT" << endl;

	int Nlattice = Nx * Ny * Nz;
	cudaMemcpy(output_rho, d_rho, Nlattice * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(output_uz, d_uz, Nlattice * sizeof(double), cudaMemcpyDeviceToHost);

	for (int z = 0; z < Nz; z++)
	{
		for (int y = 0; y < Ny; y++)
		{
			for (int x = 0; x < Nx; x++)
			{
				int k = z * Nx * Ny + y * Nx + x;
				outputfile << setprecision(4) << x + 1 << "\t" << y + 1 << "\t " << z + 1 << "\t " << output_rho[k] << "\t " << output_uz[k] << endl;
			}
		}
	}
	outputfile.close();
}

void MRTGPU::freemem()
{
	free(h_geo);
	free(h_f);
	free(h_f_post);
	free(h_rho);
	free(h_ux);
	free(h_uy);
	free(h_uz);
	free(output_rho);
	free(output_ux);
	free(output_uy);
	free(output_uz);

	cudaFree(d_geo);
	cudaFree(d_f);
	cudaFree(d_f_post);	cudaFree(d_feq);
	cudaFree(m_f); cudaFree(m_eq);
	cudaFree(inv_f); cudaFree(inv_feq);
	cudaFree(d_rho);
	cudaFree(d_ux);
	cudaFree(d_uy);
	cudaFree(d_uz);
	cudaFree(s_xx); cudaFree(s_xy); cudaFree(s_xz);
	cudaFree(s_yx); cudaFree(s_yy); cudaFree(s_yz);
	cudaFree(s_zx); cudaFree(s_zy); cudaFree(s_zz);
	cudaFree(sigma);
}