/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.12.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2023
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

// thread-block size of gebak_gather_kernel, number of rows per thread (so that it is used
// for n <= GEBAK_GATHER_THDS * GEBAK_GATHER_ROWS), and number of columns of each matrix
// that it permutes at a time
#ifndef GEBAK_GATHER_THDS
#define GEBAK_GATHER_THDS 1024
#endif
#ifndef GEBAK_GATHER_ROWS
#define GEBAK_GATHER_ROWS 8
#endif
#ifndef GEBAK_GATHER_COLS
#define GEBAK_GATHER_COLS 64
#endif

/** GEBAK_SCALE_KERNEL scales rows ilo-1:ihi-1 of V by scale (side = right)
    or by 1/scale (side = left). Rows are skipped when ilo == ihi, as in LAPACK.
    Matrices with infoA > 0 (if infoA is not null) are skipped.
    Call with a 2D grid over the rows and columns of V, and the batch in z. **/
template <typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void gebak_scale_kernel(const rocblas_side side,
                                         const I n,
                                         const I m,
                                         const I* iloA,
                                         const I* ihiA,
                                         const S* scaleA,
                                         const rocblas_stride strideS,
                                         U VV,
                                         const rocblas_stride shiftV,
                                         const I ldv,
                                         const rocblas_stride strideV,
                                         const I* infoA)
{
    const I bid = hipBlockIdx_z;
    const I i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const I j = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;

    if(infoA && infoA[bid] > 0)
        return;
    const I ilo = iloA[bid];
    const I ihi = ihiA[bid];
    if(ilo == ihi || i < ilo - 1 || i >= ihi || i >= n || j >= m)
        return;

    const S* scale = scaleA + bid * strideS;
    T* V = load_ptr_batch<T>(VV, bid, shiftV, strideV);

    S s = (side == rocblas_side_right) ? scale[i] : S(1) / scale[i];
    V[idx2D(i, j, ldv)] *= s;
}

/** GEBAK_PERMUTE_KERNEL undoes the permutations recorded in scale, in the same
    order as LAPACK: rows ilo-2 down to 0, then rows ihi to n-1. The swaps must be
    applied sequentially, but the columns of V are independent; each thread
    processes one column (used when n is too large for gebak_gather_kernel).
    Matrices with infoA > 0 (if infoA is not null) are skipped.
    Call with a 1D grid over the columns of V, and the batch in y. **/
template <typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void gebak_permute_kernel(const I n,
                                           const I m,
                                           const I* iloA,
                                           const I* ihiA,
                                           const S* scaleA,
                                           const rocblas_stride strideS,
                                           U VV,
                                           const rocblas_stride shiftV,
                                           const I ldv,
                                           const rocblas_stride strideV,
                                           const I* infoA)
{
    const I bid = hipBlockIdx_y;
    const I j = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(j >= m || (infoA && infoA[bid] > 0))
        return;

    const I ilo = iloA[bid];
    const I ihi = ihiA[bid];
    const S* scale = scaleA + bid * strideS;
    T* V = load_ptr_batch<T>(VV, bid, shiftV, strideV);

    auto swap_rows = [&](const I i) {
        const I p = static_cast<I>(scale[i]) - 1;
        if(p != i && p >= 0 && p < n)
            swap(V[idx2D(i, j, ldv)], V[idx2D(p, j, ldv)]);
    };

    for(I i = std::min(ilo - 1, n) - 1; i >= 0; i--)
        swap_rows(i);
    for(I i = std::max(ihi, I(0)); i < n; i++)
        swap_rows(i);
}

/** GEBAK_GATHER_KERNEL applies the same swaps as gebak_permute_kernel, composed
    into one permutation: as V is permuted by the swaps s_1, ..., s_K in this order,
    the row r of the result is the row perm(r) = s_1(s_2(...s_K(r))) of V. Each thread
    (x, y) owns the rows r = x + k * blockDim.x, k < ROWS, and first follows them through
    the swaps in reverse order (rows n-1 down to ihi, then rows 0 to ilo-2, staged in
    shared memory). Then, for the columns j = blockIdx.x * blockDim.y + y + l * ncols,
    ncols = gridDim.x * blockDim.y, it loads the rows perm(r) that move, and, after a
    barrier, stores them as rows r (threads along the rows, for coalesced accesses).
    Matrices with infoA > 0 (if infoA is not null) are skipped.
    Call with blockDim.x * blockDim.y <= GEBAK_GATHER_THDS, n <= blockDim.x * ROWS,
    and the batch in y. **/
template <int ROWS, typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(GEBAK_GATHER_THDS)
    gebak_gather_kernel(const I n,
                        const I m,
                        const I* iloA,
                        const I* ihiA,
                        const S* scaleA,
                        const rocblas_stride strideS,
                        U VV,
                        const rocblas_stride shiftV,
                        const I ldv,
                        const rocblas_stride strideV,
                        const I* infoA)
{
    const I bid = hipBlockIdx_y;
    const I tx = hipThreadIdx_x;
    const I ty = hipThreadIdx_y;
    const I dimx = hipBlockDim_x;
    const I tid = tx + ty * dimx;
    const I nt = dimx * hipBlockDim_y;
    if(infoA && infoA[bid] > 0)
        return;

    // numbers of swaps of rows ihi:n-1 and of rows 0:ilo-2 (nothing to do without swaps,
    // as for most matrices)
    const I ilo = iloA[bid];
    const I ihi = ihiA[bid];
    const I nb = n - std::min(std::max(ihi, I(0)), n);
    const I na = std::max(std::min(ilo - 1, n), I(0));
    if(nb + na == 0)
        return;

    const S* scale = scaleA + bid * strideS;
    T* V = load_ptr_batch<T>(VV, bid, shiftV, strideV);
    __shared__ I si[GEBAK_GATHER_THDS];
    __shared__ I sp[GEBAK_GATHER_THDS];

    // compose the swaps (an invalid index means no swap, as in gebak_permute_kernel)
    I q[ROWS];
#pragma unroll
    for(int k = 0; k < ROWS; k++)
        q[k] = tx + k * dimx;
    for(I t0 = 0; t0 < nb + na; t0 += nt)
    {
        const I t = t0 + tid;
        if(t < nb + na)
        {
            const I i = (t < nb) ? n - 1 - t : t - nb;
            const I p = static_cast<I>(scale[i]) - 1;
            si[tid] = i;
            sp[tid] = (p >= 0 && p < n) ? p : i;
        }
        __syncthreads();

        const I tn = std::min(nt, nb + na - t0);
        for(I l = 0; l < tn; l++)
        {
            const I i = si[l];
            const I p = sp[l];
#pragma unroll
            for(int k = 0; k < ROWS; k++)
                q[k] = (q[k] == i) ? p : (q[k] == p ? i : q[k]);
        }
        __syncthreads();
    }

    // permute the columns (all the threads of the block go through the loop the same
    // number of times; the real and imaginary parts are kept apart, so that they stay in
    // registers)
    const I ncols = hipGridDim_x * hipBlockDim_y;
    S vr[ROWS], vi[ROWS];
    for(I j0 = hipBlockIdx_x * hipBlockDim_y; j0 < m; j0 += ncols)
    {
        const I j = j0 + ty;
#pragma unroll
        for(int k = 0; k < ROWS; k++)
        {
            const I r = tx + k * dimx;
            if(j < m && r < n && q[k] != r)
            {
                const T x = V[idx2D(q[k], j, ldv)];
                vr[k] = std::real(x);
                vi[k] = std::imag(x);
            }
        }
        __syncthreads();

#pragma unroll
        for(int k = 0; k < ROWS; k++)
        {
            const I r = tx + k * dimx;
            if(j < m && r < n && q[k] != r)
            {
                if constexpr(rocblas_is_complex<T>)
                    V[idx2D(r, j, ldv)] = T(vr[k], vi[k]);
                else
                    V[idx2D(r, j, ldv)] = vr[k];
            }
        }
    }
}

template <typename T, typename I, typename S>
rocblas_status rocsolver_gebak_argCheck(rocblas_handle handle,
                                        const rocsolver_balance job,
                                        const rocblas_side side,
                                        const I n,
                                        const I* ilo,
                                        const I* ihi,
                                        const S* scale,
                                        const I m,
                                        T V,
                                        const I ldv,
                                        const I batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(job != rocsolver_balance_none && job != rocsolver_balance_permute
       && job != rocsolver_balance_scale && job != rocsolver_balance_both)
        return rocblas_status_invalid_value;
    if(side != rocblas_side_left && side != rocblas_side_right)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || m < 0 || ldv < n || ldv < 1 || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && m && !V) || (n && !scale) || (n && batch_count && !ilo) || (n && batch_count && !ihi))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

/** The optional info is internal (for GEEV): the matrices with info > 0 are left
    unchanged. **/
template <bool BATCHED, bool STRIDED, typename T, typename I, typename S, typename U>
rocblas_status rocsolver_gebak_template(rocblas_handle handle,
                                        const rocsolver_balance job,
                                        const rocblas_side side,
                                        const I n,
                                        const I* ilo,
                                        const I* ihi,
                                        const S* scale,
                                        const rocblas_stride strideS,
                                        const I m,
                                        U V,
                                        const rocblas_stride shiftV,
                                        const I ldv,
                                        const rocblas_stride strideV,
                                        const I batch_count,
                                        const I* info = nullptr)
{
    ROCSOLVER_ENTER("gebak", "job:", job, "side:", side, "n:", n, "m:", m, "shiftV:", shiftV,
                    "ldv:", ldv, "bc:", batch_count);

    // quick return
    if(n == 0 || m == 0 || batch_count == 0 || job == rocsolver_balance_none)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // backward balance
    if(job == rocsolver_balance_scale || job == rocsolver_balance_both)
    {
        const I blocksx = (n - 1) / BS2 + 1;
        const I blocksy = (m - 1) / BS2 + 1;
        ROCSOLVER_LAUNCH_KERNEL((gebak_scale_kernel<T>), dim3(blocksx, blocksy, batch_count),
                                dim3(BS2, BS2), 0, stream, side, n, m, ilo, ihi, scale, strideS, V,
                                shiftV, ldv, strideV, info);
    }

    // backward permutation
    if(job == rocsolver_balance_permute || job == rocsolver_balance_both)
    {
        if(n <= I(GEBAK_GATHER_THDS) * GEBAK_GATHER_ROWS)
        {
            // threads along the rows: the smallest power of 2 that is at least n, or at least
            // 32 with n <= tx * rows, for rows = 4 or GEBAK_GATHER_ROWS; the other threads of
            // the block (up to GEBAK_GATHER_THDS) along the columns
            I tx = 1, ty = 1;
            while(tx < GEBAK_GATHER_THDS && tx < n && (tx < 32 || tx * 4 < n))
                tx *= 2;
            while(tx * ty < GEBAK_GATHER_THDS && ty < m)
                ty *= 2;
            const I blocks = (std::min(m, I(GEBAK_GATHER_COLS)) - 1) / ty + 1;
            if(n <= tx * 4)
                ROCSOLVER_LAUNCH_KERNEL((gebak_gather_kernel<4, T>), dim3(blocks, batch_count),
                                        dim3(tx, ty), 0, stream, n, m, ilo, ihi, scale, strideS, V,
                                        shiftV, ldv, strideV, info);
            else
                ROCSOLVER_LAUNCH_KERNEL((gebak_gather_kernel<GEBAK_GATHER_ROWS, T>),
                                        dim3(blocks, batch_count), dim3(tx, ty), 0, stream, n, m,
                                        ilo, ihi, scale, strideS, V, shiftV, ldv, strideV, info);
        }
        else
        {
            const I blocks = (m - 1) / BS1 + 1;
            ROCSOLVER_LAUNCH_KERNEL((gebak_permute_kernel<T>), dim3(blocks, batch_count), dim3(BS1),
                                    0, stream, n, m, ilo, ihi, scale, strideS, V, shiftV, ldv,
                                    strideV, info);
        }
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
