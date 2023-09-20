/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/x64/injectors/jit_uni_dropout_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;

template <cpu_isa_t isa, typename Wmm>
void jit_uni_dropout_injector_f32<isa, Wmm>::injector_preamble(
        const injector_utils::vmm_index_set_t &vmm_idxs) {
    using namespace Xbyak::util;
    preserved_vecs_count = 0;
    vecs_to_preserve = 3;
    const auto start_idx = *(vmm_idxs.begin());
    const auto end_idx = *(vmm_idxs.rbegin()) + 1;
    start_idx_tail = vmm_idxs.begin();

    for (size_t idx = preserved_vecs_count; idx < vecs_count; idx++) {
        if (preserved_vecs_count >= vecs_to_preserve) break;
        if (start_idx <= idx && idx < end_idx) continue;

        preserved_vec_idxs[preserved_vecs_count++] = idx;
    }

    size_t preserved_vecs_count_tail = vecs_to_preserve - preserved_vecs_count;
    for (size_t i = 0; i < preserved_vecs_count_tail; i++) {
        preserved_vec_idxs[preserved_vecs_count++] = *start_idx_tail;
        ++start_idx_tail;
    }

    assert(preserved_vecs_count == vecs_to_preserve);

    if (save_state_) {
        if (preserve_p_table_) h->push(p_table);
        if (preserve_vmm_) {
            if (preserved_vecs_count)
                h->sub(h->rsp, preserved_vecs_count * vlen);

            for (size_t i = 0; i < preserved_vecs_count; ++i)
                h->uni_vmovups(
                        h->ptr[h->rsp + i * vlen], Vmm(preserved_vec_idxs[i]));
        }


        load_table_addr();
    }

    assign_regs();
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_dropout_injector_f32<isa, Wmm>::injector_preamble_tail(
        const injector_utils::vmm_index_set_iterator_t &start_idx_it) {
    size_t tail_vecs_to_preserve = std::distance(start_idx_it, start_idx_tail);
    if (tail_vecs_to_preserve == 0) return;

    const int idx_off = vecs_to_preserve - tail_vecs_to_preserve;

    if (save_state_) {
        if (idx_off) h->add(h->rsp, idx_off * vlen);

        for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
            h->uni_vmovups(Vmm(preserved_vec_idxs[idx_off + i]),
                    h->ptr[h->rsp + i * vlen]);
    }

    for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
        preserved_vec_idxs[idx_off + i] += tail_vecs_to_preserve;

    if (save_state_ && preserve_vmm_) {
        for (size_t i = 0; i < tail_vecs_to_preserve; ++i)
            h->uni_vmovups(h->ptr[h->rsp + i * vlen],
                    Vmm(preserved_vec_idxs[idx_off + i]));

        if (idx_off) h->sub(h->rsp, idx_off * vlen);
    }


    assign_regs();
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_dropout_injector_f32<isa, Wmm>::injector_postamble() {
    using namespace Xbyak::util;
    if (!save_state_) return;

    if (preserve_vmm_) {
        for (size_t i = 0; i < preserved_vecs_count; ++i)
            h->uni_vmovups(
                    Vmm(preserved_vec_idxs[i]), h->ptr[h->rsp + i * vlen]);

        if (preserved_vecs_count) h->add(h->rsp, preserved_vecs_count * vlen);
    }

    if (preserve_p_table_) h->pop(p_table);
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_dropout_injector_f32<isa, Wmm>::assign_regs() {
    vmm_aux0 = Vmm(preserved_vec_idxs[0]);
    vmm_aux1 = Vmm(preserved_vec_idxs[1]);
    vmm_mask = Vmm(preserved_vec_idxs[2]);
}


template <cpu_isa_t isa, typename Wmm>
void jit_uni_dropout_injector_f32<isa, Wmm>::compute_body(
        const injector_utils::vmm_index_set_iterator_t &start_idx_it,
        const injector_utils::vmm_index_set_iterator_t &end_idx_it,
        const Xbyak::Address &mask_raw_addr,
        bool tail) {

    const  bool is_avx512 = is_superset(isa, avx512_core);
    constexpr bool is_ymm = std::is_same<Wmm, Xbyak::Ymm>::value;
    std::for_each(start_idx_it, end_idx_it, [&](size_t idx) {
        h->uni_vpaddd(vmm_aux0, vmm_state0, vmm_state3);
        h->uni_vpsrld(vmm_aux0, vmm_aux0, 9);

        h->uni_vorps(vmm_aux0, vmm_aux0, h->ptr[p_table + vlen * 9]);
        if (is_avx512) {
            h->vcmpps(k_mask, vmm_aux0, h->ptr[p_table + vlen * 10],
                    jit_generator::_cmp_nlt_us);
            h->uni_vmulps(
                    Vmm(idx) | k_mask | h->T_z, Vmm(idx), h->ptr[p_table + vlen * 8]);
            h->uni_vmovups(vmm_aux0, h->ptr[p_table + vlen * 11]);
            h->vmovdqu8(vmm_mask, vmm_aux0 | k_mask | h->T_z);

            auto xmm_mask = Xbyak::Xmm(vmm_mask.getIdx());
            h->store_bytes(xmm_mask, mask_raw_addr, (tail) ? 1 : vlen / sizeof(uint32_t));
        } else {
            h->uni_vcmpps(vmm_mask, vmm_aux0, h->ptr[p_table + vlen * 10],
                    jit_generator::_cmp_lt_os);
            h->uni_vmulps(Vmm(idx), Vmm(idx), h->ptr[p_table + vlen * 8]);
            h->uni_vxorps(vmm_aux0, vmm_aux0, vmm_aux0);
            h->uni_vmovups(vmm_aux1, h->ptr[p_table + vlen * 11]);
            
            h->uni_vblendvps(Vmm(idx), Vmm(idx), vmm_aux0, vmm_mask);
            h->uni_vblendvps(vmm_mask, vmm_aux1, vmm_aux0, vmm_mask);
            h->uni_vpshufb(vmm_mask, vmm_mask, h->ptr[p_table + vlen * 12]);

            auto xmm_mask = Xbyak::Xmm(vmm_mask.getIdx());
            
            if (tail) {
                h->uni_vpextrb(mask_raw_addr, xmm_mask, 0);
            } else {
                h->uni_vpextrd(mask_raw_addr, xmm_mask, 0);
                if (is_ymm) {
                    auto ymm_mask = Xbyak::Ymm(vmm_mask.getIdx());
                    h->vextractf128(xmm_mask, ymm_mask, 1);
                    h->uni_vpextrd(h->ptr[mask_raw_addr.getRegExp()
                                           + 4 * sizeof(uint8_t)],
                            xmm_mask, 0);
                }
            }
        }

        // generate next state

        h->uni_vpslld(vmm_aux0, vmm_state1, 9);
        h->uni_vxorps(vmm_state2, vmm_state2, vmm_state0);
        h->uni_vxorps(vmm_state3, vmm_state3, vmm_state1);
        h->uni_vxorps(vmm_state1, vmm_state1, vmm_state2);
        h->uni_vxorps(vmm_state0, vmm_state0, vmm_state3);
        h->uni_vxorps(vmm_state2, vmm_state2, vmm_aux0);

        h->uni_vpslld(vmm_aux0, vmm_state3, 11);
        h->uni_vpsrld(vmm_state3, vmm_state3, 32 - 11);
        h->uni_vorps(vmm_state3, vmm_state3, vmm_aux0);

    });
}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_dropout_injector_f32<isa, Wmm>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs,
        const Xbyak::Address &mask_raw_addr,
        bool tail) {

    if (vmm_idxs.empty()) return;
    const auto &start_idx_it = vmm_idxs.begin();
    const auto &end_idx_it = vmm_idxs.end();
    assert(*start_idx_it < *vmm_idxs.rbegin() + 1
            && *vmm_idxs.rbegin() <= vecs_count);

    injector_preamble(vmm_idxs);
    compute_body(start_idx_tail, end_idx_it, mask_raw_addr,  tail);
    injector_preamble_tail(start_idx_it);
    compute_body(start_idx_it, start_idx_tail, mask_raw_addr,  tail);
    injector_postamble();
}


namespace {
void internal_rng_float_jump(uint32_t *state0,
        uint32_t *state1, uint32_t *state2, uint32_t *state3) {
    static const uint32_t jump_table[]
            = {0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b};
    uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i, b;

    assert(4 == sizeof(jump_table) / sizeof(*jump_table));
    for (i = 0; i < 4; ++i) {
        for (b = 0; b < 32; ++b) {
            if (jump_table[i] & (1U << b)) {
                s0 ^= *state0;
                s1 ^= *state1;
                s2 ^= *state2;
                s3 ^= *state3;
            }
            { /* draw one more integer */
                const uint32_t t = *state1 << 9;
                *state2 ^= *state0;
                *state3 ^= *state1;
                *state1 ^= *state2;
                *state0 ^= *state3;
                *state2 ^= t;
                *state3 = ((*state3 << 11) | (*state3 >> (32 - 11)));
            }
        }
    }
    *state0 = s0;
    *state1 = s1;
    *state2 = s2;
    *state3 = s3;
}


float internal_rng_scalar_float_next(
        int i, 
    uint32_t*internal_rng_state0, uint32_t *internal_rng_state1,
                uint32_t *internal_rng_state2, uint32_t *internal_rng_state3) {
    const uint32_t rng_mantissa
            = (internal_rng_state0[i] + internal_rng_state3[i]) >> 9;
    const uint32_t t = internal_rng_state1[i] << 9;
    union {
        uint32_t i;
        float f;
    } rng = {0};

    internal_rng_state2[i] ^= internal_rng_state0[i];
    internal_rng_state3[i] ^= internal_rng_state1[i];
    internal_rng_state1[i] ^= internal_rng_state2[i];
    internal_rng_state0[i] ^= internal_rng_state3[i];
    internal_rng_state2[i] ^= t;
    internal_rng_state3[i] = ((internal_rng_state3[i] << 11)
            | (internal_rng_state3[i] >> (32 - 11)));

    rng.i = 0x3f800000 | rng_mantissa;
    return rng.f - 1.0f;
}





} // namespace
template <cpu_isa_t isa, typename Wmm>
void jit_uni_dropout_injector_f32<isa, Wmm>::prepare_table(bool gen_table) {
    if (!gen_table) return;
    const uint32_t internal_rng_state0[16] = {0x68b46ad5, 0x51a0a11b,
            0x9b531a7c, 0xa247d1b2, 0x303b55b8, 0x92f9e76, 0xc3dc2511,
            0xfac8eedf, 0x9e1be4d5, 0xa70f2f1b, 0x6dfc947c, 0x54e85fb2,
            0x3575b7e8, 0xc617c26, 0xc692c741, 0xff860c8f};
    const uint32_t internal_rng_state1[16] = {0x88b21ea8, 0xabdfc34a,
            0xc1b011df, 0xe2ddcc3d, 0xafcae700, 0x8ca73ae2, 0xe6c8e877,
            0xc5a53595, 0xdaaba4c8, 0xf9c6792a, 0x93a9abbf, 0xb0c4765d,
            0xd9bdfb86, 0xfad02664, 0x90bff4f1, 0xb3d22913};
    const uint32_t internal_rng_state2[16] = {0xd3ce3419, 0x2817c080,
            0x4b0f3620, 0xb0d6c2b9, 0xb4b482e5, 0x4f6d767c, 0x2c7580dc,
            0xd7ac7445, 0x1fc3a14d, 0xe41a55d4, 0x8702a374, 0x7cdb57ed,
            0x7ec8646b, 0x851190f2, 0xe6096652, 0x1dd092cb};
    const uint32_t internal_rng_state3[16] = {0x71c2063e, 0xd1181b9b,
            0xc31bc1da, 0x63c1dc7f, 0x6234826d, 0xc2ee9fc8, 0xd0ed4589,
            0x7037582c, 0xa44d49ee, 0x497544b, 0x16948e0a, 0xb64e93af,
            0x7749f737, 0xd793ea92, 0xc59030d3, 0x654a2d76};

    const uint32_t seed_internal_rng_state0 = 0x3914cbce;
    const uint32_t seed_internal_rng_state1 = 0x236ddde2;
    const uint32_t seed_internal_rng_state2 = 0xfbd9f499;
    const uint32_t seed_internal_rng_state3 = 0xa0da1da5;
    const uint64_t len = vlen / sizeof(uint32_t);

    //static const uint32_t temp_state[]
    //        = {31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
    //                131, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121, 120,
    //                119, 118, 117, 116, 231, 230, 229, 228, 227, 226, 225, 224,
    //                223, 222, 221, 220, 219, 218, 217, 216, 331, 330, 329, 328,
    //                327, 326, 325, 324, 323, 322, 321, 320, 319, 318, 317, 316};
    //uint64_t i;
    //
    //for (i = 0; i < 16; ++i) {
    //    internal_rng_state0[i] = temp_state[i];
    //    internal_rng_state1[i] = temp_state[i + 16];
    //    internal_rng_state2[i] = temp_state[i + 32];
    //    internal_rng_state3[i] = temp_state[i + 48];

    //    internal_rng_float_jump(/* progress each sequence by 2^64 */
    //            internal_rng_state0 + i, internal_rng_state1 + i,
    //            internal_rng_state2 + i, internal_rng_state3 + i);
    //}
    //for (i = 0; i < 1; ++i) {
    //    seed_internal_rng_state0 = seed;
    //    seed_internal_rng_state1 = seed;
    //    seed_internal_rng_state2 = seed;
    //    seed_internal_rng_state3 = seed;

    //    internal_rng_float_jump(/* progress each sequence by 2^64 */
    //            &seed_internal_rng_state0, &seed_internal_rng_state1,
    //            &seed_internal_rng_state2, &seed_internal_rng_state3);
    //}
    //printf("0:\n={ ");
    //for (i = 0; i < 16; ++i) {
    //    printf("0x%x, ", internal_rng_state0[i]);
    //}
    //printf("\n");
    //printf("1:\n={ ");
    //for (i = 0; i < 16; ++i) {
    //    printf("0x%x, ", internal_rng_state1[i]);
    //}
    //printf("\n");
    //printf("2:\n={ ");
    //for (i = 0; i < 16; ++i) {
    //    printf("0x%x, ", internal_rng_state2[i]);
    //}
    //printf("\n");
    //printf("3:\n={ ");
    //for (i = 0; i < 16; ++i) {
    //    printf("0x%x, ", internal_rng_state3[i]);
    //}
    //printf("\n");
    //printf("0x%x\n", seed_internal_rng_state0);
    //printf("0x%x\n", seed_internal_rng_state1);
    //printf("0x%x\n", seed_internal_rng_state2);
    //printf("0x%x\n", seed_internal_rng_state3);
    const bool is_avx512 = is_superset(isa, avx512_core);
    h->align(64);
    h->L(l_table);
    for (size_t d = 0; d < len; d++)
        h->dd(seed_internal_rng_state0);
    for (size_t d = 0; d < len; d++)
        h->dd(seed_internal_rng_state1);
    for (size_t d = 0; d < len; d++)
        h->dd(seed_internal_rng_state2);
    for (size_t d = 0; d < len; d++)
        h->dd(seed_internal_rng_state3);
    for (size_t d = 0; d < len; d++)
        h->dd(internal_rng_state0[d]);
    for (size_t d = 0; d < len; d++)
        h->dd(internal_rng_state1[d]);
    for (size_t d = 0; d < len; d++)
        h->dd(internal_rng_state2[d]);
    for (size_t d = 0; d < len; d++)
        h->dd(internal_rng_state3[d]);

    for (size_t d = 0; d < vlen; d += sizeof(uint32_t))
        h->dd(float2int(1.));
    for (size_t d = 0; d < len; d++)
        h->dd(0x3f800000);
    for (size_t d = 0; d < len; d++)
        h->dd(float2int(1.));
    if (is_avx512) {
        for (size_t d = 0; d < len; d++)
            h->dd(0x01010101);
    } else {
        for (size_t d = 0; d < len; d++)
            h->dd(1);
        for (uint8_t d = 0; d < vlen; d++) {
            uint8_t b_mask = (d % 16 < 4) ? d % 16 * 4 : 0x80;
            h->db(&b_mask, 1);
        }
    }

}

template <cpu_isa_t isa, typename Wmm>
void jit_uni_dropout_injector_f32<isa, Wmm>::load_rng_state(size_t state0_idx,
        size_t state1_idx, size_t state2_idx, size_t state3_idx, Reg32& seed, Reg32& p, Reg32& scale) {
    vmm_state0 = Vmm(state0_idx);
    vmm_state1 = Vmm(state1_idx);
    vmm_state2 = Vmm(state2_idx);
    vmm_state3 = Vmm(state3_idx);


    h->uni_vpbroadcastd(vmm_state0, scale);
    h->uni_vpbroadcastd(vmm_state1, p);
    //h->uni_vmovups(vmm_state1, vmm_state0);
    //h->uni_vmovups(vmm_state2, h->ptr[p_table + vlen * 8]);
    //h->uni_vsubps(vmm_state0, vmm_state2, vmm_state0);
    //h->uni_vdivps(vmm_state0, vmm_state2, vmm_state0);
    h->uni_vmovups(h->ptr[p_table + vlen * 8], vmm_state0);
    //h->uni_vaddps(vmm_state1, vmm_state1, vmm_state2);
    h->uni_vmovups(h->ptr[p_table + vlen * 10], vmm_state1);

    h->uni_vpbroadcastd(vmm_state3, seed);

    h->uni_vpmulld(vmm_state0, vmm_state3, h->ptr[p_table + vlen]);
    h->uni_vpmulld(vmm_state1, vmm_state3, h->ptr[p_table + vlen * 1]);
    h->uni_vpmulld(vmm_state2, vmm_state3, h->ptr[p_table + vlen * 2]);
    h->uni_vpmulld(vmm_state3, vmm_state3, h->ptr[p_table + vlen * 3]);

    h->uni_vxorps(vmm_state0, vmm_state0, h->ptr[p_table + vlen * 4]);
    h->uni_vxorps(vmm_state1, vmm_state1, h->ptr[p_table + vlen * 5]);
    h->uni_vxorps(vmm_state2, vmm_state2, h->ptr[p_table + vlen * 6]);
    h->uni_vxorps(vmm_state3, vmm_state3, h->ptr[p_table + vlen * 7]);

}

template struct jit_uni_dropout_injector_f32<avx512_core_fp16>;
template struct jit_uni_dropout_injector_f32<avx512_core_fp16, Xbyak::Ymm>;
template struct jit_uni_dropout_injector_f32<avx512_core_fp16, Xbyak::Xmm>;
template struct jit_uni_dropout_injector_f32<avx512_core_bf16>;
template struct jit_uni_dropout_injector_f32<avx512_core>;
template struct jit_uni_dropout_injector_f32<avx512_core, Ymm>;
template struct jit_uni_dropout_injector_f32<avx512_core, Xmm>;
template struct jit_uni_dropout_injector_f32<avx2_vnni_2>;
template struct jit_uni_dropout_injector_f32<avx2_vnni_2, Xmm>;
template struct jit_uni_dropout_injector_f32<avx2>;
template struct jit_uni_dropout_injector_f32<avx2, Xmm>;
template struct jit_uni_dropout_injector_f32<avx>;
template struct jit_uni_dropout_injector_f32<avx, Xmm>;
template struct jit_uni_dropout_injector_f32<sse41>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
