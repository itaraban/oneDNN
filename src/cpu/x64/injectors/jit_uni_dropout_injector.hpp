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

#ifndef CPU_X64_JIT_UNI_DROPOUT_INJECTOR_HPP
#define CPU_X64_JIT_UNI_DROPOUT_INJECTOR_HPP

#include <assert.h>
#include <type_traits>

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/injectors/injector_utils.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, typename Wmm = typename cpu_isa_traits<isa>::Vmm>
struct jit_uni_dropout_injector_f32 {
    using Vmm = Wmm;

    // Arguments description:
    // host - jit generator which is filled with instructions
    // p - user dropout arguments
    // save_state - when true, preserves on stack vmm_aux registers preventing
    //   results spoiling. Restores them when done in injector_postamble().
    // p_table - GPR where table label is stored to get access for pre-defined
    //   constants used in alg codes.
    // use_dst - defines whether source or destination point is passed to alg
    //   code. Depends on algorithm. See `_use_dst_for_bwd` algs definition.
    jit_uni_dropout_injector_f32(jit_generator *host,  bool save_state = true,
        Xbyak::Reg64 p_table = Xbyak::util::rax,
            Xbyak::Opmask k_mask = Xbyak::Opmask(1),
            bool preserve_vmm = true,
            bool preserve_p_table = true)
        : h(host)
        , save_state_(save_state)
        , p_table(p_table)
        , k_mask(k_mask)
        , preserve_vmm_(preserve_vmm)
        , preserve_p_table_(preserve_p_table) {
        assert(is_superset(isa, sse41));
    }


   // void compute_vector_range(size_t start_idx, size_t end_idx);
    void compute_vector_range(const injector_utils::vmm_index_set_t &vmm_idxs,
            const Xbyak::Address &mask_raw_addr,
            bool tail);
    void compute_vector(size_t src_idx,
            const Xbyak::Address &mask_raw_addr,             bool tail) {
        compute_vector_range({src_idx}, mask_raw_addr, tail);
    }
    void prepare_table(bool gen_table = true);
    void load_table_addr() { h->mov(p_table, l_table); }
    void load_rng_state(size_t state0_idx, size_t state1_idx, size_t state2_idx,
            size_t state3_idx, Xbyak::Reg32 &seed, Xbyak::Reg32 &p,
            Xbyak::Reg32 &scale);

private:

    jit_generator *const h;

    const bool save_state_;
    const bool preserve_vmm_;
    const bool preserve_p_table_;

    Xbyak::Label l_table;
    const Xbyak::Reg64 p_table;
    const Xbyak::Opmask k_mask;


    static constexpr size_t vlen = vreg_traits<Vmm>::vlen;
    static constexpr size_t preserved_vecs_max = 3;
    static constexpr size_t preserved_gprs_max = 5;
    static constexpr size_t vecs_count = cpu_isa_traits<isa>::n_vregs;

    size_t vecs_to_preserve = 0;
    size_t preserved_vecs_count = 0;
    size_t preserved_vec_idxs[preserved_vecs_max] = {0};
    size_t preserved_gpr_idxs[preserved_gprs_max] = {0};
    injector_utils::vmm_index_set_iterator_t start_idx_tail;

    Vmm vmm_mask, vmm_aux0, vmm_aux1;
    Vmm vmm_state0, vmm_state1, vmm_state2, vmm_state3;

    void compute_body(
            const injector_utils::vmm_index_set_iterator_t &start_idx_it,
            const injector_utils::vmm_index_set_iterator_t &end_idx_it,
            const Xbyak::Address &mask_raw_addr, bool tail);
    void injector_preamble(const injector_utils::vmm_index_set_t &vmm_idxs);
    void injector_preamble_tail(
            const injector_utils::vmm_index_set_iterator_t &start_idx_it);
    void injector_postamble();
    void assign_regs();

};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
