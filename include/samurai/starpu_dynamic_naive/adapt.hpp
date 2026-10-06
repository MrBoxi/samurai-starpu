// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#ifndef SAMURAI_WITH_STARPU
#error "The header file <samurai/starpu_dynamic_naive/adapt.hpp> should not be included if SAMURAI_WITH_STARPU is not defined."
#endif

#include <tuple>
#include <starpu.h>
#include <samurai/mr/adapt.hpp>
#include <samurai/field/scalar_field.hpp>
#include <samurai/starpu_dynamic_naive/mesh.hpp>
#include <samurai/starpu_dynamic_naive/field.hpp>

namespace samurai
{
    namespace starpu_dynamic_naive
    {
        template <class StarpuField, class... OtherStarpuFields>
        void adapt(samurai::mra_config& mra_cfg, StarpuField& u, OtherStarpuFields&... other_fields)
        {
            // 1. Wait for all StarPU tasks to finish
            starpu_task_wait_for_all();

            // 2. Unregister handles for u and all other fields
            u.unregister_handles();
            (other_fields.unregister_handles(), ...);

            // 3. Gather u into global field
            auto& starpu_mesh = u.starpu_mesh();
            auto& global_mesh = starpu_mesh.get_global_mesh();
            using value_t = typename StarpuField::value_type;
            auto u_global = samurai::make_scalar_field<value_t>(u.name(), global_mesh);
            if (u.get_nb_task() > 0 && !u.get_field(0).get_bc().empty())
            {
                u_global.copy_bc_from(u.get_field(0));
            }
            u.gather_to(u_global);

            // 4. Adapt global field with Samurai MRAdapt
            auto MRadaptation = samurai::make_MRAdapt(u_global);
            MRadaptation(mra_cfg);

            // 5. Rebuild starpu_mesh subdomains and intersections based on new global mesh
            starpu_mesh.rebuild();

            // 6. Scatter adapted u_global to u local fields and register new handles
            u.scatter_from_and_register(u_global);

            // 7. Rebuild and register other fields (e.g. unp1) on the new subdomains
            (other_fields.rebuild_and_register(), ...);
        }

        template <class StarpuField, class... OtherStarpuFields>
        void adapt(samurai::mra_config&& mra_cfg, StarpuField& u, OtherStarpuFields&... other_fields)
        {
            samurai::mra_config cfg = mra_cfg;
            adapt(cfg, u, other_fields...);
        }


        template <class StarpuField, class... OtherStarpuFields>
        void adapt_mt([[maybe_unused]] samurai::mra_config& mra_cfg, [[maybe_unused]] StarpuField& u, [[maybe_unused]] OtherStarpuFields&... other_fields)
        {
            throw std::runtime_error("starpu_dynamic_naive::adapt_mt is not yet implemented");
        }

        template <class StarpuField, class... OtherStarpuFields>
        void adapt_mt(samurai::mra_config&& mra_cfg, StarpuField& u, OtherStarpuFields&... other_fields)
        {
            samurai::mra_config cfg = mra_cfg;
            adapt_mt(cfg, u, other_fields...);
        }


        template <class StarpuField, class... OtherStarpuFields>
        class StarpuMRAdapt
        {
        public:
            StarpuMRAdapt(StarpuField& u, OtherStarpuFields&... other_fields)
                : m_u(u)
                , m_other_fields(std::tie(other_fields...))
            {
            }

            void operator()(samurai::mra_config& cfg)
            {
                std::apply([&](auto&... others) {
                    adapt(cfg, m_u, others...);
                }, m_other_fields);
            }

            void operator()(samurai::mra_config&& cfg)
            {
                samurai::mra_config config = cfg;
                (*this)(config);
            }

            void adapt_mt(samurai::mra_config& cfg)
            {
                std::apply([&](auto&... others) {
                    starpu_dynamic_naive::adapt_mt(cfg, m_u, others...);
                }, m_other_fields);
            }

            void adapt_mt(samurai::mra_config&& cfg)
            {
                samurai::mra_config config = cfg;
                adapt_mt(config);
            }

        private:
            StarpuField& m_u;
            std::tuple<OtherStarpuFields&...> m_other_fields;
        };

        template <class StarpuField, class... OtherStarpuFields>
        auto make_MRAdapt(StarpuField& u, OtherStarpuFields&... other_fields)
        {
            return StarpuMRAdapt<StarpuField, OtherStarpuFields...>(u, other_fields...);
        }
    }
}
