#pragma once

#ifndef SAMURAI_WITH_STARPU
#error "The header file <samurai/starpu_uniform/algorithm.hpp> should not be included if SAMURAI_WITH_STARPU is not defined."
#endif

#include <starpu.h>
#include <samurai/starpu_uniform/mesh.hpp>
#include <samurai/starpu_uniform/field.hpp>

namespace samurai
{
    namespace starpu_uniform
    {
        // ----------------------------------------------------
        // GHOST EXCHANGE CODELET
        // ----------------------------------------------------
        struct GhostExchangeArgs
        {
            void* u_src_ptr;
            void* u_dst_ptr;
            const void* intersection_ptr;
            std::size_t level;
        };

        template <class StarpuField, class StarpuMesh>
        inline void ghost_exchange_cpu_func(void *buffers[], void *cl_arg)
        {
            GhostExchangeArgs args;
            starpu_codelet_unpack_args(cl_arg, &args);

            auto& u_src = *static_cast<typename StarpuField::local_field_t*>(args.u_src_ptr);
            auto& u_dst = *static_cast<typename StarpuField::local_field_t*>(args.u_dst_ptr);
            auto& intersect = *static_cast<const typename StarpuMesh::ca_type*>(args.intersection_ptr);

            // Verify that StarPU CPU worker uses the same registered pointers
            using value_type = typename StarpuField::value_type;
            assert((value_type*)STARPU_VECTOR_GET_PTR(buffers[0]) == u_src.data());
            assert((value_type*)STARPU_VECTOR_GET_PTR(buffers[1]) == u_dst.data());

            samurai::for_each_interval(
                intersect,
                [&](std::size_t level, const auto& i, const auto& index)
                {
                    u_dst(level, i, index) = u_src(level, i, index);
                });
        }

        template <class StarpuField, class StarpuMesh>
        inline starpu_codelet* get_ghost_exchange_codelet()
        {
            static starpu_codelet cl;
            static bool init = false;
            if (!init)
            {
                starpu_codelet_init(&cl);
                cl.cpu_funcs[0] = ghost_exchange_cpu_func<StarpuField, StarpuMesh>;
                cl.nbuffers = 2;
                cl.modes[0] = STARPU_R;
                cl.modes[1] = STARPU_RW;
                cl.name = "ghost_exchange";
                init = true;
            }
            return &cl;
        }

        // ----------------------------------------------------
        // SUBMIT GHOST EXCHANGE
        // ----------------------------------------------------
        template <class StarpuField>
        void submit_ghost_exchange(StarpuField& u, std::size_t level)
        {
            using StarpuMesh = typename StarpuField::mesh_t;
            int nb_task = u.get_nb_task();
            auto& starpu_mesh = u.starpu_mesh();

            auto cl_ghost = get_ghost_exchange_codelet<StarpuField, StarpuMesh>();
            for (auto& entry : starpu_mesh.get_intersections())
            {
                GhostExchangeArgs args = {
                    &u.get_field(entry.src),
                    &u.get_field(entry.dst),
                    &entry.intersection,
                    level
                };

                starpu_task_insert(cl_ghost,
                                   STARPU_R, u.get_handle(entry.src),
                                   STARPU_RW, u.get_handle(entry.dst),
                                   STARPU_VALUE, &args, sizeof(args),
                                   STARPU_TASK_COLOR, 0x0000FF, // Blue color for ghost exchange task
                                   0);
            }
        }
    }
}
