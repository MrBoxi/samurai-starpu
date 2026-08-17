// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#ifndef SAMURAI_WITH_STARPU
#error "The header file <samurai/starpu_static/algorithm.hpp> should not be included if SAMURAI_WITH_STARPU is not defined."
#endif

#include <starpu.h>
#include <samurai/starpu_static/mesh.hpp>
#include <samurai/starpu_static/field.hpp>
#include <samurai/numeric/projection.hpp>
#include <samurai/numeric/prediction.hpp>
#include <samurai/algorithm/update_outer_ghost.hpp>

namespace samurai
{
    namespace starpu_static
    {
        // ----------------------------------------------------
        // GHOST EXCHANGE CODELET & SUBMISSION
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
                cl.name = "mr_ghost_exchange";
                init = true;
            }
            return &cl;
        }

        template <class StarpuField>
        void submit_ghost_exchange(StarpuField& u, std::size_t level)
        {
            using StarpuMesh = typename StarpuField::mesh_t;
            auto& starpu_mesh = u.starpu_mesh();

            auto cl_ghost = get_ghost_exchange_codelet<StarpuField, StarpuMesh>();
            for (auto& entry : starpu_mesh.get_intersections())
            {
                if (entry.level != level) continue;

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
                                   STARPU_TASK_COLOR, 0x0000FF,
                                   0);
            }
        }

        // ----------------------------------------------------
        // LOCAL OUTER GHOSTS CODELET & SUBMISSION
        // ----------------------------------------------------
        struct OuterGhostsArgs
        {
            void* u_ptr;
            std::size_t level;
        };

        template <class StarpuField>
        inline void outer_ghosts_cpu_func(void *buffers[], void *cl_arg)
        {
            OuterGhostsArgs args;
            starpu_codelet_unpack_args(cl_arg, &args);

            auto& u = *static_cast<typename StarpuField::local_field_t*>(args.u_ptr);

            using value_type = typename StarpuField::value_type;
            assert((value_type*)STARPU_VECTOR_GET_PTR(buffers[0]) == u.data());

            std::size_t level = args.level;
            samurai::update_outer_ghosts(level, u);
        }

        template <class StarpuField>
        inline starpu_codelet* get_outer_ghosts_codelet()
        {
            static starpu_codelet cl;
            static bool init = false;
            if (!init)
            {
                starpu_codelet_init(&cl);
                cl.cpu_funcs[0] = outer_ghosts_cpu_func<StarpuField>;
                cl.nbuffers = 1;
                cl.modes[0] = STARPU_RW;
                cl.name = "mr_outer_ghosts";
                init = true;
            }
            return &cl;
        }

        template <class StarpuField>
        void submit_outer_ghosts(StarpuField& u, std::size_t level)
        {
            int nb_task = u.get_nb_task();
            auto cl_og = get_outer_ghosts_codelet<StarpuField>();

            for (int i = 0; i < nb_task; ++i)
            {
                OuterGhostsArgs args = { &u.get_field(i), level };
                starpu_task_insert(cl_og,
                                   STARPU_RW, u.get_handle(i),
                                   STARPU_VALUE, &args, sizeof(args),
                                   STARPU_TASK_COLOR, 0xFFFF00,
                                   0);
            }
        }

        // ----------------------------------------------------
        // LOCAL PROJECTION CODELET & SUBMISSION
        // ----------------------------------------------------
        struct ProjectionArgs
        {
            void* u_ptr;
            std::size_t level;
        };

        template <class StarpuField>
        inline void projection_cpu_func(void *buffers[], void *cl_arg)
        {
            ProjectionArgs args;
            starpu_codelet_unpack_args(cl_arg, &args);

            auto& u = *static_cast<typename StarpuField::local_field_t*>(args.u_ptr);
            auto& mesh = u.mesh();
            using mesh_id_t = typename StarpuField::local_field_t::mesh_t::mesh_id_t;

            using value_type = typename StarpuField::value_type;
            assert((value_type*)STARPU_VECTOR_GET_PTR(buffers[0]) == u.data());

            std::size_t level = args.level;
            if (level > mesh.min_level())
            {
                auto set_at_levelm1 = intersection(mesh[mesh_id_t::reference][level], mesh[mesh_id_t::proj_cells][level - 1]).on(level - 1);
                set_at_levelm1.apply_op(samurai::variadic_projection(u));
            }
        }

        template <class StarpuField>
        inline starpu_codelet* get_projection_codelet()
        {
            static starpu_codelet cl;
            static bool init = false;
            if (!init)
            {
                starpu_codelet_init(&cl);
                cl.cpu_funcs[0] = projection_cpu_func<StarpuField>;
                cl.nbuffers = 1;
                cl.modes[0] = STARPU_RW;
                cl.name = "mr_projection";
                init = true;
            }
            return &cl;
        }

        template <class StarpuField>
        void submit_projection(StarpuField& u, std::size_t level)
        {
            int nb_task = u.get_nb_task();
            auto cl_proj = get_projection_codelet<StarpuField>();

            for (int i = 0; i < nb_task; ++i)
            {
                ProjectionArgs args = { &u.get_field(i), level };
                starpu_task_insert(cl_proj,
                                   STARPU_RW, u.get_handle(i),
                                   STARPU_VALUE, &args, sizeof(args),
                                   STARPU_TASK_COLOR, 0x00FF00,
                                   0);
            }
        }

        // ----------------------------------------------------
        // LOCAL PREDICTION CODELET & SUBMISSION
        // ----------------------------------------------------
        struct PredictionArgs
        {
            void* u_ptr;
            std::size_t level;
        };

        template <class StarpuField>
        inline void prediction_cpu_func(void *buffers[], void *cl_arg)
        {
            PredictionArgs args;
            starpu_codelet_unpack_args(cl_arg, &args);

            auto& u = *static_cast<typename StarpuField::local_field_t*>(args.u_ptr);
            auto& mesh = u.mesh();
            using mesh_id_t = typename StarpuField::local_field_t::mesh_t::mesh_id_t;
            constexpr std::size_t pred_order = StarpuField::mesh_t::mesh_t::config_t::prediction_stencil_radius;

            using value_type = typename StarpuField::value_type;
            assert((value_type*)STARPU_VECTOR_GET_PTR(buffers[0]) == u.data());

            std::size_t level = args.level;
            auto pred_ghosts = difference(mesh[mesh_id_t::all_cells][level],
                                          union_(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::proj_cells][level]));
            auto expr = intersection(pred_ghosts, mesh[mesh_id_t::all_cells][level - 1]).on(level);
            expr.apply_op(samurai::variadic_prediction<pred_order, false>(u));
        }

        template <class StarpuField>
        inline starpu_codelet* get_prediction_codelet()
        {
            static starpu_codelet cl;
            static bool init = false;
            if (!init)
            {
                starpu_codelet_init(&cl);
                cl.cpu_funcs[0] = prediction_cpu_func<StarpuField>;
                cl.nbuffers = 1;
                cl.modes[0] = STARPU_RW;
                cl.name = "mr_prediction";
                init = true;
            }
            return &cl;
        }

        template <class StarpuField>
        void submit_prediction(StarpuField& u, std::size_t level)
        {
            int nb_task = u.get_nb_task();
            auto cl_pred = get_prediction_codelet<StarpuField>();

            for (int i = 0; i < nb_task; ++i)
            {
                PredictionArgs args = { &u.get_field(i), level };
                starpu_task_insert(cl_pred,
                                   STARPU_RW, u.get_handle(i),
                                   STARPU_VALUE, &args, sizeof(args),
                                   STARPU_TASK_COLOR, 0xFF00FF,
                                   0);
            }
        }

        // ----------------------------------------------------
        // MULTI-RESOLUTION STARPU GHOST UPDATE
        // ----------------------------------------------------
        template <class StarpuField>
        void submit_update_ghost_mr(StarpuField& u)
        {
            auto& starpu_mesh = u.starpu_mesh();
            std::size_t min_level = starpu_mesh.min_level();
            std::size_t max_level = starpu_mesh.max_level();

            // Top-down pass: exchange ghosts, update outer ghosts, exchange again, then project
            for (std::size_t level = max_level + 1; level-- > min_level;)
            {
                submit_ghost_exchange(u, level);
                submit_outer_ghosts(u, level);
                submit_ghost_exchange(u, level);

                if (level > min_level)
                {
                    submit_projection(u, level);
                }
            }

            // Bottom-up pass: predict from coarser level then exchange ghosts
            for (std::size_t level = min_level + 1; level <= max_level; ++level)
            {
                submit_prediction(u, level);
                submit_ghost_exchange(u, level);
            }
        }
    }
}
