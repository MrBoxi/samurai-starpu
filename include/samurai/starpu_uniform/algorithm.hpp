#pragma once

#ifndef SAMURAI_WITH_STARPU
#error "The header file <samurai/starpu_uniform/algorithm.hpp> should not be included if SAMURAI_WITH_STARPU is not defined."
#endif

#include <starpu.h>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <fmt/format.h>
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
            std::size_t src_offset;
            std::size_t copy_size;
        };

        template <class value_t = double>
        inline void ghost_exchange_cpu_func(void *buffers[], void *cl_arg)
        {
            GhostExchangeArgs args;
            starpu_codelet_unpack_args(cl_arg, &args);

            const auto* src_ptr = reinterpret_cast<const value_t*>(STARPU_VECTOR_GET_PTR(buffers[0])) + args.src_offset;
            auto* dst_ptr       = reinterpret_cast<value_t*>(STARPU_VECTOR_GET_PTR(buffers[1]));

            std::memcpy(dst_ptr, src_ptr, args.copy_size * sizeof(value_t));
        }

        template <class value_t = double>
        inline starpu_codelet* get_ghost_exchange_codelet()
        {
            static starpu_codelet cl;
            static bool init = false;
            if (!init)
            {
                starpu_codelet_init(&cl);
                cl.cpu_funcs[0] = ghost_exchange_cpu_func<value_t>;
                cl.nbuffers = 2;
                cl.modes[0] = STARPU_R;
                cl.modes[1] = STARPU_W;
                cl.name = "ghost_exchange";
                init = true;
            }
            return &cl;
        }

        // ----------------------------------------------------
        // GHOST EXCHANGE TASK NAME
        // ----------------------------------------------------
        inline const char* get_ghost_exchange_task_name(int src, int dst)
        {
            static std::map<std::pair<int, int>, std::string> task_names;
            auto key = std::make_pair(src, dst);
            auto it = task_names.find(key);
            if (it == task_names.end())
            {
                it = task_names.emplace(key, fmt::format("ghost_exchange_src_{}_dst_{}", src, dst)).first;
            }
            return it->second.c_str();
        }

        // ----------------------------------------------------
        // SUBMIT GHOST EXCHANGE
        // ----------------------------------------------------
        template <class StarpuField>
        void submit_ghost_exchange(StarpuField& u, std::size_t level)
        {
            using value_t = typename StarpuField::value_type;
            auto& starpu_mesh = u.starpu_mesh();

            u.partitions_submit();

            auto cl_ghost = get_ghost_exchange_codelet<value_t>();
            for (auto& entry : starpu_mesh.get_intersections())
            {
                GhostExchangeArgs args = {
                    entry.src_offset,
                    entry.copy_size
                };

                starpu_task_insert(cl_ghost,
                                   STARPU_R, u.get_sub_handle(entry.src, entry.src_part),
                                   STARPU_W, u.get_sub_handle(entry.dst, entry.dst_part),
                                   STARPU_VALUE, &args, sizeof(args),
                                   STARPU_TASK_COLOR, 0x0000FF, // Blue color for ghost exchange task
                                   STARPU_NAME, get_ghost_exchange_task_name(entry.src, entry.dst),
                                   0);
            }
        }
    }
}
