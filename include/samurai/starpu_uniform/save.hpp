#pragma once

#ifndef SAMURAI_WITH_STARPU
#error "The header file <samurai/starpu_uniform/save.hpp> should not be included if SAMURAI_WITH_STARPU is not defined."
#endif

#include <samurai/io/hdf5.hpp>
#include <samurai/field/scalar_field.hpp>
#include <samurai/starpu_uniform/mesh.hpp>
#include <samurai/starpu_uniform/field.hpp>
#include <tuple>
#include <vector>
#include <string>
#include <filesystem>
#include <memory>
#include <cassert>
#include <utility>
#include <starpu.h>
#include <unordered_map>

namespace samurai::starpu_uniform
{
    template <class StarpuField>
    auto get_local_fields(const StarpuField& starpu_field)
    {
        auto nb_task = starpu_field.get_nb_task();
        std::vector<const typename std::decay_t<StarpuField>::local_field_t*> local_fields;
        local_fields.reserve(nb_task);
        for (int i = 0; i < nb_task; ++i)
        {
            local_fields.push_back(&starpu_field.get_field(i));
        }
        return local_fields;
    }

    template <class StarpuMesh, class... StarpuFields>
    struct SaveArgs
    {
        std::filesystem::path path;
        std::string filename;
        const StarpuMesh* mesh;
        std::vector<std::string> field_names;
        std::tuple<std::vector<const typename std::decay_t<StarpuFields>::local_field_t*>...> fields;
    };

    template <class StarpuMesh, class... StarpuFields, std::size_t... Is>
    auto make_global_fields(const SaveArgs<StarpuMesh, StarpuFields...>& args, typename StarpuMesh::mesh_t& global_mesh, std::index_sequence<Is...>)
    {
        return std::make_tuple(
            [&](){
                const auto& local_fields_vec = std::get<Is>(args.fields);
                assert(!local_fields_vec.empty());
                const auto* first_field = local_fields_vec[0];
                const std::string& name = args.field_names[Is];
                using value_type = typename std::decay_t<decltype(*first_field)>::value_type;
                samurai::ScalarField<typename StarpuMesh::mesh_t, value_type> gf(name, global_mesh);
                gf.fill(0.0);
                return gf;
            }()...
        );
    }

    template <class StarpuMesh, class... StarpuFields>
    void save_cpu_func(void *buffers[], void *cl_arg)
    {
        SaveArgs<StarpuMesh, StarpuFields...>* args_ptr = nullptr;
        starpu_codelet_unpack_args(cl_arg, &args_ptr);
        
        // Take ownership of the dynamically allocated arguments to delete it on exit
        std::unique_ptr<SaveArgs<StarpuMesh, StarpuFields...>> args(args_ptr);

        const auto& mesh = *args->mesh;
        using mesh_t = typename StarpuMesh::mesh_t;
        auto& global_mesh = const_cast<mesh_t&>(mesh.get_global_mesh());

        // 1. Create the task_id field to visualize domain decomposition
        samurai::ScalarField<mesh_t, int> task_id_field("task_id", global_mesh);
        task_id_field.fill(0);

        // 2. Create the global fields matching the names and types of wrappers
        auto global_fields_tuple = make_global_fields(*args, global_mesh, std::index_sequence_for<StarpuFields...>{});

        // 3. Fill global fields and task_id from local subdomains
        for (int i = 0; i < mesh.get_nb_task(); ++i)
        {
            auto& local_mesh = const_cast<mesh_t&>(mesh.get_mesh(i));

            samurai::for_each_interval(local_mesh, [&](std::size_t level, const auto& interval, const auto& jk) {
                // Set the task_id for these cells
                task_id_field(level, interval, jk).fill(i);

                // Copy values from local subfields to the global fields
                std::apply([&](auto&... gfs) {
                    std::apply([&](const auto&... local_fields_vecs) {
                        ( (gfs(level, interval, jk) = (*local_fields_vecs[i])(level, interval, jk)), ... );
                    }, args->fields);
                }, global_fields_tuple);
            });
        }

        // 4. Save all reconstructed fields
        std::apply([&](const auto&... gfs) {
            samurai::save(args->path, args->filename, global_mesh, task_id_field, gfs...);
        }, global_fields_tuple);
    }

    template <class StarpuMesh, class... StarpuFields>
    inline starpu_codelet* get_save_codelet()
    {
        static starpu_codelet cl;
        static bool init = false;
        if (!init)
        {
            starpu_codelet_init(&cl);
            cl.cpu_funcs[0] = save_cpu_func<StarpuMesh, StarpuFields...>;
            cl.nbuffers = STARPU_VARIABLE_NBUFFERS;
            cl.name = "save_fields";
            init = true;
        }
        return &cl;
    }

    template <class StarpuMesh, class... StarpuFields>
    void insert_save(const std::filesystem::path& path, const std::string& filename, const StarpuMesh& mesh, const StarpuFields&... starpu_fields)
    {
        int nb_task = mesh.get_nb_task();
        std::size_t total_buffers = sizeof...(StarpuFields) * nb_task;

        // 1. Pack all StarPU handles and modes into descrs using starpu_data_descr
        std::vector<starpu_data_descr> descrs;
        descrs.reserve(total_buffers);
        (
            [&](){
                for (int i = 0; i < nb_task; ++i)
                {
                    descrs.push_back({starpu_fields.get_handle(i), STARPU_R});
                }
            }(), ...
        );

        // 2. Allocate save arguments on heap to pass pointer to StarPU safely
        auto* args = new SaveArgs<StarpuMesh, StarpuFields...>{
            path,
            filename,
            &mesh,
            {starpu_fields.name()...},
            std::make_tuple(get_local_fields(starpu_fields)...)
        };

        // 3. Submit save task asynchronously to StarPU's DAG using STARPU_DATA_MODE_ARRAY
        auto cl_save = get_save_codelet<StarpuMesh, StarpuFields...>();
        starpu_task_insert(cl_save,
                           STARPU_DATA_MODE_ARRAY, descrs.data(), static_cast<int>(descrs.size()),
                           STARPU_VALUE, &args, sizeof(args),
                           STARPU_TASK_COLOR, 0xFF0000, // Red color for save task
                           0);
    }
}
