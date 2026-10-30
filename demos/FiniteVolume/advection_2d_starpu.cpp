// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <array>
#include <chrono>
#include <map>
#include <string>
#include <xtensor/containers/xfixed.hpp>

#include <samurai/samurai.hpp>
#include <samurai/starpu_uniform/mesh.hpp>
#include <samurai/starpu_uniform/field.hpp>
#include <samurai/starpu_uniform/algorithm.hpp>
#include <samurai/starpu_uniform/save.hpp>
#include <samurai/starpu_uniform/view.hpp>
#include <samurai/io/hdf5.hpp>

#include <filesystem>
namespace fs = std::filesystem;

// ----------------------------------------------------
// USER ADVECTION UPDATE CODELET
// ----------------------------------------------------
struct UpdateArgs
{
    int nx;
    int ny;
    int row_stride;
    int ghost_width;
    double dt;
    double cell_length;
    double a0;
    double a1;
};

template <class value_t = double>
void advection_update_cpu_func(void *buffers[], void *cl_arg)
{
    UpdateArgs args;
    starpu_codelet_unpack_args(cl_arg, &args);

    const auto* u_data = reinterpret_cast<const value_t*>(STARPU_VECTOR_GET_PTR(buffers[0]));
    auto* unp1_data    = reinterpret_cast<value_t*>(STARPU_VECTOR_GET_PTR(buffers[1]));

    samurai::starpu_uniform::UniformFieldView<const value_t> u(u_data, args.row_stride, args.ghost_width);
    samurai::starpu_uniform::UniformFieldView<value_t> unp1(unp1_data, args.row_stride, args.ghost_width);

    double coeff = args.dt / args.cell_length;

    for (int y = 0; y < args.ny; ++y)
    {
        for (int x = 0; x < args.nx; ++x)
        {
            unp1(x, y) = u(x, y) - coeff * (args.a0 * (u(x, y) - u(x - 1, y)) + args.a1 * (u(x, y) - u(x, y - 1)));
        }
    }
}

template <class value_t = double>
starpu_codelet* get_advection_update_codelet()
{
    static starpu_codelet cl;
    static bool init = false;
    if (!init)
    {
        starpu_codelet_init(&cl);
        cl.cpu_funcs[0] = advection_update_cpu_func<value_t>;
        cl.nbuffers = 2;
        cl.modes[0] = STARPU_R;
        cl.modes[1] = STARPU_W;
        cl.name = "advection_update";
        init = true;
    }
    return &cl;
}

inline const char* get_advection_update_task_name(int i)
{
    static std::map<int, std::string> task_names;
    auto it = task_names.find(i);
    if (it == task_names.end())
    {
        it = task_names.emplace(i, fmt::format("advection_update_{}", i)).first;
    }
    return it->second.c_str();
}

template <class StarpuField>
void init(StarpuField& u)
{
    auto& starpu_mesh = u.starpu_mesh();
    int nb_task = starpu_mesh.get_nb_task();

    for (int i = 0; i < nb_task; ++i)
    {
        auto& u_local = u.get_field(i);
        u_local.fill(0.0);

        samurai::for_each_cell(
            u_local.mesh(),
            [&](auto& cell)
            {
                auto center           = cell.center();
                const double radius   = .2;
                const double x_center = 0.3;
                const double y_center = 0.3;
                if (((center[0] - x_center) * (center[0] - x_center) + (center[1] - y_center) * (center[1] - y_center)) <= radius * radius)
                {
                    u_local[cell] = 1.0;
                }
            });
    }
}

#include <unistd.h>
#include <cstring>
#include <cstdio>

int main(int argc, char* argv[])
{
    samurai::initialize("Finite volume example for the advection equation in 2d with StarPU", argc, argv);

    {
        constexpr std::size_t dim = 2;
        int nb_task = 4;

        // Simulation parameters
        const double a[dim] = {1., 1.};
        double Tf  = .1;
        const double cfl = 0.5;
        double t   = 0.;
        std::size_t level = 7;
        std::size_t max_iter = std::numeric_limits<std::size_t>::max();

        // Output parameters
        fs::path path        = fs::current_path();
        std::string filename = "FV_advection_2d_starpu";
        std::size_t nfiles   = 1;
        bool no_save = false;
        std::string save_perf_filename = "";
        std::string label = "none";

        auto& app = samurai::app;
        app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
        app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
        app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
        app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");
        app.add_option("--nb-task", nb_task, "Number of StarPU tasks")->capture_default_str()->group("Simulation parameters");
        app.add_option("--max-iter", max_iter, "Maximum number of iterations")->capture_default_str()->group("Simulation parameters");
        app.add_flag("--no-save", no_save, "Disable file saving")->group("Output");
        app.add_option("--save-perf", save_perf_filename, "Save the performance to the file")->group("Output");
        app.add_option("--label", label, "Label for the performance file")->group("Output");
        SAMURAI_PARSE(argc, argv);

        if (samurai::args::max_level != std::numeric_limits<std::size_t>::max())
        {
            level = samurai::args::max_level;
        }
        else if (samurai::args::min_level != std::numeric_limits<std::size_t>::max())
        {
            level = samurai::args::min_level;
        }


        const samurai::Box<double, dim> box({0., 0.}, {1., 1.});

        using Config = samurai::UniformConfig<dim, 1>;
        samurai::starpu_uniform::StarpuUniformMesh<Config> mesh(box, level, nb_task);

        auto u = samurai::starpu_uniform::StarpuUniformScalarField("u", mesh);
        init(u);
        auto unp1 = samurai::starpu_uniform::StarpuUniformScalarField("unp1", mesh);
        unp1.fill(0.0);

        double dt            = cfl * mesh.min_cell_length();
        const double dt_save = Tf / static_cast<double>(nfiles);

        // Initial output
        if (!no_save)
        {
            const std::string suffix = (nfiles != 1) ? "_ite_0" : "";
            samurai::starpu_uniform::insert_save(path, fmt::format("{}{}", filename, suffix), mesh, u);
        }

        std::size_t nsave = 1;
        std::size_t nt    = 0;

        // Timers to measure submission and execution times
        double total_submit_time = 0.0;
        double total_wait_time = 0.0;

        auto start_loop = std::chrono::steady_clock::now();

        while (t != Tf && nt < max_iter)
        {

            t += dt;
            if (t > Tf)
            {
                dt += Tf - t;
                t = Tf;
            }

            //std::cout << fmt::format("iteration {}: t = {}, dt = {}", nt, t, dt) << std::endl;

            // 1. Submit generic ghost exchange tasks
            samurai::starpu_uniform::submit_ghost_exchange(u, level);

            // 2. Submit user update tasks
            auto cl_update = get_advection_update_codelet<typename decltype(u)::value_type>();
            double cell_length = mesh.min_cell_length();
            for (int i = 0; i < nb_task; ++i)
            {
                const auto& grid_info = mesh.get_subdomain_grid_info(i);
                UpdateArgs args = {
                    grid_info.nx,
                    grid_info.ny_interior,
                    grid_info.row_stride,
                    grid_info.ghost_width,
                    dt,
                    cell_length,
                    a[0],
                    a[1]
                };

                starpu_task_insert(cl_update,
                                   STARPU_R, u.get_handle(i),
                                   STARPU_W, unp1.get_handle(i),
                                   STARPU_VALUE, &args, sizeof(args),
                                   STARPU_TASK_COLOR, 0x00FF00, // Green color for update task
                                   STARPU_NAME, get_advection_update_task_name(i),
                                   0);
            }

            samurai::starpu_uniform::swap(u, unp1); // swap pointers (handles and fields)


            // Save intermediate step
            if (!no_save && (t >= static_cast<double>(nsave) * dt_save || t == Tf))
            {
                const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
                samurai::starpu_uniform::insert_save(path, fmt::format("{}{}", filename, suffix), mesh, u);
            }


            nt++;
        }

        auto end_loop = std::chrono::steady_clock::now();
        total_submit_time += std::chrono::duration<double>(end_loop - start_loop).count();

        // WAIT ALL TASKS
        auto start_final_wait = std::chrono::steady_clock::now();
        starpu_task_wait_for_all();
        auto end_final_wait = std::chrono::steady_clock::now();
        total_wait_time += std::chrono::duration<double>(end_final_wait - start_final_wait).count();

        std::cout << std::endl;
        std::cout << "========= PERFORMANCE STATS =========" << std::endl;
        std::cout << "Task submission time: " << total_submit_time << " s" << std::endl;
        std::cout << "Execution time (wait): " << total_wait_time << " s" << std::endl;
        std::cout << "=====================================" << std::endl;

        if (!save_perf_filename.empty())
        {
            std::FILE* file = std::fopen(save_perf_filename.c_str(), "a");
            if (file == nullptr)
            {
                std::fprintf(stderr, "Warning: Cannot open file %s to save performance\n", save_perf_filename.c_str());
            }
            else
            {
                std::fseek(file, 0, SEEK_END);
                if (std::ftell(file) == 0)
                {
                    std::fprintf(file, "machine,version,label,num_threads,mpi_size,level,nb_tasks,max_iter,time\n");
                }

                char machine_name[256];
                if (gethostname(machine_name, 256) != 0)
                {
                    std::strcpy(machine_name, "Unknown");
                }

                int num_threads = starpu_cpu_worker_get_count();

                std::fprintf(file, "%s,starpu,%s,%d,%d,%d,%d,%d,%f\n",
                             machine_name,
                             label.c_str(),
                             num_threads,
                             1,
                             static_cast<int>(level),
                             static_cast<int>(nb_task),
                             static_cast<int>(nt),
                             total_submit_time + total_wait_time);

                std::fclose(file);
            }
        }
    }

    samurai::finalize();
    return 0;
}
