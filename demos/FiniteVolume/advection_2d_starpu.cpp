// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <array>
#include <chrono>
#include <xtensor/containers/xfixed.hpp>

#include <samurai/samurai.hpp>
#include <samurai/starpu_uniform/mesh.hpp>
#include <samurai/starpu_uniform/field.hpp>
#include <samurai/starpu_uniform/algorithm.hpp>
#include <samurai/starpu_uniform/save.hpp>
#include <samurai/io/hdf5.hpp>

#include <filesystem>
namespace fs = std::filesystem;

// ----------------------------------------------------
// USER ADVECTION UPDATE CODELET
// ----------------------------------------------------
template <class StarpuField>
struct UpdateArgs
{
    using local_field_t = typename StarpuField::local_field_t;
    local_field_t* u_ptr;
    local_field_t* unp1_ptr;
    double dt;
    double cell_length;
    double a0;
    double a1;
};

template <class StarpuField>
void advection_update_cpu_func(void *buffers[], void *cl_arg)
{
    UpdateArgs<StarpuField> args;
    starpu_codelet_unpack_args(cl_arg, &args);

    auto& u = *args.u_ptr;
    auto& unp1 = *args.unp1_ptr;
    auto& mesh = u.mesh();

    // Verify that StarPU CPU worker uses the same registered pointers
    using value_type = typename StarpuField::value_type;
    assert((value_type*)STARPU_VECTOR_GET_PTR(buffers[0]) == u.data());
    assert((value_type*)STARPU_VECTOR_GET_PTR(buffers[1]) == unp1.data());

    samurai::for_each_interval(
        mesh,
        [&](std::size_t level, const auto& i, const auto& jk)
        {
            unp1(level, i, jk) = u(level, i, jk)
                               - args.dt / args.cell_length
                                     * (args.a0 * (u(level, i, jk) - u(level, i - 1, jk)) + args.a1 * (u(level, i, jk) - u(level, i, jk - 1)));
        });
}

template <class StarpuField>
starpu_codelet* get_advection_update_codelet()
{
    static starpu_codelet cl;
    static bool init = false;
    if (!init)
    {
        starpu_codelet_init(&cl);
        cl.cpu_funcs[0] = advection_update_cpu_func<StarpuField>;
        cl.nbuffers = 2;
        cl.modes[0] = STARPU_R;
        cl.modes[1] = STARPU_W;
        cl.name = "advection_update";
        init = true;
    }
    return &cl;
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
            auto cl_update = get_advection_update_codelet<decltype(u)>();
            double cell_length = mesh.min_cell_length();
            for (int i = 0; i < nb_task; ++i)
            {
                UpdateArgs<decltype(u)> args = {
                    &u.get_field(i),
                    &unp1.get_field(i),
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
                                   0);
            }

            samurai::starpu_uniform::swap(u, unp1); // swap pointers (handles and fields)


            // Save intermediate step
            if (!no_save && (t >= static_cast<double>(nsave) * dt_save || t == Tf))
            {
                const std::string suffix = (nfiles != 1) ? fmt::format("ite_{}", nsave++) : "";
                samurai::starpu_uniform::insert_save(path, fmt::format("{}_{}", filename, suffix), mesh, u);
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
