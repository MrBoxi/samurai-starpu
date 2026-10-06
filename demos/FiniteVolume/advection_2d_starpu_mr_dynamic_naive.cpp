// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <unistd.h>
#include <xtensor/containers/xfixed.hpp>

#include <samurai/samurai.hpp>
#include <samurai/bc.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/starpu_dynamic_naive.hpp>
#include <samurai/io/hdf5.hpp>

namespace fs = std::filesystem;

template <class StarpuField>
struct UpdateArgs
{
    using local_field_t = typename StarpuField::local_field_t;
    local_field_t* u_ptr;
    local_field_t* unp1_ptr;
    double dt;
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

    using value_type = typename StarpuField::value_type;
    assert((value_type*)STARPU_VECTOR_GET_PTR(buffers[0]) == u.data());
    assert((value_type*)STARPU_VECTOR_GET_PTR(buffers[1]) == unp1.data());

    samurai::for_each_interval(
        mesh,
        [&](std::size_t level, const auto& i, const auto& jk)
        {
            double dx = mesh.cell_length(level);
            unp1(level, i, jk) = u(level, i, jk)
                               - args.dt / dx
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
        cl.name = "advection_update_mr_dynamic_naive";
        init = true;
    }
    return &cl;
}

template <class Field>
void init_field(Field& u)
{
    auto& mesh = u.mesh();
    u.fill(0.0);

    samurai::for_each_cell(
        mesh,
        [&](auto& cell)
        {
            auto center           = cell.center();
            const double radius   = .2;
            const double x_center = 0.3;
            const double y_center = 0.3;
            if (((center[0] - x_center) * (center[0] - x_center) + (center[1] - y_center) * (center[1] - y_center)) <= radius * radius)
            {
                u[cell] = 1.0;
            }
        });
}

int main(int argc, char* argv[])
{
    samurai::initialize("Finite volume advection 2D on dynamic MR mesh with StarPU (naive)", argc, argv);

    {
        constexpr std::size_t dim = 2;
        int nb_task = 4;

        // Simulation parameters
        xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
        xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1., 1.};
        std::array<double, dim> a{{1., 1.}};
        double Tf  = .1;
        double cfl = 0.5;
        double t   = 0.;
        std::size_t max_iter = std::numeric_limits<std::size_t>::max();
        bool use_adapt_mt = false;

        // Output parameters
        fs::path path        = fs::current_path();
        std::string filename = "FV_advection_2d_starpu_mr_dynamic_naive";
        std::size_t nfiles   = 1;
        bool no_save = false;
        std::string save_perf_filename = "";
        std::string label = "none";

        auto& app = samurai::app;
        app.add_option("--min-corner", min_corner, "The min corner of the box")->capture_default_str()->group("Simulation parameters");
        app.add_option("--max-corner", max_corner, "The max corner of the box")->capture_default_str()->group("Simulation parameters");
        app.add_option("--velocity", a, "The velocity of the advection equation")->capture_default_str()->group("Simulation parameters");
        app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
        app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
        app.add_option("--nb-task", nb_task, "Number of StarPU tasks / subdomains")->capture_default_str()->group("Simulation parameters");
        app.add_option("--max-iter", max_iter, "Maximum number of iterations")->capture_default_str()->group("Simulation parameters");
        app.add_flag("--adapt-mt", use_adapt_mt, "Use multi-threaded adaptation")->group("Simulation parameters");
        app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
        app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
        app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");
        app.add_flag("--no-save", no_save, "Disable file saving")->group("Output");
        app.add_option("--save-perf", save_perf_filename, "Save performance metrics to file")->group("Output");
        app.add_option("--label", label, "Label for performance file")->group("Output");
        SAMURAI_PARSE(argc, argv);

        if (use_adapt_mt)
        {
            std::cerr << "Error: --adapt-mt (multi-threaded adaptation) is not yet implemented." << std::endl;
            samurai::finalize();
            return 1;
        }

        std::size_t min_level = 4;
        std::size_t max_level = 7;
        if (samurai::args::min_level != std::numeric_limits<std::size_t>::max())
        {
            min_level = samurai::args::min_level;
        }
        if (samurai::args::max_level != std::numeric_limits<std::size_t>::max())
        {
            max_level = samurai::args::max_level;
        }

        const samurai::Box<double, dim> box(min_corner, max_corner);
        auto config = samurai::mesh_config<dim>().min_level(min_level).max_level(max_level);

        // 1. Build initial MR mesh adapted to initial condition
        auto global_mesh = samurai::mra::make_mesh(box, config);
        auto u_global    = samurai::make_scalar_field<double>("u", global_mesh);
        init_field(u_global);
        samurai::make_bc<samurai::Dirichlet<1>>(u_global, 0.);

        auto init_adaptation = samurai::make_MRAdapt(u_global);
        auto mra_config   = samurai::mra_config().epsilon(2e-4);
        init_adaptation(mra_config);

        std::cout << "Initial MR mesh built. Number of cells: " << global_mesh.nb_cells() << std::endl;

        if (!no_save)
        {
            samurai::save(path, fmt::format("{}_global_init", filename), global_mesh, u_global);
        }

        // 2. Wrap adapted MR mesh into dynamic StarPU mesh
        using Config = typename decltype(global_mesh)::config_t;
        samurai::starpu_dynamic_naive::StarpuMRMesh<Config> starpu_mesh(global_mesh, nb_task);

        // 3. Create dynamic StarPU fields
        samurai::starpu_dynamic_naive::StarpuMRScalarField<decltype(starpu_mesh), double> u(u_global, starpu_mesh);
        samurai::starpu_dynamic_naive::StarpuMRScalarField<decltype(starpu_mesh), double> unp1("unp1", starpu_mesh);
        unp1.fill(0.0);

        samurai::starpu_dynamic_naive::make_bc<samurai::Dirichlet<1>>(u, 0.);
        samurai::starpu_dynamic_naive::make_bc<samurai::Dirichlet<1>>(unp1, 0.);

        // 4. Create MRadaptation orchestrator
        auto MRadaptation = samurai::starpu_dynamic_naive::make_MRAdapt(u, unp1);

        double dt            = cfl * starpu_mesh.min_cell_length();
        const double dt_save = Tf / static_cast<double>(nfiles);

        // Initial output
        if (!no_save)
        {
            const std::string suffix = (nfiles != 1) ? "_ite_0" : "_init";
            samurai::starpu_dynamic_naive::insert_save(path, fmt::format("{}{}", filename, suffix), starpu_mesh, u);
        }

        std::size_t nsave = 1;
        std::size_t nt    = 0;

        double total_adapt_time  = 0.0;
        double total_submit_time = 0.0;
        double total_wait_time   = 0.0;

        auto start_loop = std::chrono::steady_clock::now();


        auto cl_update = get_advection_update_codelet<decltype(u)>();

        while (t < Tf && nt < max_iter)
        {
            // 1. Wait for preceding StarPU tasks to measure execution wait time
            auto start_wait = std::chrono::steady_clock::now();
            starpu_task_wait_for_all();
            auto end_wait = std::chrono::steady_clock::now();
            total_wait_time += std::chrono::duration<double>(end_wait - start_wait).count();

            // 2. Dynamic AMR adaptation step
            auto start_adapt = std::chrono::steady_clock::now();
            if (use_adapt_mt)
            {
                MRadaptation.adapt_mt(mra_config);
            }
            else
            {
                MRadaptation(mra_config);
            }
            auto end_adapt = std::chrono::steady_clock::now();
            total_adapt_time += std::chrono::duration<double>(end_adapt - start_adapt).count();

            // 3. Update dt based on new cell size after adaptation
            dt = cfl * starpu_mesh.min_cell_length();
            t += dt;
            if (t > Tf)
            {
                dt += Tf - t;
                t = Tf;
            }

            auto start_submit = std::chrono::steady_clock::now();

            // 4. Multi-resolution ghost update DAG submission
            samurai::starpu_dynamic_naive::submit_update_ghost_mr(u);

            // 5. Submit advection flux update for each task
            for (int i = 0; i < nb_task; ++i)
            {
                UpdateArgs<decltype(u)> args = {
                    &u.get_field(i),
                    &unp1.get_field(i),
                    dt,
                    a[0],
                    a[1]
                };

                starpu_task_insert(cl_update,
                                   STARPU_R, u.get_handle(i),
                                   STARPU_W, unp1.get_handle(i),
                                   STARPU_VALUE, &args, sizeof(args),
                                   STARPU_TASK_COLOR, 0x00FF00,
                                   0);
            }

            // 6. Swap StarPU handles for double buffering
            samurai::starpu_dynamic_naive::swap(u, unp1);

            auto end_submit = std::chrono::steady_clock::now();
            total_submit_time += std::chrono::duration<double>(end_submit - start_submit).count();

            // Save intermediate step
            if (!no_save && (t >= static_cast<double>(nsave) * dt_save || t == Tf))
            {
                const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
                samurai::starpu_dynamic_naive::insert_save(path, fmt::format("{}{}", filename, suffix), starpu_mesh, u);
            }

            nt++;
        }

        auto start_final_wait = std::chrono::steady_clock::now();
        starpu_task_wait_for_all();
        auto end_final_wait = std::chrono::steady_clock::now();
        total_wait_time += std::chrono::duration<double>(end_final_wait - start_final_wait).count();

        auto end_loop = std::chrono::steady_clock::now();
        double total_simulation_time = std::chrono::duration<double>(end_loop - start_loop).count();

        std::cout << std::endl;
        std::cout << "========= PERFORMANCE STATS =========" << std::endl;
        std::cout << "Adaptation time:        " << total_adapt_time << " s" << std::endl;
        std::cout << "Task submission time:   " << total_submit_time << " s" << std::endl;
        std::cout << "Execution time (wait):  " << total_wait_time << " s" << std::endl;
        std::cout << "Total simulation time:  " << total_simulation_time << " s" << std::endl;
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
                    std::fprintf(file, "machine,version,label,num_threads,mpi_size,level,nb_tasks,max_iter,time,adapt_time\n");
                }

                char machine_name[256];
                if (gethostname(machine_name, 256) != 0)
                {
                    std::strcpy(machine_name, "Unknown");
                }

                int num_threads = starpu_cpu_worker_get_count();

                std::fprintf(file, "%s,starpu_dynamic_naive,%s,%d,%d,%d,%d,%d,%f,%f\n",
                             machine_name,
                             label.c_str(),
                             num_threads,
                             1,
                             static_cast<int>(max_level),
                             static_cast<int>(nb_task),
                             static_cast<int>(nt),
                             total_simulation_time,
                             total_adapt_time);

                std::fclose(file);
            }
        }
    }

    samurai::finalize();
    return 0;
}
