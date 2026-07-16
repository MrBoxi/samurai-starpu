// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <array>
#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <xtensor/containers/xfixed.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/stencil_field.hpp>
#include <samurai/subset/node.hpp>

#include <filesystem>
namespace fs = std::filesystem;

template <class Field>
void init(Field& u)
{
    auto& mesh = u.mesh();
    u.resize();

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
                u[cell] = 1;
            }
            else
            {
                u[cell] = 0;
            }
        });
}

template <class Field>
void save(const fs::path& path, const std::string& filename, const Field& u, const std::string& suffix = "")
{
    auto& mesh = u.mesh();

#ifdef SAMURAI_WITH_MPI
    mpi::communicator world;
    samurai::save(path, fmt::format("{}_size_{}{}", filename, world.size(), suffix), mesh, u);
#else
    samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, u);
    samurai::dump(path, fmt::format("{}_restart{}", filename, suffix), mesh, u);
#endif
}

namespace
{
    std::size_t max_iter = std::numeric_limits<std::size_t>::max();
    bool no_save = false;
    std::string save_perf_filename = "";
    std::string label = "none";
}

template <std::size_t pred_stencil_size>
int main_fct(bool first_run, int argc, char* argv[])
{
    constexpr std::size_t dim = 2;
    std::size_t level = 7;

    // Simulation parameters
    xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
    xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1., 1.};
    std::array<double, dim> a{
        {1, 1}
    };
    double Tf  = .1;
    double cfl = 0.5;
    double t   = 0.;
    std::string restart_file;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "FV_advection_2d";
    std::size_t nfiles   = 1;

    if (first_run)
    {
        auto& app = samurai::app;
        app.add_option("--min-corner", min_corner, "The min corner of the box")->capture_default_str()->group("Simulation parameters");
        app.add_option("--max-corner", max_corner, "The max corner of the box")->capture_default_str()->group("Simulation parameters");
        app.add_option("--velocity", a, "The velocity of the advection equation")->capture_default_str()->group("Simulation parameters");
        app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
        app.add_option("--Ti", t, "Initial time")->capture_default_str()->group("Simulation parameters");
        app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
        app.add_option("--restart-file", restart_file, "Restart file")->capture_default_str()->group("Simulation parameters");
        app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
        app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
        app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");
        app.add_option("--max-iter", max_iter, "Maximum number of iterations")->capture_default_str()->group("Simulation parameters");
        app.add_flag("--no-save", no_save, "Disable file saving")->group("Output");
        app.add_option("--save-perf", save_perf_filename, "Save the performance to the file")->group("Output");
        app.add_option("--label", label, "Label for the performance file")->group("Output");
    }
    SAMURAI_PARSE(argc, argv);



    if (samurai::args::max_level != std::numeric_limits<std::size_t>::max())
    {
        level = samurai::args::max_level;
    }
    else if (samurai::args::min_level != std::numeric_limits<std::size_t>::max())
    {
        level = samurai::args::min_level;
    }


    filename = fmt::format("{}_pred_{}", filename, pred_stencil_size);

    const samurai::Box<double, dim> box(min_corner, max_corner);
    auto config = samurai::mesh_config<dim, pred_stencil_size>().min_level(level).max_level(level).max_stencil_size(2).disable_minimal_ghost_width();
    auto mesh = samurai::mra::make_empty_mesh(config);
    auto u    = samurai::make_scalar_field<double>("u", mesh);

    if (restart_file.empty())
    {
        mesh = samurai::mra::make_mesh(box, config);
        init(u);
    }
    else
    {
        samurai::load(restart_file, mesh, u);
    }
    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);

    double dt            = cfl * mesh.min_cell_length();
    const double dt_save = Tf / static_cast<double>(nfiles);

    auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);

    auto MRadaptation = samurai::make_MRAdapt(u);
    auto mra_config   = samurai::mra_config().epsilon(2e-4);
    MRadaptation(mra_config);
    if (!no_save)
    {
        save(path, filename, u, "_init");
    }

    std::size_t nsave = 1;
    std::size_t nt    = 0;

    auto start_loop = std::chrono::steady_clock::now();

    while (t != Tf && nt < max_iter)
    {
        MRadaptation(mra_config);

        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }

        //std::cout << fmt::format("iteration {}: t = {}, dt = {}", nt, t, dt) << std::endl;

        samurai::update_ghost_mr(u);
        unp1.resize();
        unp1 = u - dt * samurai::upwind(a, u);

        std::swap(u.array(), unp1.array());

        if (!no_save && (t >= static_cast<double>(nsave) * dt_save || t == Tf))
        {
            const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save(path, filename, u, suffix);
        }


        nt++;
    }

    auto end_loop = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(end_loop - start_loop).count();

    bool is_rank_0 = true;
#ifdef SAMURAI_WITH_MPI
    mpi::communicator world;
    is_rank_0 = (world.rank() == 0);
#endif

    if (is_rank_0)
    {
        std::cout << std::endl;
        std::cout << "========= PERFORMANCE STATS =========" << std::endl;
        std::cout << "Execution time: " << total_time << " s" << std::endl;
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

                std::string version = "seq";
#if defined(SAMURAI_WITH_MPI)
                version = "mpi";
#elif defined(SAMURAI_WITH_OPENMP)
                version = "omp";
#endif

                int num_threads = 1;
#ifdef _OPENMP
                num_threads = omp_get_max_threads();
#endif

                int mpi_size = 1;
#ifdef SAMURAI_WITH_MPI
                mpi_size = world.size();
#endif

                std::fprintf(file, "%s,%s,%s,%d,%d,%d,%d,%d,%f\n",
                             machine_name,
                             version.c_str(),
                             label.c_str(),
                             num_threads,
                             mpi_size,
                             static_cast<int>(mesh.max_level()),
                             0, // nb_tasks (N/A for non-StarPU)
                             static_cast<int>(nt),
                             total_time);

                std::fclose(file);
            }
        }
    }
    return 0;
}

int main(int argc, char* argv[])
{
    samurai::initialize("Finite volume example for the advection equation in 2d using multiresolution", argc, argv);
    main_fct<0>(true, argc, argv);
    main_fct<1>(false, argc, argv);
    samurai::finalize();
    return 0;
}
