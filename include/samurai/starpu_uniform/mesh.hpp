#pragma once

#ifndef SAMURAI_WITH_STARPU
#error "The header file <samurai/starpu_uniform/mesh.hpp> should not be included if SAMURAI_WITH_STARPU is not defined."
#endif

#include <vector>
#include <samurai/uniform_mesh.hpp>
#include <samurai/box.hpp>

namespace samurai
{
    namespace starpu_uniform
    {
        template <class Config>
        class StarpuUniformMesh
        {
        public:
            using mesh_t = UniformMesh<Config>;
            using config_t = Config;
            static constexpr std::size_t dim = Config::dim;
            using interval_t = typename Config::interval_t;
            using cl_type = LevelCellList<dim, interval_t>;
            using ca_type = LevelCellArray<dim, interval_t>;

            struct IntersectionEntry
            {
                int src;
                int dst;
                ca_type intersection;
            };

            StarpuUniformMesh() = default;
            StarpuUniformMesh(const Box<double, dim>& box, std::size_t level, int nb_task);

            int get_nb_task() const { return m_nb_task; }
            mesh_t& get_mesh(int i) { return m_meshes[i]; }
            const mesh_t& get_mesh(int i) const { return m_meshes[i]; }

            mesh_t& get_global_mesh() { return m_global_mesh; }
            const mesh_t& get_global_mesh() const { return m_global_mesh; }

            const std::vector<IntersectionEntry>& get_intersections() const { return m_intersections; }

            double min_cell_length() const
            {
                return m_global_mesh.cell_length(m_level);
            }

            void swap(StarpuUniformMesh& other) noexcept
            {
                std::swap(m_nb_task, other.m_nb_task);
                std::swap(m_level, other.m_level);
                m_meshes.swap(other.m_meshes);
                m_global_mesh.swap(other.m_global_mesh);
                m_intersections.swap(other.m_intersections);
            }

        private:
            int m_nb_task = 0;
            std::size_t m_level = 0;
            std::vector<mesh_t> m_meshes;
            mesh_t m_global_mesh;
            std::vector<IntersectionEntry> m_intersections;
        };

        template <class Config>
        StarpuUniformMesh<Config>::StarpuUniformMesh(const Box<double, dim>& box, std::size_t level, int nb_task)
            : m_nb_task(nb_task)
            , m_level(level)
            , m_global_mesh(box, level)
        {
            auto& global_domain = m_global_mesh[UniformMeshId::cells];
            m_meshes.reserve(nb_task);

            for (int rank = 0; rank < nb_task; ++rank)
            {
                cl_type subdomain_cells(level, global_domain.origin_point(), global_domain.scaling_factor());
                std::size_t subdomain_start = 0;
                std::size_t subdomain_end = 0;

                if (dim == 1)
                {
                    std::size_t n_cells = global_domain.nb_cells();
                    std::size_t n_cells_per_subdomain = n_cells / static_cast<std::size_t>(nb_task);
                    subdomain_start = n_cells_per_subdomain * static_cast<std::size_t>(rank);
                    subdomain_end = n_cells_per_subdomain * (static_cast<std::size_t>(rank) + 1);
                    if (rank == nb_task - 1)
                    {
                        subdomain_end = n_cells;
                    }
                    std::size_t cell_counter = 0;
                    for_each_meshinterval(global_domain,
                                          [&](auto mi)
                                          {
                                              for (auto i = mi.i.start; i < mi.i.end; ++i)
                                              {
                                                  if (cell_counter >= subdomain_start && cell_counter < subdomain_end)
                                                  {
                                                      subdomain_cells[mi.index].add_point(i);
                                                  }
                                                  ++cell_counter;
                                              }
                                          });
                }
                else if (dim >= 2)
                {
                    auto subdomain_nb_intervals = global_domain.nb_intervals() / static_cast<std::size_t>(nb_task);
                    subdomain_start = static_cast<std::size_t>(rank) * subdomain_nb_intervals;
                    subdomain_end = (static_cast<std::size_t>(rank) + 1) * subdomain_nb_intervals;
                    if (rank == nb_task - 1)
                    {
                        subdomain_end = global_domain.nb_intervals();
                    }
                    std::size_t k = 0;
                    for_each_meshinterval(global_domain,
                                          [&](auto mi)
                                          {
                                              if (k >= subdomain_start && k < subdomain_end)
                                              {
                                                  subdomain_cells[mi.index].add_interval(mi.i);
                                              }
                                              ++k;
                                          });
                }

                // Construct ca_type first to avoid the bug in UniformMesh(const cl_type&)
                ca_type subdomain_ca(subdomain_cells);
                m_meshes.emplace_back(subdomain_ca);
            }

            // Compute intersections between subdomains
            for (int i = 0; i < nb_task; ++i)
            {
                for (int j = 0; j < nb_task; ++j)
                {
                    if (i == j) continue;
                    auto intersect = intersection(m_meshes[j][UniformMeshId::cells], m_meshes[i][UniformMeshId::cells_and_ghosts]);
                    ca_type ca_intersect(intersect);
                    if (ca_intersect.nb_cells() > 0)
                    {
                        m_intersections.push_back({j, i, ca_intersect});
                    }
                }
            }
        }
    }
}
