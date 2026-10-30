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
        enum class SubdomainPart : std::size_t
        {
            bottom_ghost = 0,
            interior     = 1,
            top_ghost    = 2
        };

        struct SubdomainGridInfo
        {
            int nx = 0;
            int ny_interior = 0;
            int row_stride = 0;
            int ghost_width = 1;
        };

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
                SubdomainPart dst_part;
                SubdomainPart src_part;
                std::size_t src_offset = 0;
                std::size_t copy_size = 0;
            };

            StarpuUniformMesh() = default;
            StarpuUniformMesh(const Box<double, dim>& box, std::size_t level, int nb_task);

            int get_nb_task() const { return m_nb_task; }
            mesh_t& get_mesh(int i) { return m_meshes[i]; }
            const mesh_t& get_mesh(int i) const { return m_meshes[i]; }

            mesh_t& get_global_mesh() { return m_global_mesh; }
            const mesh_t& get_global_mesh() const { return m_global_mesh; }

            const std::vector<IntersectionEntry>& get_intersections() const { return m_intersections; }

            const std::array<uint32_t, 3>& get_partition_sizes(int i) const { return m_partition_sizes[i]; }

            const SubdomainGridInfo& get_subdomain_grid_info(int i) const { return m_grid_infos[i]; }

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
                m_partition_sizes.swap(other.m_partition_sizes);
                m_grid_infos.swap(other.m_grid_infos);
            }

        private:
            int m_nb_task = 0;
            std::size_t m_level = 0;
            std::vector<mesh_t> m_meshes;
            mesh_t m_global_mesh;
            std::vector<IntersectionEntry> m_intersections;
            std::vector<std::array<uint32_t, 3>> m_partition_sizes;
            std::vector<SubdomainGridInfo> m_grid_infos;
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

            // Calculate partition sizes for each subdomain
            m_partition_sizes.resize(nb_task);
            m_grid_infos.resize(nb_task);
            std::vector<int> min_y_interior(nb_task, std::numeric_limits<int>::max());
            std::vector<int> max_y_interior(nb_task, std::numeric_limits<int>::min());

            for (int rank = 0; rank < nb_task; ++rank)
            {
                if constexpr (dim == 1)
                {
                    int min_x = std::numeric_limits<int>::max();
                    int max_x = std::numeric_limits<int>::min();
                    samurai::for_each_interval(
                        m_meshes[rank][UniformMeshId::cells],
                        [&](std::size_t, const auto& interval)
                        {
                            min_x = std::min(min_x, static_cast<int>(interval.start));
                            max_x = std::max(max_x, static_cast<int>(interval.end));
                        });
                    min_y_interior[rank] = min_x;
                    max_y_interior[rank] = max_x;

                    uint32_t bottom_size = 0;
                    uint32_t interior_size = 0;
                    uint32_t top_size = 0;

                    samurai::for_each_interval(
                        m_meshes[rank][UniformMeshId::cells_and_ghosts],
                        [&](std::size_t, const auto& interval)
                        {
                            for (auto x = interval.start; x < interval.end; ++x)
                            {
                                if (x < min_x)
                                {
                                    ++bottom_size;
                                }
                                else if (x >= max_x)
                                {
                                    ++top_size;
                                }
                                else
                                {
                                    ++interior_size;
                                }
                            }
                        });

                    m_partition_sizes[rank] = {bottom_size, interior_size, top_size};

                    SubdomainGridInfo info;
                    info.nx = static_cast<int>(interior_size);
                    info.ny_interior = 1;
                    info.row_stride = static_cast<int>(bottom_size + interior_size + top_size);
                    info.ghost_width = static_cast<int>(Config::ghost_width);
                    m_grid_infos[rank] = info;
                }
                else
                {
                    samurai::for_each_interval(
                        m_meshes[rank][UniformMeshId::cells],
                        [&](std::size_t, const auto&, const auto& jk)
                        {
                            int y = static_cast<int>(jk[0]);
                            min_y_interior[rank] = std::min(min_y_interior[rank], y);
                            max_y_interior[rank] = std::max(max_y_interior[rank], y);
                        });

                    uint32_t bottom_size = 0;
                    uint32_t interior_size = 0;
                    uint32_t top_size = 0;

                    samurai::for_each_interval(
                        m_meshes[rank][UniformMeshId::cells_and_ghosts],
                        [&](std::size_t, const auto& interval, const auto& jk)
                        {
                            int y = static_cast<int>(jk[0]);
                            uint32_t count = static_cast<uint32_t>(interval.size());
                            if (y < min_y_interior[rank])
                            {
                                bottom_size += count;
                            }
                            else if (y > max_y_interior[rank])
                            {
                                top_size += count;
                            }
                            else
                            {
                                interior_size += count;
                            }
                        });

                    m_partition_sizes[rank] = {bottom_size, interior_size, top_size};

                    int nx = 0;
                    samurai::for_each_interval(
                        m_meshes[rank][UniformMeshId::cells],
                        [&](std::size_t, const auto& interval, const auto&)
                        {
                            nx = static_cast<int>(interval.size());
                        });

                    int row_stride = 0;
                    samurai::for_each_interval(
                        m_meshes[rank][UniformMeshId::cells_and_ghosts],
                        [&](std::size_t, const auto& interval, const auto&)
                        {
                            row_stride = static_cast<int>(interval.size());
                        });

                    SubdomainGridInfo info;
                    info.nx = nx;
                    info.ny_interior = max_y_interior[rank] - min_y_interior[rank] + 1;
                    info.row_stride = row_stride;
                    info.ghost_width = static_cast<int>(Config::ghost_width);
                    m_grid_infos[rank] = info;
                }
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
                        SubdomainPart dst_part = SubdomainPart::bottom_ghost;
                        std::size_t src_offset = 0;
                        std::size_t copy_size = 0;

                        if constexpr (dim == 1)
                        {
                            if (j < i)
                            {
                                dst_part = SubdomainPart::bottom_ghost;
                                copy_size = m_partition_sizes[i][0];
                                src_offset = m_partition_sizes[j][1] - copy_size;
                            }
                            else
                            {
                                dst_part = SubdomainPart::top_ghost;
                                copy_size = m_partition_sizes[i][2];
                                src_offset = 0;
                            }
                        }
                        else
                        {
                            int first_y = 0;
                            samurai::for_each_interval(ca_intersect, [&](std::size_t, const auto&, const auto& jk) {
                                first_y = static_cast<int>(jk[0]);
                            });
                            if (first_y < min_y_interior[i])
                            {
                                dst_part = SubdomainPart::bottom_ghost;
                                copy_size = m_partition_sizes[i][0];
                                src_offset = m_partition_sizes[j][1] - copy_size;
                            }
                            else
                            {
                                dst_part = SubdomainPart::top_ghost;
                                copy_size = m_partition_sizes[i][2];
                                src_offset = 0;
                            }
                        }
                        SubdomainPart src_part = SubdomainPart::interior;
                        m_intersections.push_back({j, i, ca_intersect, dst_part, src_part, src_offset, copy_size});
                    }
                }
            }
        }
    }
}
