using HDF5


function save_graph(
    adjacency_matrix,
    path,
    metadata
)
    
    h5open(path, "w") do fid
        network_group = create_group(fid, "network")
        create_dataset(network_group, "adjacency_matrix", convert(Array{Bool}, adjacency_matrix))

        for key in keys(metadata)
            attributes(network_group)[key] = metadata[key]
        end
    end

end

