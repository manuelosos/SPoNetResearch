using Distributions
include("generate_graphs.jl")




"""
Generates a G(n,p) random graph with the geometric algorithm as described in DOI: 10.1103/PhysRevE.71.036113
The graph is represented as an adjacency matrix saved as an bitarray.
"""
function generate_uniform_random_graph_geometric(
    n_nodes::Int,
    edge_probability::Float64;
    flipped=false
)
    row_index:: Int = 1
    col_index:: Int = -1

    if flipped
        adj_matrix = trues(n_nodes, n_nodes)
    else
        adj_matrix = falses(n_nodes, n_nodes)
    end

    while row_index < n_nodes
        r = rand()
        col_index = col_index + 1 + floor(log(1 - r) / log(1 - edge_probability))
        while col_index >= row_index && row_index < n_nodes
            col_index -= row_index
            row_index += 1
        end
        if row_index < n_nodes
            adj_matrix[row_index+1, col_index+1] = !flipped 
            adj_matrix[col_index+1, row_index+1] = !flipped 

        end
    end
    return adj_matrix
end


function generate_uniform_random_graph(
    n_nodes::Int,
    edge_probability::Float64
)
    if edge_probability > 0.5
        return generate_uniform_random_graph_geometric(n_nodes, 1-edge_probability, flipped=true)
    else
        return generate_uniform_random_graph_geometric(n_nodes, edge_probability)
    end
end


function connect_isolates!(
    adjacency_matrix
)
    for i=1:size(adjacency_matrix)[1]
        if count(!=(false), adjacency_matrix[i]) == 0
            tmp = rand(1: size(adjacency_matrix)[1])
            adjacency_matrix[i, tmp] = true
            adjacency_matrix[tmp, i] = true
        end
    end
    
end


function generate_graphs()
	
	n_nodes_list = [10, 100]
	

	for n_nodes in n_nodes_list
        for n_nodes_crit in n_nodes_list
            edge_probability = log(n_nodes_crit)/n_nodes_crit
            adj_matrix = generate_uniform_random_graph(n_nodes, edge_probability)
            connect_isolates!(adj_matrix)
            
            graph_name = "ER_n$(n_nodes)_p-crit$(n_nodes_crit)"
            save_graph(adj_matrix, graph_name, Dict("edge_probability" => edge_probability))
#"/workdir/bt310056/data$(graph_name).hdf5"
        end
    end

end


function main()
	

	generate_graphs()

end

main()
