using Distributions



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
            adjacency_matrix[i, rand(1: size(adjacency_matrix)[1])] = true
        end
    end
    
end





function main()


    p=0.5
    n_nodes = 100000

    t1 = time()
    adj_matrix = generate_uniform_random_graph(n_nodes, p)
    elapsed_time = time()-t1
    println(elapsed_time)


    connect_isolates!(adj_matrix)
end

main()