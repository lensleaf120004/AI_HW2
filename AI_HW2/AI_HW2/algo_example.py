import numpy as np

def random_search(loss_fn, start_points, field_size, steps=300):
    ships = np.array(start_points, dtype=float).copy()
    n_ships = len(ships)
    paths = [[] for _ in range(n_ships)]
    best_states = []

    for i, (x, y) in enumerate(ships):
        gpa = loss_fn(x, y)
        best_states.append([x, y, gpa])
        paths[i].append((x, y, gpa, x, y, gpa))
        
    # --- Homework algo_example: Implement Random Search ------
    
    # --- IMPLEMENTATION START ---
    
    for _ in range(steps):
        for i in range(n_ships):
            # pass
            x = np.random.uniform(field_size[0], field_size[1])
            y = np.random.uniform(field_size[0], field_size[1])
            ships[i] = x, y
            gpa = loss_fn(x, y)


        # --- IMPLEMENTATION END ---

            x, y = ships[i]
            gpa = loss_fn(x, y) 
            best_x, best_y, best_gpa = best_states[i]
            if gpa > best_gpa:
                best_x, best_y, best_gpa = x, y, gpa
                best_states[i] = [best_x, best_y, best_gpa]
            paths[i].append((x, y, gpa, best_x, best_y, best_gpa))
    return paths